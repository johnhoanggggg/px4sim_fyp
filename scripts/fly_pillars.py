#!/usr/bin/env python3
"""
Offboard control with 2D VFH obstacle avoidance through a pillar field.

Launches a separate matplotlib window showing a top-down view with
VFH2D histogram bins, drone position, and the environment.

Usage:
    1. Start PX4 SITL:
         cd ~/PX4-Autopilot && PX4_GZ_WORLD=pillars make px4_sitl gz_x500_tof
    2. Run this script:
         python3 fly_pillars.py
"""

import time
import math
import socket
import threading
import multiprocessing as mp

import numpy as np
from pymavlink import mavutil

from tof_reader import TofReader
from vfh2d import VFH2D
from viz2d import run_viz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SPEED = 0.4
SAFE_DISTANCE = 1.2
CONTROL_HZ = 10
WAYPOINT_TOL = 0.5

# Waypoints in NED (north, east, down, label)
WAYPOINTS = [
    (0.0,   0.0, -1.5, "Takeoff"),
    (4.0,   0.0, -1.5, "Through first gap"),
    (8.0,   1.0, -1.5, "Weave right"),
    (11.0,  0.0, -1.5, "Far end"),
    (8.0,  -1.0, -1.5, "Weave back left"),
    (4.0,   0.0, -1.5, "Return through gap"),
    (0.0,   0.0, -1.5, "Home"),
]

# ---------------------------------------------------------------------------
# MAVLink connections
# ---------------------------------------------------------------------------
recv = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
recv.port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv.wait_heartbeat()
print(f"Connected: system {recv.target_system}")

send = mavutil.mavlink_connection('udpout:127.0.0.1:14580',
                                  source_system=255, source_component=0)

TARGET_SYS = recv.target_system
TARGET_COMP = recv.target_component

recv.mav.request_data_stream_send(
    TARGET_SYS, TARGET_COMP,
    mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
running = True
use_pos_setpoints = True
target = [0.0, 0.0, -1.0, 0.0]
target_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
tof = TofReader()
vfh = VFH2D(
    resolution_deg=10,
    threshold_low=0.15,
    threshold_high=0.3,
    safe_distance=SAFE_DISTANCE,
    max_speed=MAX_SPEED,
    drone_radius=0.35,
    enlarge_bins=2,
)

# Visualizer (separate process)
viz_queue: mp.Queue = mp.Queue(maxsize=5)
viz_proc = mp.Process(target=run_viz, args=(viz_queue,), daemon=True)
viz_proc.start()

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def set_pos(x, y, z, yaw=0):
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z, 0, 0, 0, 0, 0, 0, yaw, 0)


def set_vel(vx, vy, vz, yaw=0.0):
    type_mask = 0b0000_1001_1100_0111
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0, vx, vy, vz, 0, 0, 0, yaw, 0)


_latest_yaw = 0.0


def get_pos_and_yaw():
    global _latest_yaw
    pos = None
    while True:
        msg = recv.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=(pos is None), timeout=1)
        if msg is None:
            break
        if msg.get_type() == 'LOCAL_POSITION_NED':
            pos = (msg.x, msg.y, msg.z)
        elif msg.get_type() == 'ATTITUDE':
            _latest_yaw = msg.yaw
    return pos, _latest_yaw


def get_position():
    pos, _ = get_pos_and_yaw()
    return pos


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def setpoint_loop():
    while running:
        if use_pos_setpoints:
            with target_lock:
                x, y, z, yaw = target
            try:
                set_pos(x, y, z, yaw)
            except Exception:
                pass
        time.sleep(0.05)


def heartbeat_loop():
    while running:
        try:
            send.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except Exception:
            pass
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Flight helpers
# ---------------------------------------------------------------------------

def set_target(x, y, z, yaw=0):
    with target_lock:
        target[0] = x; target[1] = y; target[2] = z; target[3] = yaw


def wait_until_reached(x, y, z, tolerance=WAYPOINT_TOL, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pos = get_position()
        if pos:
            dist = ((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2)**0.5
            print(f"  pos: ({pos[0]:.2f}, {pos[1]:.2f}, {-pos[2]:.2f}m up) dist: {dist:.2f}")
            if dist < tolerance:
                return True
        time.sleep(0.1)
    print("  Timeout reaching position")
    return False


def _push_viz(drone_n, drone_e, yaw, current_wp_idx):
    """Send state to visualizer (non-blocking)."""
    bins = vfh.get_histogram()
    chosen = vfh.get_chosen_direction()
    wps_ne = [(w[0], w[1]) for w in WAYPOINTS]
    pkt = {
        "drone_n": drone_n,
        "drone_e": drone_e,
        "yaw": yaw,
        "bins": bins,
        "chosen": chosen,
        "waypoints": wps_ne,
        "current_wp": current_wp_idx,
    }
    try:
        viz_queue.put_nowait(pkt)
    except Exception:
        pass  # drop if full


def fly_with_avoidance(waypoints, wp_offset=0):
    """Navigate waypoints using 2D VFH obstacle avoidance."""
    global use_pos_setpoints
    use_pos_setpoints = False

    for i, (wx, wy, wz, label) in enumerate(waypoints):
        wp_idx = i + wp_offset
        print(f"\n>>> [{wp_idx+1}/{len(WAYPOINTS)}] {label}")
        t0 = time.time()

        while time.time() - t0 < 60:
            pos, yaw = get_pos_and_yaw()
            if not pos:
                continue
            px, py, pz = pos

            dist = math.sqrt((px - wx)**2 + (py - wy)**2 + (pz - wz)**2)
            if dist < WAYPOINT_TOL:
                print(f"  Reached! dist={dist:.2f}")
                break

            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)

            # Goal NED → body FLU
            goal_ned = (wx - px, wy - py, wz - pz)
            frd_x =  c_yaw * goal_ned[0] + s_yaw * goal_ned[1]
            frd_y = -s_yaw * goal_ned[0] + c_yaw * goal_ned[1]
            goal_body = (frd_x, -frd_y)  # FRD→FLU: negate y

            # Obstacles
            pts = tof.get_obstacle_points(max_range=2.0)
            if len(pts) > 0:
                pts = pts[pts[:, 2] > -1.0]  # filter ground

            # VFH2D
            if len(pts) > 0:
                vel_body = vfh.update(pts, goal_body)
            else:
                goal_h = math.sqrt(goal_body[0]**2 + goal_body[1]**2)
                if goal_h > 0.01:
                    s = min(MAX_SPEED, goal_h) / goal_h
                    vel_body = (goal_body[0] * s, goal_body[1] * s)
                else:
                    vel_body = (0.0, 0.0)

            # Body FLU (vx, vy) → NED
            frd_vx = vel_body[0]
            frd_vy = -vel_body[1]  # FLU→FRD
            vel_ned = (
                c_yaw * frd_vx - s_yaw * frd_vy,
                s_yaw * frd_vx + c_yaw * frd_vy,
            )

            # Hold altitude via z velocity toward target
            vz = max(-0.5, min(0.5, (wz - pz) * 1.0))

            try:
                set_vel(vel_ned[0], vel_ned[1], vz)
            except Exception:
                pass

            # Push to visualizer
            _push_viz(px, py, yaw, wp_idx)

            min_obs = float('inf')
            if len(pts) > 0:
                min_obs = float(np.min(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)))
            print(f"  pos:({px:.2f},{py:.2f},{-pz:.2f}m) d:{dist:.2f} "
                  f"vel:({vel_ned[0]:.2f},{vel_ned[1]:.2f}) "
                  f"obs:{len(pts)} min:{min_obs:.2f}")

            time.sleep(1.0 / CONTROL_HZ)

        # Hold at waypoint
        use_pos_setpoints = True
        set_target(wx, wy, wz)
        _push_viz(wx, wy, yaw, wp_idx)
        print(f"  Holding 2s...")
        time.sleep(2)
        use_pos_setpoints = False

    use_pos_setpoints = True


def land_and_disarm():
    print("\n>>> Landing...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    t0 = time.time()
    while time.time() - t0 < 20:
        pos = get_position()
        if pos and -pos[2] < 0.15:
            print("  Touchdown!")
            break
        time.sleep(0.5)
    time.sleep(2)
    print("Disarming...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)
    time.sleep(1)


def shutdown():
    global running
    running = False
    time.sleep(0.3)
    for conn in (recv, send):
        try:
            conn.close()
        except Exception:
            pass
        try:
            conn.port.close()
        except Exception:
            pass
    viz_proc.terminate()
    print("Connections closed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try:
    for fn in (setpoint_loop, heartbeat_loop):
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    print("Waiting for ToF sensor data (up to 5s)...")
    t0 = time.time()
    while not tof.has_data() and time.time() - t0 < 5:
        time.sleep(0.5)
    if tof.has_data():
        print("ToF data received!")
    else:
        print("No ToF data yet — proceeding anyway")

    print("Streaming setpoints for 4s...")
    time.sleep(4)

    print("Setting OFFBOARD mode...")
    send.mav.set_mode_send(
        TARGET_SYS,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6 << 16)
    time.sleep(1)

    print("Arming...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)

    # Takeoff
    print("\n>>> Takeoff...")
    set_target(0, 0, -1.5)
    wait_until_reached(0, 0, -1.5, tolerance=0.3, timeout=15)

    # Fly pillar field with 2D VFH
    print("\nEnabling 2D VFH obstacle avoidance...")
    vfh.reset()
    fly_with_avoidance(WAYPOINTS[1:], wp_offset=1)

    land_and_disarm()

except KeyboardInterrupt:
    print("\nInterrupted! Landing...")
    land_and_disarm()
finally:
    shutdown()
