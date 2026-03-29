#!/usr/bin/env python3
"""
Offboard control with bidirectional 3-D DWA obstacle avoidance through
a multi-layer roof truss lattice.

DWA3D scores all directions on a spherical grid with a weighted cost
function and can back out of tight spaces.

Usage:
    1. Start PX4 SITL:
         cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss2 make px4_sitl gz_x500_tof
    2. Run this script:
         python3 fly_truss2_dwa.py
"""

import time
import math
import socket
import threading
import multiprocessing as mp

import numpy as np
from pymavlink import mavutil

from tof_reader import TofReader
from dwa3d import DWA3D
from viz2d import run_viz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SPEED = 0.5          # max speed (m/s)
SAFE_DISTANCE = 1.0
CONTROL_HZ = 10
WAYPOINT_TOL = 0.6
VEL_SMOOTH = 0.3         # EMA alpha for velocity smoothing

# Waypoints in NED (north, east, down, label)
#
# Structure (ENU): 7 pitched timber trusses running E-W at y=5,6.8..15.8
#   Span x=-3.5..3.5, bottom chord z=1.5, peak z=3.5
#   Collar ties: t0(z=2.5), t2(z=3.0), t3(z=2.5), t5(z=3.0), t6(z=2.5)
#   Drone spawns at ENU (0,0) = 5 m south of first truss.
#   NED: north=enu_y, east=enu_x, down=-enu_z
WAYPOINTS = [
    ( 0.0,   0.0,  -1.2,  "Takeoff south of structure"),
    ( 5.9,   1.0,  -2.0,  "Bay 0-1: right side, below collar z=2.5"),
    ( 7.7,   0.0,  -3.0,  "Bay 1-2: fly high, no collar on t1"),
    ( 9.5,  -1.0,  -2.5,  "Bay 2-3: left, below t2 collar z=3.0"),
    (11.3,   1.0,  -2.5,  "Bay 3-4: right, avoid noggin x=-1.5"),
    (13.1,   0.0,  -3.2,  "Bay 4-5: high, no collar on t4"),
    (14.9,  -0.5,  -2.0,  "Bay 5-6: low, below t5 collar z=3.0"),
    (16.5,   0.0,  -2.5,  "Exit north"),
    (10.4,   0.0,  -4.0,  "Return high over structure"),
    ( 0.0,   0.0,  -1.2,  "Home"),
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
fgm = DWA3D(
    n_az=72,
    n_el=18,
    max_range=1.5,
    bubble_radius=0.2,
    safe_distance=SAFE_DISTANCE,
    max_speed=MAX_SPEED,
    w_goal=1.0,
    w_obstacle=1,
    w_smooth=0.3,
    w_reverse=0.1,
    el_max_deg=89.0,
)

# Visualizer (separate process) — reuses the 2-D top-down view
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


# Truss member positions in NED for visualizer (king posts + web verticals)
# ENU (x,y) → NED (north=y, east=x)
_TRUSS_Y_VALS = (5.0, 6.8, 8.6, 10.4, 12.2, 14.0, 15.8)
_COL_POSITIONS_NED = []
for _y in _TRUSS_Y_VALS:
    _COL_POSITIONS_NED.append((_y, 0))        # king post
    _COL_POSITIONS_NED.append((_y, -1.75))    # left web vertical
    _COL_POSITIONS_NED.append((_y, 1.75))     # right web vertical


def _push_viz(drone_n, drone_e, yaw, current_wp_idx):
    """Send state to visualizer (non-blocking)."""
    bins = fgm.get_histogram()
    chosen = fgm.get_chosen_direction()
    wps_ne = [(w[0], w[1]) for w in WAYPOINTS]
    pkt = {
        "drone_n": drone_n,
        "drone_e": drone_e,
        "yaw": yaw,
        "bins": bins,
        "chosen": chosen,
        "waypoints": wps_ne,
        "current_wp": current_wp_idx,
        "obstacles": _COL_POSITIONS_NED,
        "obstacle_radius": 0.08,
        "sphere": fgm.get_sphere_data(),
    }
    try:
        viz_queue.put_nowait(pkt)
    except Exception:
        pass


def fly_with_avoidance(waypoints, wp_offset=0):
    """Navigate waypoints using 3-D FGM obstacle avoidance."""
    global use_pos_setpoints
    use_pos_setpoints = False

    prev_vel_ned = (0.0, 0.0, 0.0)   # for EMA smoothing (vx, vy, vz)

    for i, (wx, wy, wz, label) in enumerate(waypoints):
        wp_idx = i + wp_offset
        print(f"\n>>> [{wp_idx+1}/{len(WAYPOINTS)}] {label}  (DWA3D)")
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
            goal_body = (frd_x, -frd_y, -goal_ned[2])  # FRD→FLU: negate y; NED_down→FLU_up: negate z

            # Obstacles
            pts = tof.get_obstacle_points(max_range=2.0)

            # DWA3D — handles both obstacle and no-obstacle cases
            vel_body = fgm.update(pts, goal_body)

            # Body FLU → NED
            frd_vx = vel_body[0]
            frd_vy = -vel_body[1]  # FLU→FRD
            raw_vn = c_yaw * frd_vx - s_yaw * frd_vy
            raw_ve = s_yaw * frd_vx + c_yaw * frd_vy
            raw_vd = -vel_body[2]  # FLU up → NED down

            # Hard height ceiling: don't climb above waypoint altitude
            alt_error = pz - wz  # negative = drone above target
            if alt_error < -0.3:
                raw_vd = max(raw_vd, 0.5)
            elif alt_error < 0:
                raw_vd = max(raw_vd, 0.0)

            # EMA smoothing (all axes)
            a = VEL_SMOOTH
            vel_ned = (
                a * raw_vn + (1 - a) * prev_vel_ned[0],
                a * raw_ve + (1 - a) * prev_vel_ned[1],
                a * raw_vd + (1 - a) * prev_vel_ned[2],
            )
            prev_vel_ned = vel_ned

            try:
                set_vel(vel_ned[0], vel_ned[1], vel_ned[2])
            except Exception:
                pass

            _push_viz(px, py, yaw, wp_idx)

            min_obs = float('inf')
            if len(pts) > 0:
                min_obs = float(np.min(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)))
            print(f"  pos:({px:.2f},{py:.2f},{-pz:.2f}m) d:{dist:.2f} "
                  f"vel:({vel_ned[0]:.2f},{vel_ned[1]:.2f},vz:{-vel_ned[2]:.2f}) "
                  f"obs:{len(pts)} min:{min_obs:.2f}")

            time.sleep(1.0 / CONTROL_HZ)

        # Hold at waypoint (clamp altitude — never hold above waypoint)
        use_pos_setpoints = True
        pos = get_position()
        hold_z = wz if not pos else max(pos[2], wz)  # max because NED down: more negative = higher
        set_target(wx, wy, hold_z)
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
    set_target(0, 0, -1.2)
    wait_until_reached(0, 0, -1.2, tolerance=0.4, timeout=15)

    # Fly through truss lattice with 3-D FGM
    print(f"\nEnabling 3-D FGM obstacle avoidance (bubble={fgm.bubble_radius}m)...")
    fgm.reset()
    fly_with_avoidance(WAYPOINTS[1:], wp_offset=1)

    land_and_disarm()

except KeyboardInterrupt:
    print("\nInterrupted! Landing...")
    land_and_disarm()
finally:
    shutdown()
