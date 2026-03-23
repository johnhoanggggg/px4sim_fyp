#!/usr/bin/env python3
"""
Offboard control with VFH3D obstacle avoidance.

Same waypoint mission as fly_truss.py, but uses velocity commands driven
by the VFH3D algorithm to steer around obstacles detected by the 12 ToF
sensors.

Usage:
    1. Start PX4 SITL:  cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss make px4_sitl gz_x500_tof
    2. Run this script: python3 fly_avoid.py
"""

import time
import math
import socket
import threading
import sys

import numpy as np
from pymavlink import mavutil

from tof_reader import TofReader
from vfh3d import VFH3D
from gz_markers import GzMarkerViz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SPEED = 0.5          # m/s max avoidance velocity
SAFE_DISTANCE = 1.0      # m — obstacle proximity for slowdown
CONTROL_HZ = 10          # avoidance loop rate
WAYPOINT_TOL = 0.4       # m — switch to next waypoint within this distance

# Waypoints: (x, y, z_ned, label)
WAYPOINTS = [
    (0.0,  0.0, -1.5, "Rising into truss (z=2.0m)"),
    (0.0, -3.0, -1.5, "Flying to y=-3"),
    (0.0, -1.0, -1.5, "Flying to y=-1"),
    (0.0,  1.0, -1.5, "Flying to y=+1"),
    (0.0,  3.0, -1.5, "Flying to y=+3"),
    (0.0,  0.0, -1.5, "Returning to centre"),
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

# Request position stream at 10Hz on the recv channel
recv.mav.request_data_stream_send(
    TARGET_SYS, TARGET_COMP,
    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
    10, 1)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
running = True
use_pos_setpoints = True        # False during velocity-based avoidance
target = [0.0, 0.0, -1.0, 0.0]  # x, y, z, yaw — shared setpoint
target_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Sensor + avoidance modules
# ---------------------------------------------------------------------------
tof = TofReader()
vfh = VFH3D(
    resolution_deg=10,
    threshold_low=0.3,
    threshold_high=0.6,
    safe_distance=SAFE_DISTANCE,
    max_speed=MAX_SPEED,
)
viz = GzMarkerViz()

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def set_pos(x, y, z, yaw=0):
    """Send position setpoint in NED frame."""
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z, 0, 0, 0, 0, 0, 0, yaw, 0)


def set_vel(vx, vy, vz, yaw=0.0):
    """Send velocity setpoint in NED frame."""
    # ignore position (0-2), USE velocity (3-5), ignore accel (6-8),
    # no force (9), USE yaw (10), ignore yaw_rate (11)
    type_mask = 0b0000_1001_1100_0111
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        yaw, 0)


def get_position():
    """Read latest position and heading from PX4 (call from main thread only)."""
    msg = recv.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
    if msg:
        return msg.x, msg.y, msg.z
    return None


def get_yaw():
    """Read latest yaw (heading) from PX4."""
    msg = recv.recv_match(type='ATTITUDE', blocking=False)
    if msg:
        return msg.yaw
    return 0.0

# ---------------------------------------------------------------------------
# Background threads — only use 'send', never 'recv'
# ---------------------------------------------------------------------------

def setpoint_loop():
    """Stream position setpoints at 20Hz when use_pos_setpoints is True."""
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
    """Send GCS heartbeat at 2 Hz."""
    while running:
        try:
            send.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0)
        except Exception:
            pass
        time.sleep(0.5)

# ---------------------------------------------------------------------------
# Flight control helpers
# ---------------------------------------------------------------------------

def set_target(x, y, z, yaw=0):
    with target_lock:
        target[0] = x
        target[1] = y
        target[2] = z
        target[3] = yaw


def wait_until_reached(x, y, z, tolerance=WAYPOINT_TOL, timeout=15):
    """Block until position is within tolerance. Reads recv on main thread."""
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


def fly_with_avoidance(waypoints):
    """
    Navigate waypoints using VFH3D obstacle avoidance.

    Runs on the main thread. Reads position from recv, reads ToF from
    tof_reader, computes VFH3D velocity, and overrides the setpoint.
    """
    global use_pos_setpoints
    use_pos_setpoints = False  # stop position setpoints, we send velocity
    for i, (wx, wy, wz, label) in enumerate(waypoints):
        print(f"\n>>> [{i+1}/{len(waypoints)}] {label}")
        t0 = time.time()

        while time.time() - t0 < 30:
            # Read position (main thread owns recv)
            pos = get_position()
            if not pos:
                continue
            px, py, pz = pos

            # Check if waypoint reached
            dist = math.sqrt((px - wx)**2 + (py - wy)**2 + (pz - wz)**2)
            if dist < WAYPOINT_TOL:
                print(f"  Reached! dist={dist:.2f}")
                break

            # Read yaw for frame conversions
            yaw = get_yaw()
            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)

            # Goal direction in NED → rotate to body FRD → convert to FLU
            goal_ned = (wx - px, wy - py, wz - pz)
            # NED to body FRD (inverse yaw rotation)
            frd_x =  c_yaw * goal_ned[0] + s_yaw * goal_ned[1]
            frd_y = -s_yaw * goal_ned[0] + c_yaw * goal_ned[1]
            frd_z = goal_ned[2]
            # FRD to FLU: (x, -y, -z)
            goal_body = (frd_x, -frd_y, -frd_z)

            # Get obstacle points (already in body FLU frame)
            pts = tof.get_obstacle_points(max_range=4.0)

            # Filter out ground detections (body Z < -1.0m)
            if len(pts) > 0:
                pts = pts[pts[:, 2] > -1.0]

            # Visualize obstacle points in Gazebo
            if len(pts) > 0:
                viz.update(pts, (px, py, pz), yaw)

            # Compute velocity in body frame
            if len(pts) > 0:
                vel_body = vfh.update(pts, goal_body)
            else:
                goal_dist = math.sqrt(goal_ned[0]**2 + goal_ned[1]**2 + goal_ned[2]**2)
                if goal_dist > 0.01:
                    scale = min(MAX_SPEED, goal_dist) / goal_dist
                    vel_body = (goal_body[0] * scale, goal_body[1] * scale, goal_body[2] * scale)
                else:
                    vel_body = (0.0, 0.0, 0.0)

            # Convert velocity: body FLU → FRD → rotate by yaw → NED
            frd_x_out = vel_body[0]
            frd_y_out = -vel_body[1]   # FLU to FRD
            frd_z_out = -vel_body[2]
            vel = (
                c_yaw * frd_x_out - s_yaw * frd_y_out,  # NED x
                s_yaw * frd_x_out + c_yaw * frd_y_out,  # NED y
                frd_z_out,                                # NED z
            )

            # Send velocity command in NED
            try:
                set_vel(vel[0], vel[1], vel[2])
            except Exception:
                pass

            min_obs = float('inf')
            if len(pts) > 0:
                dists_obs = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
                min_obs = float(np.min(dists_obs))
            print(f"  pos: ({px:.2f},{py:.2f},{-pz:.2f}m up) dist:{dist:.2f} yaw:{math.degrees(yaw):.0f}° "
                  f"vel: ({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) obs:{len(pts)} "
                  f"min_obs:{min_obs:.2f}")
            time.sleep(1.0 / CONTROL_HZ)

        # Brief hold at waypoint using position control
        use_pos_setpoints = True
        set_target(wx, wy, wz)
        print(f"  Holding for 2s...")
        time.sleep(2)
        use_pos_setpoints = False  # resume velocity for next waypoint

    use_pos_setpoints = True  # re-enable for landing


def land_and_disarm():
    """Land, wait for touchdown, disarm."""
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
    print("Connections closed.")


# ---------------------------------------------------------------------------
# Main — all recv access stays on this thread
# ---------------------------------------------------------------------------
try:
    # Start background threads (only use 'send', never 'recv')
    for fn in (setpoint_loop, heartbeat_loop):
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    # Wait for sensor data
    print("Waiting for ToF sensor data (up to 5s)...")
    t0 = time.time()
    while not tof.has_data() and time.time() - t0 < 5:
        time.sleep(0.5)
    if tof.has_data():
        print("ToF data received!")
    else:
        print("No ToF data yet — proceeding anyway")

    # Stream setpoints (background thread already running)
    print("Streaming setpoints for 4s...")
    time.sleep(4)

    # Set OFFBOARD mode
    print("Setting OFFBOARD mode...")
    send.mav.set_mode_send(
        TARGET_SYS,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6 << 16)
    time.sleep(1)

    # Arm
    print("Arming...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)
    time.sleep(2)

    # --- Takeoff using position control ---
    print("\n>>> Takeoff to truss altitude...")
    set_target(0, 0, -1.5)
    wait_until_reached(0, 0, -1.5, tolerance=0.3, timeout=15)

    # --- Fly waypoints with VFH3D avoidance ---
    print("\nEnabling VFH3D obstacle avoidance...")
    vfh.reset()
    fly_with_avoidance(WAYPOINTS[1:])  # skip takeoff waypoint

    # --- Land ---
    land_and_disarm()

except KeyboardInterrupt:
    print("\nInterrupted! Landing...")
    land_and_disarm()
finally:
    shutdown()
