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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SPEED = 0.5          # m/s max avoidance velocity
SAFE_DISTANCE = 0.5      # m — obstacle proximity for slowdown
CONTROL_HZ = 10          # avoidance loop rate
WAYPOINT_TOL = 0.4       # m — switch to next waypoint within this distance
TAKEOFF_ALT = -1.5       # NED z for truss centre (z=2.0m up = -2.0 NED, but
                          # corridor midpoint is ~2.0m so -1.5 to -2.0)

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

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
running = True
avoidance_enabled = False        # disabled until airborne
pos_target = [0.0, 0.0, -1.0, 0.0]  # x, y, z, yaw — position setpoint for pre-avoidance
current_pos = [0.0, 0.0, 0.0]   # NED position
current_vel_cmd = [0.0, 0.0, 0.0]  # velocity command from avoidance
pos_lock = threading.Lock()
vel_lock = threading.Lock()
target_lock = threading.Lock()
waypoint_idx = 0
wp_lock = threading.Lock()

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

# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def send_velocity(vx, vy, vz, yaw=0.0):
    """Send velocity setpoint in NED frame."""
    # type_mask: ignore position (bits 0-2), use velocity (bits 3-5 = 0),
    # ignore acceleration (bits 6-8), use yaw (bit 10 = 0), ignore yaw rate (bit 11)
    type_mask = 0b0000_1100_0000_0111
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,       # position (ignored)
        vx, vy, vz,    # velocity
        0, 0, 0,        # acceleration (ignored)
        yaw, 0)


def send_position(x, y, z, yaw=0.0):
    """Send position setpoint in NED frame (used for takeoff)."""
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z, 0, 0, 0, 0, 0, 0, yaw, 0)


def get_position():
    """Read latest position from PX4."""
    msg = recv.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.5)
    if msg:
        return msg.x, msg.y, msg.z
    return None

# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def setpoint_loop():
    """Stream position setpoints at 20Hz when avoidance is disabled.
    Keeps PX4 in offboard mode during takeoff/landing."""
    while running:
        if not avoidance_enabled:
            with target_lock:
                x, y, z, yaw = pos_target
            try:
                send_position(x, y, z, yaw)
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


def position_loop():
    """Read position from PX4 as fast as possible."""
    while running:
        pos = get_position()
        if pos:
            with pos_lock:
                current_pos[0] = pos[0]
                current_pos[1] = pos[1]
                current_pos[2] = pos[2]
        time.sleep(0.02)


def avoidance_loop():
    """
    Main avoidance control loop at CONTROL_HZ.

    Reads sensor data, computes VFH3D velocity, sends commands.
    Only active when avoidance_enabled is True; otherwise sends
    position setpoints so takeoff works normally.
    """
    dt = 1.0 / CONTROL_HZ
    while running:
        if not avoidance_enabled:
            # Not yet airborne — let main thread handle position control
            time.sleep(dt)
            continue

        # Current position
        with pos_lock:
            px, py, pz = current_pos[0], current_pos[1], current_pos[2]

        # Current waypoint
        with wp_lock:
            idx = waypoint_idx
        if idx >= len(WAYPOINTS):
            # Mission complete — hover
            send_velocity(0, 0, 0)
            time.sleep(dt)
            continue

        wx, wy, wz, _ = WAYPOINTS[idx]

        # Goal direction in NED (vector from current position to waypoint)
        goal = (wx - px, wy - py, wz - pz)
        goal_dist = math.sqrt(goal[0]**2 + goal[1]**2 + goal[2]**2)

        if goal_dist < 0.01:
            goal = (0.0, 0.0, 0.0)

        # Get obstacle points from ToF sensors
        pts = tof.get_obstacle_points(max_range=4.0)

        # Run VFH3D
        if len(pts) > 0:
            vel = vfh.update(pts, goal)
        else:
            # No sensor data — fly direct toward goal (capped at max speed)
            if goal_dist > 0.01:
                scale = min(MAX_SPEED, goal_dist) / goal_dist
                vel = (goal[0] * scale, goal[1] * scale, goal[2] * scale)
            else:
                vel = (0.0, 0.0, 0.0)

        # Send velocity command
        try:
            send_velocity(vel[0], vel[1], vel[2])
        except Exception:
            pass

        # Store for display
        with vel_lock:
            current_vel_cmd[0] = vel[0]
            current_vel_cmd[1] = vel[1]
            current_vel_cmd[2] = vel[2]

        time.sleep(dt)

# ---------------------------------------------------------------------------
# Flight control
# ---------------------------------------------------------------------------

def wait_until_reached(x, y, z, tolerance=WAYPOINT_TOL, timeout=30):
    """Block until position is within tolerance of target."""
    t0 = time.time()
    while running and time.time() - t0 < timeout:
        with pos_lock:
            px, py, pz = current_pos[0], current_pos[1], current_pos[2]
        dist = math.sqrt((px - x)**2 + (py - y)**2 + (pz - z)**2)
        with vel_lock:
            vx, vy, vz = current_vel_cmd[0], current_vel_cmd[1], current_vel_cmd[2]
        pts = tof.get_obstacle_points()
        print(f"  pos: ({px:.2f}, {py:.2f}, {-pz:.2f}m up)  "
              f"dist: {dist:.2f}  vel: ({vx:.2f},{vy:.2f},{vz:.2f})  "
              f"obs_pts: {len(pts)}")
        if dist < tolerance:
            return True
        time.sleep(0.5)
    print("  Timeout reaching position")
    return False


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
# Main
# ---------------------------------------------------------------------------
try:
    # Start background threads
    for fn in (setpoint_loop, heartbeat_loop, position_loop, avoidance_loop):
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    # Wait for sensor data (optional — proceed even without it)
    print("Waiting for ToF sensor data (up to 5s)...")
    t0 = time.time()
    while not tof.has_data() and time.time() - t0 < 5:
        time.sleep(0.5)
    if tof.has_data():
        print("ToF data received!")
    else:
        print("No ToF data yet — proceeding anyway (will use direct waypoint nav)")

    # Stream position setpoints for PX4 offboard mode requirement
    # (setpoint_loop thread is already streaming pos_target at 20Hz)
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

    # --- Takeoff using position control (no avoidance yet) ---
    print("\n>>> Takeoff to truss altitude using position control...")
    takeoff_z = WAYPOINTS[0][2]  # -1.5 NED
    with target_lock:
        pos_target[2] = takeoff_z
    t0 = time.time()
    while time.time() - t0 < 15:
        with pos_lock:
            pz = current_pos[2]
        alt = -pz
        print(f"  Climbing... alt={alt:.2f}m  target={-takeoff_z:.2f}m")
        if abs(pz - takeoff_z) < 0.3:
            print("  Takeoff altitude reached!")
            break
        time.sleep(0.5)

    # --- Enable VFH3D avoidance now that we're airborne ---
    print("Enabling VFH3D obstacle avoidance...")
    avoidance_enabled = True
    vfh.reset()  # clear any stale histogram from ground readings
    time.sleep(0.5)

    # Fly waypoints with avoidance (skip first waypoint if it's just takeoff alt)
    for i, (x, y, z, label) in enumerate(WAYPOINTS):
        if i == 0:
            continue  # already at takeoff altitude
        with wp_lock:
            waypoint_idx = i
        print(f"\n>>> [{i+1}/{len(WAYPOINTS)}] {label}")
        wait_until_reached(x, y, z, timeout=20)
        print(f"  Holding for 2s...")
        time.sleep(2)

    # Mark mission complete, disable avoidance for landing
    avoidance_enabled = False
    with wp_lock:
        waypoint_idx = len(WAYPOINTS)

    # Land
    land_and_disarm()

except KeyboardInterrupt:
    print("\nInterrupted! Landing...")
    avoidance_enabled = False
    land_and_disarm()
finally:
    shutdown()
