#!/usr/bin/env python3
"""
Offboard control: takeoff and fly into the truss structure.
Truss corridor is at z=1.5-2.5m, y=-1.0 to 1.0m, x=-4.0 to 4.0m.
Background thread streams setpoints at 20Hz so offboard never drops.
"""

import time
import socket
import threading
import sys
from pymavlink import mavutil

# --- Connect ---
recv = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
recv.port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv.wait_heartbeat()
print(f"Connected: system {recv.target_system}")

send = mavutil.mavlink_connection('udpout:127.0.0.1:14580', source_system=255, source_component=0)

TARGET_SYS = recv.target_system
TARGET_COMP = recv.target_component

# --- Shared state ---
target = [0.0, 0.0, -1.0, 0.0]  # x, y, z, yaw (NED)
target_lock = threading.Lock()
running = True


def set_pos(x, y, z, yaw=0):
    send.mav.set_position_target_local_ned_send(
        0, TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z, 0, 0, 0, 0, 0, 0, yaw, 0)


# --- Background setpoint thread (20Hz) ---
def setpoint_loop():
    while running:
        with target_lock:
            x, y, z, yaw = target
        try:
            set_pos(x, y, z, yaw)
        except Exception:
            pass
        time.sleep(0.05)

sp_thread = threading.Thread(target=setpoint_loop, daemon=True)
sp_thread.start()


# --- Background GCS heartbeat thread ---
def heartbeat_loop():
    while running:
        try:
            send.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0)
        except Exception:
            pass
        time.sleep(0.5)

hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
hb_thread.start()


def set_target(x, y, z, yaw=0):
    with target_lock:
        target[0] = x
        target[1] = y
        target[2] = z
        target[3] = yaw


def get_position():
    msg = recv.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
    if msg:
        return msg.x, msg.y, msg.z
    return None


def wait_until_reached(x, y, z, tolerance=0.3, timeout=15):
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


def land_and_disarm():
    """Land, wait for touchdown, disarm, and switch to position mode."""
    print("\n>>> Landing...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)

    # Wait for landing (altitude near 0)
    t0 = time.time()
    while time.time() - t0 < 20:
        pos = get_position()
        if pos and -pos[2] < 0.15:
            print("  Touchdown!")
            break
        time.sleep(0.5)

    time.sleep(2)

    # Disarm
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


try:
    # --- Step 1: Stream setpoints (background thread already running) ---
    print("Streaming setpoints for 4s...")
    time.sleep(4)

    # --- Step 2: Set offboard mode ---
    print("Setting OFFBOARD mode...")
    send.mav.set_mode_send(
        TARGET_SYS,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6 << 16)
    time.sleep(1)

    # --- Step 3: Arm ---
    print("Arming...")
    send.mav.command_long_send(
        TARGET_SYS, TARGET_COMP,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)

    # print("Waiting for arm (or arm via QGroundControl)...")
    # t0 = time.time()
    # armed = False
    # while time.time() - t0 < 30:
    #     msg = recv.recv_match(type='HEARTBEAT', blocking=False)
    #     if msg and msg.get_srcSystem() == TARGET_SYS:
    #         if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
    #             print("Armed!")
    #             armed = True
    #             break
    #     if int(time.time() - t0) % 3 == 0:
    #         send.mav.command_long_send(
    #             TARGET_SYS, TARGET_COMP,
    #             mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    #             0, 1, 21196, 0, 0, 0, 0, 0)
    #     time.sleep(0.1)

    # if not armed:
    #     print("Arm timeout. Exiting.")
    #     shutdown()
    #     sys.exit(1)

    # --- Step 4: Fly into truss ---
    waypoints = [
        (0.0, 0.0, -1.5, "Rising into truss (z=2.0m)"),
        (0.0, -3.0, -1.5, "Flying to truss start (x=-3)"),
        (0.0, -1.0, -1.5, "Flying to truss start (x=-3)"),
        (0.0, 1.0, -1.5, "Flying to truss start (x=-3)"),
        (0.0, 3.0, -1.5, "Flying to truss end (x=+3)"),
        (0.0, 0.0, -1.5, "Returning to center"),
    ]

    for x, y, z, label in waypoints:
        print(f"\n>>> {label}")
        set_target(x, y, z)
        wait_until_reached(x, y, z, timeout=3)
        print(f"  Holding for 3s...")
        time.sleep(3)

    # --- Step 5: Land and disarm ---
    land_and_disarm()

except KeyboardInterrupt:
    print("\nInterrupted! Landing...")
    land_and_disarm()
finally:
    shutdown()