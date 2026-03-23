#!/usr/bin/env python3
"""
Read 12 ToF sensor values from Gazebo Transport topics.

Subscribes to /tof/0 .. /tof/9, /tof/up, /tof/down and exposes the latest
range data as numpy arrays and 3D obstacle points in body frame.

Requires: gz-transport and gz-msgs Python bindings (ship with Gazebo Harmonic),
          numpy.
"""

import math
import threading
import numpy as np

try:
    from gz.transport import Node
except ImportError:
    from gz.transport13 import Node

try:
    from gz.msgs.laserscan_pb2 import LaserScan
except ImportError:
    from gz.msgs10.laserscan_pb2 import LaserScan

# ---------------------------------------------------------------------------
# Sensor geometry (from model.sdf)
# ---------------------------------------------------------------------------
# Horizontal ring: 10 sensors at 36-degree increments
# Each sensor: 8x8 rays, FOV ±0.3927 rad (~±22.5°, total ~45°)

HORIZONTAL_SENSORS = {
    "0": {"yaw": 0.0},
    "1": {"yaw": 0.6283},
    "2": {"yaw": 1.2566},
    "3": {"yaw": 1.8850},
    "4": {"yaw": 2.5133},
    "5": {"yaw": 3.1416},
    "6": {"yaw": -2.5133},
    "7": {"yaw": -1.8850},
    "8": {"yaw": -1.2566},
    "9": {"yaw": -0.6283},
}

# Vertical sensors: pitch angle (rotation about Y axis in SDF)
VERTICAL_SENSORS = {
    "up":   {"pitch": -math.pi / 2},   # looking up
    "down": {"pitch":  math.pi / 2},    # looking down
}

ALL_TOPICS = [f"/tof/{k}" for k in HORIZONTAL_SENSORS] + \
             [f"/tof/{k}" for k in VERTICAL_SENSORS]

H_SAMPLES = 8
V_SAMPLES = 8
FOV_HALF = 0.3927  # rad, ±22.5°


def _ray_angles(n_samples, fov_half):
    """Return evenly spaced ray angles from -fov_half to +fov_half."""
    return np.linspace(-fov_half, fov_half, n_samples)


class TofReader:
    """Subscribe to all ToF Gazebo topics and provide range data."""

    def __init__(self):
        self._lock = threading.Lock()
        # Latest raw ranges per sensor: key="0".."9","up","down", value=np.array(64,)
        self._ranges: dict[str, np.ndarray] = {}
        self._node = Node()

        # Pre-compute ray direction unit vectors for an 8x8 grid
        h_angles = _ray_angles(H_SAMPLES, FOV_HALF)
        v_angles = _ray_angles(V_SAMPLES, FOV_HALF)
        # Shape (64, 3) — directions in sensor-local frame (forward = +X)
        dirs = []
        for v_ang in v_angles:
            for h_ang in h_angles:
                dx = math.cos(v_ang) * math.cos(h_ang)
                dy = math.cos(v_ang) * math.sin(h_ang)
                dz = math.sin(v_ang)
                dirs.append((dx, dy, dz))
        self._ray_dirs = np.array(dirs)  # (64, 3)

        # Pre-compute rotation matrices for each sensor
        self._rot = {}
        for name, cfg in HORIZONTAL_SENSORS.items():
            self._rot[name] = self._rotz(cfg["yaw"])
        for name, cfg in VERTICAL_SENSORS.items():
            self._rot[name] = self._roty(cfg["pitch"])

        # Subscribe
        for topic in ALL_TOPICS:
            sensor_name = topic.split("/")[-1]  # "0".."9","up","down"
            self._node.subscribe(LaserScan, topic,
                                 lambda msg, sn=sensor_name: self._cb(sn, msg))

    # ------- rotation helpers -------
    @staticmethod
    def _rotz(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])

    @staticmethod
    def _roty(pitch):
        c, s = math.cos(pitch), math.sin(pitch)
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    # ------- callback -------
    def _cb(self, sensor_name: str, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        with self._lock:
            self._ranges[sensor_name] = ranges

    # ------- public API -------
    def get_ranges(self) -> dict[str, np.ndarray]:
        """Return copy of latest raw ranges dict. Each value is shape (64,)."""
        with self._lock:
            return {k: v.copy() for k, v in self._ranges.items()}

    def get_obstacle_points(self, max_range: float = 4.0) -> np.ndarray:
        """
        Convert all sensor readings into 3D obstacle points in body frame.

        Returns:
            np.ndarray of shape (N, 3) — obstacle positions relative to
            the drone body centre. X=forward, Y=left, Z=up.
        """
        raw = self.get_ranges()
        if not raw:
            return np.empty((0, 3))

        all_pts = []
        for name, ranges in raw.items():
            if name not in self._rot:
                continue
            # Filter valid ranges (>0.30m rejects self-detection of drone arms/props)
            valid = (ranges > 0.30) & (ranges < max_range) & np.isfinite(ranges)
            if not np.any(valid):
                continue
            # Points in sensor-local frame
            pts_local = self._ray_dirs * ranges[:, np.newaxis]  # (64, 3)
            pts_valid = pts_local[valid]
            # Rotate to body frame
            R = self._rot[name]
            pts_body = (R @ pts_valid.T).T  # (N, 3)
            all_pts.append(pts_body)

        if not all_pts:
            return np.empty((0, 3))
        return np.vstack(all_pts)

    def has_data(self) -> bool:
        """True if at least one sensor has reported data."""
        with self._lock:
            return len(self._ranges) > 0


# ---------------------------------------------------------------------------
# Standalone test: print ranges when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    reader = TofReader()
    print("Waiting for ToF data (start the Gazebo simulation)...")
    while not reader.has_data():
        time.sleep(0.5)
    print("Receiving data. Press Ctrl+C to stop.\n")
    try:
        while True:
            ranges = reader.get_ranges()
            for name in sorted(ranges.keys(), key=lambda x: (x.isdigit(), x)):
                r = ranges[name]
                print(f"  tof/{name}: min={r.min():.2f}m  max={r.max():.2f}m  mean={r.mean():.2f}m")
            pts = reader.get_obstacle_points()
            print(f"  Total obstacle points: {len(pts)}")
            print()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Done.")
