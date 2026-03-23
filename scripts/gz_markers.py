#!/usr/bin/env python3
"""
Visualize VFH3D histogram bins as colored markers in Gazebo.

Draws spheres around the drone at each bin direction:
  Red = blocked, Green = free.
Only shows horizontal-ish bins (skip extreme up/down elevations).
"""

import math
import threading
import numpy as np

try:
    from gz.transport import Node
except ImportError:
    from gz.transport13 import Node

try:
    from gz.msgs.marker_pb2 import Marker
    from gz.msgs.marker_v_pb2 import Marker_V
    from gz.msgs.boolean_pb2 import Boolean
except ImportError:
    from gz.msgs10.marker_pb2 import Marker
    from gz.msgs10.marker_v_pb2 import Marker_V
    from gz.msgs10.boolean_pb2 import Boolean


class GzMarkerViz:
    """Draw VFH3D histogram bins as markers in the Gazebo 3D viewport."""

    VIZ_RADIUS = 1.2  # distance from drone to place bin markers

    def __init__(self, namespace="vfh_bins"):
        self._ns = namespace
        self._node = Node()
        self._busy = threading.Event()
        self._busy.set()
        self._lock = threading.Lock()
        self._pending = None
        self._prev_count = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        import time
        while True:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is not None:
                self._busy.clear()
                batch, new_count = job
                self._node.request("/marker_array", batch,
                                   Marker_V, Boolean, 100)
                self._prev_count = new_count
                self._busy.set()
            else:
                time.sleep(0.01)

    def update(self, bins: list, drone_pos_ned: tuple, yaw: float = 0.0):
        """
        Draw VFH3D bins around the drone.

        Args:
            bins: list of (azimuth, elevation, blocked) from vfh.get_blocked_bins()
            drone_pos_ned: (px, py, pz) in NED.
            yaw: drone heading (rad, NED convention).
        """
        if not self._busy.is_set():
            return

        px, py, pz = drone_pos_ned
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        r = self.VIZ_RADIUS

        batch = Marker_V()
        idx = 0

        for az, el, blocked in bins:
            # Skip extreme elevations (only show ±40° from horizontal)
            if abs(el) > math.radians(40):
                continue

            # Bin direction in body FLU
            bx = math.cos(el) * math.cos(az)
            by = math.cos(el) * math.sin(az)
            bz = math.sin(el)

            # Body FLU → FRD → rotate by yaw → NED
            frd_x, frd_y, frd_z = bx, -by, -bz
            ned_dx = c_yaw * frd_x - s_yaw * frd_y
            ned_dy = s_yaw * frd_x + c_yaw * frd_y
            ned_dz = frd_z

            ned_x = px + r * ned_dx
            ned_y = py + r * ned_dy
            ned_z = pz + r * ned_dz

            # NED → Gazebo ENU
            gz_x = ned_y
            gz_y = ned_x
            gz_z = -ned_z

            m = batch.marker.add()
            m.ns = self._ns
            m.id = idx
            m.action = Marker.ADD_MODIFY
            m.type = Marker.SPHERE
            m.pose.position.x = gz_x
            m.pose.position.y = gz_y
            m.pose.position.z = gz_z
            size = 0.12 if blocked else 0.06
            m.scale.x = size
            m.scale.y = size
            m.scale.z = size
            if blocked:
                m.material.ambient.r = 1.0
                m.material.ambient.g = 0.0
            else:
                m.material.ambient.r = 0.0
                m.material.ambient.g = 1.0
            m.material.ambient.b = 0.0
            m.material.ambient.a = 0.8
            m.material.diffuse.CopyFrom(m.material.ambient)
            m.lifetime.sec = 0
            m.lifetime.nsec = 300_000_000
            idx += 1

        # Delete stale
        for i in range(idx, self._prev_count):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        with self._lock:
            self._pending = (batch, idx)

    def clear(self):
        batch = Marker_V()
        m = batch.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._node.request("/marker_array", batch, Marker_V, Boolean, 100)
