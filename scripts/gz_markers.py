#!/usr/bin/env python3
"""
Publish Gazebo marker messages to visualize obstacle points in the 3D viewport.

Uses gz-transport /marker_array service in a background thread.
Red=close, green=far.
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
    """Draw obstacle points as colored markers in the Gazebo 3D viewport."""

    def __init__(self, namespace="tof_viz", max_points=30):
        self._ns = namespace
        self._max_pts = max_points
        self._node = Node()
        self._prev_count = 0
        self._busy = threading.Event()
        self._busy.set()  # starts as "not busy"
        self._lock = threading.Lock()
        self._pending = None
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

    def update(self, obstacle_pts_body: np.ndarray, drone_pos_ned: tuple,
               yaw: float = 0.0):
        """Queue obstacle points for drawing (non-blocking, skips if busy)."""
        if not self._busy.is_set():
            return  # skip frame if previous update still sending

        px, py, pz = drone_pos_ned
        n_pts = len(obstacle_pts_body)

        if n_pts > self._max_pts:
            idx = np.random.choice(n_pts, self._max_pts, replace=False)
            pts = obstacle_pts_body[idx]
        else:
            pts = obstacle_pts_body
        n = len(pts)

        c, s = math.cos(yaw), math.sin(yaw)
        bx, by, bz = pts[:, 0], pts[:, 1], pts[:, 2]
        frd_x, frd_y, frd_z = bx, -by, -bz
        ned_x = px + c * frd_x - s * frd_y
        ned_y = py + s * frd_x + c * frd_y
        ned_z = pz + frd_z
        # NED → Gazebo ENU
        gz_x = ned_y
        gz_y = ned_x
        gz_z = -ned_z

        dists = np.sqrt(bx**2 + by**2 + bz**2)
        t = np.clip((dists - 0.3) / 2.7, 0.0, 1.0)

        batch = Marker_V()
        for i in range(n):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.ADD_MODIFY
            m.type = Marker.SPHERE
            m.pose.position.x = float(gz_x[i])
            m.pose.position.y = float(gz_y[i])
            m.pose.position.z = float(gz_z[i])
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.material.ambient.r = float(1.0 - t[i])
            m.material.ambient.g = float(t[i])
            m.material.ambient.b = 0.0
            m.material.ambient.a = 0.9
            m.material.diffuse.r = float(1.0 - t[i])
            m.material.diffuse.g = float(t[i])
            m.material.diffuse.b = 0.0
            m.material.diffuse.a = 0.9
            m.lifetime.sec = 0
            m.lifetime.nsec = 200_000_000

        prev = self._prev_count
        for i in range(n, prev):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        with self._lock:
            self._pending = (batch, n)

    def clear(self):
        batch = Marker_V()
        m = batch.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._node.request("/marker_array", batch, Marker_V, Boolean, 100)
