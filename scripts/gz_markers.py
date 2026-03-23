#!/usr/bin/env python3
"""
Publish Gazebo marker messages to visualize obstacle points in the 3D viewport.

Uses gz-transport /marker_array service in a background thread to avoid
blocking the control loop. Red=close, green=far.
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

    def __init__(self, namespace="tof_viz", max_points=80):
        self._ns = namespace
        self._max_pts = max_points
        self._node = Node()
        self._prev_count = 0
        self._lock = threading.Lock()
        self._pending = None  # (batch_msg, new_count)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread that sends marker service calls."""
        import time
        while True:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is not None:
                batch, new_count = job
                self._node.request("/marker_array", batch,
                                   Marker_V, Boolean, 500)
                self._prev_count = new_count
            else:
                time.sleep(0.02)

    def update(self, obstacle_pts_body: np.ndarray, drone_pos_ned: tuple,
               yaw: float = 0.0):
        """
        Queue obstacle points for drawing (non-blocking).

        Args:
            obstacle_pts_body: (N,3) points in body FLU frame.
            drone_pos_ned: (px, py, pz) drone position in NED.
            yaw: drone heading in radians (NED, 0=north, CW+).
        """
        px, py, pz = drone_pos_ned
        n_pts = len(obstacle_pts_body)

        if n_pts > self._max_pts:
            idx = np.linspace(0, n_pts - 1, self._max_pts, dtype=int)
            pts = obstacle_pts_body[idx]
        else:
            pts = obstacle_pts_body
        n = len(pts)

        # Body FLU → FRD → rotate by yaw → NED → Gazebo ENU
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by, bz = pts[:, 0], pts[:, 1], pts[:, 2]
        frd_x, frd_y, frd_z = bx, -by, -bz
        ned_x = px + c * frd_x - s * frd_y
        ned_y = py + s * frd_x + c * frd_y
        ned_z = pz + frd_z
        # NED → Gazebo ENU: gz_x=east=ned_y, gz_y=north=ned_x, gz_z=up=-ned_z
        gz_x = ned_y
        gz_y = ned_x
        gz_z = -ned_z


        dists = np.sqrt(bx**2 + by**2 + bz**2)
        t = np.clip((dists - 0.3) / 2.7, 0.0, 1.0)

        batch = Marker_V()

        # Debug: blue marker at drone position to verify NED→Gazebo mapping
        dm = batch.marker.add()
        dm.ns = self._ns + "_drone"
        dm.id = 0
        dm.action = Marker.ADD_MODIFY
        dm.type = Marker.SPHERE
        dm.pose.position.x = py   # ENU: gz_x = ned_y (east)
        dm.pose.position.y = px   # ENU: gz_y = ned_x (north)
        dm.pose.position.z = -pz  # ENU: gz_z = -ned_z (up)
        # If blue ball does NOT follow the drone, try NO swap instead:
        # dm.pose.position.x = px
        # dm.pose.position.y = py
        # dm.pose.position.z = -pz
        dm.scale.x = 0.3
        dm.scale.y = 0.3
        dm.scale.z = 0.3
        dm.material.ambient.r = 0.0
        dm.material.ambient.g = 0.0
        dm.material.ambient.b = 1.0
        dm.material.ambient.a = 1.0
        dm.material.diffuse.r = 0.0
        dm.material.diffuse.g = 0.0
        dm.material.diffuse.b = 1.0
        dm.material.diffuse.a = 1.0
        dm.lifetime.sec = 1
        dm.lifetime.nsec = 0

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
            m.lifetime.nsec = 300_000_000  # 300ms expiry

        # Delete stale markers
        prev = self._prev_count
        for i in range(n, prev):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        with self._lock:
            self._pending = (batch, n)

    def clear(self):
        """Remove all markers."""
        batch = Marker_V()
        m = batch.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._node.request("/marker_array", batch, Marker_V, Boolean, 200)
