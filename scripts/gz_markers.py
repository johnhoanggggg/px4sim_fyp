#!/usr/bin/env python3
"""
Publish Gazebo marker messages to visualize obstacle points in the 3D viewport.

Uses gz-transport /marker_array service to batch-draw colored spheres at
obstacle points detected by the ToF sensors. Red=close, green=far.

Coordinate frames:
  - Gazebo world: ENU (x=east, y=north, z=up)
  - PX4 NED:      x=north, y=east, z=down
  - Body FLU:     x=forward, y=left, z=up (at yaw=0: fwd=north)
"""

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

    def update(self, obstacle_pts_body: np.ndarray, drone_pos_ned: tuple):
        """
        Draw obstacle points in Gazebo world frame (single batch call).

        Args:
            obstacle_pts_body: (N,3) points in body FLU frame.
            drone_pos_ned: (px, py, pz) drone position in NED.
        """
        px, py, pz = drone_pos_ned
        n_pts = len(obstacle_pts_body)

        # Subsample if too many points
        if n_pts > self._max_pts:
            idx = np.linspace(0, n_pts - 1, self._max_pts, dtype=int)
            pts = obstacle_pts_body[idx]
        else:
            pts = obstacle_pts_body
        n = len(pts)

        # Body FLU -> NED (yaw≈0): ned = drone + (bx, -by, -bz)
        # NED -> Gazebo: gz_x=ned_x, gz_y=ned_y, gz_z=-ned_z
        bx, by, bz = pts[:, 0], pts[:, 1], pts[:, 2]
        gz_x = px + bx
        gz_y = py + (-by)
        gz_z = -(pz + (-bz))

        dists = np.sqrt(bx**2 + by**2 + bz**2)
        t = np.clip((dists - 0.3) / 2.7, 0.0, 1.0)

        # Build batch message
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
            m.lifetime.sec = 1
            m.lifetime.nsec = 0

        # Delete stale markers from previous frame
        for i in range(n, self._prev_count):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        self._prev_count = n

        # Single batch service call
        self._node.request("/marker_array", batch, Marker_V, Boolean, 200)

    def clear(self):
        """Remove all markers."""
        batch = Marker_V()
        m = batch.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._node.request("/marker_array", batch, Marker_V, Boolean, 200)
