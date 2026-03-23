#!/usr/bin/env python3
"""
Publish Gazebo marker messages to visualize obstacle points in the 3D viewport.

Uses gz-transport /marker_array service to batch-draw colored spheres at
obstacle points. Red=close, green=far.

Frame chain:
  Body FLU (sensors) → rotate by yaw → NED world → Gazebo world
"""

import math
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

    def update(self, obstacle_pts_body: np.ndarray, drone_pos_ned: tuple,
               yaw: float = 0.0):
        """
        Draw obstacle points in Gazebo world frame.

        Args:
            obstacle_pts_body: (N,3) points in body FLU frame.
            drone_pos_ned: (px, py, pz) drone position in NED.
            yaw: drone heading in radians (NED convention, 0=north, CW positive).
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

        # Step 1: Body FLU → NED using yaw rotation
        # FLU: x=forward, y=left, z=up
        # First convert FLU to FRD (PX4 body): frd = (flu_x, -flu_y, -flu_z)
        # Then rotate FRD by yaw to NED: R_yaw * frd
        # Combined: NED = R_yaw * (bx, -by, -bz)
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by, bz = pts[:, 0], pts[:, 1], pts[:, 2]
        # FLU to FRD
        frd_x = bx
        frd_y = -by
        frd_z = -bz
        # Rotate by yaw (NED yaw: 0=north, positive=clockwise)
        ned_dx = c * frd_x - s * frd_y
        ned_dy = s * frd_x + c * frd_y
        ned_dz = frd_z

        # Step 2: Add drone position (NED)
        ned_x = px + ned_dx
        ned_y = py + ned_dy
        ned_z = pz + ned_dz

        # Step 3: NED → Gazebo ENU
        # Gazebo world is ENU: gz_x=east=ned_y, gz_y=north=ned_x, gz_z=up=-ned_z
        gz_x = ned_y
        gz_y = ned_x
        gz_z = -ned_z

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

        # Delete stale markers
        for i in range(n, self._prev_count):
            m = batch.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        self._prev_count = n
        self._node.request("/marker_array", batch, Marker_V, Boolean, 200)

    def clear(self):
        """Remove all markers."""
        batch = Marker_V()
        m = batch.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._node.request("/marker_array", batch, Marker_V, Boolean, 200)
