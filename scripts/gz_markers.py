#!/usr/bin/env python3
"""
Publish Gazebo marker messages to visualize obstacle points in the 3D viewport.

Uses gz-transport to draw colored spheres at each obstacle point detected
by the ToF sensors. Points are colored by distance (green=far, red=close).
"""

import numpy as np

try:
    from gz.transport import Node
except ImportError:
    from gz.transport13 import Node

try:
    from gz.msgs.marker_pb2 import Marker
    from gz.msgs.marker_v_pb2 import Marker_V
except ImportError:
    from gz.msgs10.marker_pb2 import Marker
    from gz.msgs10.marker_v_pb2 import Marker_V


class GzMarkerViz:
    """Draw obstacle points as colored markers in the Gazebo 3D viewport."""

    MARKER_TOPIC = "/marker"

    def __init__(self, namespace="tof_viz", max_points=200):
        self._ns = namespace
        self._max_pts = max_points
        self._node = Node()
        self._pub = self._node.advertise(self.MARKER_TOPIC, Marker_V)
        self._prev_count = 0

    def update(self, obstacle_pts_body: np.ndarray, drone_pos_ned: tuple):
        """
        Draw obstacle points in world frame.

        Args:
            obstacle_pts_body: (N,3) obstacle points in body FLU frame.
            drone_pos_ned: (px, py, pz) drone position in NED frame.
        """
        px, py, pz = drone_pos_ned
        msg = Marker_V()

        n = min(len(obstacle_pts_body), self._max_pts)

        for i in range(n):
            bx, by, bz = obstacle_pts_body[i]
            # Body FLU to NED: ned_x=body_x, ned_y=-body_y, ned_z=-body_z
            wx = px + bx
            wy = py + (-by)
            wz = pz + (-bz)

            dist = float(np.sqrt(bx**2 + by**2 + bz**2))
            # Color by distance: red=close (0.3m), green=far (3m)
            t = min(max((dist - 0.3) / 2.7, 0.0), 1.0)
            r, g, b = 1.0 - t, t, 0.0

            m = msg.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.ADD_MODIFY
            m.type = Marker.SPHERE
            m.pose.position.x = wx
            m.pose.position.y = wy
            # Gazebo uses ENU-like Z-up for rendering, NED Z is down
            m.pose.position.z = -wz
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 0.08
            mat = m.material
            mat.ambient.r = r
            mat.ambient.g = g
            mat.ambient.b = b
            mat.ambient.a = 0.8
            mat.diffuse.r = r
            mat.diffuse.g = g
            mat.diffuse.b = b
            mat.diffuse.a = 0.8
            m.lifetime.sec = 0
            m.lifetime.nsec = 500_000_000  # 0.5s auto-expire

        # Delete leftover markers from previous frame
        for i in range(n, self._prev_count):
            m = msg.marker.add()
            m.ns = self._ns
            m.id = i
            m.action = Marker.DELETE_MARKER

        self._prev_count = n
        self._pub.publish(msg)

    def clear(self):
        """Remove all markers."""
        msg = Marker_V()
        m = msg.marker.add()
        m.ns = self._ns
        m.action = Marker.DELETE_ALL
        self._pub.publish(msg)
