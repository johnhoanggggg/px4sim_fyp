#!/usr/bin/env python3
"""
SLAM Bridge Node — converts RTAB-Map visual odometry output to PX4's
VehicleVisualOdometry message for EKF2 vision fusion.

Handles the ROS ENU → PX4 NED coordinate frame transform.

Subscriptions:
    /rtabmap/odom  (nav_msgs/Odometry) — RTAB-Map visual odometry output

Publications:
    /fmu/in/vehicle_visual_odometry  (px4_msgs/VehicleVisualOdometry)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry

try:
    from px4_msgs.msg import VehicleOdometry
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


def _enu_to_ned_position(x_enu, y_enu, z_enu):
    """Convert ENU position to NED."""
    return (y_enu, x_enu, -z_enu)


def _enu_to_ned_quaternion(qx_enu, qy_enu, qz_enu, qw_enu):
    """
    Convert ROS ENU quaternion (x, y, z, w) to PX4 NED quaternion (w, x, y, z).

    The rotation from ENU to NED is a 180-deg rotation about the
    (1/sqrt(2), 1/sqrt(2), 0) axis, equivalent to:
      R_enu2ned = Rz(pi/2) * Rx(pi)  or  q_rot = (0, 0.7071, 0.7071, 0)

    For a body frame quaternion, the transform is:
      q_ned = q_enu2ned * q_enu * q_flu2frd
    where q_flu2frd handles the body frame convention difference.

    Simplified: swap x↔y, negate z, keep w.
    """
    return (qw_enu, qy_enu, qx_enu, -qz_enu)


class SlamBridgeNode(Node):
    def __init__(self):
        super().__init__('slam_bridge')

        if not HAS_PX4_MSGS:
            self.get_logger().error(
                'px4_msgs not found. Install with: '
                'cd ros2_ws && git clone https://github.com/PX4/px4_msgs.git src/px4_msgs && colcon build'
            )
            raise RuntimeError('px4_msgs required')

        self.declare_parameter('odom_topic', '/rtabmap/odom')

        odom_topic = self.get_parameter('odom_topic').value

        # Subscribe to RTAB-Map odometry
        self.create_subscription(
            Odometry, odom_topic,
            self._odom_cb,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        # Publish to PX4
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(
            VehicleOdometry,
            '/fmu/in/vehicle_visual_odometry',
            px4_qos,
        )

        self._msg_count = 0
        self.get_logger().info(
            f'SLAM bridge started: {odom_topic} → /fmu/in/vehicle_visual_odometry'
        )

    def _odom_cb(self, msg: Odometry):
        # Extract ENU pose
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel = msg.twist.twist.linear

        # Convert ENU → NED
        ned_x, ned_y, ned_z = _enu_to_ned_position(pos.x, pos.y, pos.z)
        ned_qw, ned_qx, ned_qy, ned_qz = _enu_to_ned_quaternion(
            ori.x, ori.y, ori.z, ori.w
        )

        # Build PX4 message
        vo = VehicleOdometry()
        vo.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        vo.timestamp_sample = vo.timestamp

        # Position NED
        vo.position = [float(ned_x), float(ned_y), float(ned_z)]

        # Orientation (PX4 uses [w, x, y, z])
        vo.q = [float(ned_qw), float(ned_qx), float(ned_qy), float(ned_qz)]

        # Velocity NED
        vo.velocity = [
            float(vel.y),    # ENU y → NED x
            float(vel.x),    # ENU x → NED y
            float(-vel.z),   # ENU z → NED -z
        ]

        # Pose covariance (use RTAB-Map's if available, otherwise defaults)
        # PX4 expects upper-triangular of 3x3 position covariance
        cov = msg.pose.covariance
        vo.position_variance = [
            float(cov[0]),   # xx
            float(cov[7]),   # yy
            float(cov[14]),  # zz
        ]
        vo.orientation_variance = [
            float(cov[21]),  # roll variance
            float(cov[28]),  # pitch variance
            float(cov[35]),  # yaw variance
        ]

        # Velocity variance
        vel_cov = msg.twist.covariance
        vo.velocity_variance = [
            float(vel_cov[0]),
            float(vel_cov[7]),
            float(vel_cov[14]),
        ]

        vo.pose_frame = VehicleOdometry.POSE_FRAME_NED
        vo.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED

        self._pub.publish(vo)

        self._msg_count += 1
        if self._msg_count % 50 == 0:
            self.get_logger().info(
                f'Published {self._msg_count} vision poses, '
                f'pos=({ned_x:.2f}, {ned_y:.2f}, {ned_z:.2f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SlamBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
