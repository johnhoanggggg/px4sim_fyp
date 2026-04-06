#!/usr/bin/env python3
"""
Republishes right camera CameraInfo with correct stereo baseline (Tx).

Gazebo's ros_gz_bridge doesn't populate the projection matrix P[3] field
(Tx = -fx * baseline) needed by RTAB-Map for stereo disparity. This node
subscribes to the raw CameraInfo, injects the correct Tx value, and
republishes it.

Subscriptions:
    /oakd/right/camera_info_raw  (sensor_msgs/CameraInfo)

Publications:
    /oakd/right/camera_info  (sensor_msgs/CameraInfo)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo


class StereoBaselineFixer(Node):
    def __init__(self):
        super().__init__('stereo_baseline_fixer')

        self.declare_parameter('baseline', 0.075)  # 7.5cm OAK-D Lite baseline
        self.declare_parameter('fx', 432.0)

        self._baseline = self.get_parameter('baseline').value
        self._fx = self.get_parameter('fx').value
        self._tx = -self._fx * self._baseline  # -32.4

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            CameraInfo, '/oakd/right/camera_info',
            self._cb, qos,
        )
        self._pub = self.create_publisher(CameraInfo, '/oakd/right/camera_info_fixed', 10)

        self.get_logger().info(
            f'Stereo baseline fixer: Tx={self._tx:.1f} '
            f'(baseline={self._baseline}m, fx={self._fx})'
        )

    def _cb(self, msg: CameraInfo):
        # Inject Tx into the projection matrix P
        # P is a 12-element array (3x4 row-major)
        # P[3] = Tx = -fx * baseline
        p = list(msg.p)
        p[3] = self._tx
        msg.p = p
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoBaselineFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
