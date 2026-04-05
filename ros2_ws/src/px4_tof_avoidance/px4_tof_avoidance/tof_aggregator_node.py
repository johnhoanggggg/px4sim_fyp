#!/usr/bin/env python3
"""
ROS2 node that subscribes to 12 bridged ToF LaserScan topics and publishes
an aggregated PointCloud2 of obstacle points in body frame.

This is the ROS2 equivalent of scripts/tof_reader.py, which used Gazebo
Transport directly.

Subscriptions:
    /tof/0 .. /tof/9, /tof/up, /tof/down  (sensor_msgs/LaserScan)

Publications:
    /tof/obstacles  (sensor_msgs/PointCloud2)  — all obstacle points in body FLU
"""

import math
import struct
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header

# Sensor geometry from x500_tof model.sdf
HORIZONTAL_SENSORS = {
    '0': {'yaw': 0.0},
    '1': {'yaw': 0.6283},
    '2': {'yaw': 1.2566},
    '3': {'yaw': 1.8850},
    '5': {'yaw': 3.1416},
    '7': {'yaw': -1.8850},
    '8': {'yaw': -1.2566},
    '9': {'yaw': -0.6283},
}

VERTICAL_SENSORS = {
    '4':    {'pitch': -math.pi / 4},   # forward-up 45 deg
    '6':    {'pitch':  math.pi / 4},   # forward-down 45 deg
    'up':   {'pitch': -math.pi / 2},   # straight up
    'down': {'pitch':  math.pi / 2},   # straight down
}

H_SAMPLES = 8
V_SAMPLES = 8
FOV_HALF = 0.3927  # rad


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _make_pointcloud2(points: np.ndarray, stamp) -> PointCloud2:
    """Create a PointCloud2 message from an (N, 3) float32 array."""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = 'base_link'

    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points)
    msg.is_dense = True

    if len(points) > 0:
        msg.data = points.astype(np.float32).tobytes()
    else:
        msg.data = b''

    return msg


class TofAggregatorNode(Node):
    def __init__(self):
        super().__init__('tof_aggregator')

        self.declare_parameter('max_range', 2.0)
        self.declare_parameter('min_range', 0.30)
        self.declare_parameter('publish_rate', 10.0)

        self._max_range = self.get_parameter('max_range').value
        self._min_range = self.get_parameter('min_range').value
        publish_rate = self.get_parameter('publish_rate').value

        # Pre-compute ray direction unit vectors for 8x8 grid
        h_angles = np.linspace(-FOV_HALF, FOV_HALF, H_SAMPLES)
        v_angles = np.linspace(-FOV_HALF, FOV_HALF, V_SAMPLES)
        dirs = []
        for v_ang in v_angles:
            for h_ang in h_angles:
                dx = math.cos(v_ang) * math.cos(h_ang)
                dy = math.cos(v_ang) * math.sin(h_ang)
                dz = math.sin(v_ang)
                dirs.append((dx, dy, dz))
        self._ray_dirs = np.array(dirs)  # (64, 3)

        # Pre-compute rotation matrices
        self._rot = {}
        for name, cfg in HORIZONTAL_SENSORS.items():
            self._rot[name] = _rotz(cfg['yaw'])
        for name, cfg in VERTICAL_SENSORS.items():
            self._rot[name] = _roty(cfg['pitch'])

        # Latest ranges per sensor
        self._lock = threading.Lock()
        self._ranges: dict[str, np.ndarray] = {}

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to all 12 ToF topics
        all_sensors = list(HORIZONTAL_SENSORS.keys()) + list(VERTICAL_SENSORS.keys())
        for name in all_sensors:
            topic = f'/tof/{name}'
            self.create_subscription(
                LaserScan, topic,
                lambda msg, sn=name: self._tof_cb(sn, msg),
                sensor_qos,
            )

        # Publisher
        self._pub = self.create_publisher(PointCloud2, '/tof/obstacles', 10)

        # Timer for aggregation + publish
        period = 1.0 / publish_rate
        self.create_timer(period, self._publish_obstacles)

        self.get_logger().info(
            f'ToF aggregator started: {len(all_sensors)} sensors, '
            f'range [{self._min_range:.2f}, {self._max_range:.2f}]m, '
            f'{publish_rate:.0f}Hz'
        )

    def _tof_cb(self, sensor_name: str, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        with self._lock:
            self._ranges[sensor_name] = ranges

    def _publish_obstacles(self):
        with self._lock:
            raw = {k: v.copy() for k, v in self._ranges.items()}

        if not raw:
            return

        all_pts = []
        for name, ranges in raw.items():
            if name not in self._rot:
                continue
            valid = (
                (ranges > self._min_range) &
                (ranges < self._max_range) &
                np.isfinite(ranges)
            )
            if not np.any(valid):
                continue
            pts_local = self._ray_dirs * ranges[:, np.newaxis]
            pts_valid = pts_local[valid]
            R = self._rot[name]
            pts_body = (R @ pts_valid.T).T
            all_pts.append(pts_body)

        if all_pts:
            points = np.vstack(all_pts)
        else:
            points = np.empty((0, 3))

        now = self.get_clock().now().to_msg()
        pc_msg = _make_pointcloud2(points, now)
        self._pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TofAggregatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
