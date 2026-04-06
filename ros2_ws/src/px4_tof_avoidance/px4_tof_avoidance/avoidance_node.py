#!/usr/bin/env python3
"""
ROS2 obstacle avoidance node using FGM3D / VFH3D / DWA3D algorithms.

Subscribes to aggregated ToF obstacle points and PX4 vehicle odometry,
runs the selected avoidance algorithm, and publishes trajectory setpoints
back to PX4 via uXRCE-DDS.

This is the ROS2 equivalent of scripts/fly_truss2_fgm.py's control loop.

Subscriptions:
    /tof/obstacles           (sensor_msgs/PointCloud2)
    /fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry)

Publications:
    /fmu/in/trajectory_setpoint    (px4_msgs/TrajectorySetpoint)
    /fmu/in/offboard_control_mode  (px4_msgs/OffboardControlMode)

Parameters:
    algorithm:    'fgm3d' | 'vfh3d' | 'dwa3d'
    max_speed:    maximum velocity (m/s)
    max_range:    obstacle sensing range (m)
    waypoints:    list of [x, y, z] waypoints in NED
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2

from .algorithms.fgm3d import FGM3D
from .algorithms.vfh3d import VFH3D
from .algorithms.dwa3d import DWA3D

# px4_msgs imports — these come from the px4_msgs ROS2 package
try:
    from px4_msgs.msg import (
        TrajectorySetpoint,
        OffboardControlMode,
        VehicleOdometry,
        VehicleCommand,
        VehicleStatus,
    )
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False


# Default waypoints for truss2 world (NED frame)
DEFAULT_WAYPOINTS = [
    [0.0,   0.0,  -1.2],
    [3.0,   0.0,  -1.2],
    [5.9,   1.0,  -2.0],
    [7.7,   0.0,  -3.0],
    [9.5,  -1.0,  -2.5],
    [11.3,  1.0,  -2.5],
    [13.1,  0.0,  -3.2],
    [14.9, -0.5,  -2.0],
    [16.5,  0.0,  -2.5],
    [0.0,   0.0,  -1.2],
]


def _pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 array from a PointCloud2 message."""
    if msg.width == 0:
        return np.empty((0, 3))
    buf = np.frombuffer(msg.data, dtype=np.float32)
    return buf.reshape(-1, 3)


def _euler_from_quaternion(q):
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


ALGORITHM_MAP = {
    'fgm3d': FGM3D,
    'vfh3d': VFH3D,
    'dwa3d': DWA3D,
}


class AvoidanceNode(Node):
    def __init__(self):
        super().__init__('avoidance')

        if not HAS_PX4_MSGS:
            self.get_logger().error(
                'px4_msgs not found. Install with: '
                'cd ros2_ws && git clone https://github.com/PX4/px4_msgs.git src/px4_msgs && colcon build'
            )
            raise RuntimeError('px4_msgs required')

        # Parameters
        self.declare_parameter('algorithm', 'fgm3d')
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('max_range', 2.0)
        self.declare_parameter('safe_distance', 1.0)
        self.declare_parameter('bubble_radius', 0.35)
        self.declare_parameter('waypoint_tolerance', 0.6)
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('vel_smooth', 0.3)

        algo_name = self.get_parameter('algorithm').value
        max_speed = self.get_parameter('max_speed').value
        max_range = self.get_parameter('max_range').value
        safe_dist = self.get_parameter('safe_distance').value
        bubble_r = self.get_parameter('bubble_radius').value
        self._wp_tol = self.get_parameter('waypoint_tolerance').value
        control_rate = self.get_parameter('control_rate').value
        self._vel_smooth = self.get_parameter('vel_smooth').value

        # Initialize avoidance algorithm
        if algo_name not in ALGORITHM_MAP:
            self.get_logger().error(f'Unknown algorithm: {algo_name}')
            raise ValueError(f'algorithm must be one of {list(ALGORITHM_MAP.keys())}')

        AlgoClass = ALGORITHM_MAP[algo_name]
        self._algo = AlgoClass(
            n_az=72,
            n_el=18,
            max_range=max_range,
            bubble_radius=bubble_r,
            safe_distance=safe_dist,
            max_speed=max_speed,
        )
        self.get_logger().info(f'Avoidance algorithm: {algo_name}')

        # State
        self._obstacle_pts = np.empty((0, 3))
        self._position_ned = None  # (x, y, z) NED
        self._yaw = 0.0
        self._prev_vel_ned = (0.0, 0.0, 0.0)

        # Arming / offboard state
        self._armed = False
        self._nav_state = 0
        self._offboard_setpoint_count = 0
        self._OFFBOARD_THRESHOLD = 40  # stream ~4s of setpoints before switching

        # Waypoints
        self._waypoints = DEFAULT_WAYPOINTS
        self._wp_idx = 0

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions
        self.create_subscription(
            PointCloud2, '/tof/obstacles',
            self._obstacle_cb, sensor_qos,
        )
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, sensor_qos,
        )
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._status_cb, sensor_qos,
        )

        # Publishers
        self._traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos,
        )
        self._offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos,
        )
        self._cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos,
        )

        # Control loop timer
        period = 1.0 / control_rate
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'Avoidance node started: {algo_name}, '
            f'{len(self._waypoints)} waypoints, {control_rate:.0f}Hz'
        )

    def _obstacle_cb(self, msg: PointCloud2):
        self._obstacle_pts = _pointcloud2_to_numpy(msg)

    def _odom_cb(self, msg: VehicleOdometry):
        self._position_ned = (msg.position[0], msg.position[1], msg.position[2])
        self._yaw = _euler_from_quaternion(msg.q)

    def _status_cb(self, msg: VehicleStatus):
        self._armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self._nav_state = msg.nav_state

    def _send_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._cmd_pub.publish(msg)

    def _arm(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0, param2=21196.0)  # 21196 = force arm (bypass preflight)
        self.get_logger().info('Arm command sent')

    def _set_offboard_mode(self):
        self._send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0)  # 6 = PX4_CUSTOM_MAIN_MODE_OFFBOARD
        self.get_logger().info('Offboard mode command sent')

    def _control_loop(self):
        # Always publish offboard control mode (velocity)
        ocm = OffboardControlMode()
        ocm.position = False
        ocm.velocity = True
        ocm.acceleration = False
        ocm.attitude = False
        ocm.body_rate = False
        ocm.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._offboard_pub.publish(ocm)

        self._offboard_setpoint_count += 1

        # After enough setpoints, switch to offboard first, then arm
        if self._offboard_setpoint_count >= self._OFFBOARD_THRESHOLD:
            if self._nav_state != 14:  # 14 = NAVIGATION_STATE_OFFBOARD
                self._set_offboard_mode()
            elif not self._armed:
                # Only arm after offboard mode is confirmed
                self._arm()

        if self._position_ned is None:
            return

        if not self._armed:
            # Keep publishing zero velocity until armed
            self._publish_velocity(0.0, 0.0, 0.0)
            return

        if self._wp_idx >= len(self._waypoints):
            # All waypoints reached — hold position
            self._publish_velocity(0.0, 0.0, 0.0)
            return

        px, py, pz = self._position_ned
        wx, wy, wz = self._waypoints[self._wp_idx]

        # Check if waypoint reached
        dist = math.sqrt((px - wx)**2 + (py - wy)**2 + (pz - wz)**2)
        if dist < self._wp_tol:
            self.get_logger().info(
                f'Waypoint {self._wp_idx + 1}/{len(self._waypoints)} reached'
            )
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                self.get_logger().info('All waypoints completed')
                self._publish_velocity(0.0, 0.0, 0.0)
                return
            wx, wy, wz = self._waypoints[self._wp_idx]

        # Goal NED → body FLU
        c_yaw, s_yaw = math.cos(self._yaw), math.sin(self._yaw)
        goal_ned = (wx - px, wy - py, wz - pz)
        frd_x = c_yaw * goal_ned[0] + s_yaw * goal_ned[1]
        frd_y = -s_yaw * goal_ned[0] + c_yaw * goal_ned[1]
        goal_body = (frd_x, -frd_y, -goal_ned[2])  # FRD→FLU

        # Run avoidance algorithm
        vel_body = self._algo.update(self._obstacle_pts, goal_body)

        # Body FLU → NED
        frd_vx = vel_body[0]
        frd_vy = -vel_body[1]  # FLU→FRD
        raw_vn = c_yaw * frd_vx - s_yaw * frd_vy
        raw_ve = s_yaw * frd_vx + c_yaw * frd_vy
        raw_vd = -vel_body[2]  # FLU up → NED down

        # EMA smoothing
        a = self._vel_smooth
        vel_ned = (
            a * raw_vn + (1 - a) * self._prev_vel_ned[0],
            a * raw_ve + (1 - a) * self._prev_vel_ned[1],
            a * raw_vd + (1 - a) * self._prev_vel_ned[2],
        )
        self._prev_vel_ned = vel_ned

        self._publish_velocity(vel_ned[0], vel_ned[1], vel_ned[2])

    def _publish_velocity(self, vn: float, ve: float, vd: float):
        msg = TrajectorySetpoint()
        msg.position = [float('nan')] * 3
        msg.velocity = [vn, ve, vd]
        msg.yaw = float('nan')  # let PX4 handle yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self._traj_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
