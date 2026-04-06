"""
Launch RTAB-Map stereo SLAM for Gazebo simulation with OAK-D Lite camera.

Launches:
  1. rtabmap_odom — stereo visual odometry
  2. rtabmap — SLAM with loop closure and map building
  3. slam_bridge — converts RTAB-Map pose to PX4 VehicleVisualOdometry

Prerequisites:
  - ros_gz_bridge must be running (bridging /oakd/* topics)
  - rtabmap_ros must be installed: sudo apt install ros-humble-rtabmap-ros

Usage:
  ros2 launch px4_vslam slam_sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('px4_vslam')
    config_file = os.path.join(pkg_dir, 'config', 'rtabmap_params.yaml')

    # Gazebo uses frame_id "x500_tof_0/oakd_left_link/oakd_left" etc.
    # RTAB-Map needs base_link → camera transforms via TF.
    # Publish static transforms: base_link → each camera frame
    # Left camera: (0.10, 0.0375, -0.02) from base_link
    tf_left = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.10', '0.0375', '-0.02', '0', '0', '0',
            'base_link', 'x500_tof_0/oakd_left_link/oakd_left',
        ],
    )

    # Right camera: (0.10, -0.0375, -0.02) from base_link
    tf_right = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.10', '-0.0375', '-0.02', '0', '0', '0',
            'base_link', 'x500_tof_0/oakd_right_link/oakd_right',
        ],
    )

    # IMU: (0.10, 0, -0.02) from base_link
    tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.10', '0', '-0.02', '0', '0', '0',
            'base_link', 'x500_tof_0/oakd_imu_link/oakd_imu',
        ],
    )

    # Stereo baseline fixer: injects Tx into right camera_info
    # Gazebo bridge doesn't set P[3] (Tx = -fx * baseline) needed by RTAB-Map
    baseline_fixer = Node(
        package='px4_vslam',
        executable='stereo_baseline_fixer',
        name='stereo_baseline_fixer',
        parameters=[{'baseline': 0.075, 'fx': 432.0}],
        output='screen',
    )

    # RTAB-Map stereo odometry
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[
            config_file,
            {'approx_sync': True},
        ],
        remappings=[
            ('left/image_rect', '/oakd/left/image_raw'),
            ('right/image_rect', '/oakd/right/image_raw'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info_fixed'),
            ('imu', '/oakd/imu'),
        ],
        output='screen',
    )

    # RTAB-Map SLAM
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        parameters=[
            config_file,
            {'subscribe_stereo': True, 'approx_sync': True},
        ],
        remappings=[
            ('left/image_rect', '/oakd/left/image_raw'),
            ('right/image_rect', '/oakd/right/image_raw'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info_fixed'),
            ('imu', '/oakd/imu'),
        ],
        output='screen',
    )

    # SLAM bridge (RTAB-Map odom → PX4 VehicleVisualOdometry)
    slam_bridge = Node(
        package='px4_vslam',
        executable='slam_bridge',
        name='slam_bridge',
        parameters=[{'odom_topic': '/rtabmap/odom'}],
        output='screen',
    )

    return LaunchDescription([
        tf_left,
        tf_right,
        tf_imu,
        baseline_fixer,
        rtabmap_odom,
        rtabmap_slam,
        slam_bridge,
    ])
