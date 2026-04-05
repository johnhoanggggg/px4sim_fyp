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

    # RTAB-Map stereo odometry
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[config_file],
        remappings=[
            ('left/image_rect', '/oakd/left/image_raw'),
            ('right/image_rect', '/oakd/right/image_raw'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info'),
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
            {'subscribe_stereo': True},
        ],
        remappings=[
            ('left/image_rect', '/oakd/left/image_raw'),
            ('right/image_rect', '/oakd/right/image_raw'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info'),
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
        rtabmap_odom,
        rtabmap_slam,
        slam_bridge,
    ])
