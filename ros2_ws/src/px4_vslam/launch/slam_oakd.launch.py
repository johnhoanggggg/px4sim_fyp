"""
Launch RTAB-Map stereo SLAM for real OAK-D Lite hardware.

Launches:
  1. depthai_ros — OAK-D Lite camera driver
  2. rtabmap_odom — stereo visual odometry
  3. rtabmap — SLAM
  4. slam_bridge — pose → PX4

Prerequisites:
  - depthai-ros must be installed:
      sudo apt install ros-humble-depthai-ros
  - OAK-D Lite must be connected via USB

Usage:
  ros2 launch px4_vslam slam_oakd.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('px4_vslam')
    config_file = os.path.join(pkg_dir, 'config', 'rtabmap_params.yaml')

    # OAK-D Lite camera driver
    # Uses depthai_ros_driver which publishes standard sensor_msgs
    oakd_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('depthai_ros_driver'),
                'launch', 'camera.launch.py'
            )
        ]),
        launch_arguments={
            'name': 'oakd',
            'camera_model': 'OAK-D-LITE',
            'enable_stereo': 'true',
            'enable_imu': 'true',
            'stereo_fps': '30',
            'imu_mode': '1',  # LINEAR_INTERPOLATE_ACCEL
        }.items(),
    )

    # RTAB-Map stereo odometry
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[config_file],
        remappings=[
            ('left/image_rect', '/oakd/left/image_rect'),
            ('right/image_rect', '/oakd/right/image_rect'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info'),
            ('imu', '/oakd/imu/data'),
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
            ('left/image_rect', '/oakd/left/image_rect'),
            ('right/image_rect', '/oakd/right/image_rect'),
            ('left/camera_info', '/oakd/left/camera_info'),
            ('right/camera_info', '/oakd/right/camera_info'),
            ('imu', '/oakd/imu/data'),
        ],
        output='screen',
    )

    # SLAM bridge
    slam_bridge = Node(
        package='px4_vslam',
        executable='slam_bridge',
        name='slam_bridge',
        parameters=[{'odom_topic': '/rtabmap/odom'}],
        output='screen',
    )

    return LaunchDescription([
        oakd_camera,
        rtabmap_odom,
        rtabmap_slam,
        slam_bridge,
    ])
