"""
Launch ToF aggregator and obstacle avoidance nodes.

Usage:
  ros2 launch px4_tof_avoidance avoidance.launch.py
  ros2 launch px4_tof_avoidance avoidance.launch.py algorithm:=vfh3d
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('px4_tof_avoidance')
    config_file = os.path.join(pkg_dir, 'config', 'avoidance_params.yaml')

    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='fgm3d',
        description='Avoidance algorithm: fgm3d, vfh3d, or dwa3d',
    )

    tof_aggregator = Node(
        package='px4_tof_avoidance',
        executable='tof_aggregator',
        name='tof_aggregator',
        parameters=[config_file],
        output='screen',
    )

    avoidance = Node(
        package='px4_tof_avoidance',
        executable='avoidance',
        name='avoidance',
        parameters=[
            config_file,
            {'algorithm': LaunchConfiguration('algorithm')},
        ],
        output='screen',
    )

    return LaunchDescription([
        algorithm_arg,
        tof_aggregator,
        avoidance,
    ])
