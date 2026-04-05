"""
Launch file for full truss2 simulation with ROS2 integration.

Launches:
  1. ros_gz_bridge — bridges ToF, camera, and IMU topics from Gazebo to ROS2
  2. micro_xrce_dds_agent — PX4 ↔ ROS2 uXRCE-DDS bridge
  3. RTAB-Map stereo SLAM
  4. SLAM bridge node (vision pose → PX4 EKF2)
  5. ToF aggregator + avoidance nodes

Prerequisites:
  - PX4 SITL must be started separately:
      cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss2 make px4_sitl gz_x500_tof
  - micro-XRCE-DDS agent must be installed

Usage:
  ros2 launch px4_sim_bringup sim_truss2.launch.py
  ros2 launch px4_sim_bringup sim_truss2.launch.py algorithm:=vfh3d
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('px4_sim_bringup')
    avoidance_dir = get_package_share_directory('px4_tof_avoidance')
    vslam_dir = get_package_share_directory('px4_vslam')

    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='fgm3d',
        description='Obstacle avoidance algorithm: fgm3d, vfh3d, or dwa3d',
    )

    # --- ros_gz_bridge ---
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(bringup_dir, 'config', 'gz_bridge.yaml'),
        }],
        output='screen',
    )

    # --- micro-XRCE-DDS agent (PX4 ↔ ROS2) ---
    uxrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # --- RTAB-Map stereo SLAM ---
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vslam_dir, 'launch', 'slam_sim.launch.py')
        ),
    )

    # --- ToF avoidance ---
    avoidance_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(avoidance_dir, 'launch', 'avoidance.launch.py')
        ),
        launch_arguments={'algorithm': LaunchConfiguration('algorithm')}.items(),
    )

    return LaunchDescription([
        algorithm_arg,
        gz_bridge,
        uxrce_agent,
        slam_launch,
        avoidance_launch,
    ])
