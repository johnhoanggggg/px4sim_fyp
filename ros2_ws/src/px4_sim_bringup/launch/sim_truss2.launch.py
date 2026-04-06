"""
Launch file for truss2 simulation with ROS2 integration.

Launches:
  1. ros_gz_bridge — bridges ToF, camera, IMU, and clock from Gazebo to ROS2
  2. micro-XRCE-DDS agent (PX4 ↔ ROS2)
  3. ToF aggregator + avoidance nodes

SLAM is launched separately once the drone is airborne:
  ros2 launch px4_vslam slam_sim.launch.py

Prerequisites:
  - PX4 SITL must be started separately:
      cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss2 make px4_sitl gz_x500_tof

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
from launch_ros.actions import Node  # noqa: F401


def generate_launch_description():
    avoidance_dir = get_package_share_directory('px4_tof_avoidance')

    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='fgm3d',
        description='Obstacle avoidance algorithm: fgm3d, vfh3d, or dwa3d',
    )

    # --- ros_gz_bridge ---
    # Use command-line argument style for maximum compatibility
    bridge_args = [
        # Clock (required for use_sim_time)
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # ToF sensors (12x gpu_lidar) — named /tof/s0..s9 to be valid ROS2 topics
        '/tof/s0@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s1@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s2@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s3@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s4@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s5@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s6@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s7@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s8@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/s9@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/up@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/down@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        # OAK-D Lite stereo camera
        '/oakd/left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        '/oakd/right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        '/oakd/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/oakd/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        # OAK-D Lite IMU
        '/oakd/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
    ]

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen',
    )

    # --- micro-XRCE-DDS agent (snap install) ---
    uxrce_agent = ExecuteProcess(
        cmd=['micro-xrce-dds-agent', 'udp4', '-p', '8888'],
        output='screen',
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
        avoidance_launch,
    ])
