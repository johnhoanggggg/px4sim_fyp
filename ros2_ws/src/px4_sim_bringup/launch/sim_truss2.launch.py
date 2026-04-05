"""
Launch file for full truss2 simulation with ROS2 integration.

Launches:
  1. ros_gz_bridge — bridges ToF, camera, and IMU topics from Gazebo to ROS2
  2. RTAB-Map stereo SLAM
  3. SLAM bridge node (vision pose → PX4 EKF2)
  4. ToF aggregator + avoidance nodes

Prerequisites:
  - PX4 SITL must be started separately:
      cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss2 make px4_sitl gz_x500_tof
  - In the PX4 shell, start the DDS client:
      uxrce_dds_client start -t udp -h 127.0.0.1 -p 8888

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
    # Use command-line argument style for maximum compatibility
    bridge_args = [
        # ToF sensors (12x gpu_lidar)
        '/tof/0@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/1@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/2@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/3@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/4@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/5@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/6@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/7@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/8@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/tof/9@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
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
