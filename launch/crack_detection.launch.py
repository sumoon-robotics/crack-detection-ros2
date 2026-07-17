"""
crack_detection.launch.py

Launches the three-stage crack detection + 3D measurement pipeline.
Edit config/params.yaml for your calibration and model paths.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('crack_livo_fusion')
    params_file = PathJoinSubstitution([pkg, 'config', 'params.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=params_file,
                              description='Path to parameter file'),

        # Stage 1: Segmentation (DeepLabV3+ + SAM)
        Node(
            package='crack_livo_fusion',
            executable='crack_segmentation_node.py',
            name='crack_seg',
            parameters=[LaunchConfiguration('params_file')],
            remappings=[('/left_camera/image', '/left_camera/image')],
        ),

        # Stage 2: Mask → 3D cloud fusion
        Node(
            package='crack_livo_fusion',
            executable='crack_fusion_node.py',
            name='crack_fusion',
            parameters=[LaunchConfiguration('params_file')],
        ),

        # Stage 3: Width + length measurement
        Node(
            package='crack_livo_fusion',
            executable='crack_measurement_node.py',
            name='crack_measure',
            parameters=[LaunchConfiguration('params_file')],
        ),

        # Stage 4: Live persistent map accumulation (true growing map,
        # not RViz decay-time rendering) with crack labels carried through
        Node(
            package='crack_livo_fusion',
            executable='crack_map_accumulator_node.py',
            name='crack_map_accumulator',
            parameters=[LaunchConfiguration('params_file')],
        ),
        # Range-limited point cloud for RViz (local 1.5m sliding window map)
        Node(
            package='crack_livo_fusion',
            executable='range_limited_cloud_node.py',
            name='range_limited_cloud',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
