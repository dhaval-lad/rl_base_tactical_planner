"""
Demo for spawn_entity.
Launches Gazebo and spawns a model
"""
# A bunch of software packages that are needed to launch ROS2
import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'outdoor_world_0.world'
    pkg_dir = get_package_share_directory('outdoor_robot_spawner')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')
    #os.environ["GAZEBO_RESOURCE_PATH"] = os.path.join(pkg_dir, 'worlds')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_dir, 'launch')

    gazebo = ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'],
            output='screen')

    # GAZEBO_MODEL_PATH has to be correctly set for Gazebo to be able to find the model
    #spawn_entity = Node(package='gazebo_ros', node_executable='spawn_entity.py',
    #                    arguments=['-entity', 'demo', 'x', 'y', 'z'],       #for initial oriantation change "desired_angle" param in spawn_demo.py line 91
    #                    output='screen')
    spawn_entity = Node(package='outdoor_robot_spawner', executable='spawn_demo',
                        arguments=['GrassHopper', '', '-19.0', '52.5', '0.1'],
                        output='screen')

    return LaunchDescription([
        gazebo,
        spawn_entity,

        # Static transform between different link is needed here in simulation, because in spawn_demo.py URDF of robot model will be converted into SDF.
        # In which the static transform between the links is not getting converted into SDF. So, static transform between the links needs to be published saperately. 
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_1st',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']                     # this 'map' to 'odom' TF is needed for dynamic map creation. 
            ),                          # this is required in simulation only because in here we do not use dual EKF. In real GrassHopper dual EKF is running for odometry which will generate this TF.
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_2nd',
            arguments=['0', '0', '0.075', '0', '0', '0', 'base_footprint', 'base_link']
            ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_3rd',
            # arguments=['0.263', '0', '0.245', '0', '0', '0', 'base_link', 'laser']   # without upper base + laser at front
            arguments=['0', '0', '0.48', '0', '0', '0', 'base_link', 'laser']          # with upper base + laser at center
            ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_4th',
            arguments=['0', '0', '0.17', '0', '0', '0', 'base_link', 'imu_link']
            ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_5th',
            arguments=['0.315', '0', '0.155', '0', '0', '0', 'base_link', 'camera_link']
            ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_6th',
            arguments=['0', '0', '0', '-1.57', '0', '-1.57', 'camera_link', 'camera_depth_link']
            ),
        Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher_7th',
            arguments=['0', '0', '0.165', '0', '0', '0', 'base_link', 'upper_base_link']
            )
    ])
