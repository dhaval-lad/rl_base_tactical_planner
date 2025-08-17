"""
Launches the training node for reinforcement learning.
This launch file is responsible for starting the training process.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Create a LaunchDescription object to hold launch actions
    ld = LaunchDescription()

    # Get the path to the training parameters YAML file
    training_parameters = os.path.join(
        get_package_share_directory('outdoor_robot_spawner'),
        'config',
        'training_parameters.yaml'
    )

    # Define the training node, passing in the parameters file
    start_training = Node(
        package='outdoor_robot_spawner',
        executable='start_training',
        parameters=[training_parameters]
    )

    # Add the training node to the launch description
    ld.add_action(start_training)

    # Return the launch description to be executed by ROS2 launch system
    return ld