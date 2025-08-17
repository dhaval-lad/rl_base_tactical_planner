"""
Launches the trained agent node.
This launch file is responsible for starting the trained agent process.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Create a LaunchDescription object to hold launch actions
    ld = LaunchDescription()

    # Define the trained agent node
    # This node will run the executable 'trained_agent' from the 'outdoor_robot_spawner' package
    trained_agent = Node(
        package='outdoor_robot_spawner',
        executable='trained_agent',
        # You can add parameters or remappings here if needed in the future
    )

    # Add the trained agent node to the launch description
    ld.add_action(trained_agent)

    # Return the launch description to be executed by ROS2 launch system
    return ld