import os 
import sys
import rclpy
from gazebo_msgs.srv import SpawnEntity
import math
import subprocess
import tempfile

# Fixed source paths
PKG_SRC_DIR = '/home/dhaval_lad/dhaval_ws/src/Outdoor_navigation_decision_making/outdoor_robot_spawner'
GRASSHOPPER_DESC_DIR = '/home/dhaval_lad/dhaval_ws/src/grasshopper_description'

def main():
    """
    Main function for spawning a robot and a target in Gazebo using ROS 2.
    """

    # Get input arguments from user (robot name, namespace, x, y, z)
    argv = sys.argv[1:]
    
    # Initialize ROS 2 Python client library
    rclpy.init()
        
    # Create a ROS 2 node for spawning entities
    node = rclpy.create_node("entity_spawner")
    node.get_logger().info("Starting the entity spawner node.")

    # Create a client for the `/spawn_entity` service
    node.get_logger().info('Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    # Wait until the `/spawn_entity` service is available
    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")

    # Get the path to the URDF file for the robot model
    urdf_file_path = os.path.join(GRASSHOPPER_DESC_DIR, "urdf", "grass_hopper.urdf.xacro")

    # Function to convert URDF to SDF using xacro and gz tools
    def convert_urdf_to_sdf(urdf_file):
        # Expand xacro macros in the URDF file
        urdf_cmd = ['xacro', '--inorder', urdf_file]
        urdf_result = subprocess.run(urdf_cmd, capture_output=True, text=True)
        urdf_output = urdf_result.stdout

        # Write the expanded URDF to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_urdf_file:
            temp_urdf_file.write(urdf_output)
            temp_urdf_file_path = temp_urdf_file.name

        # Convert the expanded URDF to SDF format using gz
        sdf_cmd = ['gz', 'sdf', '-p', temp_urdf_file_path]
        sdf_result = subprocess.run(sdf_cmd, capture_output=True, text=True)
        sdf_output = sdf_result.stdout

        # Remove the temporary URDF file
        os.remove(temp_urdf_file_path)

        # Return the SDF XML string
        return sdf_output
    
    # Convert the robot's URDF to SDF format
    sdf_xml = convert_urdf_to_sdf(urdf_file_path)
    
    # ----------------- SPAWN ROBOT MODEL -----------------
    # Create a request to spawn the robot entity
    request = SpawnEntity.Request()
    request.name = argv[0]  # Robot name from command line argument
    request.xml = sdf_xml   # SDF XML string for the robot model
    request.robot_namespace = argv[1]  # Robot namespace from argument
    request.initial_pose.position.x = float(argv[2])  # Initial X position
    request.initial_pose.position.y = float(argv[3])  # Initial Y position
    request.initial_pose.position.z = float(argv[4])  # Initial Z position

    # Set the initial orientation (steering angle) of the robot (default 0 degrees)
    desired_angle = float(math.radians(0))
    request.initial_pose.orientation.z = float(math.sin(desired_angle / 2))
    request.initial_pose.orientation.w = float(math.cos(desired_angle / 2))

    # Send the spawn request for the robot
    node.get_logger().info("Sending service request to `/spawn_entity` for robot")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    # ----------------- SPAWN TARGET MODEL -----------------
    # Get the path to the SDF file for the target object
    target_sdf_file_path = os.path.join(PKG_SRC_DIR, "models", "Target", "model.sdf")

    # Create a request to spawn the target entity
    request = SpawnEntity.Request()
    request.name = "Target"
    # Read the SDF XML for the target from file
    request.xml = open(target_sdf_file_path, 'r').read()
    # Set the initial position for the target (change as needed)
    request.initial_pose.position.x = float(-11.0)
    request.initial_pose.position.y = float(56.0)
    request.initial_pose.position.z = float(0.01)

    # Send the spawn request for the target
    node.get_logger().info("Sending service request to `/spawn_entity` for target")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    # Shutdown the node after spawning both entities
    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
