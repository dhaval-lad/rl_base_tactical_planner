"""
ROS 2 node to spawn a mobile robot with 270 degree LIDAR and depth camera.

Author:
  - Tommaso Van Der Meer
  - tommaso.vandermeer@student.unisi.it
Modified/contributor:
  - Dhaval Lad (URDF to SDF)
"""
import os # Operating system library
import sys # Python runtime environment library
import rclpy # ROS Client Library for Python

# Package Management Library
from ament_index_python.packages import get_package_share_directory 

# Gazebo's service to spawn a robot
from gazebo_msgs.srv import SpawnEntity

import math

import subprocess
import tempfile

def main():

    """ Main for spawning a robot node """
    # Get input arguments from user
    argv = sys.argv[1:]
    
    # Start node
    rclpy.init()
        
    # Create the node
    node = rclpy.create_node("entity_spawner")
    node.get_logger().info("HERE WE GOOO")

    # Show progress in the terminal window
    node.get_logger().info(
        'Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    # Get the spawn_entity service
    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")

    ## SPAWN ROBOT
    # for SDF file of robot model then use the below line of code to spawn the robot.
    # sdf_file_path = os.path.join(
    #     get_package_share_directory("hospital_robot_spawner"), "models",
    #     "4WD_robot", "model.sdf")

    # for URDF file of robot model use the following code to spawn the robot. 
    urdf_file_path = os.path.join(get_package_share_directory('linorobot2_description'), "urdf", "robots", "grass_hopper.urdf.xacro")

    # convert the URDF robot model to SDF and then spawn the robot. 
    def convert_urdf_to_sdf(urdf_file):
        # Use subprocess to call 'xacro' command-line tool to expand macros
        urdf_cmd = ['xacro', '--inorder', urdf_file]
        urdf_result = subprocess.run(urdf_cmd, capture_output=True, text=True)
        urdf_output = urdf_result.stdout

        # Create a temporary file to save the expanded URDF content
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_urdf_file:
            temp_urdf_file.write(urdf_output)
            temp_urdf_file_path = temp_urdf_file.name

        # Now convert the expanded URDF to SDF
        sdf_cmd = ['gz', 'sdf', '-p', temp_urdf_file_path]
        sdf_result = subprocess.run(sdf_cmd, capture_output=True, text=True)
        sdf_output = sdf_result.stdout

        # Remove the temporary URDF file
        os.remove(temp_urdf_file_path)

        # Return the SDF output
        return sdf_output
    
    # Convert URDF to SDF
    sdf_xml = convert_urdf_to_sdf(urdf_file_path)
    
    ## SPAWN ROBOT MODEL
    # Set data for request
    request = SpawnEntity.Request()
    request.name = argv[0]
    # request.xml = open(sdf_file_path, 'r').read()         # for SDF file of robot model unable this.
    request.xml = sdf_xml                                   # for URDF file of robot model unable this. 
    request.robot_namespace = argv[1]
    request.initial_pose.position.x = float(argv[2])
    request.initial_pose.position.y = float(argv[3])
    request.initial_pose.position.z = float(argv[4])

    # desired stering angele for spawning the robot is set to 0° for default. change it according to the environment. 
    desired_angle = float(math.radians(0))
    request.initial_pose.orientation.z = float(math.sin(desired_angle/2))
    request.initial_pose.orientation.w = float(math.cos(desired_angle/2))

    node.get_logger().info("Sending service request to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    ## SPAWN TARGET
    # Get path to the target
    target_sdf_file_path = os.path.join(
        get_package_share_directory("outdoor_robot_spawner"), "models",
        "Target", "model.sdf")

    request = SpawnEntity.Request()
    request.name = "Target"
    request.xml = open(target_sdf_file_path, 'r').read()
    # X,Y, Z default position for target to be spawn. change it acording to environment.
    request.initial_pose.position.x = float(-11.0)
    request.initial_pose.position.y = float(56.0)
    request.initial_pose.position.z = float(0.01)

    node.get_logger().info("Sending service request to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
