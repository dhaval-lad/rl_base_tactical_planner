from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
import os
from ament_index_python.packages import get_package_share_directory
import subprocess
import tempfile
from tf_transformations import euler_from_quaternion
import scipy.interpolate as si
import heapq
from vesc_msgs.msg import ObjectDetectionList

class RobotController(Node):
    """
    This class defines all the methods to:
        - Publish actions to the agent (move the robot)
        - Subscribe to sensors of the agent (get laser scans, object classification detection and robot position)
        - Reset the simulation

    Topics list:
        - /cmd_vel : linear and angular velocity of the robot
        - /odom : odometry readings of the chassis of the robot
        - /scan : laser readings
        - /detections_publisher : object class, depth and angel (processed camera data)
    
    Services used:
        - /demo/set_entity_state : sets the new state of the robot and target when an episode ends

    Services not used:
        - /reset_simulation : resets the gazebo simulation
        - /delete_entity : unspawns the robot from the simulation
        - /spawn_entity : spawns the robot in the simulation in a semi-random position
    """
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has just been created")

        # Action publisher
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        # Position subscriber
        self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 1)
        # Laser subscriber
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)
        # Object detection subscriber
        self.subscription = self.create_subscription(ObjectDetectionList,'detections_publisher',self.detections_callback,1)
        # Reset model state client - this resets the pose and velocity of a given model within the world
        self.client_state = self.create_client(SetEntityState, "/demo/set_entity_state")

        # Reset simulation client - UNUSED
        self.client_sim = self.create_client(Empty, "/reset_simulation")
        
        ## Spawn robot from SDF file.
        # Get the directory of the sdf of the robot
        # self._pkg_dir = os.path.join(
        #     get_package_share_directory("hospital_robot_spawner"), "models",
        # "4WD_robot", "model.sdf")

        ## Spawn robot from URDF file.
        self.urdf_file_path = os.path.join(get_package_share_directory('linorobot2_description'), "urdf", "robots", "grass_hopper.urdf.xacro")

        # Initialize param 
        self.speed = 0.2
        self.speed_real = 0.085 # max speed for real GrassHopper robot. 

        # initialize variable for object detect
        self.detections = {}
        self.person_detected = 0
        self.dustbin_detected = 0
        self.bicycle_detected = 0
        self.person_distance = -1.0
        self.dustbin_distance = -1.0
        self.bicycle_distance = -1.0
        self.person_angle = -1.0
        self.dustbin_angle = -1.0
        self.bicycle_angle = -1.0

        # Initialize attributes - This will be immediately re-written when the simulation starts
        self._agent_location = np.array([np.float32(-19.0),np.float32(52.5)])           # agent initial location according to the environment. 

        # Initialize attribute to store laser scan data and also devied them in respected regions. 
        self._laser_reads = np.array([np.float32(10)] * 270)                           #270°
        self.laser_reads_front = np.array([np.float32(10)] * 30)                       # 30°
        self.laser_reads_corners = np.array([np.float32(10)] * 70)                     # 70°
        self.laser_reads_from_sides = np.array([np.float32(10)] * 120)                 # 120°
        
        # initialize orientationlist for storing robot orientation.
        self.orientation_list = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # convert URDF to SDF
    def convert_urdf_to_sdf(self,urdf_file):
        # Use subprocess to call 'xacro' command-line tool to expand macros
        self.urdf_cmd = ['xacro', '--inorder', urdf_file]
        self.urdf_result = subprocess.run(self.urdf_cmd, capture_output=True, text=True)
        self.urdf_output = self.urdf_result.stdout
        
        # Create a temporary file to save the expanded URDF content
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as self.temp_urdf_file:
            self.temp_urdf_file.write(self.urdf_output)
            self.temp_urdf_file_path = self.temp_urdf_file.name

        # Now convert the expanded URDF to SDF
        self.sdf_cmd = ['gz', 'sdf', '-p', self.temp_urdf_file_path]
        self.sdf_result = subprocess.run(self.sdf_cmd, capture_output=True, text=True)
        self.sdf_output = self.sdf_result.stdout

        # Remove the temporary URDF file
        os.remove(self.temp_urdf_file_path)

        # Return the SDF output
        return self.sdf_output
    
    # Method to send the velocity command to the robot
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    # Method that saves the position of the robot each time the topic /odom receives a new message
    def pose_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x,-50,50)), np.float32(np.clip(msg.pose.pose.position.y,-66,66))])  # position cliped between max and min position according to simulated environment.
        self.orientation_q = msg.pose.pose.orientation
        self.orientation_list = np.array([float(self.orientation_q.x), float(self.orientation_q.y), float(self.orientation_q.z), float(self.orientation_q.w)], dtype=np.float32)
        (_, _, self._agent_orientation) = euler_from_quaternion(self.orientation_list)
        # self.get_logger().info("Agent position: " + str(self._agent_location))
        # self.get_logger().info("Agent orientation: " + str(math.degrees(self._agent_orientation)))
        self._done_pose = True

    # Method to save the class, depth and angle detected by camera from the topic /detections_publisher
    def detections_callback(self, msg):
        # Initialize variables for each type of detected object
        self.detections.clear()
        # for clearing old data
        self.person_detected = 0
        self.dustbin_detected = 0
        self.bicycle_detected = 0
        self.person_distance = -1
        self.dustbin_distance = -1
        self.bicycle_distance = -1
        self.person_angle = -1
        self.dustbin_angle = -1
        self.bicycle_angle = -1

        for detection in msg.detections:
            object_class = detection.object_class
            distance = detection.distance
            angle = detection.angle

            # If object_class is already in the dictionary, update it only if the new distance is smaller
            if object_class in self.detections:
                if distance < self.detections[object_class][0]:
                    self.detections[object_class] = [distance,angle]
            else:
                self.detections[object_class] = [distance,angle]

        # Print the entire dictionary of detected objects and their distances
        # self.get_logger().info(f"Detections: {self.detections}")

        # Print the detected objects and their distances
        for object_class, (distance,angle) in self.detections.items():
            # self.get_logger().info(f"Detected {object_class} at {distance:.2f} meters & {angle:.2f}°")

            if object_class == "person":
                self.person_detected = 1
                self.person_distance = distance
                self.person_angle = angle
            if object_class == "Dustbin":
                self.dustbin_detected = 1
                self.dustbin_distance = distance
                self.dustbin_angle = angle
            if object_class == "bicycle":
                self.bicycle_detected = 1
                self.bicycle_distance = distance
                self.bicycle_angle = angle


        ## print all information for debugging.
        # self.get_logger().info("Person: " + str(self.person_detected))
        # self.get_logger().info("Dustbin: " + str(self.dustbin_detected))
        # self.get_logger().info("Bicycle: " + str(self.bicycle_detected))
        # self.get_logger().info("Person Distance: " + str(self.person_distance))
        # self.get_logger().info("Dustbin Distance: " + str(self.dustbin_distance))
        # self.get_logger().info("Bicycle Distance: " + str(self.bicycle_distance))
        # self.get_logger().info("Person angle: " + str(self.person_angle))
        # self.get_logger().info("Dustbin angle: " + str(self.dustbin_angle))
        # self.get_logger().info("Bicycle angle: " + str(self.bicycle_angle))

    # Method that saves the laser reads each time the topic /scan receives a new message
    def laser_callback(self, msg: LaserScan):
        self._laser_reads = np.array(msg.ranges)
        # Converts inf values to 10
        self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
        # self.get_logger().info("Laser Read(from right to left): " + str(self._laser_reads))
        
        ## for reward calculation
        self.laser_reads_front = np.array(self._laser_reads[75:105])                                                                   ## 75° to 105°
        self.laser_reads_corners = np.concatenate((np.array(self._laser_reads[40:75]),np.array(self._laser_reads[105:140])))           ## 40° to 75° and 105° to 140°
        self.laser_reads_from_sides = np.concatenate((np.array(self._laser_reads[0:40]),np.array(self._laser_reads[140:220])))         ## 0° to 40° and 140° to 220°

        self._done_laser = True

    # Method to inflate the costmap received from the topic /costmap according to the obstacle deteced and the need of the environments.  
    def inflate_obstacles(self, costmap, inflation_radius):
        inflated_costmap = costmap.copy()
        obstacle_indices = np.argwhere(costmap == 1)

        for obstacle_index in obstacle_indices:
            row, col = obstacle_index
            for i in range(-inflation_radius, inflation_radius + 1):
                for j in range(-inflation_radius, inflation_radius + 1):
                    if np.sqrt(i**2 + j**2) <= inflation_radius:
                        new_row = row + i
                        new_col = col + j
                        if 0 <= new_row < costmap.shape[0] and 0 <= new_col < costmap.shape[1]:
                            inflated_costmap[new_row, new_col] = 1

        return inflated_costmap
    
    # To find the nearest free cell when robot origin move into the costmap (occupied cell) and there is no starting point for local path finder to find path. 
    # this will find new free cell for start point of local path and use it to find local path. 
    def find_nearest_free_cell(self, inflated_costmap, start_row, start_col):
        from collections import deque

        queue = deque([(start_row, start_col)])
        visited = set((start_row, start_col))

        while queue:
            row, col = queue.popleft()
            if inflated_costmap[row, col] == 0:
                return row, col

            for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + i, col + j
                if 0 <= new_row < inflated_costmap.shape[0] and 0 <= new_col < inflated_costmap.shape[1]:
                    if (new_row, new_col) not in visited:
                        queue.append((new_row, new_col))
                        visited.add((new_row, new_col))

        return start_row, start_col  # Return the original position if no free cell is found

    # Transform map co-ordinates to global co-ordinates
    def transform_to_global(self,x, y, originX, originY, yaw):
        global_x = x * self.resolution
        global_y = y * self.resolution
        rotated_x = global_x * np.cos(yaw) - global_y * np.sin(yaw)
        rotated_y = global_x * np.sin(yaw) + global_y * np.cos(yaw)
        return rotated_x + originX, rotated_y + originY
    
    # Translate the global coordinates relative to the local map
    def transform_to_local(self, global_x, global_y, originX, originY, yaw):
        translated_x = global_x - originX
        translated_y = global_y - originY

        # Apply the inverse rotation
        local_x = translated_x * np.cos(-yaw) - translated_y * np.sin(-yaw)
        local_y = translated_x * np.sin(-yaw) + translated_y * np.cos(-yaw)

        # Scale by the resolution to get the local grid coordinates
        local_x = local_x / self.resolution
        local_y = local_y / self.resolution

        return local_x, local_y
    
    def heuristic(self,a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    
    # To find local path around the obstacle within the map from robot position(start) to next waypoint in the path(end point)
    def astar(self,array, start, goal):
        # self.get_logger().info(f"Starting A* from {start} to {goal}")
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data = data + [start]
                data = data[::-1]
                return data
            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:                
                        if array[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        # If no path to goal was found, return closest path to goal
        if goal not in came_from:
            closest_node = None
            closest_dist = float('inf')
            for node in close_set:
                dist = self.heuristic(node, goal)
                if dist < closest_dist:
                    closest_node = node
                    closest_dist = dist
            if closest_node is not None:
                data = []
                while closest_node in came_from:
                    data.append(closest_node)
                    closest_node = came_from[closest_node]
                data = data + [start]
                data = data[::-1]
                return data
        return False

    def bspline_planning(self,array, sn):
        try:
            array = np.array(array)
            x = array[:, 0]
            y = array[:, 1]
            N = 2
            t = range(len(x))
            x_tup = si.splrep(t, x, k=N)
            y_tup = si.splrep(t, y, k=N)

            x_list = list(x_tup)
            xl = x.tolist()
            x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

            y_list = list(y_tup)
            yl = y.tolist()
            y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

            ipl_t = np.linspace(0.0, len(x) - 1, sn)
            rx = si.splev(ipl_t, x_list)
            ry = si.splev(ipl_t, y_list)
            path = [(rx[i],ry[i]) for i in range(len(rx))]
        except:
            path = array
        return path
    
    # Method to calculate linear and angular velocity to follow Global or Local path. In Simulation with the help of pure pursuit. 
    ## UNCOMENT THE BELOW FUNCTION WHEN WORKING WITH SIMUATION.
    def pure_pursuit(self, current_x, current_y, current_heading, path, index, lookahead_distance):
        closest_point = None
        v = self.speed
        for i in range(index, len(path)):
            x = path[i][0]
            y = path[i][1]
            distance = math.hypot(current_x - x, current_y - y)
            # if self.lookahead_distance < distance:
            if lookahead_distance < distance:
                closest_point = (x, y)
                index = i
                break

        if closest_point is not None:
            target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
            desired_steering_angle = target_heading - current_heading
        else:
            target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
            desired_steering_angle = target_heading - current_heading
            index = len(path) - 1

        if desired_steering_angle > math.pi:
            desired_steering_angle -= 2 * math.pi
        elif desired_steering_angle < -math.pi:
            desired_steering_angle += 2 * math.pi

        if desired_steering_angle > math.pi / 6 or desired_steering_angle < -math.pi / 6:
            sign = 1 if desired_steering_angle > 0 else -1
            desired_steering_angle = sign * math.pi / 4
            v = 0.0

        return v, desired_steering_angle, index
    
    # Method to calculate linear and angular velocity to follow Global or Local path. In real GrassHopper. 
    # This is because real GrassHopper robot do not have PID tune. So this speed and angles and look ahead distance is tune and tested on real robot. 
    ## UNCOMENT THE BELOW FUNCTION WHEN WORKING WITH REAL ROBOT.
    # def pure_pursuit(self, current_x, current_y, current_heading, path, index, lookahead_distance):   # lookahead distance = 0.20 speed 0.085
    #     closest_point = None

    #     ## Reduse starting velocity for smooth start until 1st point reached. 
    #     if index == 0:
    #         v = 0.8 * self.speed_real
    #     else:
    #         v = self.speed_real

    #     ## Search for the closest point on the path ahead of the robot within the lookahead distance
    #     for i in range(index, len(path)):
    #         x = path[i][0]
    #         y = path[i][1]
    #         distance = math.hypot(current_x - x, current_y - y)
    #         if lookahead_distance < distance:
    #             closest_point = (x, y)
    #             index = i
    #             break

    #     ## If a closest point is found, calculate desired heading
    #     if closest_point is not None:
    #         target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
    #         desired_steering_angle = target_heading - current_heading
    #     else:
    #         ## If no point is found, aim towards the last point on the path
    #         target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
    #         desired_steering_angle = target_heading - current_heading
    #         index = len(path) - 1

    #     ## Normalize the steering angle to the range of [-π, π]
    #     if desired_steering_angle > math.pi:
    #         desired_steering_angle -= 2 * math.pi
    #     elif desired_steering_angle < -math.pi:
    #         desired_steering_angle += 2 * math.pi

    #     ## due to the velocity constraints on real robot and no PID the angular speed/desired_stering_angle needs to be redused by (1/4) and clipped between (-0.075, 0.075) 
    #     ## tired and tested best congigured. 
    #     desired_steering_angle_ = desired_steering_angle/4.0
    #     desired_steering_angle_ = np.clip(desired_steering_angle_,-0.075,0.075)
        
    #     ## If steering angle is too large, reduce speed and limit steering to a maximum
    #     if desired_steering_angle > math.pi / 7.2 or desired_steering_angle < -math.pi / 7.2:  ## 25°
    #         sign = 1 if desired_steering_angle > 0 else -1
    #         desired_steering_angle_ = sign * 0.075    ## tune and tested max angular velocity for smooth path following. 
    #         v = 0.00 ## make linear velocity 0.0

    #     return v, desired_steering_angle_, index
    
    # Method to find nearest point on path so that the robot do not go for predefine first point on path and get stuck in loop.
    def find_nearest_point_index(self, path):
        min_distance = float('inf')
        nearest_index = 0
        self.current_x = self._agent_location[0]
        self.current_y = self._agent_location[1]
        # self.path = self.waypoints_locations[self._path]
        self.path = path 

        for i, (x, y) in enumerate(self.path):
            distance = math.hypot(self.current_x - x, self.current_y - y)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        return nearest_index

    # Method to set the state of the robot when an episode ends - /demo/set_entity_state service
    def call_set_robot_state_service(self, robot_pose=[-19.0, 52.5, 0.1, 0.0, 0.99]):  # Default Pose of the robot 
        while not self.client_state.wait_for_service(5.0):
            self.get_logger().warn("Waiting for service...1")

        self.get_logger().info("Service is available, preparing request")

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        # Pose (position and orientation)
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.position.z = float(robot_pose[2])
        request.state.pose.orientation.z = float(robot_pose[3])
        request.state.pose.orientation.w = float(robot_pose[4])
        # setting all velocities to 0 
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))

    # Method that elaborates the future obtained by callig the call_set_robot_state_service method
    def callback_set_robot_state(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
            self._done_set_rob_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    # Method to set the state of the target when an episode ends - /demo/set_entity_state service
    def call_set_target_state_service(self, position=[-11.0, 56.0]): # Default pose of the target
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...2")

        request = SetEntityState.Request()
        request.state.name = "Target"
        # Pose (position and orientation)
        request.state.pose.position.x = float(position[0])
        request.state.pose.position.y = float(position[1])

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))

    # Method that elaborates the future obtained by callig the call_set_target_state_service method
    def callback_set_target_state(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))
