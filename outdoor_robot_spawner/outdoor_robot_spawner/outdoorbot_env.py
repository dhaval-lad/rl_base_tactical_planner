import rclpy
from gymnasium import Env
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
from outdoor_robot_spawner.robot_controller import RobotController
import math
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from tf_transformations import euler_from_quaternion
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Fixed source path for this package
PKG_SRC_DIR = '/home/dhaval_lad/dhaval_ws/src/Outdoor_navigation_decision_making/outdoor_robot_spawner'

class OutDoorEnv(RobotController, Env):
    """
    This class defines the RL environment. Here are defined:
        - Action space
        - Observation space
        - Target location
        - Global path
    
    The methods available are:
        - step: makes a step of the current episode
        - reset: resets the simulation environment when an episode is finished
        - close: terminates the environment

    This class inherits from both RobotController and Env.
    Env is a standard class of Gymnasium library which defines the basic needs of an RL environment.
    
    RobotController is a ROS2 Node used to control the agent. It includes the following attributes:
        - _agent_location: current position of the robot (in gazebo's coordinate system)
        - _laser_reads: current scan of the LIDAR
        - _detection_publisher: store processed camera data in variables
    
    And the following methods (only the ones usefull here):
        - send_velocity_command: imposes a velocity to the agent (the robot)
        - call_reset_simulation_service: resets the simulation
        - call_reset_robot_service: resets the robot position to desired position
        - call_reset_target_service: resets the target position to desired position
        - convert_urdf_to_sdf: convert URDF to SDF and then spawan the robot
        - detections_callback: store tha processed camera data in to variables
        - inflate_obstacles: increase the inflation of the detected obstacle in map
        - find_nearest_free_cell: to find the nearest free cell when robot get stuck within the inflated obstacle
        - transform_to_global: Transform map co-ordinates to global co-ordinates
        - transform_to_local: Translate the global coordinates relative to the local 
        - astar: find local path around the obstacle from robot origin to nex waypoint in path
        - pure_pursuit: calculate controlled velocity to follow local and global path 
        - find_nearest_point_index: find nearest waypoint in global path so that it can start from in-between and don't go for pre defined 1st waypoint on path
    """

    def __init__(self):
        
        # Initialize the Robot Controller Node
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        # Local costmap subscriber
        self.local_costmap_sub = self.create_subscription(OccupancyGrid, '/costmap', self.local_map_callback, 1)

        # Publish pre define Global path to later visualize it in RViz
        self.path_pub = self.create_publisher(Path, 'selected_path', 1)
        # Timer to publish global path (2hz)
        self.path_timer = self.create_timer(0.5, self.publish_path)
        # Publish Local path found with the help of A*
        self.local_path_pub = self.create_publisher(Path, 'local_path', 1)

        # ENVIRONMENT PARAMETERS
        self.robot_name = 'GrassHopper'
        # Initializes the Target location (x,y) - effective only for randomization level 0 and 1 (see below)
        self._target_location = np.array([-11, 56.0], dtype=np.float32) 

        # Only one mode: the robot must reach a short-range target approximately 20 meters away.
        # Global path is already defined; no global path planning or multiple target evaluation mode.
        self._randomize_env_level = 1

        # If True, the observation space is normalized between [0,1] (except distance which is between [0,10], see below)
        self._normalize_obs = True

        # If True, the target will appear on the simulation - SET FALSE FOR TRAINING (slows down the training)
        self._visualize_target = False

        # 0: simple reward, 1: risk seeker, 2: adaptive heuristic (Checkout the method compute_reward)
        # 3: Hybrid Path Progress and Contextual Action Reward
        self._reward_method = 3

        # Initializes the min distance from target for which the episode is concluded with success
        # This has to be at least 0.1 more than self._minimum_dist_from_obstacles to add safety margin
        self._minimum_dist_from_target = 0.55

        # Minimum distance from an obstacle for which the episode is considered a failure.
        # The robot's rectangular shape results in three distinct collision distances relative to the laser position:
        #   - 0.45 meters at the front (robot length 0.6m / 2 + 0.15m for sensor setup and safety)
        #   - 0.40 meters at the corners (calculated according to robot geometry)
        #   - 0.35 meters at the sides (calculated according to robot geometry)
        self._minimum_dist_from_obstacles = 0.45
        self._minimum_dist_from_obstacles_for_coeners = 0.40
        self._minimum_dist_from_obstacles_for_sides = 0.35
        
        # Use short_path=True for current trained model (short, fixed path).
        # Set to False for longer, more complex paths with random starts.
        self.short_path = True

        # Parameters below are used only when self._reward_method == 2 (adaptive heuristic reward)
        self._attraction_threshold = 2      # Attraction threshold for adaptive heuristic
        self._attraction_factor = 1         # Attraction factor for adaptive heuristic
        self._repulsion_threshold = 2       # Repulsion threshold for adaptive heuristic
        self._repulsion_factor = 0.1        # Repulsion factor for adaptive heuristic
        self._distance_penalty_factor = 1/8 # Distance penalty factor for adaptive heuristic

        # Used in reward_method = 3 for checking path completion percentage
        self.max_steps_so_far = 100  # Initialize with reasonable default

        # Initialize step count
        self._num_steps = 0
        # Initialize episode count
        self._num_episodes = 0        

        # If we are evaluating the model, we want to set the SEED.
        # Since there is only one environment level, always set the random seed for reproducibility during evaluation.
        np.random.seed(4)

        # Warning for training
        if self._visualize_target == True:
            self.get_logger().info("WARNING! TARGET VISUALIZATION IS ACTIVATED, SET IT FALSE FOR TRAINING")

        # Discrete Action Space - 3 actions:
        # 0: Full stop
        # 1: Global path follow
        # 2: Local path follow
        self.action_space = Discrete(n=3)

        ### Normalized Observation Space
        if self._normalize_obs == True:
            self.observation_space = Dict(
                {
                    "laser_reads": Box(low=0, high=1, shape=(270,), dtype=np.float32),
                    "agent": Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32),
                    "person": Discrete(2),
                    "dustbin": Discrete(2),
                }
            )
        ### De-normalized Observation Space
        else:
            self.observation_space = Dict(
                {
                    "laser_reads": Box(low=0, high=10, shape=(270,), dtype=np.float32),
                    "agent": Box(low=np.array([0, -math.pi]), high=np.array([10, math.pi]), dtype=np.float32),
                    "person": Discrete(2),
                    "dustbin": Discrete(2),
                }
            )

        # Relatively smaller path for RL agent to start training and get used to it. 
        # Pairs of [x,y] co-ordinates in robot 'odom' frame
        if self.short_path == True:
            self.waypoints_locations = [
                [[-14.0,51.9],[-11.0,51.9],[-7.0,51.9],[-3.0,51.9],[1.0,51.9]]
                ]
        # Else 3 relatively longer path for Rl agent to get trained and reach its goal. Increased complexity.
        # Pairs of [x,y] co-ordinates in robot 'odom' frame
        else:
            self.waypoints_locations = [
                [[-11.0,51.9],[-7.0,51.9],[-3.0,51.9],[1.0,51.9],[5.0,51.9],[9.0,51.9],[13.0,51.9],[17.0,51.9],[18.5,52.7],[19.7,52.7],[22.0,52.7],[22.85,52.5],[22.9,50.0],[22.9,46.0],[22.9,42.0],[22.9,38.0],[22.9,34.0],
                [22.9,30.0],[22.9,26.0],[22.9,22.0],[22.9,19.5],[22.9,18.0],[22.9,16.7],[26.9,16.7],[30.9,16.7],[34.9,16.7],[38.9,16.7],[43.35,16.7],[43.35,13.2],[45.1,13.2],[45.7,13.2],[45.7,11.0]],

                [[-11.0,51.9],[-7.0,51.9],[-3.0,51.9],[1.0,51.9],[5.0,51.9],[9.0,51.9],[13.0,51.9],[17.0,51.9],[18.5,52.7],[19.7,52.7],[22.0,52.7],[22.85,52.5],[22.9,50.0],[22.9,46.0],[22.9,42.0],[22.9,38.0],[22.9,34.0],
                [22.9,30.0],[22.9,26.0],[22.9,22.0],[22.9,19.5],[22.9,18.0],[19.0,18.0],[15.0,18.0],[11.0,18.0],[7.0,18.0],[3.0,18.0],[-1.0,18.0],[-5.0,18.0],[-8.0,18.0],[-10.3,18.0],[-10.5,16.6],[-10.5,13.0],[-10.5,9.0],
                [-10.5,5.0],[-10.5,1.0],[-10.5,-3.0],[-10.5,-5.6],[-6.5,-5.6],[-2.5,-5.6],[1.5,-5.6],[5.5,-5.6],[9.5,-5.6],[13.5,-5.6],[17.5,-5.6],[21.5,-5.6],[25.5,-5.6],[26.3,-5.6],[26.3,-9.6],[26.3,-13.6],[26.3,-17.6],
                [26.3,-21.6],[26.3,-25.6],[26.3,-29.6],[26.3,-33.6],[26.3,-37.6],[26.3,-41.6],[26.3,-45.6],[26.3,-46.3],[22.3,-46.3],[18.3,-46.3],[14.3,-46.3],[10.3,-46.3],[6.3,-46.3],[2.3,-46.3],[-1.7,-46.3],[-4.5,-46.3],
                [-8.5,-46.3],[-12.5,-46.3],[-12.5,-47.6],[-12.5,-49.5],[-14.0,-49.5]],

                [[-11.0,51.9],[-7.0,51.9],[-3.0,51.9],[1.0,51.9],[5.0,51.9],[9.0,51.9],[13.0,51.9],[17.0,51.9],[18.5,52.7],[19.7,52.7],[22.0,52.7],[22.85,52.5],[22.9,50.0],[22.9,46.0],[22.9,42.0],[22.9,38.0],[22.9,34.0],
                [22.9,30.0],[22.9,26.0],[22.9,22.0],[22.9,19.5],[22.9,18.0],[19.0,18.0],[15.0,18.0],[11.0,18.0],[7.0,18.0],[3.0,18.0],[-1.0,18.0],[-5.0,18.0],[-8.0,18.0],[-10.3,18.0],[-10.5,16.6],[-10.5,13.0],[-10.5,9.0],
                [-10.5,5.0],[-10.5,1.0],[-10.5,-3.0],[-10.5,-5.6],[-6.5,-5.6],[-0.5,-5.6],[-0.5,-9.6],[-0.5,-13.6],[-0.5,-17.6],[-0.5,-21.6],[-0.5,-25.6],[-0.5,-29.6],[-0.5,-33.6],[-0.5,-37.6],[-0.5,-41.6],[-0.5,-45.0],
                [-0.5,-46.3],[-4.5,-46.3],[-8.5,-46.3],[-12.5,-46.3],[-12.5,-47.6],[-12.5,-49.5],[-14.0,-49.5]]
            ]

        # Possible spawn locations for the robot.
        # Each entry: [x, y, angle], where x and y are coordinates and angle is the orientation (radians).
        if self.short_path == True:
            # For the short path scenario, only one spawn location is used.
            self.robot_locations = [
                [-19.0, 52.5, 0]
            ]
        else:
            # For the long path scenario with increased complexity, multiple spawn locations are available.
            self.robot_locations = [
                [22.7, 55.4, -math.pi/2],
                [23.5, 23.0, -math.pi/2],
                [-3.5, 52.2, 0],
                [-19.0, 52.5, 0]
            ]

        # This variable takes into account which point in the path the robot has to reach
        self._which_waypoint = 0
        self.first_waypoint_index = 0

        # Variables to montior successes and failures
        self._successes = 0
        self._failures = 0

        # Variables related to pure-pursuit control and path planning
        self.global_path = []
        self.local_path = []
        self._path = 0
        self.global_path_index = 0
        self.local_path_index = 0
        self.global_goal_reached = False
        self.local_goal_reached = False
        self.lookahead_distance = 0.30
        self.map_data = None
        self.map_info = None

        # initilize variable to start discrete action
        self.discrete_action = 0

        # to initilize param for distance traveled
        self.previous_distance = None
        self.distance_traveled = 0

    # Callback to process the occupancy grid from '/costmap' and compute a local path using A*
    def local_map_callback(self, msg):
        # --- 1. Extract map and pose information ---
        # Store occupancy grid data and map metadata
        self.map_data = msg.data
        self.map_info = msg.info
        self.resolution = self.map_info.resolution  # meters per cell
        self.originX = self.map_info.origin.position.x  # map origin x (world frame)
        self.originY = self.map_info.origin.position.y  # map origin y (world frame)

        # Extract yaw (rotation) from map origin quaternion
        orientation_q = self.map_info.origin.orientation
        _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        # --- 2. Convert agent and target positions to map grid indices ---
        # Convert agent's world coordinates to map grid indices (column, row)
        agent_x = self._agent_location[0]
        agent_y = self._agent_location[1]
        self.column, self.row = self.transform_to_local(agent_x, agent_y, self.originX, self.originY, yaw)
        self.column = int(self.column)
        self.row = int(self.row)

        # Convert target's world coordinates to map grid indices (columnH, rowH)
        target_x = float(self._target_location[0])
        target_y = float(self._target_location[1])
        self.columnH, self.rowH = self.transform_to_local(target_x, target_y, self.originX, self.originY, yaw)

        # --- 3. Prepare and preprocess the occupancy grid ---
        # Reshape flat map data to 2D array (height x width)
        self.data = np.array(self.map_data).reshape(self.map_info.height, self.map_info.width)
        # Convert occupancy values to meters (for inflation)
        self.data = self.data * self.resolution
        # Ensure agent's cell is free (set to 0)
        self.data[self.row][self.column] = 0 
        # Mark unknown or high values as obstacles (1)
        self.data[self.data < 0] = 1
        self.data[self.data > 5] = 1

        # --- 4. Set inflation radius based on person detection ---
        # If a person is detected close by, use a larger inflation radius for safety
        if self.person_detected == 1 and 0 < self.person_distance <= 4:
            inflation_distance_meters = 0.5  # larger inflation for safe overtaking
        else:
            inflation_distance_meters = 0.3  # default inflation
        inflation_radius_cells = int(inflation_distance_meters / self.resolution)

        # --- 5. Inflate obstacles on the map ---
        inflated_costmap = self.inflate_obstacles(self.data, inflation_radius=inflation_radius_cells)

        # If the robot is inside an inflated obstacle, move to nearest free cell
        if inflated_costmap[self.row, self.column] == 1:
            self.get_logger().warn("Robot is inside an inflated obstacle. Adjusting start position.")
            self.row, self.column = self.find_nearest_free_cell(inflated_costmap, self.row, self.column)

        # --- 6. Find local path using A* search ---
        self.local_path = self.astar(inflated_costmap, (self.row, self.column), (self.rowH, self.columnH))

        # If no path found or path is too short, log warning and exit
        if not self.local_path or len(self.local_path) < 10:
            self.get_logger().warn("Path not found.")
            return

        # --- 7. Convert local path grid indices to world coordinates and smooth ---
        # Transform path from grid indices to world coordinates
        self.local_path = [self.transform_to_global(p[1], p[0], self.originX, self.originY, yaw) for p in self.local_path]
        # Smooth the path using B-spline interpolation
        self.local_path = self.bspline_planning(self.local_path, len(self.local_path) * 5)

        # --- 8. Publish the local path for visualization in RViz2 ---
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for pose in self.local_path:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'map'
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            path_msg.poses.append(pose_msg)
        self.local_path_pub.publish(path_msg)

    # Publishes the selected global path to RViz2 for visualization.
    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for waypoint in self.waypoints_locations[self._path]:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    # Step function for RL environment: executes one action and returns the reward, done, observation, info
    def step(self, action):
        # 1. Increment step counter
        self._num_steps += 1

        # 2. Store the discrete action
        self.discrete_action = action

        # 3. Action selection and velocity command calculation
        # Action 0: Full stop
        if self.discrete_action == 0:
            # Stop the robot by setting both linear and angular velocities to zero
            action_ = (0.0, 0.0)
            self.get_logger().info("Full stop action.")

        # Action 1: Follow the global path using pure pursuit controller
        elif self.discrete_action == 1:
            # Get current robot pose
            self.current_x = self._agent_location[0]
            self.current_y = self._agent_location[1]
            self.current_yaw = self._agent_orientation

            # If global path exists and goal not reached, compute velocity command
            if self.global_path and not self.global_goal_reached:
                twist = Twist()
                # Calculate linear and angular velocities using pure pursuit
                twist.linear.x, twist.angular.z, self.global_path_index = self.pure_pursuit(
                    self.current_x, self.current_y, self.current_yaw,
                    self.global_path, self.global_path_index, self.lookahead_distance
                )
                # Check if robot is close enough to the global goal
                if math.hypot(self.current_x - self.global_path[-1][0], self.current_y - self.global_path[-1][1]) < 0.05:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.global_goal_reached = True
                    self.get_logger().info("Global Goal reached.")
                action_ = (twist.linear.x, twist.angular.z)
                self.get_logger().info("Global Path Follow.")

        # Action 2: Follow the local path using pure pursuit controller
        elif self.discrete_action == 2:
            # Get current robot pose
            self.current_x = self._agent_location[0]
            self.current_y = self._agent_location[1]
            self.current_yaw = self._agent_orientation

            # If local path exists and goal not reached, compute velocity command
            if self.local_path and not self.local_goal_reached:
                twist = Twist()
                # Calculate linear and angular velocities using pure pursuit
                twist.linear.x, twist.angular.z, self.local_path_index = self.pure_pursuit(
                    self.current_x, self.current_y, self.current_yaw,
                    self.local_path, self.local_path_index, self.lookahead_distance
                )
                # Check if robot is close enough to the local goal
                if math.hypot(self.current_x - self.local_path[-1][0], self.current_y - self.local_path[-1][1]) < 0.05:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.local_goal_reached = True
                    self.get_logger().info("Local Goal reached.")
                action_ = (twist.linear.x, twist.angular.z)
                self.get_logger().info("Local Path Follow.")

        # 4. Send the calculated velocity command to the robot
        self.send_velocity_command(action_)

        # 5. Spin the node to update sensor readings and robot state
        self.spin()

        # 6. Update robot's polar coordinates relative to the target
        self.transform_coordinates()

        # 7. Get the latest observation (sensor data, agent state, object detection, etc.)
        observation = self._get_obs()

        # 8. Gather info dictionary with all relevant state variables for reward and diagnostics
        info = self._get_info()

        # 9. Update the distance traveled in this step
        self.distance_traveled = self.update_distance(info)

        # 10. Compute the reward for this step
        reward = self.compute_rewards(info)
        self.get_logger().info("Reward per step: " + str(reward))

        # 11. If in evaluation mode (randomize_env_level==1), update statistics
        if self._randomize_env_level == 1:
            self.compute_statistics(info)

        # 12. Determine if the episode is done (collision, reached target, etc.)
        # If at the last waypoint, check for success or collision
        if self._which_waypoint == len(self.waypoints_locations[self._path]) - 1:
            done = bool(
                (info["distance"] < self._minimum_dist_from_target) or
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            )
        else:
            # Otherwise, check for collision, and if reached current waypoint, move to next
            done = bool(
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            )
            # If reached current waypoint, update to next waypoint
            if info["distance"] < self._minimum_dist_from_target:
                self._which_waypoint += 1
                self.global_path_index += 1
                self.randomize_target_location()
                # Optionally update target visualization in simulation
                if self._visualize_target:
                    self.call_set_target_state_service(self._target_location)

        # 13. Return observation, reward, done, truncated (always False), and info
        return observation, reward, done, False, info

    def render(self):
        # Function to render env steps
        # Our env is always rendered if you start gazebo_world.launch.py
        pass

    # Reset method for the RL environment.
    # This function is called at the end of each episode to reset the simulation and all relevant variables,
    # preparing the environment for the next episode.
    def reset(self, seed=None, options=None):
        self.get_logger().info("Resetting the environment")

        # 1. Update the record of the maximum number of steps taken in any episode so far.
        self.update_max_steps(self._num_steps)

        # 2. Increment the episode counter to keep track of how many episodes have been run.
        self._num_episodes += 1

        # 3. Randomize the robot's starting pose for this episode.
        #    This function returns a pose2d object (x, y, theta) for the robot's initial position.
        pose2d = self.randomize_robot_location()
        
        # 4. Reset the flag that tracks whether the robot's state has been set successfully.
        self._done_set_rob_state = False

        # 5. Call the ROS2 service to set the robot's position in the simulation to the new pose.
        #    This ensures the robot starts at the correct location for the new episode.
        self.call_set_robot_state_service(pose2d)  # Service imported from robot_controller.py

        # 6. Wait (spin) until the /set_entity_state service responds and the robot's state is set.
        #    This prevents the environment from returning stale or random observations.
        while self._done_set_rob_state == False:
            rclpy.spin_once(self)

        # 7. Add a short, non-blocking wait to ensure all sensor data (e.g., LIDAR, odometry) is up-to-date.
        #    This helps avoid using outdated sensor readings at the start of the episode.
        start_time = time.time()
        wait_duration = 0.1  # Total wait time in seconds
        increment = 0.1      # Time to wait per spin_once call
        while time.time() - start_time < wait_duration:
            rclpy.spin_once(self, timeout_sec=increment)  # Process ROS callbacks and yield control

        # 8. Randomly select a new global path for the robot to follow.
        #    This is only valid for randomization level 1 (evaluation mode).
        self._path = np.random.randint(0, len(self.waypoints_locations))
        # self._path = 0   # Uncomment to manually select a specific path
        self.get_logger().info(f"Selected path index: {self._path}")

        # 9. Reset the waypoint index to 0 (start of the path).
        #    This is only valid for randomization level 1.
        self._which_waypoint = 0
        
        # 10. Set the global path for this episode based on the selected path index.
        self.global_path = self.waypoints_locations[self._path]

        # 11. Find the nearest point on the global path to the robot's current position.
        #     This ensures the robot starts at the closest waypoint, not necessarily the first.
        self.global_path_index = self.find_nearest_point_index(self.global_path)
        # self.get_logger().info(f"Global path index reset: {self.global_path_index}")

        # 12. Find the nearest point on the local path to the robot's current position.
        #     This is useful for local planning and obstacle avoidance.
        self.local_path_ = self.local_path
        self.local_path_index = self.find_nearest_point_index(self.local_path_)
        # self.get_logger().info(f"Local path index reset: {self.local_path_index}")
        
        # 13. Assign the nearest global path index to the current waypoint and record the first waypoint index.
        self._which_waypoint = self.global_path_index
        self.first_waypoint_index = self.global_path_index

        # 14. Randomize the target location for this episode.
        self.randomize_target_location()

        # 15. If target visualization is enabled, update the target's position in the simulation for visualization.
        if self._visualize_target == True:
            # Call the service to spawn the target at the new waypoint location.
            self.call_set_target_state_service(self._target_location)

        # 16. Spin the node to process any remaining ROS callbacks and update internal state.
        self.spin()

        # 17. Calculate the robot's polar coordinates relative to the target.
        #     This is important for observation and reward calculation.
        self.transform_coordinates()

        # 18. Compute the initial observation and info dictionary for the new episode.
        observation = self._get_obs()
        info = self._get_info()

        # 19. Reset the step counter for the new episode.
        self._num_steps = 0

        # 20. Log that the reset function is complete.
        self.get_logger().info("Exiting reset function")
        
        # 21. Return the initial observation and info dictionary to the RL agent.
        return observation, info
    
    # Updates and returns the distance traveled by the agent since the last step.
    def update_distance(self, info):
        current_distance = info["distance"]  # Get current distance to target
        if self.previous_distance is None:
            # If this is the first step, there is no previous distance to compare
            distance_traveled = 0
        else:
            # Calculate the absolute difference between previous and current distance
            distance_traveled = abs(self.previous_distance - current_distance)
        
        # Store the current distance for the next update
        self.previous_distance = current_distance

        return distance_traveled

    # Gather and construct the current observation for the RL agent at each step of the episode.
    def _get_obs(self):
        # Construct the observation dictionary with relevant sensor and state information.
        obs = { 
            "laser_reads": self._laser_reads,                # Raw LIDAR scan data
            # "laser_front": self.laser_reads_front,         # (Optional) Front laser readings
            # "laser_corners": self.laser_reads_corners,     # (Optional) Corner laser readings
            # "laser_sides": self.laser_reads_from_sides,    # (Optional) Side laser readings
            "agent": self._polar_coordinates,                # Polar coordinates (distance, angle) to target
            "person": int(self.person_detected),             # Person detected flag (0 or 1)
            "dustbin": int(self.dustbin_detected),           # Dustbin detected flag (0 or 1)
            # Additional objects and their distances/angles can be added here if needed
        }

        # Uncomment the following lines for debugging specific observation components:
        # self.get_logger().info("Laser_front_observation : " + str(obs["laser_front"]))
        # self.get_logger().info("Laser_corners_observation : " + str(obs["laser_corners"]))
        # self.get_logger().info("Laser_sides_observation : " + str(obs["laser_sides"]))
        # self.get_logger().info("observation : " + str(obs))

        # Normalize the observation if normalization is enabled
        if self._normalize_obs == True:
            obs = self.normalize_observation(obs)
            # self.get_logger().info("Normalized observation : " + str(obs))

        # self.get_logger().info("Agent Location: " + str(self._agent_location))  # For debugging agent position

        return obs

    # _get_info: Returns a dictionary with detailed information about the agent's current state.
    # This includes distance and angle to the target, sensor readings, detection flags, and path info.
    def _get_info(self):
        # Compose and return a dictionary with all relevant info for logging, debugging, or reward calculation
        return {
            "distance": self._radius,  # Distance from agent to target (meters)
            "angle": self._theta,      # Angle from agent to target (radians)
            "distance_traveled": self.distance_traveled,  # Distance moved since last step
            "laser": self._laser_reads,                   # Full LIDAR scan (raw)
            "laser_front_new": self.laser_reads_front,    # LIDAR readings at the front
            "laser_corners": self.laser_reads_corners,    # LIDAR readings at the corners
            "laser_sides_new": self.laser_reads_from_sides, # LIDAR readings at the sides
            "person": self.person_detected,               # Person detected (bool/int)
            "person_depth": self.person_distance,         # Distance to detected person
            "person_angle": self.person_angle,            # Angle to detected person
            "dustbin": self.dustbin_detected,             # Dustbin detected (bool/int)
            "dustbin_depth": self.dustbin_distance,       # Distance to detected dustbin
            "dustbin_angle": self.dustbin_angle,          # Angle to detected dustbin
            "action": self.discrete_action,               # Last action taken by agent (discrete)
            "perpendicular_distance": self.perpendicular_distance, # Perpendicular distance to path
            "robot_side_from_global_path": self.position, # Robot's side relative to global path
            "waypoint_index": self._which_waypoint,       # Current waypoint index in path
            "first_waypoint_index": self.first_waypoint_index, # Index of first waypoint
            "length_of_path": len(self.waypoints_locations[self._path]) # Total number of waypoints in path
        }

    # Spins the node until it receives new sensor data (executes both laser and odom callbacks)
    def spin(self):
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose == False) or (self._done_laser == False):
            rclpy.spin_once(self)

    # transform_coordinates: Computes polar coordinates (distance and angle) from robot to target,
    # and calculates the perpendicular distance and side (left/right/on) of the robot relative to the global path.
    def transform_coordinates(self):
        # --- Polar Coordinates Calculation ---
        # Compute the Euclidean distance between the robot and the target (radius in polar coordinates)
        self._radius = math.dist(self._agent_location, self._target_location)

        # Compute the target's X coordinate in the robot's local Cartesian frame
        # This is a rotation of the global vector (target - robot) by -robot_orientation
        self._robot_target_x = (
            math.cos(-self._agent_orientation) * (self._target_location[0] - self._agent_location[0])
            - math.sin(-self._agent_orientation) * (self._target_location[1] - self._agent_location[1])
        )

        # Compute the target's Y coordinate in the robot's local Cartesian frame
        self._robot_target_y = (
            math.sin(-self._agent_orientation) * (self._target_location[0] - self._agent_location[0])
            + math.cos(-self._agent_orientation) * (self._target_location[1] - self._agent_location[1])
        )

        # Compute the angle (theta) from the robot to the target in the robot's local frame
        self._theta = math.atan2(self._robot_target_y, self._robot_target_x)

        # Store the polar coordinates as a numpy array for use in the Gym environment
        self._polar_coordinates = np.array([self._radius, self._theta], dtype=np.float32)

        # --- Perpendicular Distance and Side Calculation ---
        # Unpack coordinates for readability
        x_r, y_r = self._agent_location           # Robot position
        x_1, y_1 = self._target_location          # Target position (current waypoint)
        x_2, y_2 = self._other_point_on_path      # Another point on the path (previous or next waypoint)

        # Initialize perpendicular distance
        self.perpendicular_distance = 0

        # Calculate the numerator and denominator for the perpendicular distance formula
        # Formula: |(x2-x1)*(y1-yr) - (x1-xr)*(y2-y1)| / sqrt((x2-x1)^2 + (y2-y1)^2)
        numerator = abs(((x_2 - x_1) * (y_1 - y_r)) - ((x_1 - x_r) * (y_2 - y_1)))
        denominator = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        # Compute the perpendicular distance from the robot to the line segment (global path)
        self.perpendicular_distance = numerator / denominator

        # --- Determine Robot Side Relative to Path ---
        # Use the cross product to determine if the robot is to the left, right, or on the path
        if self._which_waypoint == 0:
            # For the first waypoint, use the next segment
            D = (x_1 - x_2) * (y_r - y_2) - (y_1 - y_2) * (x_r - x_2)
        else:
            # For other waypoints, use the previous segment
            D = (x_2 - x_1) * (y_r - y_1) - (y_2 - y_1) * (x_r - x_1)

        # Assign side and adjust perpendicular distance sign accordingly
        if D > 0:
            self.position = "Right"
            self.perpendicular_distance = -self.perpendicular_distance  # Right side: negative distance
        elif D < 0:
            self.position = "Left"
            # Left side: keep distance positive
        else:
            self.position = "On the Line"
            # On the line: distance is zero

        # --- Debugging (uncomment for verbose output) ---
        # self.get_logger().info("Agent is on " + str(self.position) + " side of Global path.")
        # self.get_logger().info("Theta: " + str(math.degrees(self._theta)) + " Radius: " + str(self._radius))
        # self.get_logger().info("Distance from target: " + str(math.sqrt(self._robot_target_x**2 + self._robot_target_y**2)))
        # self.get_logger().info("Xt: " + str(self._robot_target_x) + " - Yt: " + str(self._robot_target_y))
        # self.get_logger().info("Polar coordinates: " + str(self._polar_coordinates))
        # self.get_logger().info("Perpendicular distance: " + str(self.perpendicular_distance))

    # Randomize target location for the single environment level (level 1, "default path following").
    def randomize_target_location(self):
        if self._randomize_env_level == 1:
            # The new waypoint is already set for the current path and waypoint index.
            if self._which_waypoint == 0:
                self._other_point_on_path = np.array([
                    self.waypoints_locations[self._path][self._which_waypoint + 1][0],
                    self.waypoints_locations[self._path][self._which_waypoint + 1][1]
                    ], dtype=np.float32)
            else:
                self._other_point_on_path = np.array([
                    self.waypoints_locations[self._path][self._which_waypoint - 1][0],
                    self.waypoints_locations[self._path][self._which_waypoint - 1][1]
                    ], dtype=np.float32)

            self._target_location = np.array([
                self.waypoints_locations[self._path][self._which_waypoint][0],
                self.waypoints_locations[self._path][self._which_waypoint][1]
                ], dtype=np.float32)  # Base position
            self.get_logger().info(f"Randomized target location: {self._target_location}")
            self.get_logger().info(f"Other point on path: {self._other_point_on_path}")

    # Randomize robot's initial location for environment level 1 (only one level used)
    def randomize_robot_location(self):
        if self._randomize_env_level == 1:
            # Randomly decides which location to pick for spawning
            self._location = np.random.randint(0, len(self.robot_locations))
            position_x = float(self.robot_locations[self._location][0])
            position_y = float(self.robot_locations[self._location][1])
            position_z = float(0.1)
            angle = float(self.robot_locations[self._location][2])
            orientation_z = float(math.sin(angle / 2))
            orientation_w = float(math.cos(angle / 2))
        return [position_x, position_y, position_z, orientation_z, orientation_w]

    
    # Compute reward for the current step based on selected reward method
    def compute_rewards(self, info):
        # Simple reward: +1 for reaching target, -1 for hitting obstacle, 0 otherwise
        if self._reward_method == 0:
            if (info["distance"] < self._minimum_dist_from_target):
                reward = 1  # Target reached
                self.get_logger().info("TARGET REACHED")
            # Check if any laser detects obstacle too close
            elif bool(
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            ):
                reward = -1  # Hit obstacle
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                reward = 0  # Continue episode

        # Risk seeker: smaller penalty for obstacle, same for target
        elif self._reward_method == 1:
            if (info["distance"] < self._minimum_dist_from_target):
                reward = 1  # Target reached
                self.get_logger().info("TARGET REACHED")
            elif bool(
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            ):
                reward = -0.1  # Hit obstacle, less penalty
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                reward = 0  # Continue

        # Heuristic: reward for target, big penalty for obstacle, otherwise shaped reward
        elif self._reward_method == 2:
            if (info["distance"] < self._minimum_dist_from_target):
                reward = 1000 - self._num_steps  # Large reward minus steps taken
                self.get_logger().info("TARGET REACHED Reward: " + str(reward))
            elif bool(
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            ):
                reward = -10000  # Large penalty for collision
                self.get_logger().info("HIT AN OBSTACLE")
            else:
                # Distance penalty
                instant_reward = -info["distance"] * self._distance_penalty_factor
                # Attraction: only if close to target
                if info["distance"] < self._attraction_threshold:
                    attraction_reward = self._attraction_factor / info["distance"]
                else:
                    attraction_reward = 0
                # Repulsion: sum for all close obstacles
                contributions = [
                    ((-self._repulsion_factor / read**2) * ((1/read) - (1/self._repulsion_threshold)))
                    for read in info["laser"] if read <= self._repulsion_threshold
                ]
                repulsion_reward = sum(contributions)
                # Total reward, capped at 1
                reward = min(instant_reward + attraction_reward + repulsion_reward, 1)

        # Hybrid reward method: combines path progress with context-sensitive action rewards
        elif self._reward_method == 3:
            # Calculate percentage of path completed, normalized to [0, 100]
            path_completion_percentage = (
                (info["waypoint_index"] - info["first_waypoint_index"]) /
                (info["length_of_path"] - info["first_waypoint_index"])
            ) * 100
            # Normalize the number of steps taken by the maximum steps seen so far
            normalized_steps = self._num_steps / self.max_steps_so_far

            # Case 1: Target reached
            if (info["distance"] < self._minimum_dist_from_target):
                # Reward is based on path completion minus normalized steps, scaled down
                reward = (path_completion_percentage - normalized_steps) / 10
                self.get_logger().info("TARGET REACHED Reward: " + str(reward))
            # Case 2: Collision with obstacle detected
            elif bool(
                (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
                (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
                (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
            ):
                reward = -10  # Large penalty for collision
                self.get_logger().info("HIT AN OBSTACLE")
            # Case 3: Ongoing episode, compute action-based rewards
            else:
                # --- Action: Fullstop (action == 0) ---
                reward_stop = 0
                if info["action"] == 0:  # Fullstop
                    reward_stop = 1  # Base reward for stopping
                    # Penalize unnecessary stop (not near any obstacle)
                    if min(info["laser_front_new"]) > 1.0 or min(info["laser_corners"]) > 0.8:
                        reward_stop = -reward_stop
                    else:
                        # Reward for stopping near both person and dustbin
                        if info["person"] == 1 and info["dustbin"] == 1:
                            reward_stop = (reward_stop * (2 / (info["person_depth"] + info["dustbin_depth"]))) * 2
                        # Reward for stopping near person in front
                        elif info["person"] == 1 and 75 < info["person_angle"] < 105:
                            reward_stop = reward_stop * (1 / info["person_depth"])
                        # Reward for stopping near dustbin in front
                        elif info["dustbin"] == 1 and 75 < info["dustbin_angle"] < 105:
                            reward_stop = (reward_stop * (1 / info["dustbin_depth"])) / 2
                        # Otherwise, reward for stopping near any obstacle in front
                        else:
                            reward_stop = reward_stop * (1 / min(info["laser_front_new"]))

                # --- Action: Global Path Follow (action == 1) ---
                reward_global = 0
                if info["action"] == 1:  # Global Path Follow
                    reward_global = 1  # Base reward for following global path
                    # Check if heading is well aligned with path
                    if -math.pi / 12 < info["angle"] < math.pi / 12:
                        # Penalize if path ahead is blocked
                        if min(info["laser_front_new"]) < 2:
                            reward_global = -reward_global
                        # Slight penalty if cluttered on sides/corners
                        elif min(info["laser_corners"]) < 1 or min(info["laser_sides_new"]) < 0.8:
                            reward_global = reward_global * 0.8
                        # Bonus for clear path ahead
                        else:
                            reward_global = reward_global * 1.2
                    else:
                        # Penalize misaligned heading if blocked
                        if min(info["laser_front_new"]) < 2:
                            reward_global = -reward_global * 1.2
                        # Small reward for trying in cluttered environment
                        if min(info["laser_corners"]) < 1 or min(info["laser_sides_new"]) < 0.8:
                            reward_global = 1e-3
                        # Otherwise, keep base reward
                        else:
                            reward_global = reward_global

                # --- Action: Local Path Follow (action == 2) ---
                reward_local = 0
                if info["action"] == 2:  # Local Path Follow
                    reward_local = 1  # Base reward for following local path
                    # If heading is well aligned
                    if -math.pi / 12 < info["angle"] < math.pi / 12:
                        # If close to obstacle in front, reward for maneuvering near person/dustbin
                        if min(info["laser_front_new"]) < 2:
                            if info["person"] == 1 and info["dustbin"] == 1:
                                reward_local = (reward_local * (2 / (info["person_depth"] + info["dustbin_depth"]))) / 2
                            elif info["person"] == 1 and 75 < info["person_angle"] < 105:
                                reward_local = reward_local * (1 / info["person_depth"]) * 2
                            elif info["dustbin"] == 1 and 75 < info["dustbin_angle"] < 105:
                                reward_local = (reward_local * (1 / info["dustbin_depth"])) * 4
                            else:
                                reward_local = reward_local * (1 / min(info["laser_front_new"])) * 2
                        # No change if cluttered on sides/corners
                        elif min(info["laser_corners"]) < 1 or min(info["laser_sides_new"]) < 0.8:
                            reward_local = reward_local
                        # Penalize unnecessary local path following
                        else:
                            reward_local = -reward_local
                    else:
                        # Slight bonus if blocked but not aligned
                        if min(info["laser_front_new"]) < 2:
                            reward_local = reward_local * 1.2
                        # No change if cluttered on sides/corners
                        elif min(info["laser_corners"]) < 1 or min(info["laser_sides_new"]) < 0.8:
                            reward_local = reward_local
                        # Small reward for trying local path in open space but not aligned
                        else:
                            reward_local = 1e-3

                # Combine action-based rewards with different weights
                # reward = α * reward_global + β * reward_local + γ * reward_stop
                # Here, α=1, β=1.5, γ=1.25 (can be tuned)
                reward = reward_global + 1.5 * reward_local + 1.25 * reward_stop

                # Optionally: add angle or perpendicular distance penalty here for further shaping

        return reward

    # Function to update the maximum number of steps encountered so far.
    # This is useful for tracking the longest episode or for dynamic curriculum learning.
    def update_max_steps(self, num_steps):
        # Check if the current number of steps is greater than the recorded maximum
        if num_steps > self.max_steps_so_far:
            # Update the maximum steps encountered so far
            self.max_steps_so_far = num_steps
        # Uncomment the line below for debugging/logging purposes
        # self.get_logger().info("Max Steps So Far: " + str(self.max_steps_so_far))
    
    # Normalize the robot's observation values to the range [0, 1] for consistent input to learning algorithms.
    def normalize_observation(self, observation):
        # Normalize distance from target (original range: 0 to 100)
        observation["agent"][0] = observation["agent"][0] / 10

        # Normalize angle from target (original range: -pi to pi)
        # Shifts and scales angle to [0, 1]
        observation["agent"][1] = (observation["agent"][1] + math.pi) / (2 * math.pi)

        # Normalize all laser readings (original range: 0 to 100)
        observation["laser_reads"] = observation["laser_reads"] / 10

        return observation

    # This method is called during every step to check for success and failure.
    # It computes statistics for agent performance during evaluation or path planning.
    # It increments the success or failure counters based on whether the agent reached the target or hit an obstacle.
    def compute_statistics(self, info):
        ## This method is used to compute statistics when in Evaluation mode 1
        
        # Check if the agent is at the last waypoint and within the minimum distance from the target
        # If so, count this as a successful episode
        if (self._which_waypoint == len(self.waypoints_locations[self._path]) - 1) and (info["distance"] < self._minimum_dist_from_target):
            self._successes += 1  # Increment the success counter
            # Optionally, you could also increment completed paths or log agent position here
        
        # Check if the agent has collided with an obstacle
        # This is determined by checking if any of the laser readings (front, sides, corners) are below their respective minimum safe distances
        elif bool(
            (min(info["laser_front_new"]) < self._minimum_dist_from_obstacles) or
            (min(info["laser_sides_new"]) < self._minimum_dist_from_obstacles_for_sides) or
            (min(info["laser_corners"]) < self._minimum_dist_from_obstacles_for_coeners)
        ):
            self._failures += 1  # Increment the failure counter
            # This means the agent hit an obstacle and the episode is considered a failure

    # This function generates and saves a horizontal bar chart summarizing the agent's performance
    # (success, failure, and truncated rates) over a set of evaluation episodes. 
    # The function logs the rates, creates a plot, and saves it as a PNG file for later review.
    def plot_and_save_results(self):
        # n_episode_error is calculated as the number of episodes not included in evaluation.
        # Here, 50 is the number of episodes you want to evaluate (set in trained_agent.py).
        # If you want to evaluate 100 episodes, replace 50 with 100.
        n_episode_error = (self._num_episodes - 1) - 50

        # Calculate the total number of episodes that were actually evaluated
        total_episodes = self._num_episodes - 1 - n_episode_error  

        # Calculate the number of truncated episodes (episodes that were neither success nor failure)
        truncated_episodes = total_episodes - self._successes - self._failures

        # Compute the success, failure, and truncated rates as percentages
        # Avoid division by zero by checking if total_episodes > 0
        success_rate = (self._successes / total_episodes) * 100 if total_episodes > 0 else 0
        fail_rate = (self._failures / total_episodes) * 100 if total_episodes > 0 else 0
        truncate_rate = (truncated_episodes / total_episodes) * 100 if total_episodes > 0 else 0

        # Log the calculated rates for debugging and record-keeping
        self.get_logger().info(f"Success Rate: {success_rate:.2f}%")
        self.get_logger().info(f"Failure Rate: {fail_rate:.2f}%")
        self.get_logger().info(f"Truncated Rate: {truncate_rate:.2f}%")

        # Prepare data for plotting
        labels = ['Success', 'Failure', 'Truncated']  # Categories for the bar chart
        values = [success_rate, fail_rate, truncate_rate]  # Corresponding rates
        colors = ['green', 'red', 'orange']  # Bar colors for each category
        
        # Define bar width and spacing for the horizontal bar chart
        bar_width = 0.3  
        spacing = 0.15  # Space between bars

        fig, ax = plt.subplots(figsize=(9, 5))  # Adjust figure size for better layout

        # Calculate y-axis positions for each bar, ensuring proper spacing
        y_positions = np.arange(len(labels)) * (bar_width + spacing)

        # Draw the horizontal bars
        bars = ax.barh(y_positions, values, height=bar_width, color=colors)

        # Annotate each bar with its percentage value
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 2,  # Position text slightly to the right of the bar
                bar.get_y() + bar.get_height() / 2,  # Vertically center the text
                f'{value:.2f}%', 
                va='center', ha='left', fontsize=10, color='black', fontweight='bold'
            )

        # Add a summary of total episodes and counts in the top-right corner of the plot
        text_details = (
            f"Total Episodes: {total_episodes}\n"
            f"Success: {self._successes}\n"
            f"Failure: {self._failures}\n"
            f"Truncated: {truncated_episodes}"
        )
        ax.text(
            90,  # X position (near the right edge of the plot)
            y_positions[-1] + bar_width,  # Y position (above the last bar)
            text_details,
            fontsize=10, color='black',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8)  # Add a white background for readability
        )

        # Set y-axis labels and formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Percentage')
        ax.set_title('Agent Performance Overview')

        # Set x-axis and y-axis limits for better visualization
        plt.xlim(0, 100)  # X-axis from 0% to 100%
        plt.ylim(-0.5, y_positions[-1] + bar_width * 2)  # Add space above and below bars

        # Save the plot to a file with a timestamp in the filename
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        # Use fixed src package path
        package_path = PKG_SRC_DIR
        eval_dir = os.path.join(package_path, 'evaluation_plots')
        os.makedirs(eval_dir, exist_ok=True)
        save_path = os.path.join(eval_dir, f"performance_plot_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        self.get_logger().info(f"Plot saved at: {save_path}")

        # Optionally, show the plot interactively (commented out for headless operation)
        # plt.show()  

    # Close the environment: log stats, plot results, and clean up resources.
    def close(self):
        # n_episode_error is calculated as the number of episodes not included in evaluation.
        # Here, 50 is the number of episodes you want to evaluate (set in trained_agent.py).
        # If you want to evaluate 100 episodes, replace 50 with 100.
        n_episode_error = (self._num_episodes - 1) - 50
        self.get_logger().info("Start Reset errors: " + str(n_episode_error))

        # Print achieved data and statistics if in evaluation/randomization mode
        if self._randomize_env_level >= 1:
            total_episodes = self._num_episodes - 1 - n_episode_error
            self.get_logger().info("Number of episodes: " + str(total_episodes))
            self.get_logger().info("Successes: " + str(self._successes))
            self.get_logger().info("Failures: " + str(self._failures))
            
            # Also log truncated episodes and rates
            truncated_episodes = total_episodes - self._successes - self._failures
            self.get_logger().info("Truncated episodes: " + str(truncated_episodes))
            self.get_logger().info("Success Rate: " + str(self._successes / total_episodes))
            self.get_logger().info("Failure Rate: " + str(self._failures / total_episodes))

        # Generate and save a performance plot summarizing the results
        self.plot_and_save_results()

        # Destroy all clients, publishers, and subscribers to clean up resources
        # (Uncomment the following lines if you want to explicitly destroy these resources)
        # self.destroy_client(self.client_sim)
        # self.destroy_client(self.client_state)
        # self.destroy_publisher(self.action_pub)
        # self.destroy_subscription(self.pose_sub)
        # self.destroy_subscription(self.laser_sub)

        # Destroy the node itself to complete shutdown
        self.destroy_node()