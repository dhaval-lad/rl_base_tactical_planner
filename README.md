# Outdoorbot Tactical Planner

> An infrastructure to train RL agents for tactical decision making in outdoor environments using Software-in-the-Loop (SIL) simulation (built with ROS2, Gazebo, OpenAI Gym, and Stable Baselines3).

## Description
This repository contains an application using ROS2 Humble, Gazebo, OpenAI Gym and Stable Baselines3 to train reinforcement learning agents for tactical decision making in outdoor environments. The system utilizes a custom 4WD robot called GrassHopper equipped with camera and LiDAR sensors to perceive the environment and make intelligent navigation decisions in complex outdoor scenarios.

The GrassHopper robot features a four-wheel drive system and is equipped with a 270° LiDAR for comprehensive obstacle detection and a forward-facing camera for visual perception. The LiDAR provides distance measurements ranging from 0.08 to 10 meters, while the camera captures RGB images and depth information for object detection and environmental understanding.

This repository includes the following elements:
* A comprehensive outdoor simulation environment featuring sidewalk scenarios, obstacles, and dynamic elements.
* An advanced Gym environment tailored for tactical decision making, integrating both visual and LiDAR data.
* Trained reinforcement learning agents capable of navigating complex outdoor environments by avoiding obstacles, following paths, and making tactical decisions based on sensor fusion.
* The agent is responsible for making tactical decisions by selecting actions from a set of available discrete actions in the action space, which include: Global Path Follow, Local Path Follow, and Full Stop.

The application serves as a robust framework for developing and testing autonomous navigation systems in outdoor environments using Software-in-the-Loop simulation, providing a realistic testing environment for outdoor robotics applications.

## Current Status

This repository was created as part of my Master's thesis work, and as such, it is not actively maintained or updated. The primary purpose of sharing this project is to assist others who may be embarking on similar research or development journeys in the field of outdoor robotics and ROS2-based reinforcement learning. Since the project has fulfilled its academic purpose, I do not plan to provide further updates or new features in the future.

For a comprehensive understanding of the methodology, implementation details, and experimental results, please refer to my thesis document, which is available [here](.documents/Optimizing_Outdoor_Navigation__A_Reinforcement_Learning_Approach_to_Tachtical_Planning_in_Mobile_Robots.pdf).

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Package Overview](#package-overview)
- [Getting Started](#getting-started)
  - [Run a random agent](#run-a-random-agent)
  - [Run a trained agent](#run-a-trained-agent)
  - [Train a new agent](#train-a-new-agent)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Re-train an Existing Agent](#re-train-an-existing-agent)
- [References](#references)

## Dependencies

### System Requirements
* **Ubuntu 22.04 LTS** - [Ubuntu 22.04](https://releases.ubuntu.com/22.04/) 
* **ROS2 Humble** - [Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
* **Gazebo** - [Installation Guide](http://classic.gazebosim.org/tutorials?tut=ros2_installing&cat=connect_ros)

### Required ROS2 Packages
The following packages must be installed in your ROS2 workspace before using this package:

* **GrassHopper Description** - [Repository](https://github.com/dhaval-lad/grasshopper_description.git)
  * Provides the URDF description and mesh files for the GrassHopper 4WD robot
  * Required for robot visualization and simulation

* **GrassHopper Gazebo** - [Repository](https://github.com/dhaval-lad/grasshopper_gazebo.git)
  * Contains Gazebo simulation worlds and launch files
  * Provides the outdoor environment simulation

* **Actor Collision Plugin** - [Repository](https://github.com/dhaval-lad/actor_collision_plugin.git)
  * Enables dynamic actor simulation in Gazebo
  * Required for pedestrian simulation in outdoor environments

* **Dynamic Map** - [Repository](https://github.com/dhaval-lad/Dynamic-Map.git)
  * Provides dynamic mapping capabilities
  * Used for real-time environment representation

* **ObjectDetection Messages** - [Repository](https://github.com/dhaval-lad/objectdetection_msgs.git)
  * Custom ROS2 message definitions for object detection
  * Required for communication between detection and planning nodes

* **Object Detection** - [Repository](https://github.com/dhaval-lad/object_detection.git)
  * YOLOv5-based object detection implementation
  * Provides visual perception capabilities for the robot

### Python Dependencies
* **Stable Baselines3** - [Installation Guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
  * Reinforcement learning framework for training agents
* **Tensorboard** - [Installation Guide](https://pypi.org/project/tensorboard/)
  * Training monitoring and visualization tool
* **Optuna** - [Installation Guide](https://optuna.org/#installation)
  * Hyperparameter optimization framework (optional)

## Installation

### Prerequisites
Before installing this package, ensure you have:
* A properly configured ROS2 workspace - [ROS2 workspace creation tutorial](https://www.youtube.com/watch?v=3GbrKQ7G2P0)
* All system requirements and dependencies installed as described in the [Dependencies](#dependencies) section above

### Install Main Package
Clone the main tactical planner package into your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/dhaval-lad/rl_base_tactical_planner.git
```

### Configure Package Paths

Before building the package, you must update several hardcoded file paths to match your system's directory structure. This ensures that the simulation and all scripts can correctly locate required models and resources.

#### Actor Model Paths
- outdoor_world_0.world → lines 6519, 6523 → update actor model paths

#### Python Source Paths
Several Python files use hardcoded absolute paths for `PKG_SRC_DIR` and `GRASSHOPPER_DESC_DIR`. You must update these variables to match your local workspace structure. The table below lists the files and line numbers where these changes are required:

- outdoorbot_env.py (line 16) → PKG_SRC_DIR
- robot_controller.py (line 19) → GRASSHOPPER_DESC_DIR
- spawn_demo.py (lines 10, 11) → PKG_SRC_DIR, GRASSHOPPER_DESC_DIR
- start_training.py (line 20) → PKG_SRC_DIR
- trained_agent.py (line 37) → PKG_SRC_DIR

These paths need to be updated to point to the correct location of the different directory, package and models.

### Build the Workspace
Build your ROS2 workspace to install the package:

```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

### Verify Installation
To verify that everything is working correctly, launch the outdoor simulation world:

```bash
ros2 launch outdoor_robot_spawner gazebo_world.launch.py
```

The outdoor world should open in Gazebo with the GrassHopper robot visible in the environment.

> **Note**: Additional configuration for RL training, hyperparameter tuning, and model evaluation will be explained in the usage sections later.

## Package Overview

The Outdoorbot Tactical Planner package is organized into **5 main nodes** and **4 launch files**, all of which are thoroughly commented in the source code to enhance clarity and ease of understanding. These comments explain the logic, parameters, and flow within each node, making it easier for users and developers to follow the implementation and customize the system as needed. Together, these components provide a comprehensive reinforcement learning framework for outdoor navigation, with each node and launch file serving a distinct role in the training, evaluation, and deployment pipeline.

### Core Nodes

#### 1. **OutDoorEnv** (`outdoorbot_env.py`)
**Purpose**: Main reinforcement learning environment that defines the Gym interface and provides all RL-related core functions.
- **Overview**: This node contains the essential RL methods such as `step()`, `reset()`, reward calculation, observation and action space definitions, as well as utilities for predefined global path, `_get_obs`, and `_get_info`. These functions form the backbone of the RL training loop and agent-environment interaction.
- **Inherits from**: `RobotController` and `gymnasium.Env`
- **Key Responsibilities**:
  - Defines action space (discrete actions: Global Path Follow, Local Path Follow, Full Stop)
  - Manages observation space (LiDAR data, camera detections, path information)
  - Implements reward functions with multiple strategies (simple, risk-seeker, heuristic, hybrid)
  - Handles episode termination conditions (success/failure)
  - Provides RL interface methods: `step`, `reset`, `_get_obs`, `_get_info`
  - Maintains and utilizes a predefined global path for navigation
- **Configuration**: Supports different reward methods
- **Costmap & Local Path**: Subscribes to the costmap for obstacle data and publishes the local path generated by A* for navigation.

#### 2. **RobotController** (`robot_controller.py`)
**Purpose**: Supporting class for OutDoorEnv, providing low-level robot control and sensor data management. It contains essential functionalities required for the RL model to operate, helping to keep the package modular and maintainable.
- **Key Responsibilities**:
  - Publishes velocity commands to `/cmd_vel`
  - Subscribes to sensor topics (`/odom`, `/scan`, `/detections_publisher`)
  - Manages robot state and position tracking
  - Implements path planning algorithms (A* for local path, Pure Pursuit for path following)
  - Handles obstacle inflation and collision avoidance
  - Provides simulation reset and entity management services
- **Sensor Integration**: Processes LiDAR scans, object detections, and odometry data
- **Path Planning and Following**: Contains A* (for local path) and Pure Pursuit (for path following); both are called from OutDoorEnv.

#### 3. **TrainingNode** (`start_training.py`)
**Purpose**: Main training orchestrator for reinforcement learning
- **Training Modes**:
  - `random_agent`: Testing/debugging mode (no training)
  - `training`: Train new model from scratch
  - `retraining`: Continue training from saved model
  - `hyperparam_tuning`: Optimize PPO hyperparameters
- **Key Features**:
  - Integrates with Stable Baselines3 PPO algorithm
  - Real-time reward plotting and monitoring
  - Hyperparameter optimization with Optuna
  - Automatic model saving and logging

#### 4. **SpawnDemo** (`spawn_demo.py`)
**Purpose**: Robot and target spawning, as well as world initialization
- **Key Responsibilities**:
  - Converts URDF to SDF format for Gazebo compatibility
  - Spawns the GrassHopper robot and target entity in the simulation world (used for both robot and target spawning)
  - Sets initial positions for both robot and target
  - Manages entity state, including simulation reset and respawn logic
- **Integration**: Technically interfaces with the Gazebo simulation environment to programmatically instantiate and manage simulation entities

#### 5. **TrainedAgent** (`trained_agent.py`)
**Purpose**: Node for deploying and systematically evaluating trained reinforcement learning models in simulation.
- **Key Responsibilities**:
  - Loads pre-trained RL models (e.g., PPO agents) from checkpoints
  - Sets up the simulation environment in evaluation mode, ensuring reproducibility
  - Runs the trained policy for a specified number of episodes, automatically handling environment resets
  - Collects and logs episode metrics such as cumulative reward, success rate, and collisions
  - Generates performance plots for quantitative analysis of agent behavior
- **Usage**: Used for testing, benchmarking, and validating trained agents before deployment or further development

### Launch Files

#### 1. **gazebo_world.launch.py**
**Purpose**: Launches the complete simulation environment with GUI
- **Components**:
  - Starts Gazebo with outdoor world (`outdoor_world_0.world`)
  - Spawns GrassHopper robot at specified coordinates
  - Sets up all necessary TF transforms for sensor integration
  - Configures model paths and environment variables
- **Use Case**: Provides an outdoor-like simulation environment for the robot or RL agent to interact with during development, debugging, visualization, and training. Enables realistic testing and training of navigation and decision-making algorithms in a simulated outdoor world.

#### 2. **headless_world.launch.py**
**Purpose**: Launches simulation without GUI for faster training
- **Components**:
  - Same as `gazebo_world.launch.py` but without visual interface
  - Optimized for training performance
  - Reduces computational overhead
- **Use Case**: Faster training and less computational power

#### 3. **start_training.launch.py**
**Purpose**: Initiates the reinforcement learning training process
- **Components**:
  - Launches the TrainingNode with configuration parameters
  - Sets up the complete training environment
- **Use Case**: Starting training sessions with different configurations

#### 4. **trained_agent.launch.py**
**Purpose**: Deploys and evaluates trained models in simulation
- **Components**:
  - Loads specified trained model
  - Launches evaluation environment
  - Sets up performance monitoring
- **Use Case**: Testing trained agents and performance evaluation

This architecture enables flexible development, training, and deployment of reinforcement learning agents for outdoor navigation tasks.

## Getting Started

The application includes various training modes that require configuration before running. Before running anything, you need to edit specific files to set the desired mode.

### Run a random agent
Running a random agent helps to understand the basic concepts of the Gym environment.

**Configuration**:
* Open `start_training.py`, go to line 31 and set the `self._training_mode` attribute to `"random_agent"`
* In `start_training.py`, go to line 225 and set `episodes = 5` to specify the number of episodes the random agent will run before closing the node. Change this value if you want to run a different number of episodes.
* Open `outdoorbot_env.py`, set `self._visualize_target = True` for debugging
* Build the package before running any launch files:
```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

**Run Commands**:
```bash
# Terminal 1: Launch simulation
ros2 launch outdoor_robot_spawner gazebo_world.launch.py

# Terminal 2: Run random agent
ros2 launch outdoor_robot_spawner start_training.launch.py
```

### Run a trained agent
Test pre-trained models from the `rl_models` folder.

**Configuration**:
* Open `trained_agent.py`:
  - Go to line 58 and change `model_filename` to your desired model (e.g., `'PPO_test_14022025_002626.zip'`)
  - Go to line 89 and set `n_episodes = 50` (this is the default number of evaluation episodes). **If you change this value, you must also update the number `50` to your new value in `outdoorbot_env.py` at lines 958 and 1052 to keep evaluation statistics consistent.**
* Open `outdoorbot_env.py`, set `self._visualize_target = True`
* Build the package before running any launch files:
```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

**Run Commands**:
```bash
# Terminal 1: Launch simulation
ros2 launch outdoor_robot_spawner gazebo_world.launch.py

# Terminal 2: Run evaluation
ros2 launch outdoor_robot_spawner trained_agent.launch.py
```

### Train a new agent
Train a new reinforcement learning model from scratch.

**Configuration**:
* Open `start_training.py`, go to line 31 and set the `self._training_mode` attribute to `"training"`
* Open `outdoorbot_env.py`, set `self._visualize_target = False`
* On line 103 set `self.short_path = True` for short path training (current model uses this)
* On line 86 set `self._reward_method = 3` for hybrid reward (recommended)
* Build the package before running any launch files:
```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

**Run Commands**:
```bash
# Terminal 1: Launch headless simulation (faster training)
ros2 launch outdoor_robot_spawner headless_world.launch.py
# Or, to launch with the Gazebo GUI:
ros2 launch outdoor_robot_spawner gazebo_world.launch.py

# Terminal 2: Start training
ros2 launch outdoor_robot_spawner start_training.launch.py
```

### Hyperparameter Tuning

You can tune the PPO hyperparameters to optimize agent performance.  
To enable hyperparameter tuning, on line 31 set `self._training_mode = "hyperparam_tuning"` in `start_training.py`.  
This mode will run multiple training sessions with different hyperparameter combinations and report the best results.

**How to use**:
- Edit `start_training.py` and set `self._training_mode = "hyperparam_tuning"` (line 31).
- Optionally, adjust the hyperparameter search space in the code. (line 320 to 342)
- Build the package before running any launch files:
```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

**Run Commands**:
```bash
# Terminal 1: Launch headless simulation (faster training)
ros2 launch outdoor_robot_spawner headless_world.launch.py
# Or, to launch with the Gazebo GUI:
ros2 launch outdoor_robot_spawner gazebo_world.launch.py

# Terminal 2: Start training
ros2 launch outdoor_robot_spawner start_training.launch.py
```
**Note:** This mode is resource-intensive and may take a long time depending on the search space.

---

### Re-train an Existing Agent

You can continue training from a previously saved model checkpoint.  
To do this, on line 31 set `self._training_mode = "retraining"` in `start_training.py`.  
This will load the last saved model and resume training with the current settings.

**How to use:**
- Edit `start_training.py` and set `self._training_mode = "retraining"` (line 31).
- Make sure your previous model checkpoint exists in the expected directory (see `start_training.py` for details).
- Build the package before running any launch files:
```bash
cd ~/ros2_ws
colcon build --packages-select outdoor_robot_spawner
```

**Run Commands**:
```bash
# Terminal 1: Launch headless simulation (faster training)
ros2 launch outdoor_robot_spawner headless_world.launch.py
# Or, to launch with the Gazebo GUI:
ros2 launch outdoor_robot_spawner gazebo_world.launch.py

# Terminal 2: Start training
ros2 launch outdoor_robot_spawner start_training.launch.py
```

Training will resume from the last checkpoint.

**Note:** You can adjust training parameters or environment settings before resuming.

### Important Notes

**Training Modes** (in `start_training.py`):
- `"random_agent"`: Testing/debugging (no training)
- `"training"`: Train new model from scratch
- `"retraining"`: Continue training from saved model
- `"hyperparam_tuning"`: Optimize PPO hyperparameters

**Environment Settings** (in `outdoorbot_env.py`):
- `short_path = True`: Short, fixed path (current model)
- `short_path = False`: Complex training with 3 long paths and multiple spawn locations
- `_reward_method = 3`: Hybrid reward (designed for this task)

**Evaluation Settings** (in `trained_agent.py`):
- `n_episodes = 50`: Number of evaluation episodes
- **Important**: If you change `n_episodes`, also update lines 958 and 1052 in `outdoorbot_env.py`

## References
- **Hospital Path Planner (Thesis Reference)**: [Repository](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning.git)
