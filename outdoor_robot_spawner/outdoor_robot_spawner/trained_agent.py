#!usr/bin/env python3

"""
This script loads a trained PPO agent and evaluates its performance in the custom OutDoorEnv-v0 environment.

USAGE:
------
1. Make sure you have built your ROS2 workspace and sourced the environment.
2. Place your trained PPO model (e.g., PPO_test_14022025_002626.zip) in the 'rl_models' directory of your package.
3. Run this script with: `python3 trained_agent.py`
4. The script will:
    - Register the custom gym environment.
    - Load the trained model.
    - Evaluate the agent for a specified number of episodes.
    - Print out statistics (mean, std, min, max reward, and mean episode length).
    - Clean up and shut down the ROS2 node.

NOTES:
------
- If you want to test a different model, change the `model_filename` variable below.
- Make sure the environment and model are compatible (e.g., trained with the same observation/action space).

"""

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np

# Fixed source path for this package
PKG_SRC_DIR = '/home/dhaval_lad/dhaval_ws/src/Outdoor_navigation_decision_making/outdoor_robot_spawner'

class TrainedAgent(Node):
    """
    ROS2 Node for evaluating a trained RL agent.
    """
    def __init__(self):
        # Initialize the node with custom parameters
        super().__init__("trained_outdoorbot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

def main(args=None):
    # Initialize ROS2
    rclpy.init()
    node = TrainedAgent()
    node.get_logger().info("Trained agent node has been created")

    # Path to the directory where trained models are stored (fixed src path)
    trained_models_dir = os.path.join(PKG_SRC_DIR, 'rl_models')

    # Specify the model filename you want to test
    model_filename = 'PPO_test_14022025_002626.zip'  # <-- Change this to your model file if needed

    # Full path to the trained model
    trained_model_path = os.path.join(trained_models_dir, model_filename)

    # Register the custom gym environment
    register(
        id="OutDoorEnv-v0",
        entry_point="outdoor_robot_spawner.outdoorbot_env:OutDoorEnv",
        max_episode_steps=4000,  # Maximum number of steps per episode to achieve the goal
    )

    # Create and wrap the environment with a monitor for logging
    env = gym.make('OutDoorEnv-v0')
    env = Monitor(env)

    # Optional: Check if the environment is valid (raises error if not)
    check_env(env)

    # NOTE: If you trained your model on a different ROS2 distro (e.g., humble vs foxy),
    # the action and observation spaces may not deserialize correctly.
    # To avoid this, pass them as custom_objects so they are not loaded from the model file.
    custom_obj = {
        'action_space': env.action_space,
        'observation_space': env.observation_space
    }

    # Load the trained PPO model
    model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)

    # Number of episodes to evaluate
    n_episodes = 50

    # Evaluate the trained agent
    # Returns: (list of episode rewards, list of episode lengths)
    Mean_ep_rew, Num_steps = evaluate_policy(
        model,
        env=env,
        n_eval_episodes=n_episodes,
        return_episode_rewards=True,
        deterministic=True
    )

    # Print evaluation statistics
    node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
    node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
    node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

    # Close the environment and clean up
    env.close()
    node.get_logger().info("The script is completed, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()