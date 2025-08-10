#!usr/bin/env python3

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

class TrainedAgent(Node):

    def __init__(self):
        super().__init__("trained_hospitalbot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

def main(args=None):
    rclpy.init()
    node = TrainedAgent()
    node.get_logger().info("Trained agent node has been created")

    # We get the dir where the models are saved
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/Hospitalbot-Path-Planning/outdoor_robot_spawner'
    trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'PPO_test_14022025_002626_2.zip')   # change the model name you want to test.

    # Register the gym environment
    register(
        id="OutDoorEnv-v0",
        entry_point="outdoor_robot_spawner.outdoorbot_env:OutDoorEnv",
        max_episode_steps=4000,             # max number of steps per episode to achive the goal. 
    )

    env = gym.make('OutDoorEnv-v0')
    env = Monitor(env)

    check_env(env)

    # This is done to bypass the problem between using two different distros of ROS (humble and foxy)
    # They use different python versions, for this reason the action and observation space cannot be deserialized from the trained model
    # The solution is passing them as custom_objects, so that they won't be loaded from the model
    custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}

    # Here we load the trained model
    model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)
    n_episodes = 50
    # Evaluating the trained agent
    Mean_ep_rew, Num_steps = evaluate_policy(model, env=env, n_eval_episodes=n_episodes, return_episode_rewards=True, deterministic=True)

    # Print harvested data
    node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
    node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
    node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

    # Close env to print harvested info and destroy the hospitalbot node
    env.close()

    node.get_logger().info("The script is completed, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()