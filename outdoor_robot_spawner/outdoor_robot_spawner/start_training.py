#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

# Fixed source path for this package
PKG_SRC_DIR = '/home/dhaval_lad/dhaval_ws/src/Outdoor_navigation_decision_making/outdoor_robot_spawner'

# TrainingNode is the main ROS2 node for training and evaluation
class TrainingNode(Node):
    def __init__(self):
        # Initialize the node with custom parameters
        super().__init__("outdoorbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # Set the training mode:
        # "random_agent" - run the RL agent for testing/debugging (no training or saving)
        # "training" - train a new RL model from scratch
        # "retraining" - continue training from a previously saved model
        # "hyperparam_tuning" - search for better PPO hyperparameters before training
        self._training_mode = "random_agent"

# Callback to log and plot rewards during training in real time
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

        # Directory to save reward plots
        save_path = os.path.join(PKG_SRC_DIR, 'reward_plots')
        self.save_path = os.path.abspath(os.path.expanduser(save_path))

        # Lists to store episode rewards and lengths
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_step_rewards = []
        self.current_rewards = 0
        self.current_length = 0
        self.current_step_rewards = []

        # Set up the first figure for total reward and episode length
        plt.ion()
        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 10))
        self.reward_line, = self.axs1[0].plot([], [], 'r-')
        self.length_line, = self.axs1[1].plot([], [], 'b-')
        self.axs1[0].set_xlabel('Episode')
        self.axs1[0].set_ylabel('Total Reward')
        self.axs1[0].set_title('Total Reward vs Episode')
        self.axs1[0].grid(True)
        self.axs1[1].set_xlabel('Episode')
        self.axs1[1].set_ylabel('Episode Length')
        self.axs1[1].set_title('Episode Length vs Episode')
        self.axs1[1].grid(True)
        self.fig1.canvas.draw()

        # Set up the second figure for per-step reward
        self.fig2, self.ax2 = plt.subplots(figsize=(14, 5))
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Per Step Reward')
        self.ax2.set_title('Per Step Reward vs Steps')
        self.ax2.grid(True)
        self.fig2.canvas.draw()

    def _on_step(self) -> bool:
        # Update current reward and step count
        self.current_rewards += self.locals['rewards'][0]
        self.current_length += 1
        self.current_step_rewards.append(self.current_rewards)

        # If episode ends, store results and reset counters
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            self.episode_step_rewards.append(self.current_step_rewards.copy())

            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: reward = {self.current_rewards}")

            self.current_rewards = 0
            self.current_length = 0
            self.current_step_rewards = []

        # Update the plots in real time
        self.update_plot()
        return True

    def update_plot(self):
        # Update reward and length plots
        self.reward_line.set_xdata(np.arange(1, len(self.episode_rewards) + 1))
        self.reward_line.set_ydata(self.episode_rewards)
        self.length_line.set_xdata(np.arange(1, len(self.episode_lengths) + 1))
        self.length_line.set_ydata(self.episode_lengths)

        self.axs1[0].relim()
        self.axs1[0].autoscale_view()
        self.axs1[1].relim()
        self.axs1[1].autoscale_view()
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        # Update per-step reward plot for each episode
        self.ax2.clear()
        colors = plt.cm.jet(np.linspace(0, 1, len(self.episode_step_rewards) + 1))
        for idx, (step_rewards, color) in enumerate(zip(self.episode_step_rewards, colors)):
            self.ax2.plot(step_rewards, label=f'Episode {idx+1}', color=color)
        if self.current_step_rewards:
            self.ax2.plot(self.current_step_rewards, label=f'Episode {len(self.episode_rewards)+1}', color=colors[len(self.episode_step_rewards)])
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Total Reward')
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.ax2.grid(True)
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

    def save_plot(self):
        # Save the reward and step plots with timestamp
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        self.fig1.savefig(os.path.join(self.save_path, f"reward_length_plot_{timestamp}.png"))
        self.fig2.savefig(os.path.join(self.save_path, f"step_reward_plot_{timestamp}.png"))

# Real-time plotter for random agent runs
class RealTimePlotterRandomAgent:
    def __init__(self):
        # Directory to save plots
        save_path = os.path.join(PKG_SRC_DIR, 'reward_plots')
        self.save_path = os.path.abspath(os.path.expanduser(save_path))

        # Set up the plot for random agent
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Total Reward')
        self.ax.grid(True)
        self.ax.set_title('Total Reward per Step for Each Episode')

        # Prepare colors and lines for up to 10 episodes
        self.colors = plt.cm.jet(np.linspace(0, 1, 10))
        self.lines = [self.ax.plot([], [], label=f'Episode {i+1}', color=self.colors[i])[0] for i in range(10)]
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.step_rewards = [[] for _ in range(10)]
        self.step_numbers = [[] for _ in range(10)]

    def update_plot(self, episode, step, reward):
        # Update the plot with new reward for the given episode and step
        self.step_rewards[episode].append(reward)
        self.step_numbers[episode].append(step)
        self.lines[episode].set_xdata(self.step_numbers[episode])
        self.lines[episode].set_ydata(self.step_rewards[episode])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self):
        # Save the plot with a timestamp
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        self.fig.savefig(os.path.join(self.save_path, f"reward_length_plot_{timestamp}.png"))

# Main function to run training, evaluation, or hyperparameter tuning
def main(args=None):
    # Initialize ROS2 and create the training node
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Set up directories for saving models and logs
    trained_models_dir = os.path.join(PKG_SRC_DIR, 'rl_models')
    log_dir = os.path.join(PKG_SRC_DIR, 'logs')

    # Create directories if they do not exist
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Register the custom gym environment
    register(
        id="OutDoorEnv-v0",
        entry_point="outdoor_robot_spawner.outdoorbot_env:OutDoorEnv",
        max_episode_steps=5000,
    )
    node.get_logger().info("The environment has been registered")

    # Create and wrap the environment with a monitor
    env = gym.make('OutDoorEnv-v0')
    env = Monitor(env)

    # Check if the environment is valid
    check_env(env)
    node.get_logger().info("Environment check finished")

    # Only create reward logger if not running random agent
    if node._training_mode != "random_agent":
        reward_logger = RewardLoggerCallback(verbose=1)

    # Set up callbacks for evaluation and stopping training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        best_model_save_path=trained_models_dir,
        n_eval_episodes=40
    )

    # Run the selected training mode
    if node._training_mode == "random_agent":
        # MODE 1: RANDOM AGENT
        # Runs a random agent for a fixed number of episodes to test the environment and get a baseline.
        episodes = 5
        node.get_logger().info("Starting the RANDOM AGENT now")
        plotter = RealTimePlotterRandomAgent()
        try:
            for ep in range(episodes):
                done = False
                truncated = False
                total_reward = 0
                step_count = 0
                while not done and not truncated:
                    obs, reward, done, truncated, info = env.step(env.action_space.sample())
                    total_reward += reward
                    plotter.update_plot(ep, step_count, total_reward)
                    step_count += 1
                    if done or truncated:
                        node.get_logger().info(f"Episode {ep+1} finished with total reward: {total_reward}")
                        obs = env.reset()
        except KeyboardInterrupt:
            pass
        finally:
            # Optionally save the plot here
            pass

    elif node._training_mode == "training":
        # MODE 2: TRAINING FROM SCRATCH
        # Trains a new PPO model with specified hyperparameters and saves the model and reward plots.
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            n_steps=13796,
            gamma=0.9903371598256028,
            gae_lambda=0.9400495641396684,
            ent_coef=3.9410688714778495e-08,
            vf_coef=0.5601469263946912,
            learning_rate=0.0005254698873323493,
            clip_range=0.312078687542749
        )
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        try:
            model.learn(
                total_timesteps=int(4000000),
                reset_num_timesteps=True,
                callback=[eval_callback, reward_logger],
                tb_log_name=f"PPO_test_{timestamp}",
                progress_bar=True
            )
        except KeyboardInterrupt:
            # Save model if interrupted
            model.save(f"{trained_models_dir}/PPO_test_{timestamp}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Always save model and plots at the end
            model.save(f"{trained_models_dir}/PPO_test_{timestamp}")
            reward_logger.save_plot()

    elif node._training_mode == "retraining":
        # MODE 3: RETRAINING (CONTINUE TRAINING EXISTING MODEL)
        # Loads an existing PPO model and continues training it, saving progress and reward plots.
        node.get_logger().info("Retraining an existent model")
        trained_model_path = os.path.join(PKG_SRC_DIR, 'rl_models', 'PPO_test_14022025_002626_2.zip')
        custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
        model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)
        try:
            model.learn(
                total_timesteps=int(20000000),
                reset_num_timesteps=True,
                callback=[eval_callback, reward_logger],
                tb_log_name="PPO_test_14022025_002626_3"
            )
        except KeyboardInterrupt:
            # Save model if interrupted
            model.save(f"{trained_models_dir}/PPO_test_14022025_002626_3")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            model.save(f"{trained_models_dir}/PPO_test_14022025_002626_3")
            reward_logger.save_plot()

    elif node._training_mode == "hyperparam_tuning":
        # MODE 4: HYPERPARAMETER TUNING
        # Runs hyperparameter tuning using Optuna to find the best PPO hyperparameters.
        env.close()
        del env
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_agent, n_trials=10, n_jobs=1)
        node.get_logger().info("Best Hyperparameters: " + str(study.best_params))

    # Clean up and shut down the node
    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

# Define hyperparameter search space for Optuna (broad search)
def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1), # Default: 0.5
    }

# Define a refined hyperparameter search space for Optuna (narrow search)
def optimize_ppo_refinement(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 14336), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.96, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 9e-4), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.15, 0.37), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.94, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.00001), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0.55, 0.65), # Default: 0.5
    }

# Optuna objective function for hyperparameter tuning
def optimize_agent(trial):
    try:
        # Create a new environment for each trial
        env_opt = gym.make('OutDoorEnv-v0')
        LOG_DIR = os.path.join(PKG_SRC_DIR, 'logs')
        SAVE_PATH = os.path.join(PKG_SRC_DIR, 'tuning', 'trial_{}'.format(trial.number))
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        # Choose hyperparameters to test
        model_params = optimize_ppo_refinement(trial)

        # Create and train the model
        model = PPO("MultiInputPolicy", env_opt, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=150000)

        # Evaluate the model performance
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        env_opt.close()
        del env_opt
        model.save(SAVE_PATH)
        return mean_reward

    except Exception as e:
        # Return a very low reward if something fails
        return -10000

if __name__ == "__main__":
    main()