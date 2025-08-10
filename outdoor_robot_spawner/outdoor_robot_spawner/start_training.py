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

class TrainingNode(Node):

    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Defines which action the script will perform "random_agent", "training", "retraining" or "hyperparam_tuning"
        self._training_mode = "random_agent"

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        save_path = '~/ros2_ws/src/Hospitalbot-Path-Planning/outdoor_robot_spawner/reward_plots'
        self.save_path = os.path.abspath(os.path.expanduser(save_path))
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_step_rewards = []
        self.current_rewards = 0
        self.current_length = 0
        self.current_step_rewards = []

        # Set up the first figure
        plt.ion()
        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 10))
        self.reward_line, = self.axs1[0].plot([], [], 'r-')
        self.length_line, = self.axs1[1].plot([], [], 'b-')
        self.axs1[0].set_xlabel('Episode')
        self.axs1[0].set_ylabel('Total Reward')
        self.axs1[0].set_title('Total Reward vs Episode')
        self.axs1[0].grid(True)  # Add grid
        self.axs1[1].set_xlabel('Episode')
        self.axs1[1].set_ylabel('Episode Length')
        self.axs1[1].set_title('Episode Length vs Episode')
        self.axs1[1].grid(True)  # Add grid
        self.fig1.canvas.draw()

        # Set up the second figure
        self.fig2, self.ax2 = plt.subplots(figsize=(14, 5))   #(horizontal,vertical)
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Per Step Reward')
        self.ax2.set_title('Per Step Reward vs Steps')
        self.ax2.grid(True)  # Add grid
        self.fig2.canvas.draw()

    def _on_step(self) -> bool:
        # Accumulate rewards and steps
        self.current_rewards += self.locals['rewards'][0]
        self.current_length += 1
        self.current_step_rewards.append(self.current_rewards)

        # Update the plot at each step
        # self.update_plot()

        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            self.episode_step_rewards.append(self.current_step_rewards.copy())

            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: reward = {self.current_rewards}")

            self.current_rewards = 0
            self.current_length = 0
            self.current_step_rewards = []
        self.update_plot()
        return True
    
    def update_plot(self):
        self.reward_line.set_xdata(np.arange(1, len(self.episode_rewards) + 1))  # Shift x-axis by 1
        self.reward_line.set_ydata(self.episode_rewards)

        self.length_line.set_xdata(np.arange(1, len(self.episode_lengths) + 1))  # Shift x-axis by 1
        self.length_line.set_ydata(self.episode_lengths)

        self.axs1[0].relim()
        self.axs1[0].autoscale_view()
        self.axs1[1].relim()
        self.axs1[1].autoscale_view()

        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        # Update the step reward plot in the second figure
        self.ax2.clear()
        colors = plt.cm.jet(np.linspace(0, 1, len(self.episode_step_rewards) + 1))
        for idx, (step_rewards, color) in enumerate(zip(self.episode_step_rewards, colors)):
            self.ax2.plot(step_rewards, label=f'Episode {idx+1}', color=color)
        if self.current_step_rewards:
            self.ax2.plot(self.current_step_rewards, label=f'Episode {len(self.episode_rewards)+1}', color=colors[len(self.episode_step_rewards)])
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Total Reward')
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.ax2.grid(True)  # Add grid

        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

    def save_plot(self):
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        self.fig1.savefig(os.path.join(self.save_path, f"reward_length_plot_{timestamp}.png"))
        self.fig2.savefig(os.path.join(self.save_path, f"step_reward_plot_{timestamp}.png"))

class RealTimePlotterRandomAgent:
    def __init__(self):
        save_path = '~/ros2_ws/src/Hospitalbot-Path-Planning/outdoor_robot_spawner/reward_plots'
        self.save_path = os.path.abspath(os.path.expanduser(save_path))
        # Set up the plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Total Reward')
        self.ax.grid(True)
        self.ax.set_title('Total Reward per Step for Each Episode')

        self.colors = plt.cm.jet(np.linspace(0, 1, 10))  # Generate 10 colors for 10 episodes
        # self.lines = []                                                                                        ## case-1
        self.lines = [self.ax.plot([], [], label=f'Episode {i+1}', color=self.colors[i])[0] for i in range(10)]  ## case-2
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))                                                  ## case-2
        self.step_rewards = [[] for _ in range(10)]                                                              ## case-2
        self.step_numbers = [[] for _ in range(10)]                                                              ## case-2

    def update_plot(self, episode, step, reward):                                ## for ploting reward at real-time case-2
        self.step_rewards[episode].append(reward)
        self.step_numbers[episode].append(step)
        self.lines[episode].set_xdata(self.step_numbers[episode])
        self.lines[episode].set_ydata(self.step_rewards[episode])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.grid(True)
        self.fig.canvas.draw()  # Draw the updated plot
        self.fig.canvas.flush_events()  # Flush GUI events to update the plot

    # def plot_episode(self, steps, rewards, episode):              ## for ploting reward at the end of every episod  case-1
    #     line, = self.ax.plot(steps, rewards, label=f'Episode {episode+1}', color=self.colors[episode])
    #     self.lines.append(line)
    #     self.ax.legend()
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()

    def save_plot(self):
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        self.fig.savefig(os.path.join(self.save_path, f"reward_length_plot_{timestamp}.png"))

def main(args=None):

    # Initialize the training node to get the desired parameters
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Create the dir where the trained RL models will be saved
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/Hospitalbot-Path-Planning/outdoor_robot_spawner'
    trained_models_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # First we register the gym environment created in hospitalbot_env module
    register(
        id="OutDoorEnv-v0",
        entry_point="outdoor_robot_spawner.outdoorbot_env:OutDoorEnv",
        max_episode_steps=5000,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make('OutDoorEnv-v0')
    env = Monitor(env)

    # Sample Observation and Action space for Debugging
    #node.get_logger().info("Observ sample: " + str(env.observation_space.sample()))
    #node.get_logger().info("Action sample: " + str(env.action_space.sample()))

    # Here we check if the custom gym environment is fine
    check_env(env)
    node.get_logger().info("Environment check finished")

    if node._training_mode != "random_agent":
        # this callback is for ploting the training data onto the plot in real time.
        reward_logger = RewardLoggerCallback(verbose=1)

    # Now we create two callbacks which will be executed during training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=trained_models_dir, n_eval_episodes=40)
    
    
    if node._training_mode == "random_agent":
        # N° Episodes
        episodes = 5
        ## Execute a random agent
        node.get_logger().info("Starting the RANDOM AGENT now")
        plotter = RealTimePlotterRandomAgent()
        try:
            for ep in range(episodes):
                done = False
                truncated = False
                total_reward = 0  # To keep track of the total reward per episode
                # step_rewards = []                                                                        ## case-1
                # step_numbers = []                                                                        ## case-1
                step_count = 0                                                                             ## case-1 & case-2
                while not done and not truncated:
                    obs, reward, done, truncated, info = env.step(env.action_space.sample())
                    # node.get_logger().info("Agent state: [" + str(info["distance"]) + ", " + str(info["angle"]) + "]")
                    total_reward += reward
                    # node.get_logger().info(f"Total reward: {total_reward}")
                    plotter.update_plot(ep, step_count, total_reward)                                      ## case-2
                    # step_rewards.append(total_reward)                                                    ## case-1 
                    # step_numbers.append(step_count)                                                      ## case-1
                    step_count += 1
                    if done or truncated:
                        node.get_logger().info(f"Episode {ep+1} finished with total reward: {total_reward}")
                        # plotter.plot_episode(step_numbers, step_rewards, ep)                             ## case-1
                        obs = env.reset()
        except KeyboardInterrupt:
            pass
        finally:
            #plotter.save_plot()
            pass

    elif node._training_mode == "training":
        # Train the model
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=13796, gamma=0.9903371598256028, gae_lambda=0.9400495641396684, ent_coef=3.9410688714778495e-08, vf_coef=0.5601469263946912, learning_rate=0.0005254698873323493, clip_range=0.312078687542749)   # short path & 270
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        # Execute training
        try:
            model.learn(total_timesteps=int(4000000), reset_num_timesteps=True, callback=[eval_callback, reward_logger], tb_log_name=f"PPO_test_{timestamp}", progress_bar=True)
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/PPO_test_{timestamp}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Save the trained model
            model.save(f"{trained_models_dir}/PPO_test_{timestamp}")
            # Ensure the reward logger plots are saved
            reward_logger.save_plot()

    elif node._training_mode == "retraining":
        ## Re-train an existent model
        node.get_logger().info("Retraining an existent model")
        # Path in which we find the model
        trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'PPO_test_14022025_002626_2.zip')
        # Here we load the rained model
        custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}
        model = PPO.load(trained_model_path, env=env, custom_objects=custom_obj)
        # Execute training
        try:
            model.learn(total_timesteps=int(20000000), reset_num_timesteps=True, callback=[eval_callback, reward_logger], tb_log_name="PPO_test_14022025_002626_3")     # rename every time the file name 
        except KeyboardInterrupt:
            # If you notice that the training is sufficiently well interrupt to save
            model.save(f"{trained_models_dir}/PPO_test_14022025_002626_3")              # rename every time the file name 
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Save the trained model
            model.save(f"{trained_models_dir}/PPO_test_14022025_002626_3")              # rename every time the file name 
            reward_logger.save_plot()

    elif node._training_mode == "hyperparam_tuning":
        # Delete previously created environment
        env.close()
        del env
        # Hyperparameter tuning using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_agent, n_trials=10, n_jobs=1)
        # Print best params
        node.get_logger().info("Best Hyperparameters: " + str(study.best_params))

    # Shutting down the node
    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

def optimize_ppo(trial):
    ## This method defines the range of hyperparams to search fo the best tuning
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1), # Default: 0.5
    }

def optimize_ppo_refinement(trial):
    ## This method defines a smaller range of hyperparams to search fo the best tuning
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 14336), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.96, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 9e-4), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.15, 0.37), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.94, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.00001), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0.55, 0.65), # Default: 0.5
    }

def optimize_agent(trial):
    ## This method is used to optimize the hyperparams for our problem
    try:
        # Create environment
        env_opt = gym.make('OutDoorEnv-v0')
        # Setup dirs
        HOME_DIR = os.path.expanduser('~')
        PKG_DIR = 'ros2_ws/src/Hospitalbot-Path-Planning/outdoor_robot_spawner'
        LOG_DIR = os.path.join(HOME_DIR, PKG_DIR, 'logs')
        SAVE_PATH = os.path.join(HOME_DIR, PKG_DIR, 'tuning', 'trial_{}'.format(trial.number))
        # Setup the parameters
        #model_params = optimize_ppo(trial)
        model_params = optimize_ppo_refinement(trial)
        # Setup the model
        model = PPO("MultiInputPolicy", env_opt, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=150000)
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)
        # Close env and delete
        env_opt.close()
        del env_opt

        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -10000

if __name__ == "__main__":
    main()