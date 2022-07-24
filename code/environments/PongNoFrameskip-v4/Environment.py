"""
rllib for students at the ICE by Artur Niederfahrenhorst
This file defines a custom environment that RLLib can use in Rollout Workers to produce experiences.
"""

import numpy as np
import gym
import csv
import datetime
from pathlib import Path
import os

from .Preprocessing import preprocess_images


class CustomEnvironment:
	"""
	This environment wraps the CarRacing-v0 environment.
	It adds functionality for interaction with RLLib, such as the action and observation space.
	Sensory values are added directly to the observation.
	See: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
	"""
	
	def __init__(self, env_config):
		self.last_visual_observation = None
		self.environment = gym.make(env_config.get('ENVIRONMENT_NAME'))
		self.action_space = self.environment.action_space
		self.stepcount = 0
		self.negative_rewards_in_a_row = 0
		self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(42, 42, 3))
		
		self.env_config = env_config
		self.reward_sum = 0
		self.start_time = datetime.datetime.now()
		
		self.logdir = env_config["logdir"]
		Path(self.logdir).mkdir(parents=True, exist_ok=True)
	
	def reset(self):
		"""
		Reset the environment, but also skip the first x action. Add sensory values to the observation.
		:return: observation
		"""
		self.stepcount = 0
		self.negative_rewards_in_a_row = 0
		self.last_visual_observation = self.environment.reset()
		
		for _ in range(max(0, self.env_config.get('SKIP_FIRST_X_ACTIONS', 0))):
			# Save the last overvation for video rendering, turn left, do not break but do not press gas (stay in place)
			self.last_visual_observation, reward, done, info = self.environment.step(self.environment.action_space.low)
			
		# In order to skip bad preprocessing done by RLLib, we do our own preprocessing in the workers
		# return preprocess_images(np.expand_dims(np.array(self.last_visual_observation, dtype=np.float), axis=0), self.env_config)[0]
		return preprocess_images(np.expand_dims(np.array(self.last_visual_observation, dtype=np.float), axis=0), self.env_config)[0]
	
	def step(self, action):
		"""
		Repeat an action for x steps and return the final observation and accumulated rewards from these actions.
		:param action: action to take by our agent
		:return: final observation, accumulated reward, done, info
		"""
		acc_reward = 0
		for _ in range(max(1, self.env_config.get('ACTION_REPEAT_FOR_X_STEPS', 1))):
			# Save the last observation for video rendering
			self.last_visual_observation, reward, done, info = self.environment.step(action)
			acc_reward += reward
			if done:
				continue
		
		if self.env_config.get('RENDER', False):
			self.environment.render()
		
		# To accelerate training, we stop episodes if they do not go well (50 negative rewards in a row)
		if acc_reward < 0:
			self.negative_rewards_in_a_row += 1
		else:
			self.negative_rewards_in_a_row = 0
		
		if self.negative_rewards_in_a_row > self.env_config.get('STOP_AFTER_X_NEGATIVE_REWARDS', 0) > 0:
			self.negative_rewards_in_a_row = 0
			done = True
		
		if self.env_config.get("LOG_EPISODE_REWARDS", False):
			self.reward_sum += acc_reward
		
		if done and self.env_config.get("LOG_EPISODE_REWARDS", False):
			time_since_start = datetime.datetime.now() - self.start_time
			with open(os.path.join(self.logdir, '_rewards.csv'), 'a', newline='') as reward_file:
				csv_writer = csv.writer(reward_file)
				csv_writer.writerow([os.getpid(), time_since_start.total_seconds(), self.reward_sum])
				reward_file.flush()
			self.reward_sum = 0
		
		return preprocess_images(np.expand_dims(np.array(self.last_visual_observation, dtype=np.float), axis=0), self.env_config)[0], acc_reward, done, info
