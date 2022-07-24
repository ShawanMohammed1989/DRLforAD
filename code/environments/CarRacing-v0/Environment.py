"""
Master Thesis project by Artur Niederfahrenhorst
This file defines a custom environment that RLLib can use in Rollout Workers to produce experiences.
"""

import numpy as np
import gym
import csv
import datetime
from pathlib import Path
import os

# In order to view images of the VAE we need our own tensorboard
base_logdir = "../results/episode_rewards/"
Path(base_logdir).mkdir(parents=True, exist_ok=True)


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
		self.observation_space = gym.spaces.Dict({
			'image': self.environment.observation_space,
			'speed': gym.spaces.Box(low=-200, high=200., shape=(1,)),
			'gyro': gym.spaces.Box(low=-200, high=200., shape=(1,)),
			'steering': gym.spaces.Box(low=-2, high=2, shape=(1,))
		})
		
		self.env_config = env_config
		self.reward_sum = 0
		self.start_time = datetime.datetime.now()
		self.log_dir = base_logdir + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
	
	def reset(self):
		"""
		Reset the environment, but also skip the first x action. Add sensory values to the observation.
		:return: observation
		"""
		self.stepcount = 0
		self.negative_rewards_in_a_row = 0
		reset_image = self.environment.reset()
		self.last_visual_observation = reset_image
		
		for _ in range(max(0, self.env_config.get('SKIP_FIRST_X_ACTIONS', 0))):
			# Save the last overvation for video rendering, turn left, do not break but do not press gas (stay in place)
			self.last_visual_observation, reward, done, info = self.environment.step(self.environment.action_space.low)
		
		# Add information to the observation
		speed = self.get_speed()
		gyro = self.get_gyro()
		steering = self.get_steering()
		
		# The observation accommodates sensory values to experiment with SDF (See Kemal's thesis)
		observation = {'image': np.asanyarray(self.last_visual_observation),
		               'speed': np.asanyarray([speed]),
		               'gyro': np.asanyarray([gyro]),
		               'steering': np.asanyarray([steering])}
		
		return observation
	
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
			self.render(mode='human')
		
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
			with open(self.log_dir + '_rewards.csv', 'a', newline='') as reward_file:
				csv_writer = csv.writer(reward_file)
				csv_writer.writerow([os.getpid(), time_since_start.total_seconds(), self.reward_sum])
				reward_file.flush()
			self.reward_sum = 0
		
		# Add information to the observation
		speed = self.get_speed()
		gyro = self.get_gyro()
		steering = self.get_steering()
		
		# The observation accommodates sensory values to experiment with SDF (See Kemal's thesis)
		observation = {'image': np.asanyarray(self.last_visual_observation),
		               'speed': np.asanyarray([speed]),
		               'gyro': np.asanyarray([gyro]),
		               'steering': np.asanyarray([steering])}
		
		return observation, acc_reward, done, info
	
	def render(self, mode='human'):
		"""
		Depending on mode, either render envornment on screen, or return an RGB tensor.
		:param mode: render mode
		:return: None, or an RGB tensor
		"""
		if mode == 'rgb_array':
			return np.array(self.last_visual_observation)  # return RGB frame suitable for video
		elif mode == 'human':
			self.environment.render()
	
	def get_speed(self):
		return np.sqrt(np.square(self.environment.car.hull.linearVelocity[0]) +
		               np.square(self.environment.car.hull.linearVelocity[1])) / 120
	
	def get_gyro(self):
		return self.environment.car.hull.angularVelocity
	
	def get_steering(self):
		return self.environment.car.wheels[0].joint.angle

