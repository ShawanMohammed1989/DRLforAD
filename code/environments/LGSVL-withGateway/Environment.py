import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import lgsvl
import ray
import os
import random
import math
import cv2

CONFIG = {
"scene": "BorregasAve",
"port" : 8181,

"action_space" :
  spaces.Box(
    np.array([-1,-1]), 
    np.array([+1,+1,]),
    dtype=np.float32,
  ), # steering, throttle (+) / brake (-)

"observation_space" : 
  spaces.Box(
      low=0,
      high=255,
      shape=(100, 300, 3),
      dtype=np.uint8
    ) # RGB image from front camera
}

class CustomEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, config=CONFIG):
    self.simulator_interface = config["SIMULATOR_INTERFACE_HANDLE"]
    
    config = {**config, **CONFIG}
 
    self.seed()

    self.action_space = config["action_space"]
    self.observation_space = config["observation_space"]

  def step(self, action):
    return ray.get(self.simulator_interface.step.remote(action))

  def reset(self):
    return ray.get(self.simulator_interface.reset.remote())

  def close(self):
    return ray.get(self.simulator_interface.close.remote())


