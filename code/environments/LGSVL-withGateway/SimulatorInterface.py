import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import lgsvl
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


class SimulatorInterface():
  metadata = {'render.modes': ['human']}

  def __init__(self, config=CONFIG):

    config = {**config, **CONFIG}
    self.env = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), config["port"])
    if self.env.current_scene == config["scene"]:
      self.env.reset()
    else:
      self.env.load(config["scene"])

    self.spawns = self.env.get_spawn()
    self.vehicles = dict()
    self._occupied = list()
    self.seed()
    self.control = lgsvl.VehicleControl()

    self.action_space = config["action_space"]
    self.observation_space = config["observation_space"]

    self.width = self.observation_space.shape[1]
    self.height = self.observation_space.shape[0]

    self.reward = 0
    self.done = False


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def step(self, action):
    """
    Run one step of the simulation.
    """
    self.done = False
    info = {}
    prev_reward = self.reward
    
    jsonable = self.action_space.to_jsonable(action)
    self.control.steering = jsonable[0]

    if (jsonable[1] > 0):
      self.control.throttle = jsonable[1]
      self.control.braking = 0.0
    else:
      self.control.throttle = 0.0
      self.control.braking = abs(jsonable[1])


    self.ego.apply_control(self.control, sticky=True)
    self.ego.on_collision(self._on_collision)
    self.env.run(time_limit=0.1)

    observation = self._get_observation()

    self._calculate_reward()
    step_reward = self.reward - prev_reward

    return observation, step_reward, self.done, info


  def reset(self):
    """
    Resets environment for a new episode.
    """
    self.reward = 0
    self.vehicles.clear()
    self._occupied.clear()
    self.spawns.clear()
    self.env.reset()
    self.spawns = self.env.get_spawn()
    self._setup_ego()
    count = random.randint(1,10)
    while count > 0:
      self._setup_npc()
      count -= 1
    
    return self._get_observation()


  def _on_collision(self, agent1, agent2, contact):
    """
    Collision callback -- results in a negative reward.
    """
    self.reward -= 50
    self.done = True
    name1 = self.vehicles[agent1]
    name2 = self.vehicles[agent2] if agent2 is not None else "OBSTACLE"
    print("{} collided with {} at {}".format(name1, name2, contact))


  def _calculate_reward(self, mult = 1.0):
    """
    Reward is calculated based on distance travelled.
    """
    self.reward += mult * self._distance_travelled()


  def _distance_travelled(self):
    """
    Helper function to calculate the distance travelled by the ego
    vehicle. Makes an API call for position at each step.
    """
    last_pos = self.ego_position
    self.ego_position = self.ego.transform.position
    return self._proximity(last_pos, self.ego_position)
    

  def render(self, mode='human'):
    pass


  def close(self):
    self.env.stop()


  def _setup_ego(self, name = "Lexus2016RXHybrid (Autoware)", spawn_index = 0, random_spawn = False):
    """
    Spawns ego vehicle at the specified (by default index 0) spawn point in the Unity scene.
    """
    state = lgsvl.AgentState()
    if (random_spawn):
      state.transform = self.spawns[random.randint(0, len(self.spawns) - 1)]
    else:
      state.transform = self.spawns[spawn_index]
    
    self.ego = self.env.add_agent(name, lgsvl.AgentType.EGO, state)
    self.vehicles[self.ego] = "EGO"
    self._occupied.append(state.transform.position)
    self.sensors = self.ego.get_sensors()
    for s in self.sensors:
      if (s.name == "Main Camera"):
        self.camera = s
        break
    self.ego_position = state.transform.position


  def _setup_npc(self, npc_type = None, position = None, follow_lane = True,
                 speed = None, speed_upper = 25.0, speed_lower = 7.0,
                 randomize = False, min_dist = 10.0, max_dist = 40.0):
    
    """
    Spawns an NPC vehicle of a specific type at a specific location with an
    option to have it follow lane annotations in the Unity scene at a given
    speed.

    Not specifying any input results in a random selection of NPC type, a
    random spawn location within the [min_dist, max_dist] range of the ego
    vehicle, and a random speed selected within the [speed_lower, speed_upper]
    range.
    """
    
    npc_types = {"Sedan", "Hatchback", "SUV", "Jeep", "BoxTruck", "SchoolBus"}
    
    if (not npc_type):
      npc_type = random.sample(npc_types, 1)[0]

    if (randomize or not position):
      sx = self.ego.transform.position.x
      sy = self.ego.transform.position.y
      sz = self.ego.transform.position.z
      
      while (not position):
        angle = random.uniform(0.0, 2*math.pi)
        dist = random.uniform(min_dist, max_dist)
        point = lgsvl.Vector(sx + dist * math.cos(angle), sy, sz + dist * math.sin(angle))
        transform = self.env.map_point_on_lane(point)

        px = transform.position.x
        py = transform.position.y
        pz = transform.position.z


        mindist = 0.0
        maxdist = 10.0
        dist = random.uniform(mindist, maxdist)
        angle = math.radians(transform.rotation.y)
        position = lgsvl.Vector(px - dist * math.cos(angle), py, pz + dist * math.sin(angle))

        for pos in self._occupied:
          if (position and self._proximity(position, pos) < 7):
            position = None
        
        
    state = lgsvl.AgentState()
    state.transform = self.env.map_point_on_lane(position)
    n = self.env.add_agent(npc_type, lgsvl.AgentType.NPC, state)

    if (follow_lane):
      if (not speed):
        speed = random.uniform(speed_lower, speed_upper)
      n.follow_closest_lane(True, speed)
  
    self.vehicles[n] = npc_type
    self._occupied.append(position)


  def _proximity(self, position1, position2):
    """
    Helper function for calculating Euclidean distance between two Vector objects.
    """
    return math.sqrt((position1.x - position2.x)**2 + (position1.y - position2.y)**2 + (position1.z - position2.z)**2)


  def _get_observation(self):
    """
    Makes API call to simulator to capture a camera image which is saved to disk,
    loads the captured image from disk and returns it as an observation.
    """
    filename = os.path.dirname(os.path.realpath(__file__)) + '/../../tmp.jpg'
    self.camera.save(filename, quality = 75)
    im = cv2.imread(filename, 1)
    im = cv2.resize(im, (self.width, self.height))
    im = im.astype(np.float32)
    im /= 255.0

    return im
    

