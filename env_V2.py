import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
from ObservationAdaptors import *
from VehicleControl import *
from LoadOpenDrive2 import *
from ObjectSpawn import *
from LoadOpenDrive2 import *
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

SUPPORTED_SIGNS_COUNT = 5
LEAST_HEIGHT = -10

register(
    id="CarlaEnvV2",
    entry_point="env_V2:CarlaEnvV2"
)

class CarlaEnvV2(gym.Env):
    '''
    render_modes : {"human" , None} : in it will render using pygame if the parameter was set to human!
    '''
    metadata = {"render_modes": ["human"], "render_fps": }

    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=40000):
        super(CarlaEnvV2, self).__init__()


        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        
        load_opendrive_map(map_path)
        # sleep()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        spawn_vehicles(self.client, vehicles_count) 
        self.walkers = spawn_pedestrians(self.world, walkers_count)
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Box(low=0, high=3, shape=(2,), dtype=np.int32)

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "speed_y": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "presence": spaces.Box(low=0, high=1, shape=(3, 6), dtype=np.float32),
            "lane_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  # SUPPORTED_SIGNS_COUNT traffic signs encoded as one-hot
        })

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        

    