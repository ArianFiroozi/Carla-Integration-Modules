from gymnasium import spaces
import gymnasium
import numpy as np
from ObservationAdaptors import *
from VehicleControl import *
from LoadOpenDrive2 import *
from ObjectSpawn import *
from LoadOpenDrive2 import *
import time

import carla, random

SUPPORTED_SIGNS_COUNT = 5

class CarlaEnv(gymnasium.Env):
    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=50000000):
        super(CarlaEnv, self).__init__()
        
        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        
        load_opendrive_map(map_path)
        
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        spawn_vehicles(self.client, vehicles_count)
        self.walkers = spawn_pedestrians(self.world, walkers_count)
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Box(low=0, high=2, shape=(2,), dtype=np.int32)

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "speed_y": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "presence": spaces.Box(low=0, high=1, shape=(3, 6), dtype=np.float32),
            "lane_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  # SUPPORTED_SIGNS_COUNT traffic signs encoded as one-hot
        })
        self.reset()

    def reset(self, seed = 12):
        self.current_step = 0
        destroy_all_actors(self.client)
        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        spawn_vehicles(self.client, self.vehicles_count)
        self.walkers = spawn_pedestrians(self.world, self.walkers_count)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        speed_action = int(action[0])
        turn_action = int(action[1])
        self.vehicle_controller.exec_command(self.vehicle_controller.speed_action_convertor(speed_action))
        self.vehicle_controller.exec_command(self.vehicle_controller.turn_action_convertor(turn_action))

        prev_obs = self._get_observation()

        self.world.tick()
        # Step pedestrians
        step_peds(self.world, self.walkers)

        # Calculate reward
        reward = self.vehicle_controller.get_reward()

        # Additional penalty or reward for traffic signs
        traffic_signs = self._get_nearby_traffic_signs()
        reward += self._process_traffic_signs(traffic_signs)

        # Get observation
        obs = self._get_observation()

        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Return step info
        return obs, reward, done, {}


    def _get_observation(self):
        x_speed_matrix, y_speed_matrix, presence_matrix = get_speed_matrices(self.ego_vehicle)
        lane_angle = get_lane_angle(self.ego_vehicle, self.world.get_map())
        traffic_signs = self._encode_traffic_signs()

        return {
            "speed_x": x_speed_matrix,
            "speed_y": y_speed_matrix,
            "presence": presence_matrix,
            "lane_angle": np.array(lane_angle),
            "traffic_signs": traffic_signs,
            "max_speed": 100
        }

    def _get_nearby_traffic_signs(self):
        return get_nearby_signs(self.ego_vehicle, self.world.get_map(), radius=10)

    def _encode_traffic_signs(self):
        # traffic_signs = self._get_nearby_traffic_signs() TODO
        encoded_signs = np.zeros(10)  # Assuming 10 possible traffic sign types
        return encoded_signs
        for sign in traffic_signs:
            if sign.type.isdigit():
                sign_index = int(sign.type) % 10  # Map sign type to an index
                encoded_signs[sign_index] = 1

        return encoded_signs

    def _process_traffic_signs(self, traffic_signs):
        # penalty = 0
        # for sign in traffic_signs:
        #     if sign.type == "1000001":  # Example: stop sign
        #         penalty -= 5 if self.vehicle_controller.control.throttle > 0 else 0
        #     elif sign.type == "1000002":  # Example: speed limit
        #         speed = self.ego_vehicle.get_velocity()
        #         speed_kmh = 3.6 * ((speed.x**2 + speed.y**2)**0.5)
        #         if speed_kmh > 50:  # Assuming speed limit of 50 km/h
        #             penalty -= 1
        # return penalty
        return 0

    def render(self, mode="human"):
        pass  # Visualization logic can go here if needed

    def close(self):
        pass
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

def run(map_path, walkers_count, vehicles_count, steps, device):
    env = CarlaEnv(map_path, walkers_count, vehicles_count, max_steps=steps)
    # check_env(env, warn=True)  
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_carla/", device=device)
    model.learn(total_timesteps=100000)
    model.save("sac_carla_model")
map_path = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/LoadOpenDrive2/simple_map.xodr"
run(map_path, 10, 10, 10000, "cuda")