from gymnasium import spaces
import gymnasium
import numpy as np
import torch
from ObservationAdaptors import *
from VehicleControl import *
from LoadOpenDrive2 import *
from ObjectSpawn import *
from LoadOpenDrive2 import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from IPython.display import clear_output

import carla

SUPPORTED_SIGNS_COUNT = 5
LEAST_HEIGHT = -10

with open("training.pid", "w") as f:
    f.write(str(os.getpid()))

class CarlaEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=40000, init_speed=0.5):
        super(CarlaEnv, self).__init__()
        
        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        self.init_speed=init_speed
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        load_opendrive_map(map_path, self.client)
        self.world = self.client.get_world()

        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        self.vehicles = spawn_vehicles(self.client, vehicles_count)
        self.walkers = spawn_pedestrians(self.world, walkers_count)
        self.max_steps = max_steps
        self.current_step = 0

        self.__set_world_settings()

        self.action_space = spaces.Box(low=0, high=3, shape=(2,), dtype=np.int32)

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-torch.inf, high=torch.inf, shape=(3, 6), dtype=np.float32),
            "speed_y": spaces.Box(low=-torch.inf, high=torch.inf, shape=(3, 6), dtype=np.float32),
            "presence": spaces.Box(low=0, high=1, shape=(3, 6), dtype=np.float32),
            "lane_angle": spaces.Box(low=-torch.pi, high=torch.pi, shape=(1,), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  # SUPPORTED_SIGNS_COUNT traffic signs encoded as one-hot
        })
        self.last_heartbeat_time = time.time()
        with open("heartbeat.txt", "w") as f:
            f.write(str(self.last_heartbeat_time))

    def reset(self, seed = 12):
        # print(f'reseting...')
        self.current_step = 0

        # load_opendrive_map(map_path, self.client) #TODO: remove this, but keep in mind this breaks the spawns
        self.vehicle_controller.sensor_c.destroy()
        self.vehicle_controller.sensor_l.destroy()
        if hasattr(self.ego_vehicle, 'is_listening') and self.ego_vehicle.is_listening:
            self.ego_vehicle.stop()
        if self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()

        self.world = self.client.get_world()


        self.ego_vehicle = spawn_ego_vehicle(self.world, self.init_speed)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)

        # self.vehicles=spawn_vehicles(self.client, self.vehicles_count)
        # self.walkers = spawn_pedestrians(self.world, self.walkers_count)
        # clear_output(wait=True)
        return self._get_observation(), {}

    def __set_world_asynch(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.client.get_trafficmanager().set_synchronous_mode(False)
        self.world.apply_settings(settings)
    
    def __set_world_settings(self, no_rendering_mode=False, fixed_delta_seconds=0.1): #TODO: parameters
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.client.get_trafficmanager().set_synchronous_mode(True)
        settings.no_rendering_mode = no_rendering_mode
        settings.fixed_delta_seconds = fixed_delta_seconds
        settings.substepping = True
        settings.max_substep_delta_time = 0.1
        settings.max_substeps = 10
        self.world.apply_settings(settings)

    def step(self, action): 
        prev_obs = self._get_observation()        
        done = self.current_step >= self.max_steps
        speed_action = int(action[0])
        turn_action = int(action[1])
        self.vehicle_controller.exec_command(self.vehicle_controller.speed_action_convertor(speed_action))
        self.vehicle_controller.exec_command(self.vehicle_controller.turn_action_convertor(turn_action))
        try:
            # print("ticking")
            self.world.tick(5.0)
            # print("ticked")
        except ...:
            print ("tick fail")
            self.reset()
            return prev_obs, 0, False, False, {}
        step_peds(self.world, self.walkers)
        if self.vehicle_controller.collision_happened:
            done = True
            # print(f'step colision')
            self.vehicle_controller.collision_happened=False
        if (self.ego_vehicle.get_location().z <= LEAST_HEIGHT and not done):
            # print("oftadam")
            done = True
            
        reward = self.vehicle_controller.get_reward(prev_obs)
        if done:
            # print(f'decresed colision penalty')
            reward += -100
            
        traffic_signs = self._get_nearby_traffic_signs()
        reward += self._process_traffic_signs(traffic_signs)

        obs = self._get_observation()
        # if (obs["presence"].sum() > 7):
        #     print (f'presence : {obs["presence"]}')
        #print (f'got obs on step')# {self.current_step} : {obs}')
        self.current_step += 1
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= 10.0:
            with open("heartbeat.txt", "w") as f:
                f.write(str(current_time))
            self.last_heartbeat_time = current_time
        
        truncated = False
        if self.current_step>200: #not allowing an episode run more than 200 steps. TODO: checkfor accuracy
            done=True

        return obs, reward, done , truncated, {}

    def _get_observation(self):
        x_speed_matrix, y_speed_matrix, presence_matrix = get_speed_matrices(self.ego_vehicle)
        lane_angle = get_lane_angle(self.ego_vehicle, self.world.get_map())
        traffic_signs = self._encode_traffic_signs()

        return {
            "speed_x": x_speed_matrix,
            "speed_y": y_speed_matrix,
            "presence": presence_matrix,
            "lane_angle": torch.asarray(lane_angle),
            "traffic_signs": traffic_signs,
            "max_speed": 100
        }

    def _get_nearby_traffic_signs(self):
        return get_nearby_signs(self.ego_vehicle, self.world.get_map(), radius=10)

    def _encode_traffic_signs(self):
        # traffic_signs = self._get_nearby_traffic_signs() TODO
        encoded_signs = np.zeros(SUPPORTED_SIGNS_COUNT)
        return encoded_signs
        for sign in traffic_signs:
            if sign.type.isdigit():
                sign_index = int(sign.type) % 10
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
import os

def create_checkpoints_folder(base_path='./checkpoints/checkpoint'):
    folder_index = 0
    while True:
        folder_name = f"{base_path}{folder_index}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        folder_index += 1

def get_latest_checkpoint(base_path='./checkpoints/checkpoint'):
    folder_index = 0
    folder_name=""
    while True:
        folder_name = f"{base_path}{folder_index}"
        if not os.path.exists(folder_name):
            break
        folder_index += 1
    
    file_num=1
    while True:
        file_name = f"{base_path}{folder_index-1}/ppo_carla_checkpoint_{file_num}000_steps.zip"
        if not os.path.exists(file_name):
            break
        file_num += 1
    if file_num==1 and folder_index==0:
        return ""
    return f"{base_path}{folder_index-1}/ppo_carla_checkpoint_{file_num-1}000_steps.zip"

def run(map_path, walkers_count, vehicles_count, steps, device, init_speed):
    env = CarlaEnv(map_path, walkers_count, vehicles_count, max_steps=steps, init_speed=init_speed)
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    model_path = "ppo_carla_model"

    latest_checkpoint=get_latest_checkpoint()
    if latest_checkpoint != "":
        print(f"Loading existing model: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, verbose=2, env=env, n_epochs=10)
    else:
        print("Creating new model...")
        model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log="./ppo_carla/", n_epochs=10)
    
    try:
        checkpoints_folder = create_checkpoints_folder()
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints_folder, name_prefix='ppo_carla_checkpoint')

        # model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log="./ppo_carla/")
        model.learn(total_timesteps=steps, callback=checkpoint_callback)
        print ("saving model")
        model.save("ppo_carla_model")
    except ...:
        print(f"Error during model training: ")
map_path = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/LoadOpenDrive2/simple_map.xodr"
run(map_path, 0, 10, 2000000, "cuda", 0.7)