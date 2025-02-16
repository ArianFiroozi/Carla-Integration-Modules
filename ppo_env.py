from gymnasium import spaces
import gymnasium
import numpy as np
from ObservationAdaptors import *
from VehicleControl import *
from LoadOpenDrive2 import *
from ObjectSpawn import *
from LoadOpenDrive2 import *
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import carla, random

SUPPORTED_SIGNS_COUNT = 5
LEAST_HEIGHT = -10

class CarlaEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=40000):
        super(CarlaEnv, self).__init__()
        
        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        load_opendrive_map(map_path, self.client)
        self.world = self.client.get_world()

        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        self.vehicles = spawn_vehicles(self.client, vehicles_count)
        self.walkers = spawn_pedestrians(self.world, walkers_count)
        self.max_steps = max_steps
        self.current_step = 0


        self.action_space = spaces.MultiDiscrete([4,4])

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "speed_y": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
            "presence": spaces.Box(low=0, high=1, shape=(3, 6), dtype=np.float32),
            "lane_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  # SUPPORTED_SIGNS_COUNT traffic signs encoded as one-hot
        })
        

    def reset(self, seed = 12):
        print(f'reseting')
        self.current_step = 0

        # print(f'actors count is : {len(self.world.get_actors())}')
        # actor_filters=['sensor.other.collision', 'vehicle.*', 'walker.*']
        # for filter in actor_filters:
        #     for actor in self.world.get_actors().filter(filter):
        #         if actor.is_alive:
        #             # if actor.type_id=='controller.ai.walker':
        #             #     try: 
        #             #         actor.stop()p
        #             #     except ...:
        #             #         print("ai not attached")
        #             actor.destroy()
        # self.ego_vehicle.destroy()
        # self.client.get_trafficmanager().reload()
        # self.world.destroy()
        # try:
        #     self.world = self.client.reload_world()
        # except Exception as e : 
        #     print (f'reloading error was : {e}')
        # print(f'reloaded the world')
        # settings = self.world.get_settings()
        # settings.fixed_delta_seconds=10
        # self.world.apply_settings(settings)

        # print([actor.type_id for actor in self.world.get_actors()])
        try :
            load_opendrive_map(map_path, self.client)
            print('loaded')
            self.world = self.client.get_world()

            self.ego_vehicle = spawn_ego_vehicle(self.world)
            print('ego spawned')

            # transform=self.ego_vehicle.get_transform()
            # location=carla.Location(x=transform.location.x-10, y=transform.location.y-10,z=10)
            # self.world.get_spectator().set_transform(carla.Transform(location, carla.Rotation(yaw=0.0, pitch=0)))

            self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
            print(f'reset stage 1')
            self.vehicles=spawn_vehicles(self.client, self.vehicles_count)
            self.walkers = spawn_pedestrians(self.world, self.walkers_count)
            print(f'reset stage 2')
        except ...:
            print(f'exception on reset')
            self.reset()
        return self._get_observation(), {}

    def step(self, action): 
        # time.sleep(0.1)
        # spectator = self.world.get_spectator() 
        # transform = self.ego_vehicle.get_transform() 
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))) 
        # time.sleep(0.5)
        # #print (f'geting prev observation on step {self.current_step} :')
        prev_obs = self._get_observation()
        # #print (f'got prev obs on step')
        #print(f'ego z location = {self.ego_vehicle.get_location().z}')
        # for actor in actors:
            #print(f'actor {actor.type_id} location is : {actor.get_location().z}', end="-")
        
        done = self.current_step >= self.max_steps
        #print (f'take action on step {self.current_step} : {action}')
        speed_action = action[0]
        turn_action = action[1]
        #print(f"executed command is: {self.vehicle_controller.speed_action_convertor(speed_action)}")
        self.vehicle_controller.exec_command(self.vehicle_controller.speed_action_convertor(speed_action))
        self.vehicle_controller.exec_command(self.vehicle_controller.turn_action_convertor(turn_action))
        # {self.current_step} : {prev_obs}')
        # if (self.world)
        ##print (f'ticking world on step {self.current_step} :')
        try:
            self.world.tick()
            #print (f'World has ticked')
        except ...:
            self.reset()
            #print("Exception supressed")
            return prev_obs, 0, False, False, {}
        # Step pedestrians
        #print (f'steping peds on step {self.current_step} :')
        step_peds(self.world, self.walkers)
        if self.vehicle_controller.collision_happened:
            done = True
            print(f'step colision')
            self.vehicle_controller.collision_happened=False
        if (self.ego_vehicle.get_location().z <= LEAST_HEIGHT and not done):
            print("oftadam")
            # return prev_obs, 0, True, False, {}
            done = True
        # Calculate reward
        #print (f'calculationg rewards on step {self.current_step} :')
        reward = self.vehicle_controller.get_reward(prev_obs)
        if done:
            print(f'decresed colision penalty')
            reward += -100
        ##print (f'got reward {reward} on step {self.current_step} :')

        # Additional penalty or reward for traffic signs
        traffic_signs = self._get_nearby_traffic_signs()
        reward += self._process_traffic_signs(traffic_signs)

        # Get observation
        #print (f'geting observation on step {self.current_step} :')
        obs = self._get_observation()
        # if (obs["presence"].sum() > 7):
        #     print (f'presence : {obs["presence"]}')
        #print (f'got obs on step')# {self.current_step} : {obs}')
        # Check if done
        self.current_step += 1
        #print (f'startin step {self.current_step} :')
        
        truncated = False  # Update this based on your custom truncation logic, if any.
        #print(f'returning')
        
        # #print(obs, reward, done, truncated)
        return obs, reward, done , truncated, {}



    def _get_observation(self):
        x_speed_matrix, y_speed_matrix, presence_matrix = get_speed_matrices(self.ego_vehicle)
        ##print (f'hora out index ta inja nist')
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
        encoded_signs = np.zeros(SUPPORTED_SIGNS_COUNT)  # Assuming 10 possible traffic sign types
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
    
    # Wrap the environment with Monitor and DummyVecEnv
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    ##print(f"Environment type: {type(env)}")
    
    # Check the environment
    # check_env(env, warn=True)
    
    try:
        # Use MultiInputPolicy for dict observation space
        model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log="./ppo_carla/")
        model.learn(total_timesteps=steps)
        ##print(f'saving model')
        model.save("ppo_carla_model")
            # sleep(0.1)
    except ...:
        print(f"Error during model training: ")
map_path = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/LoadOpenDrive2/simple_map.xodr"
run(map_path, 0, 0, 40000, "cuda")