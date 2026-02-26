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
from manual_controller import *
import random
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import os
import carla
import os
from pathlib import Path
import time  





REPO_ROOT = Path(__file__).resolve().parent

RUN_DIR = REPO_ROOT / "runs"
RUN_DIR.mkdir(exist_ok=True)

PID_PATH = RUN_DIR / "training.pid"
HEARTBEAT_PATH = RUN_DIR / "heartbeat.txt"

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

MAX_ITER_IN_EPISODE=5000
SUPPORTED_SIGNS_COUNT = 5
LEAST_HEIGHT = -10

with open(PID_PATH, "w") as f:
    f.write(str(os.getpid()))

class CarlaEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=40000, init_speed=0.5):
        super(CarlaEnv, self).__init__()
        
        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        self.init_speed=init_speed
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)  

        self.world = self.client.get_world()
        
        if map_path:
            map_path = str(Path(map_path))
            if map_path.lower().endswith(".xodr") and os.path.exists(map_path):
                print(f"Loading OpenDRIVE map: {map_path}")
                load_opendrive_map(map_path, self.client)
                self.world = self.client.get_world()
            else:
                print(f"Skipping OpenDRIVE load (file not found or not .xodr): {map_path}")
                print("Using current CARLA map:", self.world.get_map().name)
        else:
            print("No map_path provided. Using current CARLA map:", self.world.get_map().name)

        self._apply_sync(fixed_dt=0.05)
        print("SYNC:", self.world.get_settings().synchronous_mode, "dt:", self.world.get_settings().fixed_delta_seconds)

        self.ego_vehicle = spawn_ego_vehicle(self.world)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        self.vehicles = spawn_vehicles(self.client, vehicles_count)
        self.walkers = spawn_pedestrians(self.world, walkers_count)
        self.max_steps = max_steps
        self.current_step = 0

        # self.__set_world_settings()

        self.action_space = spaces.MultiDiscrete([5,4])

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-torch.inf, high=torch.inf, shape=(25, 11), dtype=np.float32),
            "speed_y": spaces.Box(low=-torch.inf, high=torch.inf, shape=(25, 11), dtype=np.float32),
            "presence": spaces.Box(low=0, high=1, shape=(25, 11), dtype=np.float32),
            "lane_angle": spaces.Box(low=-torch.pi, high=torch.pi, shape=(1,), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  # SUPPORTED_SIGNS_COUNT traffic signs encoded as one-hot
            "ego_speed_x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "ego_speed_y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "ego_in_lane_position_x": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),  # Lateral offset, assuming lane width ~10m
            "throttle": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "brake": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "steering_angle": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "reverse": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.last_heartbeat_time = time.time()
        with open(HEARTBEAT_PATH, "w") as f:
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


        self.ego_vehicle = spawn_ego_vehicle(self.world, round(random.uniform(0, 1) / 0.3) * 0.3)
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
    
    
    
    def _apply_sync(self, fixed_dt=0.05):
        # always grab the current world (after map load)
        self.world = self.client.get_world()
        settings = self.world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt

        # ✅ physics stability
        settings.substepping = True
        settings.max_substep_delta_time = 0.01  # 100 Hz physics
        settings.max_substeps = int(fixed_dt / settings.max_substep_delta_time) + 1

        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
    
    
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
        
        # 1. Define end-of-episode variables (both default to False)
        terminated = False  # Indicates failure (collision or falling off the map)
        truncated = False   # Indicates time ran out (max steps reached without failure)

        speed_action = int(action[0])
        turn_action = int(action[1])
        
        self.vehicle_controller.exec_command(self.vehicle_controller.speed_action_convertor(speed_action))
        self.vehicle_controller.exec_command(self.vehicle_controller.turn_action_convertor(turn_action))
        
        try:
            self.world.tick()
        except Exception as e:
            print(f"tick fail: {e}")
            # If tick fails, do not reward or penalize, just reset the environment
            self.reset()
            return prev_obs, 0, False, False, {}

        step_peds(self.world, self.walkers)

        # 2. Check for failure (Terminated)
        if self.vehicle_controller.collision_happened:
            terminated = True
            self.vehicle_controller.collision_happened = False
        
        if (self.ego_vehicle.get_location().z <= LEAST_HEIGHT and not terminated):
            terminated = True
            
        # 3. Calculate reward
        reward = self.vehicle_controller.get_reward(prev_obs)
        
        # Apply heavy penalty only on failure
        if terminated:
            reward += -100
            
        traffic_signs = self._get_nearby_traffic_signs()
        reward += self._process_traffic_signs(traffic_signs)

        obs = self._get_observation()
        
        self.current_step += 1
        
        # 4. Update Heartbeat
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= 10.0:
            with open(HEARTBEAT_PATH, "w") as f:
                f.write(str(current_time))
            self.last_heartbeat_time = current_time
        
        # 5. Check for timeout (Truncated)
        if self.current_step >= self.max_steps:
            truncated = True

        # Gymnasium standard requires returning terminated and truncated separately
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        x_speed_matrix, y_speed_matrix, presence_matrix, vx_local, vy_local = \
            get_speed_matrices(self.ego_vehicle)
        lane_angle = get_lane_angle(self.ego_vehicle, self.world.get_map())
        traffic_signs = self._encode_traffic_signs()
        
        control = self.ego_vehicle.get_control()
        throttle = control.throttle
        brake = control.brake
        steering = control.steer
        reverse = 1.0 if control.reverse else 0.0
        
        map = self.world.get_map()
        waypoint = map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        lane_center = waypoint.transform.location
        ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
        theta = np.radians(ego_yaw)
        ego_location = self.ego_vehicle.get_transform().location
        # Vector from lane center to ego vehicle in global coordinates
        dx_global = ego_location.x - lane_center.x
        dy_global = ego_location.y - lane_center.y
        # Rotate to local coordinates (lateral offset is along local y-axis)
        lateral_offset = -dx_global * np.sin(theta) + dy_global * np.cos(theta)

        return {
            "speed_x": x_speed_matrix,
            "speed_y": y_speed_matrix,
            "presence": presence_matrix,
            "lane_angle": torch.asarray(lane_angle),
            "traffic_signs": traffic_signs,
            "max_speed": np.array([100.0], dtype=np.float32),
            "ego_speed_x": np.array([vx_local], dtype=np.float32),
            "ego_speed_y": np.array([vy_local], dtype=np.float32),
            "ego_in_lane_position_x": np.array([lateral_offset], dtype=np.float32),
            "throttle": np.array([throttle], dtype=np.float32),
            "brake": np.array([brake], dtype=np.float32),
            "steering_angle": np.array([steering], dtype=np.float32),
            "reverse": np.array([reverse], dtype=np.float32),
        }

    def _get_nearby_traffic_signs(self):
        return get_nearby_signs(self.ego_vehicle, self.world.get_map(), radius=10)

    def _encode_traffic_signs(self):
        # 1. Fetch nearby traffic signs
        traffic_signs = self._get_nearby_traffic_signs()
        
        # 2. Initialize the one-hot encoded array
        encoded_signs = np.zeros(SUPPORTED_SIGNS_COUNT)
        
        # 3. Process each sign and update the array
        for sign in traffic_signs:
            # Check if the type is a valid digit to avoid ValueError
            if sign.type.isdigit():
                # Use modulo SUPPORTED_SIGNS_COUNT to prevent index out of bounds
                sign_index = int(sign.type) % SUPPORTED_SIGNS_COUNT
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
            if file_num==1 and folder_index != 0:
                folder_index-=1
                continue
            else:
                break
        file_num += 1
    if file_num==1 and folder_index==0:
        return ""
    return f"{base_path}{folder_index-1}/ppo_carla_checkpoint_{file_num-1}000_steps.zip"

def run(map_path, walkers_count, vehicles_count, steps, device, init_speed, manual_mode=False):
    env = CarlaEnv(map_path, walkers_count, vehicles_count, max_steps=steps, init_speed=init_speed)
    
    if manual_mode:
        # Use the manual controller
        controller = ManualController(env)
        controller.run()
        
    else :
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        model_path = "ppo_carla_model"
        latest_checkpoint=get_latest_checkpoint()
        if latest_checkpoint != "":
            print(f"Loading existing model: {latest_checkpoint}")
            model = PPO.load(latest_checkpoint, verbose=2, env=env, n_epochs=10, batch_size=128, learning_rate=3e-4, clip_range=0.2)
        else:
            print("Creating new model...")
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=2,
                tensorboard_log="./ppo_carla/",
                n_epochs=10,
                batch_size=128,
                learning_rate=3e-4,
                clip_range=0.2,
            )    
        try:
            checkpoints_folder = create_checkpoints_folder()
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints_folder, name_prefix='ppo_carla_checkpoint')

            # model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log="./ppo_carla/")
            model.learn(total_timesteps=steps, callback=checkpoint_callback)
            print ("saving model")
            model.save("ppo_carla_model")
        except ...:
            print(f"Error during model training: ")
            
            
            
            
            

# map_path = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/LoadOpenDrive2/lab-map.xodr"
if __name__ == "__main__":
    map_path = r"C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4\Content\Carla\Maps\OpenDrive\Town01_Opt.xodr"
    run(map_path, 0, 0, 2000000, "cuda", 0)