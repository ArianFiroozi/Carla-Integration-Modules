from multiprocessing.util import info

from gymnasium import spaces
import gymnasium
import numpy as np
import torch
from CarlaEnv.ObservationAdaptors import *
from CarlaEnv.VehicleControl import *
from CarlaEnv.LoadOpenDrive2 import *
from CarlaEnv.ObjectSpawn import *
import os
import carla
from pathlib import Path
import time  
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = REPO_ROOT / "extra_stats"

RUN_DIR = RL_ROOT / "runs"
RUN_DIR.mkdir(exist_ok=True)

PID_PATH = RUN_DIR / "training.pid"
HEARTBEAT_PATH = RUN_DIR / "heartbeat.txt"

CHECKPOINTS_DIR = RL_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

MAX_ITER_IN_EPISODE=5000
SUPPORTED_SIGNS_COUNT = 5
LEAST_HEIGHT = -5

with open(PID_PATH, "w") as f:
    f.write(str(os.getpid()))

class CarlaEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, map_path, walkers_count, vehicles_count, max_steps=40000, init_speed=0.5, action_mode="discrete",
                 random_ego_spawn=True, random_vehicle_spawn=True, smooth_steering=False, no_rendering=True):
        super(CarlaEnv, self).__init__()

        self.walkers_count = walkers_count
        self.vehicles_count = vehicles_count
        self.init_speed = init_speed
        self.action_mode = action_mode  # "discrete" or "continuous"
        self.smooth_steering = smooth_steering
        # The observation is built entirely from actor states + map queries (only collision &
        # lane-invasion sensors, no cameras), so we can run the server WITHOUT rendering.
        # This massively reduces GPU load and is the main fix for RDP/virtual-GPU freezes.
        self.no_rendering = no_rendering
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)  # was 10.0; tolerate occasional slow ticks before erroring

        self.random_ego_spawn = random_ego_spawn
        self.random_vehicle_spawn = random_vehicle_spawn
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

        # Do not spawn the ego vehicle here. Initialize as None.
        self.ego_vehicle = None
        self.vehicle_controller = None
        self.vehicles = None
        self.walkers = None
        
        self.max_steps = max_steps
        self.current_step = 0
        self._tick_fail_count = 0   # consecutive world.tick() failures (wedged-server guard)
        self.map = self.world.get_map()
        
        # Steering smoothing state
        self.prev_steer = 0.0
        
        # Set action space based on mode
        if self.action_mode == "discrete":
            self.action_space = spaces.MultiDiscrete([5,4])
        elif self.action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0]), 
                high=np.array([1.0, 1.0, 1.0]), 
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        self.observation_space = spaces.Dict({
            "speed_x": spaces.Box(low=-torch.inf, high=torch.inf, shape=(25, 11), dtype=np.float32),
            "speed_y": spaces.Box(low=-torch.inf, high=torch.inf, shape=(25, 11), dtype=np.float32),
            "presence": spaces.Box(low=0, high=9, shape=(25, 11), dtype=np.int64),
            "lane_angle": spaces.Box(low=-torch.pi, high=torch.pi, shape=(1,), dtype=np.float32),
            "max_speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            "traffic_signs": spaces.Box(low=0, high=1, shape=(SUPPORTED_SIGNS_COUNT,), dtype=np.float32),  
            "ego_speed_x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "ego_speed_y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "ego_in_lane_position_x": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),  
            "throttle": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "brake": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "steering_angle": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "reverse": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        self.last_heartbeat_time = time.time()
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(str(self.last_heartbeat_time))
                
    def _cleanup_world_actors(self):
        """
        Destroy ALL vehicles, sensors and walkers currently on the server — including 'zombie'
        actors left behind by a previously force-killed run. A fresh process has no handle to
        those zombies, so destroying only self-tracked actors is not enough; this clears them
        and frees occupied spawn points / orphaned listening sensors.
        """
        try:
            actors = self.world.get_actors()
            victims = (list(actors.filter("sensor.*")) +
                       list(actors.filter("walker.*")) +
                       list(actors.filter("vehicle.*")))
            for a in victims:
                try:
                    if a.type_id.startswith("sensor.") and a.is_listening:
                        a.stop()
                except Exception:
                    pass
            if victims:
                self.client.apply_batch([carla.command.DestroyActor(a.id) for a in victims])
        except Exception as e:
            print(f"[WARN] world actor cleanup failed: {e}")

    def reset(self, seed=None):
        self.current_step = 0
        self.prev_steer = 0.0

        # RANDOMIZE SEED (so NPC spawns differ each episode)
        if seed is not None:
            random.seed(seed)

        # Stop our own sensor callbacks first (so they don't fire mid-teardown), then do a
        # GLOBAL cleanup that also removes zombie actors orphaned by a previous force-killed run.
        if self.vehicle_controller is not None:
            for s in ("sensor_c", "sensor_l"):
                sensor = getattr(self.vehicle_controller, s, None)
                if sensor is not None:
                    try:
                        if sensor.is_listening:
                            sensor.stop()
                    except Exception:
                        pass

        self._cleanup_world_actors()
        self.vehicles = []
        self.walkers = []
        self.ego_vehicle = None
        # Tick so CARLA actually removes them
        self.world.tick()

        self.vehicles = spawn_vehicles(self.client, self.vehicles_count, random_spawn=self.random_vehicle_spawn)
        self.walkers = spawn_pedestrians(self.world, self.walkers_count)
        self.world.tick()

        self.ego_vehicle = spawn_ego_vehicle(self.world, self.init_speed, random_spawn=self.random_ego_spawn)
        self.vehicle_controller = VehicleController(self.world, self.ego_vehicle)
        print("[reset] controller+sensors created", flush=True)   # DEBUG: remove once hang is localized

        # Tick again so ego sensors + physics start stable
        self.world.tick()
        print("[reset] post-spawn tick done", flush=True)         # DEBUG

        obs = self._get_observation()
        print("[reset] observation built; reset complete", flush=True)  # DEBUG
        return obs, {}

    def _apply_sync(self, fixed_dt=0.05):
        # always grab the current world (after map load)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt
        settings.no_rendering_mode = self.no_rendering   # kill GPU rendering (no cameras are used)
        settings.substepping = True
        settings.max_substep_delta_time = 0.01   # 100 Hz physics
        settings.max_substeps = int(fixed_dt / settings.max_substep_delta_time) + 1
        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        # Hybrid physics: NPCs far from the ego get cheap kinematic updates instead of full
        # physics -> large CPU/GPU savings and far fewer sync-mode deadlocks with traffic.
        try:
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(70.0)
        except Exception as e:
            print(f"[WARN] Could not enable TM hybrid physics: {e}")
    

    def _process_action(self, action):
        """
        Apply deterministic post-processing to the raw agent action.
        This ensures the agent works with raw actions, and only the
        environment applies constraints like throttle floor and
        brake/throttle exclusivity.
        
        Args:
            action: np.array [throttle, brake, steer] in range [0,1], [0,1], [-1,1]
        
        Returns:
            np.array [throttle, brake, steer] post-processed
        """
        throttle = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        steer = float(np.clip(action[2], -1.0, 1.0))

        # NET LONGITUDINAL CONTROL (replaces brake-over-throttle exclusivity).
        # The old rule (brake>0.1 -> throttle=0) turned a throttle+brake hedge into a FREE,
        # crash-proof full brake, which the agent kept rediscovering as a safe haven. Now the
        # pedals combine into a net command, so a hedge resolves to its (weak) net effect and is
        # no longer a free stop. Genuine braking still works; committing to throttle still drives.
        net = throttle - brake
        if net >= 0.0:
            throttle, brake = net, 0.0
            # Throttle floor: very small throttle doesn't overcome CARLA's dead-zone.
            if 0.05 < throttle < 0.13:
                throttle = 0.13
        else:
            throttle, brake = 0.0, -net

        return np.array([throttle, brake, steer], dtype=np.float32)
        
    def step(self, action=None , new_action_mode =None): 
        prev_obs = self._get_observation()        
        # 1. Define end-of-episode variables
        terminated = False  
        truncated = False   

        # Only execute manual control if 'action' is provided!
        # This prevents overwriting the Traffic Manager when recording Autopilot.
        
        action_mode = None

        if new_action_mode is not None:
            action_mode = new_action_mode
        else:
            action_mode = self.action_mode
            
        if action is not None:
            if action_mode == "discrete":
                speed_action = int(action[0])
                turn_action = int(action[1])
                self.vehicle_controller.exec_command(self.vehicle_controller.speed_action_convertor(speed_action))
                self.vehicle_controller.exec_command(self.vehicle_controller.turn_action_convertor(turn_action))
            elif self.action_mode == "continuous":
                # Capture the RAW agent pedals BEFORE exclusivity zeroes the throttle, so the
                # reward can punish a throttle+brake hedge the env would otherwise hide for free.
                self.vehicle_controller.raw_throttle = float(np.clip(action[0], 0.0, 1.0))
                self.vehicle_controller.raw_brake = float(np.clip(action[1], 0.0, 1.0))

                # Apply post-processing to raw action
                action = self._process_action(action)

                throttle = float(action[0])
                brake = float(action[1])
                steer = float(action[2])
                self.vehicle_controller.exec_continuous_command(throttle, brake, steer)
        
        if self.current_step == 0:
            print("[step] first world.tick() of episode...", flush=True)   # DEBUG: remove once localized
        try:
            self.world.tick()
            self._tick_fail_count = 0
        except Exception as e:
            self._tick_fail_count += 1
            print(f"[WARN] world.tick() failed ({self._tick_fail_count}/3): {e}")
            # Repeated failures mean the server is wedged (typically after a force-kill left it
            # in synchronous mode). No client-side reset can recover that, so stop cleanly: the
            # trainer saves an emergency checkpoint and you restart CARLA + --resume-dir.
            if self._tick_fail_count >= 3:
                raise RuntimeError(
                    "world.tick() failed 3x in a row — the CARLA server is likely wedged. "
                    "Restart CarlaUE4.exe, then resume with --resume-dir."
                )
            # One transient failure: try a clean reset (now also clears zombie actors).
            self.reset()
            reward, info = self.vehicle_controller.get_reward(prev_obs)
            return prev_obs, reward, False, False, info

        step_peds(self.world, self.walkers)

        # 2. Check for failure (Terminated)
        if self.vehicle_controller.collision_happened:
            terminated = True
        
        if (self.ego_vehicle.get_location().z <= LEAST_HEIGHT and not terminated):
            terminated = True
            
        # 3. Calculate reward
        reward,info = self.vehicle_controller.get_reward(prev_obs)
        if info.get('is_terminal_crash', 0) == 1:
            terminated = True
            self.vehicle_controller.collision_happened = False

        # Record the ACTUAL executed action (post throttle-floor/exclusivity) so the
        # replay buffer can store what was executed instead of the raw agent output.
        # In continuous mode `action` was reassigned by _process_action above.
        if action is not None and self.action_mode == "continuous":
            info['executed_action'] = np.asarray(action, dtype=np.float32)

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
            
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        x_speed_matrix, y_speed_matrix, presence_matrix, vx_local, vy_local = get_speed_matrices(self.ego_vehicle)
        lane_angle = get_lane_angle(self.ego_vehicle, self.world.get_map())
        traffic_signs = self._encode_traffic_signs()
        
        control = self.ego_vehicle.get_control()
        throttle = control.throttle
        brake = control.brake
        steering = control.steer
        reverse = 1.0 if control.reverse else 0.0
        
        map = self.map
        waypoint = map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True)
        lane_center = waypoint.transform.location
        transform = self.ego_vehicle.get_transform()
        ego_yaw = transform.rotation.yaw
        ego_location = transform.location

        theta = np.radians(ego_yaw)
        dx_global = ego_location.x - lane_center.x
        dy_global = ego_location.y - lane_center.y
        lateral_offset = -dx_global * np.sin(theta) + dy_global * np.cos(theta)

        return {
            "speed_x": x_speed_matrix,
            "speed_y": y_speed_matrix,
            "presence": presence_matrix,
            "lane_angle": np.array([lane_angle], dtype=np.float32),
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
        traffic_signs = self._get_nearby_traffic_signs()
        encoded_signs = np.zeros(SUPPORTED_SIGNS_COUNT)
        for sign in traffic_signs:
            if sign.type.isdigit():
                sign_index = int(sign.type) % SUPPORTED_SIGNS_COUNT
                encoded_signs[sign_index] = 1
        return encoded_signs

    def _process_traffic_signs(self, traffic_signs):
        return 0

    def render(self, mode="human"):
        pass

    def close(self):
        # Revert the server to ASYNCHRONOUS mode on exit. A sync-mode server with no client is
        # a prime cause of the next run deadlocking, so always undo it on a clean shutdown.
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass
        try:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(False)
        except Exception:
            pass