# autopilot_controller.py

import sys
from pathlib import Path

CARLA_ROOT = Path(r"C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla")

sys.path.insert(0, str(CARLA_ROOT))

import time
import random
import numpy as np
import carla
from agents.navigation.behavior_agent import BehaviorAgent

DUMMY_SPEED, DUMMY_TURN = 4,3


class AutopilotController:
    def __init__(self, env,
                 record_dir="expert_demos",
                 base_name="autopilot",
                 max_steps=2000,
                 noise_schedule=None,
                 sleep=0.0):

        self.env = env
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)

        self.base_name = base_name
        self.max_steps = max_steps
        self.sleep = sleep


        # Noise scheduler = list of (episode_threshold, std_value)
        self.noise_schedule = noise_schedule or [
            (0,   0.00),
            (50,  0.01),
            (150, 0.02),
            (300, 0.03),
            (600, 0.04)
        ]

        self.ep_idx = 0


    def get_noise_std(self, episode):
        std = 0.0
        for ep_threshold, value in self.noise_schedule:
            if episode >= ep_threshold:
                std = value
        return std


    def apply_noise(self, control, episode):
        std = self.get_noise_std(episode)

        if std == 0:
            return control  # no noise early on

        noisy = carla.VehicleControl()
        noisy.throttle = np.clip(control.throttle + np.random.normal(0, std * 0.5), 0, 1)
        noisy.brake    = np.clip(control.brake    + np.random.normal(0, std * 0.2), 0, 1)
        noisy.steer    = np.clip(control.steer    + np.random.normal(0, std),      -1, 1)
        noisy.reverse  = control.reverse

        return noisy


    def set_random_destination(self, min_distance=30.0, max_tries=20):
        world = self.env.world
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        start_transform = self.env.ego_vehicle.get_transform()
        start_loc = start_transform.location

        for _ in range(max_tries):
            dest_transform = random.choice(spawn_points)
            dest_loc = dest_transform.location

            if start_loc.distance(dest_loc) < min_distance:
                continue  # too close, try another

            # Use BehaviorAgent's own method; it handles route creation
            self.agent.set_destination(start_loc, dest_loc)
            return True

        return False  # could not find a good destination




    def run(self, episodes=50, record=True):

        for ep in range(episodes):
            print(f"\n=== Episode {ep} ===")

            obs, info = self.env.reset()

            for _ in range(10): # Let physics settle
                self.env.world.tick()
                
                
            self.agent = BehaviorAgent(self.env.ego_vehicle, behavior="normal")

            success = self.set_random_destination()
            if not success:
                print("[ERROR] Could not set a valid destination, skipping episode")
                continue


            

            data_steps = []
            done = False
            t = 0

            max_failures_per_episode = 20
            failure_count = 0

            while not done and t < self.max_steps:

                try:
                    expert_control = self.agent.run_step()
                except AttributeError as e:
                    print("[WARN] BehaviorAgent.run_step() failed, resetting destination:", e)
                    failure_count += 1

                    if failure_count > max_failures_per_episode:
                        print("[ERROR] Too many BehaviorAgent failures, ending episode early")
                        break

                    success = self.set_random_destination()
                    if not success:
                        print("[ERROR] Could not find a valid destination; ending episode early.")
                        break

                    # Skip this timestep entirely
                    continue

                # If we get here, we have a valid control
                failure_count = 0  # reset failures

                # DEBUG waypoint visualization as you already have
                debug = self.env.world.debug
                wp_info = self.agent.get_local_planner().get_incoming_waypoint_and_direction(steps=3)
                if wp_info and wp_info[0]:
                    target_wp = wp_info[0]
                    debug.draw_point(
                        target_wp.transform.location + carla.Location(z=1.0),
                        size=0.1,
                        color=carla.Color(255,0,0),
                        life_time=0.1
                    )

                print(expert_control)
                noisy = self.apply_noise(expert_control, episode=ep)

                action = np.array([
                    float(noisy.throttle),
                    float(noisy.brake),
                    float(noisy.steer)
                ], dtype=np.float32)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if record:
                    obs_copy = dict(obs)
                    obs_copy["throttle"] = np.array([expert_control.throttle], dtype=np.float32)
                    obs_copy["brake"] = np.array([expert_control.brake], dtype=np.float32)
                    obs_copy["steering_angle"] = np.array([expert_control.steer], dtype=np.float32)
                    obs_copy["reverse"] = np.array([float(expert_control.reverse)], dtype=np.float32)

                    data_steps.append({
                        "obs": obs_copy,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": done,
                        "t": t
                    })

                obs = next_obs
                t += 1
                time.sleep(self.sleep)


            if record and len(data_steps) > 0:
                self._save_episode(data_steps)

            self.ep_idx += 1


    def _save_episode(self, steps):

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = self.record_dir / f"{self.base_name}_{ts}_ep{self.ep_idx:05d}.npz"

        T = len(steps)
        obs_keys = steps[0]["obs"].keys()

        arrays = {}

        for key in obs_keys:

            stacked = np.stack([s["obs"][key] for s in steps], axis=0)

            arrays[f"obs_{key}"] = stacked.astype(np.float32)


        dummy_action = np.array([DUMMY_SPEED, DUMMY_TURN], dtype=np.int64)

        arrays["actions"] = np.tile(dummy_action, (T, 1))

        arrays["rewards"] = np.array(
            [s["reward"] for s in steps],
            dtype=np.float32
        )

        arrays["terminated"] = np.array(
            [s["terminated"] for s in steps],
            dtype=bool
        )

        arrays["truncated"] = np.array(
            [s["truncated"] for s in steps],
            dtype=bool
        )

        arrays["dones"] = np.array(
            [s["done"] for s in steps],
            dtype=bool
        )

        arrays["t"] = np.arange(T, dtype=np.int32)

        np.savez_compressed(out_file, **arrays)

        print(f"[Saved] {out_file} (T={T})")
