import sys
from pathlib import Path

CARLA_ROOT = Path(r"C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla")
sys.path.insert(0, str(CARLA_ROOT))

import time
import numpy as np
import carla


DUMMY_SPEED, DUMMY_TURN = 3, 2


class AutopilotController:
    def __init__(self, env, record_dir="expert_demos", base_name="autopilot", max_steps=2000, sleep=0.0):
        self.env = env
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)

        self.base_name = base_name
        self.max_steps = max_steps
        self.sleep = sleep

        self.tm = env.client.get_trafficmanager()
        self.tm_port = self.tm.get_port()

        self.ep_idx = 0

    def _npify_obs(self, obs: dict):
        """Convert observations to stable numpy format (same logic as ManualController)."""
        out = {}

        for k, v in obs.items():

            if isinstance(v, np.ndarray):
                arr = v
            else:
                try:
                    import torch
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                    else:
                        arr = np.asarray(v)
                except Exception:
                    arr = np.asarray(v)

            if arr.dtype.kind in ("f", "c"):
                arr = arr.astype(np.float32, copy=False)
            elif arr.dtype.kind in ("i", "u"):
                arr = arr.astype(np.int64, copy=False)

            out[k] = arr

        return out

    def run(self, episodes=50, record=True):

        for ep in range(episodes):

            print(f"\n=== Episode {ep} ===")

            obs, _ = self.env.reset()
            
            ego = self.env.ego_vehicle

            # Enable TM autopilot
            ego.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(ego, True)
            self.tm.distance_to_leading_vehicle(ego, 3.0)
            self.tm.ignore_lights_percentage(ego, 100)
            



            # Make ego speed slightly random
            self.tm.vehicle_percentage_speed_difference(
                ego,
                np.random.randint(-10, 10)
            )
            
            # p = np.random.randint(5, 20)

            # self.tm.random_left_lanechange_percentage(ego, p)
            # self.tm.random_right_lanechange_percentage(ego, p)


            steps = []
            done = False
            t = 0

            while not done and t < self.max_steps:

                # Step the environment with Action=None
                # This ensures the environment updates physics, but DOES NOT overwrite the Autopilot!
                
                if t % 200 == 0:
                    new_offset = np.random.uniform(-1.2, 1.2)
                    self.tm.vehicle_lane_offset(ego, new_offset)
                
                
                
                next_obs, reward, terminated, truncated, info = self.env.step(None)

                done = bool(terminated or truncated)

                # Get the actual control that Traffic Manager applied during this step
                control = ego.get_control()

                if record:

                    obs_copy = self._npify_obs(obs)

                    # store control signals inside observations (same structure as manual)
                    obs_copy["throttle"] = np.array([control.throttle], dtype=np.float32)
                    obs_copy["brake"] = np.array([control.brake], dtype=np.float32)
                    obs_copy["steering_angle"] = np.array([control.steer], dtype=np.float32)
                    obs_copy["reverse"] = np.array([float(control.reverse)], dtype=np.float32)

                    steps.append({
                        "obs": obs_copy,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done": bool(done),
                        "t": int(t),
                    })

                obs = next_obs
                t += 1

                if self.sleep > 0:
                    time.sleep(self.sleep)

            if record and len(steps) > 0:
                self._save_episode(steps)

            self.ep_idx += 1

    def _save_episode(self, steps):

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = self.record_dir / f"{self.base_name}_{ts}_ep{self.ep_idx:05d}.npz"

        T = len(steps)
        obs_keys = steps[0]["obs"].keys()

        arrays = {}

        # Stack observations
        for key in obs_keys:
            stacked = np.stack([s["obs"][key] for s in steps], axis=0)
            arrays[f"obs_{key}"] = stacked.astype(np.float32)

        # Keep dummy actions to match manual dataset format
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

        arrays["t"] = np.array(
            [s["t"] for s in steps],
            dtype=np.int32
        )

        np.savez_compressed(out_file, **arrays)

        print(f"[Saved] {out_file} (T={T})")
