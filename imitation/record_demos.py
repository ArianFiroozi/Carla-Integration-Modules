import time
from pathlib import Path
import argparse

import keyboard
import numpy as np
import carla
from .autopilot_controller import AutopilotController
from . import config

class ManualController:
    """
    Manual driving controller with optional demo recording.
    Keys:
      - Speed:   Up = speed_up, Down = speed_down, Space = stop, R = reverse, else = constant
      - Turning: Left = left, Right = right, else = go_straight (auto-center)
      - Quit:    Q
      - Reset:   U (after episode ends)
    """
    def __init__(self, env,
                 demo_dir=config.MANUAL_DEMO_DIR,
                 sleep_seconds=config.MANUAL_SLEEP_SECONDS,
                 print_every=config.MANUAL_PRINT_EVERY,
                 debug_grids=config.MANUAL_DEBUG_GRIDS):

        self.env = env
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self.print_every = print_every
        self.debug_grids = debug_grids

    def _npify_obs(self, obs: dict):
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

    def _update_spectator(self):
        world = self.env.world
        ego_vehicle = self.env.ego_vehicle

        spectator = world.get_spectator()
        tr = ego_vehicle.get_transform()
        forward = tr.get_forward_vector()

        cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
        cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def _get_action_from_keyboard(self):
        # speed mapping
        if keyboard.is_pressed('up'):
            speed_action = 0  # Accelerate
        elif keyboard.is_pressed('down'):
            speed_action = 1  # Brake
        elif keyboard.is_pressed('space'):
            speed_action = 2  # stop
        elif keyboard.is_pressed('r'):
            speed_action = 3  # reverse
        else :
            speed_action = 4 # constant
            
        if keyboard.is_pressed('left'):
            turn_action = 1   # Steer left
        elif keyboard.is_pressed('right'):
            turn_action = 0   # Steer right
        elif keyboard.is_pressed('f'):
            turn_action = 3  # Go straight
        elif keyboard.is_pressed('t'):
            turn_action = 2 # Do not turn
        else :
            turn_action = 3

        return np.array([speed_action, turn_action], dtype=np.int64)

    def _save_episode(self, ep_idx: int, steps, base_name: str):
        if len(steps) == 0:
            return

        obs_keys = steps[0]["obs"].keys()
        data = {}

        for k in obs_keys:
            data[f"obs_{k}"] = np.stack([s["obs"][k] for s in steps], axis=0)

        data["actions"] = np.stack([s["action"] for s in steps], axis=0).astype(np.int64)
        data["rewards"] = np.array([s["reward"] for s in steps], dtype=np.float32)
        data["terminated"] = np.array([s["terminated"] for s in steps], dtype=np.bool_)
        data["truncated"] = np.array([s["truncated"] for s in steps], dtype=np.bool_)
        data["dones"] = np.array([s["done"] for s in steps], dtype=np.bool_)
        data["t"] = np.array([s["t"] for s in steps], dtype=np.int32)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.demo_dir / f"{base_name}_{ts}_ep{ep_idx:04d}.npz"
        np.savez_compressed(out_path, **data)
        print(f"[saved] {out_path}  (T={len(steps)})")

    def _debug_obs(self, obs, prefix=""):
        print(prefix, "keys:", list(obs.keys()))
        for k, v in obs.items():
            arr = np.asarray(v)
            try:
                print(f"{k:24s} shape={arr.shape} dtype={arr.dtype} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
            except Exception:
                print(f"{k:24s} shape={arr.shape} dtype={arr.dtype}")

    def _print_grid(self, p):
        p = np.asarray(p)
        for r in range(p.shape[0]):
            row = " ".join(f"{p[r, c]:.0f}" for c in range(p.shape[1]))
            print(row)

    def run(self, record=config.MANUAL_RECORD, base_name=config.MANUAL_BASE_NAME):
        print("Manual control mode activated.")
        print("Keys: Up/Down/Space/R for speed | Left/Right for steer | Q quit | U reset after episode")

        if record:
            print(f"Recording ON -> saving to: {self.demo_dir.resolve()} (base_name={base_name})")
        else:
            print("Recording OFF")

        obs, _ = self.env.reset()
        self._update_spectator()

        ep_idx = 0
        t = 0
        steps = []
        done = False

        while True:
            if keyboard.is_pressed("q"):
                print("Quit pressed.")
                if record and len(steps) > 0:
                    self._save_episode(ep_idx, steps, base_name)
                break

            action = self._get_action_from_keyboard()

            if not done:
                next_obs, reward, terminated, truncated, info = self.env.step(action.tolist())
                done = bool(terminated or truncated)
                self._update_spectator()

                if record:
                    steps.append({
                        "obs": self._npify_obs(obs),
                        "action": action,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done": bool(done),
                        "t": int(t),
                    })

                obs = next_obs
                t += 1

                if self.print_every and (t % self.print_every == 0):
                    self._debug_obs(obs, prefix=f"[t={t}]")
                    if self.debug_grids and "presence" in obs:
                        self._print_grid(obs["presence"])

            if done:
                if record:
                    self._save_episode(ep_idx, steps, base_name)
                    ep_idx += 1

                print("Episode ended. Press 'u' to reset or 'q' to quit.")
                while True:
                    if keyboard.is_pressed("q"):
                        print("Exiting.")
                        return
                    if keyboard.is_pressed("u"):
                        obs, _ = self.env.reset()
                        self._update_spectator()
                        done = False
                        t = 0
                        steps = []
                        print("Reset.")
                        break
                    time.sleep(0.1)
            time.sleep(self.sleep_seconds)














if __name__ == "__main__":
    from CarlaEnv.env import CarlaEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=config.CARLA_MAP_PATH)
    parser.add_argument("--walkers", type=int, default=config.CARLA_WALKERS)
    parser.add_argument("--vehicles", type=int, default=config.CARLA_VEHICLES)
    parser.add_argument("--max-steps", type=int, default=config.CARLA_MAX_STEPS)
    parser.add_argument("--init-speed", type=float, default=config.CARLA_INIT_SPEED)
    parser.add_argument("--mode", type=str, default=config.RECORD_DRIVE_MODE)
    parser.add_argument("--action-mode", type=str, default=config.ACTION_MODE)
    parser.add_argument("--demo-dir", type=str, default=config.MANUAL_DEMO_DIR)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--base-name", type=str, default=config.MANUAL_BASE_NAME)
    parser.add_argument("--episodes", type=int, default=config.DEFAULT_AUTOPILOT_EPISODES)
    args = parser.parse_args()
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=args.walkers,
        vehicles_count=args.vehicles,
        max_steps=args.max_steps,
        init_speed=args.init_speed,
        action_mode=args.action_mode,
        random_ego_spawn=config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=config.RANDOM_VEHICLE_START_POS
    )

    if args.mode == "autopilot":
        print("[INFO] Initializing Autopilot Controller...")
        controller = AutopilotController(
            env,
            record_dir=str(config.AUTOPILOT_DEMO_DIR), 
            base_name="autopilot", 
            max_steps=args.max_steps
        )
        controller.run(episodes=args.episodes) 
        exit(0)

    elif args.mode == "manual":
        print("[INFO] Initializing Manual Controller...")
        controller = ManualController(
            env,
            demo_dir=args.demo_dir
        )
        record_flag = args.record if args.record else config.MANUAL_RECORD
        controller.run(
            record=record_flag,
            base_name=args.base_name
        )
    else:
        print(f"[ERROR] Unknown mode '{args.mode}'. Use 'autopilot' or 'manual'.")
        exit(1)
    