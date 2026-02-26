import os
import time
from pathlib import Path
import numpy as np
import keyboard
import carla


# from manual_controller import ManualController

# ---- config ----
DEMO_DIR = Path("demos")
DEMO_DIR.mkdir(exist_ok=True)

SAVE_EVERY_STEPS = 2000  # chunking if you want long runs; per-episode saving is usually enough
SLEEP_SECONDS = 0.001     
PRINT_EVERY = 500     

def _npify_obs(obs: dict):
    """Ensure every obs value is a numpy array (float32)."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            arr = v
        else:
            # torch tensors or scalars -> numpy
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.asarray(v)
            except Exception:
                arr = np.asarray(v)

        # cast floats to float32, ints to int64
        if arr.dtype.kind in ("f", "c"):
            arr = arr.astype(np.float32, copy=False)
        elif arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.int64, copy=False)
        out[k] = arr
    return out

def save_episode(ep_idx: int, steps, base_name: str):
    """steps: list of dicts with keys: obs, action, reward, done, t"""
    if len(steps) == 0:
        return

    # stack obs dict keys
    obs_keys = steps[0]["obs"].keys()
    data = {}

    for k in obs_keys:
        data[f"obs_{k}"] = np.stack([s["obs"][k] for s in steps], axis=0)

    data["actions"] = np.stack([s["action"] for s in steps], axis=0).astype(np.int64)
    data["rewards"] = np.array([s["reward"] for s in steps], dtype=np.float32)
    data["dones"] = np.array([s["done"] for s in steps], dtype=np.bool_)
    data["t"] = np.array([s["t"] for s in steps], dtype=np.int32)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = DEMO_DIR / f"{base_name}_{ts}_ep{ep_idx:04d}.npz"
    np.savez_compressed(out_path, **data)
    print(f"[saved] {out_path}  (T={len(steps)})")



def update_spectator(world, ego_vehicle):
    spectator = world.get_spectator()
    tr = ego_vehicle.get_transform()

    # third-person chase cam behind car
    forward = tr.get_forward_vector()
    cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
    cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))


def record_manual(env, base_name="demo"):
    """
    Manual driving with recording.
    Keys match your manual_controller:
      up/down/space/r for speed, left/right/f/t for turning
      u = reset after episode
      q = quit
    """
    ep_idx = 0

    obs, _ = env.reset()
    done = False
    t = 0
    steps = []

    print("Recording demos...")
    print("Quit: q | Reset episode: u | (driving keys same as manual_controller)")

    while True:
        if keyboard.is_pressed("q"):
            print("Quit pressed. Saving current episode and exiting.")
            save_episode(ep_idx, steps, base_name)
            break

        # ---- action mapping (same as your ManualController) ----
        if keyboard.is_pressed('up'):
            speed_action = 0
        elif keyboard.is_pressed('down'):
            speed_action = 1
        elif keyboard.is_pressed('space'):
            speed_action = 2
        elif keyboard.is_pressed('r'):
            speed_action = 3
        else:
            speed_action = 4

        if keyboard.is_pressed('left'):
            turn_action = 1
        elif keyboard.is_pressed('right'):
            turn_action = 0
        elif keyboard.is_pressed('f'):
            turn_action = 3
        elif keyboard.is_pressed('t'):
            turn_action = 2
        else:
            turn_action = 2

        action = np.array([speed_action, turn_action], dtype=np.int64)

        if not done:
            next_obs, reward, done, truncated, info = env.step(action.tolist())

            update_spectator(env.world, env.ego_vehicle)
            
            steps.append({
                "obs": _npify_obs(obs),
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "t": int(t),
            })
            obs = next_obs
            t += 1

            if t % PRINT_EVERY == 0:
                print(f"t={t} reward={reward:.2f} done={done}")

        if done:
            # save ep
            save_episode(ep_idx, steps, base_name)
            ep_idx += 1

            # wait for reset or quit
            print("Episode ended. Press 'u' to reset, 'q' to quit.")
            while True:
                if keyboard.is_pressed("q"):
                    print("Exiting.")
                    return
                if keyboard.is_pressed("u"):
                    obs, _ = env.reset()
                    done = False
                    t = 0
                    steps = []
                    print("Reset.")
                    break
                time.sleep(0.1)

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    # Import your env runner
    from env import CarlaEnv

    map_path = r"C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4\Content\Carla\Maps\OpenDrive\Town01_Opt.xodr"
    env = CarlaEnv(map_path=map_path, walkers_count=0, vehicles_count=0, max_steps=2000, init_speed=0)

    record_manual(env, base_name="town01_manual")