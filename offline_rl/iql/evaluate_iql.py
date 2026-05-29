# # File: offline_rl/iql/evaluate_iql.py

# import os
# import time
# from collections import Counter
# from pathlib import Path
# import argparse
# import json
# import numpy as np
# import torch
# import cv2
# from torch.utils.tensorboard import SummaryWriter
# import carla
# import datetime

# from CarlaEnv.env import CarlaEnv
# from config import offline_rl_config as cfg
# from utils.obs_wrapper import CarlaObsWrapper
# from offline_rl.iql.iql_agent import IQLAgent

# # -------------------------------------------------------------
# # Path Resolution Helpers (Mirrored from BC/SAC)
# # -------------------------------------------------------------

# def get_latest_experiment(root):
#     folders = [f for f in Path(root).iterdir() if f.is_dir()]
#     if not folders:
#         raise FileNotFoundError(f"No experiment folders found in {root}")
#     return sorted(folders)[-1]

# def get_best_checkpoint(model_dir):
#     best = model_dir / "best_model.pt"
#     if best.exists():
#         return best
#     ckpts = list(model_dir.glob("checkpoint_step_*.pt"))
#     if not ckpts:
#         raise FileNotFoundError(f"No checkpoints in {model_dir}")
#     return sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))[-1]

# def resolve_eval_paths(args):
#     root = Path("experiments/offline_rl")
    
#     if args.model_path:
#         model_path = Path(args.model_path)
#         exp_dir = model_path.parents[1]
#     else:
#         exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
#         model_path = get_best_checkpoint(exp_dir / "models")
    
#     config_path = exp_dir / "config.json"
#     eval_dir = exp_dir / "eval"
#     eval_dir.mkdir(exist_ok=True)
    
#     return model_path, config_path, eval_dir

# # -------------------------------------------------------------
# # Recording & Visualization
# # -------------------------------------------------------------

# def create_video_recorder(env, save_path, width=640, height=360, fps=20):
#     world = env.world
#     ego_vehicle = env.ego_vehicle
#     bp_lib = world.get_blueprint_library()
#     cam_bp = bp_lib.find("sensor.camera.rgb")
#     cam_bp.set_attribute("image_size_x", str(width))
#     cam_bp.set_attribute("image_size_y", str(height))
#     cam_bp.set_attribute("fov", "90")

#     cam_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
#     camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

#     video = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#     def callback(image):
#         array = np.frombuffer(image.raw_data, dtype=np.uint8)
#         array = array.reshape((image.height, image.width, 4))
#         frame = array[:, :, :3][:, :, ::-1] # RGBA to BGR
#         video.write(frame)

#     camera.listen(callback)
#     return camera, video

# def update_spectator(env):
#     world = env.world
#     ego_vehicle = env.ego_vehicle
#     if ego_vehicle is None: return
#     spectator = world.get_spectator()
#     tr = ego_vehicle.get_transform()
#     forward = tr.get_forward_vector()
#     cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
#     cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
#     spectator.set_transform(carla.Transform(cam_loc, cam_rot))

# # -------------------------------------------------------------
# # Core Logic
# # -------------------------------------------------------------

# def run_episode(env, agent, wrapper, max_steps, video_path=None):
#     obs, _ = env.reset()
#     wrapper.reset()
    
#     camera, video = (None, None)
#     if video_path:
#         camera, video = create_video_recorder(env, video_path)

#     rewards = []
#     for t in range(max_steps):
#         grid, scalars = wrapper.preprocess(obs)
#         grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

#         with torch.no_grad():
#             # IQL evaluation typically uses the deterministic mean from the actor
#             _, _, action_t = agent.actor.sample(grid_t, scalars_t)
#             action = action_t.cpu().numpy()[0]

#         obs, reward, terminated, truncated, _ = env.step(action)
#         rewards.append(float(reward))
#         update_spectator(env)

#         if terminated or truncated:
#             break

#     if camera:
#         camera.stop()
#         camera.destroy()
#         video.release()

#     return {
#         "return": float(np.sum(rewards)),
#         "length": len(rewards),
#         "reason": "terminated" if terminated else ("truncated" if truncated else "max_steps")
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, default=None)
#     parser.add_argument("--exp_id", type=str, default=None)
#     parser.add_argument("--episodes", type=int, default=5)
#     parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
#     parser.add_argument("--record", action="store_true", default=True)
#     args = parser.parse_args()

#     # 1. Resolve Paths & Config
#     model_path, config_path, eval_dir = resolve_eval_paths(args)
#     with open(config_path, "r") as f:
#         run_cfg = json.load(f)
    
#     norm_stats = run_cfg.get("dataset_meta", {}).get("normalization_stats", {})
    
#     # 2. Setup Logging
#     run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     current_eval_dir = eval_dir / run_stamp
#     current_eval_dir.mkdir(parents=True)
#     video_dir = current_eval_dir / "videos"
#     video_dir.mkdir(exist_ok=True)
    
#     tb_writer = SummaryWriter(str(current_eval_dir / "tb"))

#     # 3. Initialize Agent & Env
#     agent = IQLAgent(device=cfg.DEVICE)
#     ckpt = torch.load(model_path, map_location=cfg.DEVICE)
#     agent.actor.load_state_dict(ckpt["actor"] if "actor" in ckpt else ckpt)
#     agent.actor.eval()

#     wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=cfg.DEVICE, action_mode="continuous")
#     env = CarlaEnv(map_path=args.map, vehicles_count=cfg.CARLA_VEHICLES,walkers_count=cfg.CARLA_WALKERS, action_mode="continuous")

#     all_returns = []
#     print(f"\n[EVAL] Starting IQL Eval: {model_path.name}")

#     try:
#         for ep in range(args.episodes):
#             v_path = video_dir / f"ep_{ep+1:03d}.mp4" if args.record else None
#             res = run_episode(env, agent, wrapper, cfg.CARLA_MAX_STEPS, video_path=v_path)
            
#             all_returns.append(res["return"])
#             print(f"Ep {ep+1}: Return={res['return']:.2f}, Len={res['length']}, Reason={res['reason']}")
            
#             tb_writer.add_scalar("eval/episode_return", res["return"], ep + 1)
#             tb_writer.add_scalar("eval/episode_length", res["length"], ep + 1)

#         # Final Summary
#         summary = {
#             "avg_return": float(np.mean(all_returns)),
#             "std_return": float(np.std(all_returns)),
#             "model_used": str(model_path)
#         }
#         with open(current_eval_dir / "summary.json", "w") as f:
#             json.dump(summary, f, indent=2)
            
#         print(f"\nFinal Avg Return: {summary['avg_return']:.2f}")

#     finally:
#         env.close()
#         tb_writer.close()

# if __name__ == "__main__":
#     main()


# File: offline_rl/iql/evaluate_iql.py

import os
import time
from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import carla
import datetime

from CarlaEnv.env import CarlaEnv
from config import offline_rl_config as cfg
from utils.obs_wrapper import CarlaObsWrapper
from offline_rl.iql.iql_agent import IQLAgent

def load_norm_stats_from_bc_checkpoint():
    """Try to find normalization stats from the BC experiment folder."""
    try:
        bc_ckpt = Path(cfg.BC_CHECKPOINT_PATH)
        exp_dir = bc_ckpt.parents[1]
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            stats = data.get("dataset_meta", {}).get("normalization_stats", {})
            return stats
    except Exception:
        pass
    return {}

def create_third_person_camera(env, save_path, width=640, height=360, fps=20):
    """Create a third-person chase camera attached to the ego vehicle."""
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("enable_postprocess_effects", "True")

    cam_transform = carla.Transform(
        carla.Location(x=-6, z=3),
        carla.Rotation(pitch=-15)
    )

    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
    video = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame = array[:, :, :3]   
        frame = frame[:, :, ::-1] 
        video.write(frame)

    camera.listen(callback)
    return camera, video

def update_spectator(env):
    """Update the spectator camera to follow the vehicle."""
    world = env.world
    ego_vehicle = env.ego_vehicle
    if ego_vehicle is None:
        return

    spectator = world.get_spectator()
    tr = ego_vehicle.get_transform()
    forward = tr.get_forward_vector()

    cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
    cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))

def select_action(agent, grid, scalars, evaluate=True):
    """Helper function to get actions from IQL's SACActor"""
    agent.actor.eval()
    with torch.no_grad():
        if evaluate:
            # Get deterministic mean action for evaluation
            _, _, action = agent.actor.sample(grid, scalars)
        else:
            # Get stochastic action
            action, _, _ = agent.actor.sample(grid, scalars)
    agent.actor.train()
    return action.cpu().numpy()[0]

def run_eval_episode(env, agent, wrapper, max_steps, record_video=False, video_path=None, update_spectator_flag=True):
    """Run a single evaluation episode."""
    obs, _ = env.reset()
    wrapper.reset()

    camera = None
    video = None
    if record_video and video_path is not None:
        camera, video = create_third_person_camera(env, video_path)

    rewards = []
    action_history = []
    terminated_flag = False
    truncated_flag = False

    for t in range(max_steps):
        grid, scalars = wrapper.preprocess(obs)
        grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

        # Get deterministic action from IQL Actor
        action = select_action(agent, grid_t, scalars_t, evaluate=True)
        
        action_history.append({
            "step": t,
            "raw_action": action.tolist() if hasattr(action, 'tolist') else list(action),
            "env_action": action if isinstance(action, list) else action.tolist()
        })

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))

        if update_spectator_flag:
            update_spectator(env)

        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break

    if camera is not None:
        camera.stop()
        camera.destroy()
    if video is not None:
        video.release()

    return {
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "length": len(rewards),
        "end_reason": "terminated" if terminated_flag else ("truncated" if truncated_flag else "max_steps"),
        "rewards": rewards,
        "actions": action_history,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate IQL agent in CARLA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained best_model.pt")
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--record", action="store_true", help="Record video of evaluation episodes")
    parser.add_argument("--no-spectator", action="store_true", help="Don't update spectator camera")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Initialize Agent and Load Model
    agent = IQLAgent(device=device)
    
    # Robust loading logic (handles both dict and direct state_dict)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    if "actor" in ckpt:
        agent.actor.load_state_dict(ckpt["actor"])
        print("[INFO] Loaded Actor weights from full checkpoint dict.")
    else:
        agent.actor.load_state_dict(ckpt)
        print("[INFO] Loaded Actor weights directly from state_dict.")
        
    agent.actor.eval()

    # 2. Prepare Environment and Wrapper
    norm_stats = load_norm_stats_from_bc_checkpoint()
    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=device, action_mode="continuous")

    model_dir = Path(args.model_path).parent.parent
    eval_dir = model_dir / "eval"
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_eval_dir = eval_dir / f"iql_eval_{run_stamp}"
    current_eval_dir.mkdir(parents=True, exist_ok=True)
    
    if args.record:
        (current_eval_dir / "videos").mkdir(exist_ok=True)
    
    print(f"[INFO] Saving eval logs and videos to: {current_eval_dir}")

    env = CarlaEnv(
        map_path=args.map,
        walkers_count=cfg.CARLA_WALKERS,
        vehicles_count=cfg.CARLA_VEHICLES,
        max_steps=args.max_steps,
        init_speed=cfg.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=cfg.RANDOM_EGO_START_POS,
        random_vehicle_spawn=cfg.RANDOM_VEHICLE_START_POS
    )

    all_returns = []
    all_lengths = []
    end_reasons = Counter()
    overall_t0 = time.time()

    print(f"\n{'='*50}\nStarting IQL Evaluation: {args.episodes} episodes\n{'='*50}\n")

    try:
        for ep in range(args.episodes):
            print(f"=== Episode {ep+1}/{args.episodes} ===")
            
            video_path = current_eval_dir / "videos" / f"episode_{ep+1:03d}.mp4" if args.record else None
            
            result = run_eval_episode(
                env=env, agent=agent, wrapper=wrapper, max_steps=args.max_steps,
                record_video=args.record, video_path=video_path,
                update_spectator_flag=not args.no_spectator,
            )

            episode_path = current_eval_dir / f"episode_{ep+1:03d}.json"
            with open(episode_path, "w") as f:
                json.dump({
                    "return": result["return"],
                    "mean_reward": result["mean_reward"],
                    "length": result["length"],
                    "end_reason": result["end_reason"],
                }, f, indent=2)

            all_returns.append(result["return"])
            all_lengths.append(result["length"])
            end_reasons[result["end_reason"]] += 1

            print(f"Return: {result['return']:.2f} | Length: {result['length']} | End: {result['end_reason']}")
            
        total_time = time.time() - overall_t0

        # Summary
        print(f"\n{'='*50}\nEVALUATION SUMMARY\n{'='*50}")
        print(f"Avg return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
        print(f"Avg length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
        print(f"End reasons: {dict(end_reasons)}")
        
        summary_path = current_eval_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "model_path": str(args.model_path),
                "episodes": args.episodes,
                "avg_return": float(np.mean(all_returns)),
                "avg_length": float(np.mean(all_lengths)),
                "end_reasons": dict(end_reasons),
                "wall_time_sec": total_time,
            }, f, indent=2)

    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()