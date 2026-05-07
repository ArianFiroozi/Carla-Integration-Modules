# File: rl/sac/eval_sac.py

import time
from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from CarlaEnv.env import CarlaEnv
from agents.sac.sac_agent import SACAgent
from config import sac_config as cfg
from config import bc_config
import datetime
from torch.utils.tensorboard import SummaryWriter
import carla
import cv2

from utils.obs_wrapper import CarlaObsWrapper

DEVICE = cfg.DEVICE


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


def get_latest_experiment(root):
    """Find the latest experiment directory."""
    folders = [f for f in Path(root).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No experiment folders found.")
    return sorted(folders)[-1]


def get_best_or_last_checkpoint(model_dir):
    """Get best model or latest checkpoint."""
    best = model_dir / "best_model.pt"
    if best.exists():
        return best, "best_model"
    
    ckpts = list(model_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    ckpts_sorted = sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts_sorted[-1], ckpts_sorted[-1].stem


def resolve_paths(args):
    """Resolve model path and experiment directory."""
    if args.model_path:
        model_path = Path(args.model_path)
        exp_dir = model_path.parents[1]
        config_path = exp_dir / "config.json"
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        return model_path, config_path, eval_dir

    root = Path(args.experiments_root)
    exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
    model_path, _ = get_best_or_last_checkpoint(exp_dir / "models")
    config_path = exp_dir / "config.json"
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    return model_path, config_path, eval_dir


def create_third_person_camera(env, save_path, width=640, height=360, fps=20):
    """
    Create a third-person chase camera attached to the ego vehicle.
    Camera positioned behind and above the vehicle.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    # Camera settings
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("enable_postprocess_effects", "True")

    # Position camera behind and above the vehicle
    # x=-6 (6 meters behind), z=3 (3 meters above ground)
    # pitch=-15 (slightly looking down)
    cam_transform = carla.Transform(
        carla.Location(x=-6, z=3),
        carla.Rotation(pitch=-15)
    )

    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

    # Setup video writer
    video = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame = array[:, :, :3]   # Remove alpha channel
        frame = frame[:, :, ::-1] # BGRA → BGR for OpenCV
        video.write(frame)

    camera.listen(callback)
    return camera, video


def create_top_down_camera(env, save_path, width=640, height=360, fps=20):
    """
    Create a top-down camera for a bird's eye view.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")

    # Position camera directly above the vehicle
    cam_transform = carla.Transform(
        carla.Location(z=20),  # 20 meters above
        carla.Rotation(pitch=-90)  # Looking straight down
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
    """
    Update the spectator (free camera in CARLA window) to follow the vehicle.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle
    if ego_vehicle is None:
        return

    spectator = world.get_spectator()
    tr = ego_vehicle.get_transform()
    forward = tr.get_forward_vector()

    # Position spectator behind and above
    cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
    cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))


def run_eval_episode(env, agent, wrapper, max_steps, record_video=False, video_path=None, update_spectator_flag=True):
    """Run a single evaluation episode."""
    obs, _ = env.reset()
    wrapper.reset()

    # Setup camera and video recording if requested
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

        # Use deterministic actions for evaluation
        action = agent.select_action(grid_t, scalars_t, evaluate=True)[0]
        env_action = wrapper.process_continuous_output(action)
        
        action_history.append({
            "step": t,
            "raw_action": action.tolist() if hasattr(action, 'tolist') else list(action),
            "env_action": env_action if isinstance(env_action, list) else env_action.tolist()
        })

        obs, reward, terminated, truncated, info = env.step(env_action)
        rewards.append(float(reward))

        # Update spectator camera in CARLA window
        if update_spectator_flag:
            update_spectator(env)

        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break

    # Clean up camera
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
    parser = argparse.ArgumentParser(description="Evaluate SAC agent")
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--exp_id", type=str, default=None, help="Specific experiment ID to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--experiments_root", type=str, default=str(cfg.SAVE_DIR))
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    parser.add_argument("--record", action="store_true", default=cfg.RECORD_EVAL_VID,help="Record video of evaluation episodes")
    parser.add_argument("--no-spectator", action="store_true", help="Don't update spectator camera")
    
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + config
    model_path, config_path, eval_dir = resolve_paths(args)
    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")

    # Load normalization stats
    norm_stats = load_norm_stats_from_bc_checkpoint()
    if not norm_stats:
        print("[WARN] No normalization stats found. Using empty stats.")

    # Create agent and load weights
    agent = SACAgent(device=device)
    agent.load(model_path)
    agent.actor.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    # Create wrapper
    wrapper = CarlaObsWrapper(
        norm_stats=norm_stats, 
        device=device, 
        action_mode="continuous"
    )

    # Create evaluation directory
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_eval_dir = eval_dir / run_stamp
    current_eval_dir.mkdir(exist_ok=True)
    if args.record:
        (current_eval_dir / "videos").mkdir(exist_ok=True)
    print(f"Saving eval logs to: {current_eval_dir}")

    # TensorBoard
    tb_dir = current_eval_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(str(tb_dir))

    # Create environment
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
    num_episodes = args.episodes

    print(f"\n{'='*50}")
    print(f"Starting evaluation: {num_episodes} episodes")
    if args.record:
        print("Video recording: ENABLED")
    print(f"{'='*50}\n")

    try:
        for ep in range(num_episodes):
            print(f"=== Episode {ep+1}/{num_episodes} ===")
            
            # Setup video path if recording
            video_path = None
            if args.record:
                video_path = current_eval_dir / "videos" / f"episode_{ep+1:03d}.mp4"
            
            result = run_eval_episode(
                env=env,
                agent=agent,
                wrapper=wrapper,
                max_steps=args.max_steps,
                record_video=args.record,
                video_path=video_path,
                update_spectator_flag=not args.no_spectator,
            )

            # Save episode results
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

            print(
                f"Episode {ep+1}: return={result['return']:.2f}, "
                f"mean_reward={result['mean_reward']:.3f}, "
                f"length={result['length']}, "
                f"end={result['end_reason']}"
            )
            if args.record:
                print(f"  Video saved: {video_path}")
            
            # TensorBoard episode metrics
            tb_writer.add_scalar("eval/episode_return", result["return"], ep + 1)
            tb_writer.add_scalar("eval/episode_length", result["length"], ep + 1)

        total_time = time.time() - overall_t0

        # Summary statistics
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Episodes: {num_episodes}")
        print(f"Avg return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
        print(f"Min return: {np.min(all_returns):.2f}")
        print(f"Max return: {np.max(all_returns):.2f}")
        print(f"Avg length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
        print(f"End reasons: {dict(end_reasons)}")
        print(f"Wall time: {total_time:.1f}s")
        print(f"Model: {model_path}")

        # Save summary
        summary = {
            "model_path": str(model_path),
            "episodes": num_episodes,
            "avg_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "min_return": float(np.min(all_returns)),
            "max_return": float(np.max(all_returns)),
            "avg_length": float(np.mean(all_lengths)),
            "std_length": float(np.std(all_lengths)),
            "end_reasons": dict(end_reasons),
            "wall_time_sec": total_time,
            "carla_vehicles": cfg.CARLA_VEHICLES,
            "carla_walkers": cfg.CARLA_WALKERS,
            "video_recorded": args.record,
        }

        summary_path = current_eval_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # TensorBoard summary metrics
        tb_writer.add_scalar("eval/avg_return", np.mean(all_returns), 0)
        tb_writer.add_scalar("eval/std_return", np.std(all_returns), 0)
        tb_writer.add_scalar("eval/avg_length", np.mean(all_lengths), 0)
        
        tb_writer.flush()
        tb_writer.close()

        print(f"\nSaved summary to: {summary_path}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()