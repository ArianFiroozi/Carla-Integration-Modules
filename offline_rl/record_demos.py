import argparse
from pathlib import Path
from CarlaEnv.env import CarlaEnv
from offline_rl.controllers.imitation_controller import ImitationController
from config import offline_rl_config


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
        return best
    
    ckpts = list(model_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    ckpts_sorted = sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts_sorted[-1]


def resolve_model_path(args):
    """
    Resolve model path following BC evaluation logic:
    1. Explicit --model-path argument
    2. Config BC_MODEL_PATH
    3. Auto-discover from latest BC experiment
    """
    if args.model_path:
        return Path(args.model_path)
    
    if offline_rl_config.BC_CHECKPOINT_PATH is not None:
        path = Path(offline_rl_config.BC_CHECKPOINT_PATH)
        if path.exists():
            return path
    
    # Auto-discover from latest experiment
    root = Path(args.experiments_root)
    exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
    return get_best_or_last_checkpoint(exp_dir / "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect offline RL dataset using BC policy + exploration")
    parser.add_argument("--map", type=str, default=offline_rl_config.CARLA_MAP_PATH)
    parser.add_argument("--vehicles", type=int, default=offline_rl_config.CARLA_VEHICLES)
    parser.add_argument("--max-steps", type=int, default=offline_rl_config.CARLA_MAX_STEPS)
    parser.add_argument("--model-path", type=str, default=None, 
                        help="Path to BC model .pt file (auto-discovers if not provided)")
    parser.add_argument("--exp-id", type=str, default=None, 
                        help="Specific BC experiment ID (uses latest if not provided)")
    parser.add_argument("--experiments-root", type=str, 
                        default=str(offline_rl_config.BC_CHECKPOINTS_ROOT))
    parser.add_argument("--episodes", type=int, default=offline_rl_config.COLLECT_EPISODES, 
                        help="Number of episodes to record")
    parser.add_argument("--out-dir", type=str, default=str(offline_rl_config.SAVE_DIR))
    parser.add_argument("--epsilon", type=float, default=offline_rl_config.EPSILON, 
                        help="Probability of taking a random action (0 = pure BC, 1 = random)")
    args = parser.parse_args()

    # Resolve model path
    model_path = resolve_model_path(args)
    print(f"[INFO] Using model: {model_path}")

    print("[INFO] Initializing CARLA Environment for Offline Data Collection...")
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=offline_rl_config.CARLA_WALKERS,
        vehicles_count=args.vehicles,
        max_steps=args.max_steps,
        init_speed=offline_rl_config.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=offline_rl_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=offline_rl_config.RANDOM_VEHICLE_START_POS
    )

    print(f"[INFO] Initializing ImitationController (epsilon={args.epsilon}, episodes={args.episodes})")
    controller = ImitationController(
        env=env,
        model_path=str(model_path),
        record_dir=args.out_dir,
        base_name="offline_noisy_data",
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        device=offline_rl_config.DEVICE
    )

    try:
        controller.run(episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user. Saved successfully up to now.")
    finally:
        env.close()