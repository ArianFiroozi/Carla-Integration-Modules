import time
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats

from CarlaEnv.env import CarlaEnv
from agents.sac.sac_agent import SACAgent
from config import sac_config as cfg
from config import bc_config
from config import general_config
from utils.reward_compiler import compile_reward
from utils.obs_wrapper import CarlaObsWrapper
from utils.seed_utils import seed_everything

# ==============================================================================
# 1. HELPER FUNCTIONS (From eval_sac.py)
# ==============================================================================

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
    folders = [f for f in Path(root).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No experiment folders found.")
    return sorted(folders)[-1]

def get_best_or_last_checkpoint(model_dir):
    best = model_dir / "best_model.pt"
    if best.exists():
        return best, "best_model"
    ckpts = list(model_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    ckpts_sorted = sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts_sorted[-1], ckpts_sorted[-1].stem

def resolve_paths(args):
    if args.model_path:
        model_path = Path(args.model_path)
        exp_dir = model_path.parents[1]
        config_path = exp_dir / "config.json"
        return model_path, config_path

    root = Path(args.experiments_root)
    exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
    model_path, _ = get_best_or_last_checkpoint(exp_dir / "models")
    config_path = exp_dir / "config.json"
    return model_path, config_path


# ==============================================================================
# 2. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="In-Memory Reward Compiler Verifier")
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--total-steps", type=int, default=10000, help="Total steps to harvest for verification")
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--exp_id", type=str, default=None, help="Specific experiment ID to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--experiments_root", type=str, default=str(cfg.SAVE_DIR))
    
    args = parser.parse_args()
    seed_everything(bc_config.GLOBAL_SEED)

    print("\n" + "="*80)
    print("🚀 STARTING PURE IN-MEMORY REWARD VERIFIER")
    print("="*80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Using device: {device}")

    # Load paths & norm stats
    model_path, config_path = resolve_paths(args)
    print(f"[LOG] Resolved Model Path: {model_path}")
    print(f"[LOG] Resolved Config Path: {config_path}")

    norm_stats = load_norm_stats_from_bc_checkpoint()
    if not norm_stats:
        print("[WARN] No normalization stats found. Using empty stats.")
    else:
        print("[LOG] Normalization stats loaded successfully.")

    # Initialize Agent
    print("[LOG] Initializing SAC Agent...")
    agent = SACAgent(device=device)
    agent.load(model_path)
    agent.actor.eval()
    print("[LOG] Model loaded and set to EVAL mode.")

    # Initialize Wrapper & Env
    print("[LOG] Spinning up CARLA environment (Headless)...")
    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=device, action_mode="continuous")
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=cfg.CARLA_WALKERS,
        vehicles_count=cfg.CARLA_VEHICLES,
        max_steps=args.max_steps,
        init_speed=cfg.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=cfg.RANDOM_EGO_START_POS,
        random_vehicle_spawn=cfg.RANDOM_VEHICLE_START_POS,
        no_rendering=True
    )

    # In-memory arrays
    true_rewards = []
    approx_rewards = []
    true_components = {}
    approx_components = {}

    print(f"\n[LOG] Harvesting {args.total_steps} steps into RAM...")
    
    obs, _ = env.reset(seed=bc_config.GLOBAL_SEED)
    wrapper.reset()
    
    steps_collected = 0
    episodes_run = 1
    t0 = time.time()

    try:
        while steps_collected < args.total_steps:
            grid, scalars = wrapper.preprocess(obs)
            grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

            # Get deterministic action
            action = agent.select_action(grid_t, scalars_t, evaluate=True)[0]
            action_list = action.tolist() if hasattr(action, 'tolist') else list(action)

            next_obs, raw_reward, terminated, truncated, info = env.step(action_list)
            
            # Pack state into a unified dict simulating the dataset
            unified_dict = info.copy()
            unified_dict.update({
                'obs_ego_speed_x': float(obs.get("ego_speed_x", [0.0])[0]),
                'obs_ego_speed_y': float(obs.get("ego_speed_y", [0.0])[0]),
                'obs_lane_angle': float(obs.get("lane_angle", [0.0])[0]),
                'obs_ego_in_lane_position_x': float(obs.get("ego_in_lane_position_x", [0.0])[0]),
                'obs_throttle': float(action_list[0]),
                'obs_brake': float(action_list[1]),
                'obs_reverse': 1.0 if float(action_list[0]) < 0 else 0.0,
                'terminated': float(terminated),
                'obs_presence': obs.get("presence", np.zeros((25, 11)))
            })

            # COMPILE ON THE FLY
            r_true, m_true = compile_reward(unified_dict, general_config, mode="info", is_tensor=False)
            r_approx, m_approx = compile_reward(unified_dict, general_config, mode="obs", is_tensor=False)

            true_rewards.append(r_true)
            approx_rewards.append(r_approx)

            # Initialize dicts on first step
            if steps_collected == 0:
                true_components = {k: [] for k in m_true.keys()}
                approx_components = {k: [] for k in m_true.keys()}

            for k in true_components.keys():
                true_components[k].append(m_true.get(k, 0.0))
                approx_components[k].append(m_approx.get(k, 0.0))

            steps_collected += 1

            if steps_collected % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  -> Collected {steps_collected}/{args.total_steps} steps... ({elapsed:.1f}s elapsed)")

            if terminated or truncated:
                print(f"[LOG] Episode {episodes_run} ended (Terminated: {terminated}). Resetting...")
                obs, _ = env.reset()
                wrapper.reset()
                episodes_run += 1
            else:
                obs = next_obs

    finally:
        print("\n[LOG] Shutting down CARLA...")
        try:
            env.close()
        except Exception:
            pass

    print("[LOG] Crunching Statistical Arrays...")
    true_rewards = np.array(true_rewards)
    approx_rewards = np.array(approx_rewards)

    # ==============================================================================
    # 3. STATISTICAL REPORT
    # ==============================================================================
    print("\n" + "="*90)
    print(f"{'REWARD COMPONENT':<25} | {'MAE':<10} | {'MEAN BIAS':<10} | {'PEARSON (r)':<25}")
    print("="*90)

    total_mae = np.mean(np.abs(true_rewards - approx_rewards))
    total_bias = np.mean(approx_rewards - true_rewards)
    total_r, _ = scipy_stats.pearsonr(true_rewards, approx_rewards)
    print(f"{'OVERALL REWARD':<25} | {total_mae:<10.4f} | {total_bias:<10.4f} | {total_r:<15.4f}")
    print("-" * 90)

    plot_data = {"overall": (true_rewards, approx_rewards, total_mae, total_r)}
    
    for k in true_components.keys():
        t_arr = np.nan_to_num(np.array(true_components[k]), nan=0.0)
        a_arr = np.nan_to_num(np.array(approx_components[k]), nan=0.0)
        
        mae = np.mean(np.abs(t_arr - a_arr))
        bias = np.mean(a_arr - t_arr)
        
        if np.var(t_arr) > 1e-6 and np.var(a_arr) > 1e-6:
            r, _ = scipy_stats.pearsonr(t_arr, a_arr)
            r_str = f"{r:.4f}"
            if r < 0.85: r_str += "  ⚠️ LOW CORRELATION"
            elif r > 0.98: r_str += "  ✅ EXCELLENT"
        else:
            r = np.nan
            r_str = "N/A (No Variance)"
            
        print(f"{k:<25} | {mae:<10.4f} | {bias:<10.4f} | {r_str}")
        plot_data[k] = (t_arr, a_arr, mae, r)

    print("="*90)

    # ==============================================================================
    # 4. RENDER PLOT
    # ==============================================================================
    print("\n[LOG] Rendering Plot to screen... (Close the UI window to exit script)")
    
    num_plots = len(plot_data)
    cols = 4
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for idx, (name, (t_arr, a_arr, mae, r)) in enumerate(plot_data.items()):
        ax = axes[idx]
        ax.scatter(t_arr, a_arr, alpha=0.3, color='steelblue', edgecolor='none')
        
        min_val = min(t_arr.min(), a_arr.min()) - 0.1
        max_val = max(t_arr.max(), a_arr.max()) + 0.1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="y=x")
        
        r_title = f"{r:.3f}" if not np.isnan(r) else "N/A"
        ax.set_title(f"{name.upper()}\nMAE: {mae:.4f} | r: {r_title}", fontsize=11, fontweight='bold')
        ax.set_xlabel("True (3D info)")
        ax.set_ylabel("Approx (2D obs)")
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend()

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    print("[LOG] Done. Exiting gracefully.")

if __name__ == "__main__":
    main()