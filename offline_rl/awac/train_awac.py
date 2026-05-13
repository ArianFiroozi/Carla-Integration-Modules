import os
import glob
import json
import time
import datetime
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config import offline_rl_config as cfg
from rl.sac.replay_buffer import SACReplayBuffer
from utils.obs_wrapper import CarlaObsWrapper
from offline_rl.awac.awac_agent import AWACAgent

def load_norm_stats():
    try:
        bc_ckpt = Path(cfg.BC_CHECKPOINT_PATH)
        exp_dir = bc_ckpt.parents[1]
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            return data.get("dataset_meta", {}).get("normalization_stats", {})
    except Exception as e:
        print(f"[WARN] Could not load norm stats: {e}")
    return {}

def populate_buffer(buffer, wrapper, data_dir):
    data_files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"[INFO] Found {len(data_files)} episode files in {data_dir}", flush=True)
    
    total_transitions = 0
    for i, f in enumerate(data_files):
        d = np.load(f)
        obs_keys = [k for k in d.files if k.startswith("obs_")]
        T = len(d["rewards"])
        
        wrapper.reset()
        obs = {k.replace("obs_", ""): d[k][0] for k in obs_keys}
        grid, scalars = wrapper.preprocess(obs)
        
        for t in range(T - 1):
            action = d["actions"][t]
            reward = d["rewards"][t]
            
            terminated = d["terminated"][t]
            truncated = d["truncated"][t]
            done = float(terminated or truncated)
            
            next_obs = {k.replace("obs_", ""): d[k][t+1] for k in obs_keys}
            next_grid, next_scalars = wrapper.preprocess(next_obs)
            
            buffer.add(grid, scalars, action, reward, next_grid, next_scalars, done)
            
            grid, scalars = next_grid, next_scalars
            total_transitions += 1

    print(f"[INFO] Successfully loaded {total_transitions} transitions into Replay Buffer.", flush=True)

# -------------------------------------------------------------
# Logging Helpers
# -------------------------------------------------------------
def make_experiment_dir():
    run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = Path(cfg.REPO_ROOT) / "experiments" / "offline_rl" / f"{run_stamp}_awac"
    
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "tb").mkdir(exist_ok=True)
    (exp_dir / "eval").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    return exp_dir

def save_config(exp_dir):
    config_path = exp_dir / "config.json"
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items() if k.isupper() and not k.startswith('_')}
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=4)
    return cfg_dict

def log_config_to_tensorboard(tb_writer, config_dict):
    config_str = json.dumps(config_dict, indent=2, default=str)
    tb_writer.add_text("Config/Hyperparameters", f"```json\n{config_str}\n```", 0)
    for k, v in config_dict.items():
        if isinstance(v, (int, float, bool)):
            tb_writer.add_scalar(f"Config/{k}", float(v), 0)
        else:
            tb_writer.add_text(f"Config/{k}", str(v), 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    agent = AWACAgent(device=device)
    buffer = SACReplayBuffer(capacity=500_000, device=device)
    
    norm_stats = load_norm_stats()
    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=device, action_mode="continuous")

    # 1. Setup Logging Directories
    exp_dir = make_experiment_dir()
    cfg_dict = save_config(exp_dir)
    print(f"[INFO] Experiment Directory created at: {exp_dir}")

    tb_dir = exp_dir / "tb"
    model_dir = exp_dir / "models"
    writer = SummaryWriter(str(tb_dir))
    log_config_to_tensorboard(writer, cfg_dict)

    # 2. Inject BC Weights
    bc_path = Path(cfg.BC_CHECKPOINT_PATH)
    if bc_path.exists():
        print(f"[INFO] Injecting Imitation Learning knowledge from {bc_path}")
        agent.load_actor_from_bc(bc_path, strict=False)
    else:
        print(f"[WARN] BC checkpoint not found at {bc_path}. Initializing Actor randomly! (NOT RECOMMENDED)")

    # 3. Populate Replay Buffer
    print("[INFO] Populating buffer from offline data...")
    populate_buffer(buffer, wrapper, cfg.SAVE_DIR)

    if len(buffer) == 0:
        print("[ERROR] Buffer is empty! Check your data directory.")
        return
    
    # 4. Start Training
    train_steps = getattr(cfg, "OFFLINE_TRAIN_STEPS", 100_000)
    print(f"[INFO] Starting AWAC Offline Training for {train_steps} steps...")
    
    t0 = time.time()
    for step in range(1, train_steps + 1):
        losses = agent.update(buffer)
        
        # Log to TensorBoard
        if step % 100 == 0:
            writer.add_scalar("train/actor_loss", losses["actor_loss"], step)
            writer.add_scalar("train/critic_loss", losses["critic_loss"], step)
            writer.add_scalar("train/mean_weight", losses.get("mean_weight", 0), step)
            writer.add_scalar("train/mean_adv", losses.get("mean_adv", 0), step)
            
        # Log to Console
        if step % 1000 == 0:
            elapsed = time.time() - t0
            print(f"Step {step:06d}/{train_steps} | "
                  f"Actor: {losses['actor_loss']:.4f} | "
                  f"Critic: {losses['critic_loss']:.4f} | "
                  f"Weight: {losses.get('mean_weight', 0):.4f} | "
                  f"Time: {elapsed:.1f}s")
            
        # Save Model Checkpoint
        if step % 5000 == 0:
            save_path = model_dir / f"awac_model_step_{step}.pt"
            agent.save(save_path)
            print(f"[INFO] Saved checkpoint -> {save_path}")

    # Save Final Model
    final_path = model_dir / "best_model.pt"
    agent.save(final_path)
    print(f"[INFO] AWAC Model successfully saved to {final_path}")
    
    writer.close()
    print("[INFO] AWAC Training Completed Successfully!")

if __name__ == "__main__":
    main()