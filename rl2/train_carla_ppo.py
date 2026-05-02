import sys
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CarlaEnv.env import CarlaEnv
from rl2.actor_critic import UTCarActorCritic
from imitation.datasets.bc_dataset import BCDatasetContinuous
from imitation.evaluate_imitation import ObsHistory, extract_grid_and_scalars
from config import bc_config

# ==========================================
# setting PPO
# ==========================================
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 64
LR = 1e-4             # تو فاز RL لرنینگ ریت رو یکم کمتر می‌دیم تا وزن‌های IL خراب نشن
BC_COEF = 2.0         # ضریب دیتای متخصص
MAX_STEPS = 1000      # تعداد قدم‌ها تو هر دور جمع‌آوری دیتا
TOTAL_ITERATIONS = 500 # تعداد کل حلقه‌های آموزش
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gae(rewards, values, dones):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
        delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
    
    returns = advantages + values
    return advantages, returns

def get_il_config_and_stats(model_dir):
    """خوندن فایل کانفیگ مدل IL برای نرمال‌سازی دیتا"""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        print(f"[WARN] No config.json found at {config_path}")
        return {}
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("dataset_meta", {}).get("normalization_stats", {})

def train_carla_ppo():
    print("1. Loading IL Configuration and Dataset...")
    IL_MODEL_PATH = str(ROOT / "experiments" / "YOUR_EXPERIMENT_FOLDER" / "models" / "best_model.pt")
    IL_CONFIG_DIR = str(ROOT / "experiments" / "YOUR_EXPERIMENT_FOLDER")
    
    norm_stats = get_il_config_and_stats(IL_CONFIG_DIR)
    
    expert_dataset = BCDatasetContinuous(str(bc_config.CONTINUOUS_DATASET_PATH), one_hot_presence=bc_config.USE_ONE_HOT_GRID)
    expert_loader = DataLoader(expert_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    expert_iter = iter(expert_loader)
    
    print("\n2. Initializing CARLA Environment and Actor-Critic...")
    env = CarlaEnv(
        map_path=bc_config.CARLA_MAP_PATH,
        walkers_count=bc_config.CARLA_WALKERS,
        vehicles_count=bc_config.CARLA_VEHICLES,
        max_steps=bc_config.CARLA_MAX_STEPS,
        init_speed=bc_config.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=bc_config.RANDOM_VEHICLE_START_POS
    )
    
    model = UTCarActorCritic(latent_dim=128).to(DEVICE)
    if os.path.exists(IL_MODEL_PATH):
        model.load_il_weights(IL_MODEL_PATH)
    else:
        print(f"[WARN] IL model not found at {IL_MODEL_PATH}. Starting from scratch!")
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # پوشه ذخیره مدل‌های RL
    save_dir = ROOT / "rl2" / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    print("\n3. Starting PPO + BC Training Loop...")
    for iteration in range(TOTAL_ITERATIONS):
        obs_grid, obs_scalars, actions, log_probs, rewards, values, dones = [], [], [], [], [], [], []
        
        obs, _ = env.reset()
        history = ObsHistory(window_size=bc_config.WINDOW_SIZE, norm_stats=norm_stats)
        
        # رول‌اوت تو محیط واقعی CARLA
        for step in range(MAX_STEPS):
            grid_np, scalars_np = extract_grid_and_scalars(obs, history, norm_stats)
            
            grid_t = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            scalar_t = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(grid_t, scalar_t)
                
            action_np = action.cpu().numpy()[0]
            next_obs, reward, done, truncated, _ = env.step(action_np)
            
            obs_grid.append(grid_t)
            obs_scalars.append(scalar_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor([reward], dtype=torch.float32).to(DEVICE))
            values.append(value.flatten())
            dones.append(torch.tensor([float(done or truncated)], dtype=torch.float32).to(DEVICE))
            
            if done or truncated:
                obs, _ = env.reset()
                history.reset()
            else:
                obs = next_obs

        obs_grid = torch.cat(obs_grid)
        obs_scalars = torch.cat(obs_scalars)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(log_probs)
        rewards = torch.cat(rewards)
        values = torch.cat(values)
        dones = torch.cat(dones)

        advantages, returns = compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_ppo_loss, total_bc_loss = 0, 0
        for epoch in range(PPO_EPOCHS):
            _, new_log_probs, entropy, new_values = model.get_action_and_value(obs_grid, obs_scalars, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages
            loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()
            loss_v = F.mse_loss(new_values.flatten(), returns)
            loss_entropy = -0.01 * entropy.mean()
            ppo_loss = loss_pi + 0.5 * loss_v + loss_entropy

            try:
                expert_batch = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                expert_batch = next(expert_iter)
                
            expert_grid = expert_batch[0].to(DEVICE)
            expert_scalars = expert_batch[1].to(DEVICE)
            expert_targets = expert_batch[2].to(DEVICE)

            expert_latent = model.extractor(expert_grid, expert_scalars)
            expert_pred_mean = model.actor_mean(expert_latent)
            bc_loss = F.smooth_l1_loss(expert_pred_mean, expert_targets, beta=0.1)

            loss = ppo_loss + (BC_COEF * bc_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_ppo_loss += ppo_loss.item()
            total_bc_loss += bc_loss.item()
            
        avg_reward = rewards.sum().item() / (dones.sum().item() + 1e-5)
        print(f"Iteration {iteration+1}/{TOTAL_ITERATIONS} | Avg Reward: {avg_reward:.2f} | PPO Loss: {total_ppo_loss/PPO_EPOCHS:.4f} | BC Loss: {total_bc_loss/PPO_EPOCHS:.4f}")

        if (iteration + 1) % 10 == 0:
            ckpt_path = save_dir / f"ppo_bc_model_iter_{iteration+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Saved] Model checkpoint -> {ckpt_path}")

if __name__ == "__main__":
    train_carla_ppo()