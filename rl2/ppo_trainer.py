import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dummy_env import DummyCarlaEnv
from actor_critic import UTCarActorCritic

# ==========================================
# ハイパーپارامترهای PPO و ترکیب BC
# ==========================================
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 64
LR = 3e-4
BC_COEF = 2.0  # ضریب اهمیت دیتای متخصص (لاندا). هرچی بیشتر باشه ماشین محتاط‌تر میشه

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gae(rewards, values, dones):
    """محاسبه Advantage با روش Generalized Advantage Estimation"""
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

def get_dummy_expert_batch(batch_size=32):
    """
    تولید دیتای فیکِ متخصص برای تست آفلاین.
    مانی بعداً اینجا رو به دیتالودر واقعی وصل می‌کنه.
    """
    grid = torch.rand(batch_size, 5, 25, 11).to(DEVICE)
    scalars = torch.rand(batch_size, 4).to(DEVICE)
    target_actions = torch.rand(batch_size, 3).to(DEVICE) 
    return grid, scalars, target_actions

def train_ppo_with_bc():
    print("1. Initializing Environment and Model...")
    env = DummyCarlaEnv(max_steps=100)
    model = UTCarActorCritic(latent_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
  
    print("\n2. Starting Collection Phase (Rollouts)...")
    obs_grid, obs_scalars, actions, log_probs, rewards, values, dones = [], [], [], [], [], [], []
    
    obs = env.reset()
    
    for step in range(100):
        grid_t = torch.tensor(obs["grid"]).unsqueeze(0).to(DEVICE)
        scalar_t = torch.tensor(obs["scalars"]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(grid_t, scalar_t)
            
        next_obs, reward, done, _ = env.step(action.cpu().numpy()[0])
        
        obs_grid.append(grid_t)
        obs_scalars.append(scalar_t)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.tensor([reward], dtype=torch.float32).to(DEVICE))
        values.append(value.flatten())
        dones.append(torch.tensor([float(done)], dtype=torch.float32).to(DEVICE))
        
        obs = next_obs if not done else env.reset()

    obs_grid = torch.cat(obs_grid)
    obs_scalars = torch.cat(obs_scalars)
    actions = torch.cat(actions)
    old_log_probs = torch.cat(log_probs)
    rewards = torch.cat(rewards)
    values = torch.cat(values)
    dones = torch.cat(dones)

    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    print("\n3. Starting Update Phase (PPO + BC Loss)...")
    for epoch in range(PPO_EPOCHS):
        
        _, new_log_probs, entropy, new_values = model.get_action_and_value(obs_grid, obs_scalars, actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()
        
        loss_v = F.mse_loss(new_values.flatten(), returns)
        loss_entropy = -0.01 * entropy.mean()
        
        ppo_loss = loss_pi + 0.5 * loss_v + loss_entropy

       
        expert_grid, expert_scalars, expert_targets = get_dummy_expert_batch()
        
        expert_latent = model.extractor(expert_grid, expert_scalars)
        expert_pred_mean = model.actor_mean(expert_latent)
        
        bc_loss = F.smooth_l1_loss(expert_pred_mean, expert_targets, beta=0.1)

     
        total_loss = ppo_loss + (BC_COEF * bc_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/{PPO_EPOCHS} | Total Loss: {total_loss.item():.4f} (PPO: {ppo_loss.item():.4f}, BC: {bc_loss.item():.4f})")

    print("\n✅ PPO Trainer successfully executed! The offline framework is ready for CARLA.")

if __name__ == "__main__":
    train_ppo_with_bc()