import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import offline_rl_config as cfg
from agents.sac.sac_agent import SACActor, SACCritic
from networks.feature_extractor import FeatureExtractor

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def squashed_gaussian_nll(mean, log_std, target, action_scale, action_bias, eps=1e-6):
    y = (target - action_bias) / action_scale
    y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    z = atanh(y)
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + eps) - torch.log(action_scale + eps)
    return -log_prob.sum(dim=1)

def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, (1 - tau))
    return weight * (diff ** 2)
 
class IQLValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            cnn_channels=getattr(cfg, "CNN_CHANNELS", [16, 32, 64]),
            kernel_sizes=getattr(cfg, "KERNEL_SIZES", [3, 3, 3]),
            n_mlp_layers=getattr(cfg, "SCALAR_N_MLP_LAYERS", 2),
            mlp_hidden_size=getattr(cfg, "SCALAR_MLP_HIDDEN_SIZE", 32)
        )
        self.v_head = nn.Sequential(
            nn.Linear(cfg.LATENT_DIM, getattr(cfg, "HEAD_MLP_HIDDEN_SIZE", 64)),
            nn.ReLU(),
            nn.Linear(getattr(cfg, "HEAD_MLP_HIDDEN_SIZE", 64), 1)
        )
    def forward(self, grid, scalars):
        latent = self.extractor(grid, scalars)
        return self.v_head(latent)

class IQLAgent:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.actor = SACActor(
            grid_channels=cfg.GRID_CHANNELS, scalar_dim=cfg.SCALAR_DIM, latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM, action_low=cfg.ACTION_LOW, action_high=cfg.ACTION_HIGH
        ).to(self.device)

        self.critic = SACCritic(
            grid_channels=cfg.GRID_CHANNELS, scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM, action_dim=cfg.ACTION_DIM
        ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.value_net = IQLValueNet().to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=cfg.CRITIC_LR)

        self.train_step = 0

    def update(self, replay_buffer):
        self.train_step += 1
        grid, scalars, actions_exp, rewards, next_grid, next_scalars, dones = replay_buffer.sample(cfg.BATCH_SIZE)

        with torch.no_grad():
            q1_target, q2_target = self.critic_target(grid, scalars, actions_exp)
            q_target = torch.min(q1_target, q2_target)
        
        v = self.value_net(grid, scalars)

        # 1. Value Network Update (Expectile)
        value_loss = expectile_loss(q_target - v, cfg.IQL_TAU).mean()
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=10.0)
        self.value_opt.step()

        # 2. Critic Update (MSE to r + gamma * V_next)
        with torch.no_grad():
            next_v = self.value_net(next_grid, next_scalars)
            q_backup = rewards + (1.0 - dones) * cfg.GAMMA * next_v

        q1, q2 = self.critic(grid, scalars, actions_exp)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        # 3. Actor Update (Advantage Weighted Regression)
        with torch.no_grad():
            adv = q_target - v.detach()
            weights = torch.exp(adv * cfg.IQL_BETA).clamp(max=100.0)

        mean, log_std = self.actor(grid, scalars)
        nll = squashed_gaussian_nll(mean, log_std, actions_exp, self.actor.action_scale, self.actor.action_bias)
        
        actor_loss = (weights.squeeze() * nll).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        # 4. Target Network Update
        if self.train_step % cfg.TARGET_UPDATE_INTERVAL == 0:
            for s, t in zip(self.critic.parameters(), self.critic_target.parameters()):
                t.data.copy_(cfg.TAU * s.data + (1.0 - cfg.TAU) * t.data)

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_weight": weights.mean().item(),
            "mean_q": q_target.mean().item(),
            "mean_v": v.mean().item(),
            "mean_adv": adv.mean().item()
        }

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_actor_from_bc(self, bc_checkpoint_path, strict=False):
        """Loads pretrained BC weights to accelerate IQL training."""
        ckpt = torch.load(bc_checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
            
        translated_state = {}
        for k, v in state.items():
            if k.startswith("extractor."):
                translated_state[k.replace("extractor.", "feature_extractor.", 1)] = v
            elif k.startswith("actor.head."):
                translated_state[k.replace("actor.head.", "head.mean_head.", 1)] = v
            elif k.startswith("actor."):
                translated_state[k.replace("actor.", "head.", 1)] = v
            else:
                translated_state[k] = v
                
        missing, unexpected = self.actor.load_state_dict(translated_state, strict=strict)
        print(f"\n[INFO] BC Weights Loaded into IQL Actor Successfully from {bc_checkpoint_path}!")