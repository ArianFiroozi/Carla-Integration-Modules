import copy
import torch
import torch.optim as optim
import torch.nn.functional as F

from config import offline_rl_config as cfg 
from agents.sac.sac_agent import SACActor, SACCritic

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def squashed_gaussian_nll(mean, log_std, target, action_scale, action_bias, eps=1e-5):
    y = (target - action_bias) / action_scale
    y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    z = 0.5 * (torch.log1p(y) - torch.log1p(-y))
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + eps) - torch.log(action_scale + eps)
    return -torch.clamp(log_prob.sum(dim=1), min=-100, max=100)

class AWACAgent:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.actor = SACActor(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM,
            action_low=cfg.ACTION_LOW,
            action_high=cfg.ACTION_HIGH,
        ).to(self.device)

        self.critic = SACCritic(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = optim.Adam(
            self.actor.parameters(), 
            lr=cfg.ACTOR_LR, 
            weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-5)
        )
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        self.train_step = 0

    def update(self, replay_buffer):
        self.train_step += 1
        grid, scalars, actions_exp, rewards, next_grid, next_scalars, dones = replay_buffer.sample(cfg.BATCH_SIZE)

        # ==========================================
        # 1. Update Critic (Always)
        # ==========================================
        with torch.no_grad():
            next_action, _, _ = self.actor.sample(next_grid, next_scalars)
            q1_t, q2_t = self.critic_target(next_grid, next_scalars, next_action)
            q_t = torch.min(q1_t, q2_t)
            target_q = rewards + (1.0 - dones) * cfg.GAMMA * q_t

        q1, q2 = self.critic(grid, scalars, actions_exp)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
        self.critic_opt.step()

        # ==========================================
        # Critic Warm-up 
        # ==========================================
        warmup_steps = getattr(cfg, "CRITIC_WARMUP_STEPS", 20000)
        if self.train_step < warmup_steps:
            return {
                "critic_loss": critic_loss.item(), 
                "actor_loss": 0.0, "mean_weight": 0.0, "mean_adv": 0.0
            }

        # ==========================================
        # 2. Update Actor (AWAC - Improved Stability)
        # ==========================================
        with torch.no_grad():
            num_samples = 4
            v_pis = []
            for _ in range(num_samples):
                pi_a, _, _ = self.actor.sample(grid, scalars)
                v_pis.append(torch.min(*self.critic(grid, scalars, pi_a)))
            v_pi = torch.stack(v_pis).mean(dim=0)
            
            q_exp = torch.min(*self.critic(grid, scalars, actions_exp))
            adv = q_exp - v_pi
            
            awac_lambda = getattr(cfg, "AWAC_LAMBDA", 2.0)
            scaled_adv = adv / awac_lambda
            weights = torch.exp(scaled_adv - scaled_adv.max()).clamp(max=20.0)

        mean, log_std = self.actor(grid, scalars)
        nll = squashed_gaussian_nll(mean, log_std, actions_exp, self.actor.action_scale, self.actor.action_bias)
        actor_loss = (weights.squeeze() * nll).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ==========================================
        # 3. Update Target Networks (Soft Update)
        # ==========================================
        if self.train_step % cfg.TARGET_UPDATE_INTERVAL == 0:
            for s, t in zip(self.critic.parameters(), self.critic_target.parameters()):
                t.data.copy_(cfg.TAU * s.data + (1.0 - cfg.TAU) * t.data)

        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "mean_weight": weights.mean().item(), "mean_adv": adv.mean().item(),
            "mean_q": q1.mean().item()
        }

    def save(self, path):
        torch.save(self.actor.state_dict(), path)
    
    def load_actor_from_bc(self, bc_checkpoint_path, strict=False):
        """
        Loads the pretrained BC weights into the AWAC Actor to prevent Critic divergence
        and accelerate offline training.
        """
        ckpt = torch.load(bc_checkpoint_path, map_location=self.device, weights_only=False)
        
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
            
        translated_state = {}
        for k, v in state.items():
            if k.startswith("extractor."):
                new_key = k.replace("extractor.", "feature_extractor.", 1)
                translated_state[new_key] = v
                
            elif k.startswith("actor.head."):
                new_key = k.replace("actor.head.", "head.mean_head.", 1)
                translated_state[new_key] = v
                
            elif k.startswith("actor."):
                new_key = k.replace("actor.", "head.", 1)
                translated_state[new_key] = v
            else:
                translated_state[k] = v
                
        missing, unexpected = self.actor.load_state_dict(translated_state, strict=strict)
        
        print(f"\n[INFO] BC Weights Loaded into AWAC Actor Successfully from {bc_checkpoint_path}!")
        if missing:
            print(f"[WARN] Missing keys during load: {missing}")