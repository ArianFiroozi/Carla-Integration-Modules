import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from config import awac_config as cfg
from networks.feature_extractor import FeatureExtractor
from networks.actor_heads import BCGaussianContinuousHead
from networks.critic_heads import TwinQCriticHead


def squashed_gaussian_nll(mean, log_std, target, action_scale, action_bias, eps=1e-5):
    """Calculates the Negative Log Likelihood (NLL) of the target action given the policy."""
    y = (target - action_bias) / action_scale
    y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    z = 0.5 * (torch.log1p(y) - torch.log1p(-y))
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + eps) - torch.log(action_scale + eps)
    return -torch.clamp(log_prob.sum(dim=1), min=-100, max=100)


class AWACActor(nn.Module):
    """Actor = FeatureExtractor + Gaussian Head (mean/log_std)"""
    def __init__(
        self,
        grid_channels=5,
        scalar_dim=8,
        latent_dim=128,
        action_dim=3,
        log_std_min=-5,
        log_std_max=2,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            latent_dim=latent_dim,
            cnn_channels=cfg.CNN_CHANNELS,
            kernel_sizes=cfg.KERNEL_SIZES,
            n_mlp_layers=cfg.SCALAR_N_MLP_LAYERS,
            mlp_hidden_size=cfg.SCALAR_MLP_HIDDEN_SIZE
        )
        self.head = BCGaussianContinuousHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            n_mlp_layers=cfg.HEAD_N_MLP_LAYERS,
            mlp_hidden_size=cfg.HEAD_MLP_HIDDEN_SIZE
        )

        if action_low is None: action_low = [-1.0] * action_dim
        if action_high is None: action_high = [1.0] * action_dim

        action_low = torch.tensor(action_low, dtype=torch.float32)
        action_high = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, grid, scalars):
        latent = self.feature_extractor(grid, scalars)
        mean, log_std = self.head(latent)
        return mean, log_std

    def sample(self, grid, scalars):
        mean, log_std = self.forward(grid, scalars)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()                
        a = torch.tanh(z)                   

        action = a * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        # log_prob mapping for critic targeting
        log_prob = normal.log_prob(z).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        log_prob -= torch.log(self.action_scale + 1e-6).sum(-1, keepdim=True)

        return action, log_prob, mean_action


class AWACCritic(nn.Module):
    """Critic = FeatureExtractor + TwinQHead"""
    def __init__(self, grid_channels=5, scalar_dim=8, latent_dim=128, action_dim=3):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            latent_dim=latent_dim,
            cnn_channels=cfg.CNN_CHANNELS,
            kernel_sizes=cfg.KERNEL_SIZES,
            n_mlp_layers=cfg.SCALAR_N_MLP_LAYERS,
            mlp_hidden_size=cfg.SCALAR_MLP_HIDDEN_SIZE
        )   
        self.head = TwinQCriticHead(latent_dim=latent_dim, action_dim=action_dim)

    def forward(self, grid, scalars, action):
        latent = self.feature_extractor(grid, scalars)
        q1, q2 = self.head(latent, action)
        return q1, q2


class AWACAgent:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Actor
        self.actor = AWACActor(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM,
            log_std_min=cfg.LOG_STD_MIN,
            log_std_max=cfg.LOG_STD_MAX,
            action_low=cfg.ACTION_LOW,
            action_high=cfg.ACTION_HIGH,
        ).to(self.device)

        # Critic + target
        self.critic = AWACCritic(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR, weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-5))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        self.train_step = 0
        
        # Logging State Trackers
        self.last_actor_loss = 0.0
        self.last_actor_grad_norm = 0.0
        self.last_awac_weight = 0.0
        self.last_awac_adv = 0.0

    def select_action(self, grid, scalars, evaluate=False):
        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(grid, scalars)
            else:
                action, _, _ = self.actor.sample(grid, scalars)
        self.actor.train()
        return action.cpu().numpy()

    def update(self, replay_buffer):
        self.train_step += 1
        
        # In AWAC, actions drawn from buffer are treated as the "expert" behavior
        grid, scalars, actions_exp, rewards, next_grid, next_scalars, dones = replay_buffer.sample(cfg.BATCH_SIZE)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        
        grad_norms = {}
        is_warmed_up = (self.train_step >= cfg.CRITIC_WARMUP_STEPS)
        update_critic = (self.train_step % getattr(cfg, "CRITIC_UPDATE_EVERY", 1) == 0)
        update_actor = (self.train_step % getattr(cfg, "ACTOR_UPDATE_EVERY", 1) == 0) and is_warmed_up

        # =========================================================
        # 1. CRITIC UPDATE
        # =========================================================
        critic_loss = torch.tensor(0.0)
        if update_critic:
            with torch.no_grad():
                next_action, _, _ = self.actor.sample(next_grid, next_scalars)
                q1_t, q2_t = self.critic_target(next_grid, next_scalars, next_action)
                q_t = torch.min(q1_t, q2_t)  # Standard AWAC target uses no alpha
                target_q = rewards + (1.0 - dones) * cfg.GAMMA * q_t

            q1, q2 = self.critic(grid, scalars, actions_exp)
            
            if getattr(cfg, "USE_HUBER_LOSS", True):
                critic1_loss = F.smooth_l1_loss(q1, target_q, beta=1.0)
                critic2_loss = F.smooth_l1_loss(q2, target_q, beta=1.0)
                critic_loss = critic1_loss + critic2_loss
            else:
                critic_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            
            critic_grad_norm = self._compute_grad_norm(self.critic.parameters())
            grad_norms["critic_grad_norm"] = critic_grad_norm
            
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_opt.step()
        else:
            grad_norms["critic_grad_norm"] = 0.0

        if not is_warmed_up:
            grad_norms["actor_grad_norm"] = 0.0
            return {
                "critic_loss": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
                "actor_loss": 0.0,
                "awac_weight": 0.0,
                "awac_adv": 0.0,
                **grad_norms
            }

        # =========================================================
        # 2. ACTOR UPDATE (AWAC Advantage Weighting)
        # =========================================================
        if update_actor:
            with torch.no_grad():
                num_samples = getattr(cfg, "AWAC_NUM_SAMPLES", 4)
                v_pis = []
                # Estimate V(s) by sampling multiple actions from current policy
                for _ in range(num_samples):
                    pi_a, _, _ = self.actor.sample(grid, scalars)
                    q1_pi, q2_pi = self.critic(grid, scalars, pi_a)
                    v_pis.append(torch.min(q1_pi, q2_pi))
                
                v_pi = torch.stack(v_pis).mean(dim=0)
                
                # Q-value of the expert action from the buffer
                q1_exp, q2_exp = self.critic(grid, scalars, actions_exp)
                q_exp = torch.min(q1_exp, q2_exp)
                
                # Calculate Advantage: A(s, a_exp) = Q(s, a_exp) - V(s)
                adv = q_exp - v_pi
                
                awac_lambda = getattr(cfg, "AWAC_LAMBDA", 2.0)
                scaled_adv = adv / awac_lambda
                
                # directly to prevent exp() overflow. exp(3.0) is roughly 20.0.
                # This ensures one outlier doesn't drag the whole batch down to a weight of 0.
                weights = torch.exp(scaled_adv.clamp(max=3.0))
                

            # Calculate Negative Log Likelihood of the expert action under current policy
            mean, log_std = self.actor.forward(grid, scalars)
            nll = squashed_gaussian_nll(mean, log_std, actions_exp, self.actor.action_scale, self.actor.action_bias)
            
            # Final AWAC Actor Loss
            actor_loss = (weights.squeeze() * nll).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            
            actor_grad_norm = self._compute_grad_norm(self.actor.parameters())
            grad_norms["actor_grad_norm"] = actor_grad_norm
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            self.last_actor_loss = actor_loss.item()
            self.last_awac_weight = weights.mean().item()
            self.last_awac_adv = adv.mean().item()
            self.last_actor_grad_norm = actor_grad_norm
        else:
            grad_norms["actor_grad_norm"] = self.last_actor_grad_norm

        # =========================================================
        # 3. TARGET NETWORK UPDATE
        # =========================================================
        if update_critic and self.train_step % cfg.TARGET_UPDATE_INTERVAL == 0:
            self.soft_update(self.critic, self.critic_target, cfg.TAU)

        return {
            "critic_loss": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
            "actor_loss": self.last_actor_loss,
            "awac_weight": self.last_awac_weight,
            "awac_adv": self.last_awac_adv,
            **grad_norms
        }

    def _compute_grad_norm(self, parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    @staticmethod
    def soft_update(source, target, tau):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "train_step" in ckpt:
            self.train_step = ckpt["train_step"]

    def load_actor_from_bc(self, bc_checkpoint_path, strict=False):
        """Translates keys from ImitationPolicy (BC) to AWACActor (RL)."""
        ckpt = torch.load(bc_checkpoint_path, map_location=self.device, weights_only=False)
        
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            
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
        
        print("\n[INFO] BC Weights Loaded into AWAC Actor Successfully!")
        if missing:
            print(f"[WARN] Missing keys during load (Usually safe if just log_std): {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys in BC checkpoint: {unexpected}\n")