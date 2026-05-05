import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from config import sac_config as cfg
from networks.feature_extractor import FeatureExtractor
from networks.actor_heads import BCGaussianContinuousHead
from networks.critic_heads import TwinQCriticHead

 
class SACActor(nn.Module):
    """
    Actor = FeatureExtractor + Gaussian Head (mean/log_std)
    Uses tanh-squashed Gaussian policy with action scaling.
    """
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

        # action scaling buffers
        if action_low is None:
            action_low = [-1.0] * action_dim
        if action_high is None:
            action_high = [1.0] * action_dim

        action_low = torch.tensor(action_low, dtype=torch.float32)
        action_high = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)
        if torch.any(action_scale <= 0):
            raise ValueError(f"Invalid action bounds: low={action_low}, high={action_high}")

    def forward(self, grid, scalars):
        latent = self.feature_extractor(grid, scalars)
        mean, log_std = self.head(latent)
        return mean, log_std

    def sample(self, grid, scalars):
        mean, log_std = self.forward(grid, scalars)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()                # reparameterization
        a = torch.tanh(z)                   # [-1,1]

        # scale to environment bounds
        action = a * self.action_scale + self.action_bias

        # log_prob with tanh correction + scaling correction
        log_prob = normal.log_prob(z).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        log_prob -= torch.log(self.action_scale + 1e-6).sum(-1, keepdim=True)

        # deterministic action
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class SACCritic(nn.Module):
    """
    Critic = FeatureExtractor + TwinQHead
    """
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
        self.head = TwinQCriticHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

    def forward(self, grid, scalars, action):
        latent = self.feature_extractor(grid, scalars)
        q1, q2 = self.head(latent, action)
        return q1, q2


class SACAgent:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Actor
        self.actor = SACActor(
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
        self.critic = SACCritic(
            grid_channels=cfg.GRID_CHANNELS,
            scalar_dim=cfg.SCALAR_DIM,
            latent_dim=cfg.LATENT_DIM,
            action_dim=cfg.ACTION_DIM,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR, weight_decay=cfg.WEIGHT_DECAY)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR, weight_decay=cfg.WEIGHT_DECAY)

        # Entropy (alpha)
        self.auto_entropy = cfg.AUTO_ENTROPY
        if self.auto_entropy:
            self.target_entropy = -cfg.ACTION_DIM * cfg.TARGET_ENTROPY_SCALE
            self.log_alpha = torch.tensor(cfg.INIT_ALPHA, dtype=torch.float32, device=self.device).log().requires_grad_(True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.ALPHA_LR)
        else:
            self._alpha = cfg.INIT_ALPHA

        self.train_step = 0

    @property
    def alpha(self):
        if self.auto_entropy:
            return self.log_alpha.exp()
        return torch.tensor(self._alpha, device=self.device)


    def select_action(self, grid, scalars, evaluate=False):
        """
        grid: torch.Tensor [B,C,H,W]
        scalars: torch.Tensor [B,S]
        returns numpy action
        """
        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(grid, scalars)
            else:
                action, _, _ = self.actor.sample(grid, scalars)
        self.actor.train()
        return action.cpu().numpy()

    def update(self, replay_buffer, action_processor=None):
        """
        One SAC update step
        """
        self.train_step += 1

        grid, scalars, actions, rewards, next_grid, next_scalars, dones = replay_buffer.sample(cfg.BATCH_SIZE)

        # ---------------------- Critic update ----------------------
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_grid, next_scalars)
            if action_processor is not None:
                next_action = action_processor(next_action)
            q1_t, q2_t = self.critic_target(next_grid, next_scalars, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = rewards + (1.0 - dones) * cfg.GAMMA * q_t

        q1, q2 = self.critic(grid, scalars, actions)
        critic_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)



        self.critic_opt.step()
        # --- Critic Warm-up ---
        if self.train_step < cfg.WARMUP_STEPS:
            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.alpha.item(),
            }

        # ---------------------- Actor update -----------------------
        new_action, logp, _ = self.actor.sample(grid, scalars)
        if action_processor is not None:
            new_action = action_processor(new_action)
        q1_pi, q2_pi = self.critic(grid, scalars, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ---------------------- Alpha update -----------------------
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            # print(f"log_alpha: {self.log_alpha.item():.4f}, grad: {self.log_alpha.grad}")
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # ---------------------- Target update ----------------------
        if self.train_step % cfg.TARGET_UPDATE_INTERVAL == 0:
            self.soft_update(self.critic, self.critic_target, cfg.TAU)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "alpha": self.alpha.item(),
        }

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
            "log_alpha": self.log_alpha.detach().cpu() if self.auto_entropy else None,
            "alpha_opt": self.alpha_opt.state_dict() if self.auto_entropy else None,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.auto_entropy and ckpt.get("log_alpha") is not None:

            
            # Copy data into existing tensor instead
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
            # Optimizer already exists and points to correct tensor
            if ckpt.get("alpha_opt") is not None:
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])


    def load_actor_from_bc(self, bc_checkpoint_path, strict=False):
        """
        Optional: load BC weights into SAC actor.
        Translates keys from ImitationPolicy (BC) to SACActor (RL).
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
            # --------------------------------------------------------
                
            elif k.startswith("actor."):
                new_key = k.replace("actor.", "head.", 1)
                translated_state[new_key] = v
            else:
                translated_state[k] = v
                
        missing, unexpected = self.actor.load_state_dict(translated_state, strict=strict)
        
        print("\n[INFO] BC Weights Loaded into SAC Actor Successfully!")
        if missing:
            print(f"[WARN] Missing keys during load (Usually safe if just log_std): {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys in BC checkpoint: {unexpected}\n")
        
        
