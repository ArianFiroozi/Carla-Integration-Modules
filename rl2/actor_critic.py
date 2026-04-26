import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imitation.models.feature_extractor import FeatureExtractor
from imitation.models.actor_heads import BCDecoupledContinuousHead

class UTCarActorCritic(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.extractor = FeatureExtractor(
            grid_channels=5, scalar_dim=4, latent_dim=latent_dim,
            cnn_channels=[32, 64, 128], kernel_sizes=[3, 3, 3],
            n_mlp_layers=2, mlp_hidden_size=64
        )
        
        self.actor_mean = BCDecoupledContinuousHead(
            latent_dim=latent_dim, action_dim=3, n_mlp_layers=2, mlp_hidden_size=128
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(1, 3))
        
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def load_il_weights(self, best_model_path):
        """تزریق وزن‌های پیش‌آموزش دیده شده از فاز IL به شبکه Actor"""
        device = next(self.parameters()).device
        ckpt = torch.load(best_model_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        
        extractor_dict = {k.replace("extractor.", ""): v for k, v in state_dict.items() if k.startswith("extractor.")}
        actor_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        
        self.extractor.load_state_dict(extractor_dict, strict=False)
        self.actor_mean.load_state_dict(actor_dict, strict=False)
        print("[Warm Start] IL weights loaded successfully into RL Actor!")

    def get_action_and_value(self, grid, scalars, action=None):
        """تولید اکشن احتمالی و محاسبه لگاریتم احتمال و آنتروپی برای فرمول PPO"""
        latent = self.extractor(grid, scalars)
        action_mean = self.actor_mean(latent)
        
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(latent)