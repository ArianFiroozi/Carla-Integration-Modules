import torch.nn as nn

from networks.feature_extractor import FeatureExtractor
from networks.actor_heads import *

class ImitationPolicy(nn.Module):
    """
    Policy used for Behavioral Cloning training. 
    """
    def __init__(
        self,
        mode="discrete",
        is_gaussian=False,
        decoupled=False,
        grid_channels=9,
        scalar_dim=4,
        n_speed=None,
        n_turn=None,
        cnn_channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        head_n_mlp_layers=2,
        head_mlp_hidden_size=64,
        scalar_n_mlp_layers = 2,
        scalar_mlp_hidden_size = 64,
        latent_dim=128,
        action_low=None, action_high=None
    ):
        super().__init__()
        self.mode = mode
        self.is_gaussian = is_gaussian
        
        
        if action_low is None:  action_low = [0.0, 0.0, -1.0]
        if action_high is None: action_high = [1.0, 1.0,  1.0]
        action_low  = torch.tensor(action_low, dtype=torch.float32)
        action_high = torch.tensor(action_high, dtype=torch.float32)
        
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias",  (action_high + action_low) / 2.0)
        
        
        self.extractor = FeatureExtractor(
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            latent_dim=latent_dim,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            n_mlp_layers=scalar_n_mlp_layers,
            mlp_hidden_size=scalar_mlp_hidden_size
        ) 

        if mode == "discrete":
            self.actor = DiscreteActorHead(
                latent_dim=latent_dim, n_speed=n_speed, n_turn=n_turn,
                n_mlp_layers=head_n_mlp_layers, mlp_hidden_size=head_mlp_hidden_size
            )
        else:
            if is_gaussian:
                self.actor = BCGaussianContinuousHead(
                    latent_dim=latent_dim,
                    n_mlp_layers=head_n_mlp_layers,
                    mlp_hidden_size=head_mlp_hidden_size,
                    decoupled=decoupled
                )
            else:
                self.actor = BCContinuousHead(
                    latent_dim=latent_dim,
                    n_mlp_layers=head_n_mlp_layers,
                    mlp_hidden_size=head_mlp_hidden_size,
                    decoupled=decoupled
                )

    def forward(self, grid, scalars):
        latent = self.extractor(grid, scalars)

        if self.mode == "continuous" and self.is_gaussian:
            mean, log_std = self.actor(latent)
            # squash mean to env bounds
            a = torch.tanh(mean)
            mean = a * self.action_scale + self.action_bias
            return mean, log_std

        elif self.mode == "continuous":
            raw = self.actor(latent)
            a = torch.tanh(raw)
            action = a * self.action_scale + self.action_bias
            return action

        else:
            return self.actor(latent)
