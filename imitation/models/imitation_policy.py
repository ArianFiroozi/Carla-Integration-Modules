import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .actor_heads import *

class ImitationPolicy(nn.Module):
    """
    Policy used for Behavioral Cloning training. 
    """
    def __init__(
        self,
        mode="discrete",
        is_gaussian=False,
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
        latent_dim=128
    ):
        super().__init__()
        self.mode = mode
        self.is_gaussian = is_gaussian

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
                    n_mlp_layers=head_n_mlp_layers, mlp_hidden_size=head_mlp_hidden_size
                )
            else:
                self.actor = BCDecoupledContinuousHead(
                    latent_dim=latent_dim,
                    n_mlp_layers=head_n_mlp_layers, 
                    mlp_hidden_size=head_mlp_hidden_size 
                )

    def forward(self, grid, scalars):
        latent = self.extractor(grid, scalars)
        if self.mode == "continuous" and self.is_gaussian:
            return self.actor(latent, mode="bc")
        return self.actor(latent)
       