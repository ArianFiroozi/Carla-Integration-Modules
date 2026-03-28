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
        grid_channels=1,
        scalar_dim=4,
        n_speed=None,
        n_turn=None,
    ):
        super().__init__()

        if mode not in ["discrete", "continuous"]:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode
        self.is_gaussian = is_gaussian

        self.extractor = FeatureExtractor(
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            latent_dim=128
        )

        if mode == "discrete":
            self.actor = DiscreteActorHead(latent_dim=128, n_speed=n_speed, n_turn=n_turn)
        else:
            if is_gaussian:
                self.actor = BCGaussianContinuousHead(latent_dim=128)
            else:
                self.actor = BCContinuousHead(latent_dim=128)

    def forward(self, grid, scalars):
        
        latent = self.extractor(grid, scalars)
        
        if self.mode == "continuous" and self.is_gaussian:
            return self.actor(latent, mode="bc")
        
        return self.actor(latent)
