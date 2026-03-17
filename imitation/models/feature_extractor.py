import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Shared encoder used by BC and PPO.
    Processes:
        - occupancy grid via CNN
        - scalar features via MLP
    Produces a fused latent vector.
    """

    def __init__(self, grid_channels=1, scalar_dim=4, latent_dim=128):
        super().__init__()

        # CNN for spatial grid
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        # Dynamically determine CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, 25, 11)
            out = self.cnn(dummy)
            cnn_dim = int(np.prod(out.shape[1:]))

        # Scalar encoder
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion network
        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, grid, scalars):

        grid_feat = self.cnn(grid).flatten(1)
        scalar_feat = self.scalar_mlp(scalars)

        fused = torch.cat([grid_feat, scalar_feat], dim=1)

        return self.fuse(fused)
