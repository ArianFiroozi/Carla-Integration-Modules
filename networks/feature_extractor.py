import torch
import torch.nn as nn
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, grid_channels=15, scalar_dim=4, latent_dim=128, 
                 cnn_channels=[32, 64, 128], kernel_sizes=[3, 3, 3], 
                 n_mlp_layers=2, mlp_hidden_size=64):
        super().__init__()

        cnn_layers = []
        in_channels = grid_channels
        for out_channels, k_size in zip(cnn_channels, kernel_sizes):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=k_size//2))
            cnn_layers.append(nn.ReLU(inplace=True))
            if out_channels != cnn_channels[0]: 
                cnn_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            
            dummy = torch.zeros(1, grid_channels, 25, 11)
            out = self.cnn(dummy)
            cnn_dim = int(np.prod(out.shape[1:]))

        scalar_layers = []
        in_dim = scalar_dim
        for _ in range(n_mlp_layers):
            scalar_layers.append(nn.Linear(in_dim, mlp_hidden_size))
            scalar_layers.append(nn.ReLU(inplace=True))
            in_dim = mlp_hidden_size
            
        self.scalar_mlp = nn.Sequential(*scalar_layers)
        scalar_out_dim = mlp_hidden_size if n_mlp_layers > 0 else scalar_dim
        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + scalar_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, grid, scalars):
        grid_feat = self.cnn(grid).flatten(1)
        scalar_feat = self.scalar_mlp(scalars)
        fused = torch.cat([grid_feat, scalar_feat], dim=1)
        return self.fuse(fused)
