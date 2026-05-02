import torch
import torch.nn as nn


def build_mlp(input_dim, hidden_size, n_layers):
    layers = []
    in_dim = input_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU(inplace=True))
        in_dim = hidden_size
    return nn.Sequential(*layers), in_dim


class QCriticHead(nn.Module):
    """
    Q-value head for SAC:
      input: latent + action
      output: scalar Q-value
    """
    def __init__(
        self,
        latent_dim=128,
        action_dim=3,
        n_mlp_layers=2,
        mlp_hidden_size=256,
    ):
        super().__init__()
        self.shared_net, out_dim = build_mlp(
            input_dim=latent_dim + action_dim,
            hidden_size=mlp_hidden_size,
            n_layers=n_mlp_layers,
        )
        self.q_head = nn.Linear(out_dim, 1)

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=1)
        feat = self.shared_net(x)
        q = self.q_head(feat)
        return q


class TwinQCriticHead(nn.Module):
    """
    Twin Q critics (Q1, Q2) for SAC.
    """
    def __init__(
        self,
        latent_dim=128,
        action_dim=3,
        n_mlp_layers=2,
        mlp_hidden_size=256,
    ):
        super().__init__()
        self.q1 = QCriticHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_mlp_layers=n_mlp_layers,
            mlp_hidden_size=mlp_hidden_size,
        )
        self.q2 = QCriticHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_mlp_layers=n_mlp_layers,
            mlp_hidden_size=mlp_hidden_size,
        )

    def forward(self, latent, action):
        q1 = self.q1(latent, action)
        q2 = self.q2(latent, action)
        return q1, q2
