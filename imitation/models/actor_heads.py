import torch
import torch.nn as nn


class DiscreteActorHead(nn.Module):
    """
    Outputs logits for high-level discrete actions.
    Example:
        speed_action (5 classes)
        turn_action (4 classes)
    """

    def __init__(self, latent_dim=128, n_speed=5, n_turn=4):
        super().__init__()

        self.speed_head = nn.Linear(latent_dim, n_speed)
        self.turn_head = nn.Linear(latent_dim, n_turn)

    def forward(self, latent):

        speed_logits = self.speed_head(latent)
        turn_logits = self.turn_head(latent)

        return speed_logits, turn_logits


class BCContinuousHead(nn.Module):
    """
    Continuous head for Behavioral Cloning.
    Directly outputs control values:
        throttle
        brake
        steer
    """

    def __init__(self, latent_dim=128, action_dim=3):
        super().__init__()

        self.head = nn.Linear(latent_dim, action_dim)

    def forward(self, latent):

        actions = self.head(latent)

        return actions


class PPOContinuousHead(nn.Module):
    """
    Continuous head for PPO.
    Outputs Gaussian distribution parameters.
    """

    def __init__(self, latent_dim=128, action_dim=3):
        super().__init__()

        self.mean_head = nn.Linear(latent_dim, action_dim)

        # global trainable log_std parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, latent):

        mean = self.mean_head(latent)

        # broadcast std across batch
        std = torch.exp(self.log_std).expand_as(mean)

        return mean, std
