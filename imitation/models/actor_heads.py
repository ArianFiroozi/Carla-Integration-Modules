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


class DiscreteActorHead(nn.Module):
    """
    Outputs logits for high-level discrete actions.
    Example:
        speed_action (5 classes)
        turn_action (4 classes)
    """
    def __init__(self, latent_dim=128, n_speed=5, n_turn=4, n_mlp_layers=2, mlp_hidden_size=64):
        super().__init__()
        self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
        
        self.speed_head = nn.Linear(out_dim, n_speed)
        self.turn_head = nn.Linear(out_dim, n_turn)

    def forward(self, latent):
        feat = self.shared_net(latent)
        speed_logits = self.speed_head(feat)
        turn_logits = self.turn_head(feat)
        return speed_logits, turn_logits



class BCContinuousHead(nn.Module):
    """
    Continuous head for Behavioral Cloning.
    Directly outputs control values bounded to their valid physical ranges:
        throttle: [0, 1]
        brake: [0, 1]
        steer: [-1, 1]
    """
    def __init__(self, latent_dim=128, action_dim=3, n_mlp_layers=2, mlp_hidden_size=64):
        super().__init__()
        self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
        self.head = nn.Linear(out_dim, action_dim)
    def forward(self, latent):
        feat = self.shared_net(latent)
        raw_actions = self.head(feat)
        throttle = torch.sigmoid(raw_actions[:, 0:1])      # Bounds to [0, 1]
        brake = torch.sigmoid(raw_actions[:, 1:2])         # Bounds to [0, 1]
        steer = torch.tanh(raw_actions[:, 2:3])            # Bounds to [-1, 1]
        actions = torch.cat([throttle, brake, steer], dim=1)
        return actions
        
    
class BCGaussianContinuousHead(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3, log_std_min=-3, log_std_max=1, 
                 n_mlp_layers=2, mlp_hidden_size=64):
        super().__init__()
        
        self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)

        self.mean_head = nn.Linear(out_dim, action_dim)
        self.log_std_head = nn.Linear(out_dim, action_dim)
        self.log_std_head.bias.data.fill_(-1.0)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, latent, mode="bc"):
        
        feat = self.shared_net(latent)
        
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # BC MODE: Bound mean to valid CARLA action ranges
        if mode == "bc":
            throttle = torch.sigmoid(mean[:, 0:1])
            brake    = torch.sigmoid(mean[:, 1:2])
            steer    = torch.tanh(mean[:, 2:3])
            mean = torch.cat([throttle, brake, steer], dim=1)

        return mean, std



class BCDecoupledContinuousHead(nn.Module):
    """
    Decoupled head for Behavioral Cloning.
    Separates the MLP for Speed (Throttle/Brake) and Steering 
    to prevent action collapse and gradient dominance.
    """
    def __init__(self, latent_dim=128, action_dim=3, n_mlp_layers=2, mlp_hidden_size=64):
        super().__init__()
        self.speed_net, speed_out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
        self.speed_head = nn.Linear(speed_out_dim, 2) # Throttle, Brake
        
        self.steer_net, steer_out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
        self.steer_head = nn.Linear(steer_out_dim, 1) # Steer

    def forward(self, latent):
        speed_feat = self.speed_net(latent)
        speed_raw = self.speed_head(speed_feat)
        throttle = torch.sigmoid(speed_raw[:, 0:1])
        brake = torch.sigmoid(speed_raw[:, 1:2])
        
        steer_feat = self.steer_net(latent)
        steer_raw = self.steer_head(steer_feat)
        steer = torch.tanh(steer_raw[:, 0:1])
        
        return torch.cat([throttle, brake, steer], dim=1)