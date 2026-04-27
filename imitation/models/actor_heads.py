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
    def __init__(
        self,
        latent_dim=128,
        n_speed=5,
        n_turn=4,
        n_mlp_layers=2,
        mlp_hidden_size=64,
        decoupled=False,  # kept for API symmetry, not used
    ):
        """
        Outputs logits for high-level discrete actions.
        Example:
            speed_action (5 classes)
            turn_action (4 classes)
        """
        super().__init__()
        self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
        self.speed_head = nn.Linear(out_dim, n_speed)
        self.turn_head = nn.Linear(out_dim, n_turn)

    def forward(self, latent):
        feat = self.shared_net(latent)
        return self.speed_head(feat), self.turn_head(feat)



class BCContinuousHead(nn.Module):
    """
    Continuous Behavioral Cloning head.
    decoupled=False → shared MLP
    decoupled=True  → separate speed & steering MLPs
    Directly outputs control values bounded to their valid physical ranges:
        throttle: [0, 1]
        brake: [0, 1]
        steer: [-1, 1]
    """
    def __init__(
        self,
        latent_dim=128,
        action_dim=3,
        n_mlp_layers=2,
        mlp_hidden_size=64,
        decoupled=False,
    ):
        super().__init__()
        self.decoupled = decoupled

        if decoupled:
            self.speed_net, sd = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.speed_head = nn.Linear(sd, 2)  # throttle, brake

            self.steer_net, st = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.steer_head = nn.Linear(st, 1)
        else:
            self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.head = nn.Linear(out_dim, action_dim)

    def forward(self, latent):
        if self.decoupled:
            speed_feat = self.speed_net(latent)
            speed_raw = self.speed_head(speed_feat)

            steer_feat = self.steer_net(latent)
            steer_raw = self.steer_head(steer_feat)

            throttle = torch.sigmoid(speed_raw[:, 0:1])
            brake    = torch.sigmoid(speed_raw[:, 1:2])
            steer    = torch.tanh(steer_raw[:, 0:1])
        else:
            feat = self.shared_net(latent)
            raw = self.head(feat)

            throttle = torch.sigmoid(raw[:, 0:1])      # Bounds to [0, 1]
            brake = torch.sigmoid(raw[:, 1:2])         # Bounds to [0, 1]
            steer = torch.tanh(raw[:, 2:3])            # Bounds to [-1, 1]

        return torch.cat([throttle, brake, steer], dim=1)
    
        
    
class BCGaussianContinuousHead(nn.Module):
    """
    Gaussian BC head with optional decoupling.
    """
    def __init__(
        self,
        latent_dim=128,
        action_dim=3,
        log_std_min=-3,
        log_std_max=1,
        n_mlp_layers=2,
        mlp_hidden_size=64,
        decoupled=False,
    ):
        super().__init__()
        self.decoupled = decoupled
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if decoupled:
            self.speed_net, sd = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.speed_mean = nn.Linear(sd, 2)
            self.speed_logstd = nn.Linear(sd, 2)

            self.steer_net, st = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.steer_mean = nn.Linear(st, 1)
            self.steer_logstd = nn.Linear(st, 1)
        else:
            self.shared_net, out_dim = build_mlp(latent_dim, mlp_hidden_size, n_mlp_layers)
            self.mean_head = nn.Linear(out_dim, action_dim)
            self.log_std_head = nn.Linear(out_dim, action_dim)

        # bias init
        for m in self.modules():
            if isinstance(m, nn.Linear) and "logstd" in m._get_name().lower():
                m.bias.data.fill_(-1.0)

    def forward(self, latent, mode="bc"):
        if self.decoupled:
            sf = self.speed_net(latent)
            sm = self.speed_mean(sf)
            sl = self.speed_logstd(sf)

            tf = self.steer_net(latent)
            tm = self.steer_mean(tf)
            tl = self.steer_logstd(tf)

            mean = torch.cat([sm, tm], dim=1)
            log_std = torch.cat([sl, tl], dim=1)
        else:
            feat = self.shared_net(latent)
            mean = self.mean_head(feat)
            log_std = self.log_std_head(feat)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Bound mean for BC
        if mode == "bc":
            throttle = torch.sigmoid(mean[:, 0:1])
            brake    = torch.sigmoid(mean[:, 1:2])
            steer    = torch.tanh(mean[:, 2:3])
            mean = torch.cat([throttle, brake, steer], dim=1)

        return mean, std




