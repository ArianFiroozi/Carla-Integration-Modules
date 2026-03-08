from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


class BCCNN(nn.Module):
    def __init__(self, scalar_dim: int, n_speed=5, n_turn=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (32,12,5)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # (64,6,2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 25, 11)
            out = self.cnn(dummy)
            cnn_dim = int(np.prod(out.shape[1:]))

        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.head_speed = nn.Linear(128, n_speed)
        self.head_turn  = nn.Linear(128, n_turn)

    def forward(self, grid, scalars):
        x = grid.unsqueeze(1)          # (B,1,25,11)
        x = self.cnn(x).flatten(1)     # (B,cnn_dim)
        s = self.scalar_mlp(scalars)   # (B,64)
        h = self.fuse(torch.cat([x, s], dim=1))
        return self.head_speed(h), self.head_turn(h)


# ---- Policy Wrapper ----
@dataclass
class BCPolicyConfig:
    n_speed: int = 5
    n_turn: int = 4
    presence_max: float = 9.0   # adjust if your grid max changes
    deterministic: bool = True  # argmax


class BCPolicy:
    def __init__(self, ckpt_path: str, device: str = "auto", cfg: BCPolicyConfig = BCPolicyConfig()):
        self.cfg = cfg

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        scalar_dim = int(ckpt["scalar_dim"])
        meta = ckpt.get("meta", {})

        self.model = BCCNN(
            scalar_dim=scalar_dim,
            n_speed=meta.get("n_speed", cfg.n_speed),
            n_turn=meta.get("n_turn", cfg.n_turn),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _npify(self, x):
        # supports torch tensors, lists, scalars
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def obs_to_tensors(self, obs: dict):
        # presence grid
        p = self._npify(obs["presence"]).astype(np.float32)  # (25,11)
        maxv = self.cfg.presence_max if self.cfg.presence_max > 0 else max(1.0, float(np.max(p)))
        p = p / maxv
        grid = torch.from_numpy(p).float().unsqueeze(0).to(self.device)  # (1,25,11)

        # scalars (match dataset build order as much as possible)
        scalars_list = []
        for k in ["lane_angle", "ego_in_lane_position_x", "ego_speed_x", "ego_speed_y"]:
            if k in obs:
                arr = self._npify(obs[k]).astype(np.float32).reshape(-1)
                scalars_list.append(arr)
        if "traffic_signs" in obs:
            arr = self._npify(obs["traffic_signs"]).astype(np.float32).reshape(-1)
            scalars_list.append(arr)

        if len(scalars_list) == 0:
            scalars = np.zeros((1, 1), dtype=np.float32)
        else:
            scalars = np.concatenate(scalars_list, axis=0)[None, :].astype(np.float32)

        scalars = torch.from_numpy(scalars).float().to(self.device)  # (1,d)
        return grid, scalars

    @torch.no_grad()
    def predict(self, obs: dict):
        grid, scalars = self.obs_to_tensors(obs)
        speed_logits, turn_logits = self.model(grid, scalars)

        if self.cfg.deterministic:
            speed = int(speed_logits.argmax(dim=1).item())
            turn  = int(turn_logits.argmax(dim=1).item())
        else:
            # sample
            speed = int(torch.distributions.Categorical(logits=speed_logits).sample().item())
            turn  = int(torch.distributions.Categorical(logits=turn_logits).sample().item())

        return np.array([speed, turn], dtype=np.int64)