import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from pathlib import Path
from config import bc_config






class BaseDataset(Dataset):
    def __init__(
        self,
        npz_path,
        one_hot_presence=True,
        include_traffic_signs=False,
        num_classes=4
    ):
        data = np.load(npz_path)

        self.one_hot_presence = one_hot_presence
        self.include_traffic_signs = include_traffic_signs
        
        meta_path = Path(npz_path).with_suffix(".meta.json")
        self.norm_stats = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.norm_stats = meta.get("normalization_stats", {})
        else:
            print(f"[WARN] Meta file not found at {meta_path}. Dynamic normalization might fail!")

        # GRID
        self.presence = data["obs_presence"].astype(np.int64)
        self.speed_x = data["obs_speed_x"].astype(np.float32)
        self.speed_y = data["obs_speed_y"].astype(np.float32)

        self.num_classes = max(num_classes, int(self.presence.max()) + 1)

        # SCALARS
        scalars = []
        scalar_keys_in_order = []

        def add_scalar(key):
            if key in data.files:
                arr = data[key].astype(np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]
                scalars.append(arr)
                scalar_keys_in_order.append(key)

        add_scalar("obs_lane_angle")
        add_scalar("obs_ego_in_lane_position_x")
        add_scalar("obs_ego_speed_x")
        add_scalar("obs_ego_speed_y")

        if bc_config.USE_SPATIAL_FEATURES:
            add_scalar("obs_dist_front")
            add_scalar("obs_dist_left")
            add_scalar("obs_dist_right")
            add_scalar("obs_dist_back")

        if include_traffic_signs:
            add_scalar("obs_traffic_signs")

        if len(scalars) == 0:
            self.scalars = np.zeros((self.presence.shape[0], 1), dtype=np.float32)
        else:
            self.scalars = np.concatenate(scalars, axis=1)
            # Apply normalization to the concatenated scalar array
            for i, key in enumerate(scalar_keys_in_order):
                self.scalars[:, i] = self._normalize(self.scalars[:, i], key)

        if not self.one_hot_presence:
            maxv = float(np.max(self.presence))
            if maxv > 0:
                self.presence = self.presence.astype(np.float32) / maxv
            else:
                self.presence = self.presence.astype(np.float32)

        self.data = data

    def __len__(self):
        return self.presence.shape[0]
    
    def _normalize(self, v, key):
        """Unified normalization function based on config.SCALING_METHOD."""
        if bc_config.SCALING_METHOD == "min_max":
            if key in self.norm_stats:
                s_min = self.norm_stats[key]["min"]
                s_max = self.norm_stats[key]["max"]
                if s_max - s_min > 1e-6:
                    norm_v = 2.0 * (v - s_min) / (s_max - s_min) - 1.0
                    return np.clip(norm_v, -1.0, 1.0)
            return np.zeros_like(v) # Fallback if stats are missing

        elif bc_config.SCALING_METHOD == "z_score":
            if key in self.norm_stats:
                mean = self.norm_stats[key]["mean"]
                std = self.norm_stats[key]["std"]
                if std > 1e-6:
                    return (v - mean) / std
            return np.zeros_like(v) # Fallback if stats are missing
            
        elif bc_config.SCALING_METHOD == "fixed" and "speed" in key:
            return v / bc_config.MAX_SPEED

        # Default: return unnormalized value or zero for scalars in 'fixed' mode
        return v if "speed" in key else np.zeros_like(v)
        
    def _get_processed_grid(self, idx):
        pres = torch.from_numpy(self.presence[idx]).long()
        
        # Normalize speed grids using the unified method
        vx_np = self._normalize(self.speed_x[idx], "obs_speed_x")
        vy_np = self._normalize(self.speed_y[idx], "obs_speed_y")
        
        vx = torch.from_numpy(vx_np).float()
        vy = torch.from_numpy(vy_np).float()

        if self.one_hot_presence:
            # Note: build_dataset remaps ego from 9 to 3. This is correct.
            mask_v = (pres == 1).float()
            mask_w = (pres == 2).float()
            mask_e = (pres == 3).float()
            stacked = torch.stack([mask_v, mask_w, mask_e, vx, vy], dim=1)
        else:
            presence_norm = pres.float()
            stacked = torch.stack([presence_norm, vx, vy], dim=1)

        W, C, H, Wd = stacked.shape
        return stacked.view(W * C, H, Wd)








class BCDataset(BaseDataset):
    """Dataset for Discrete High-Level Actions"""
    def __init__(
        self,
        npz_path,
        one_hot_presence=True, 
        include_traffic_signs=False,
        num_classes=4
    ):
        super().__init__(
            npz_path,
            one_hot_presence=one_hot_presence, 
            include_traffic_signs=include_traffic_signs,
            num_classes=num_classes
        )
        self.actions = self.data["actions"].astype(np.int64)

    def __getitem__(self, idx):
        grid = self._get_processed_grid(idx).float()
        scalars = torch.from_numpy(self.scalars[idx])
        action = torch.from_numpy(self.actions[idx])

        return grid, scalars, action



def _process_actions_np(targets):

    # TODO: fix this simliar to how it is done it env.py
    # targets: (N,3) [throttle, brake, steer]
    t = np.clip(targets[:, 0], 0.0, 1.0)
    b = np.clip(targets[:, 1], 0.0, 1.0)
    s = np.clip(targets[:, 2], -1.0, 1.0)

    # throttle floor
    mask = (t < 0.13) & (t > 0.05)
    t[mask] = 0.13

    # brake/throttle exclusivity
    brake_active = b > 0.1
    t = t * (~brake_active)
    b = b * brake_active

    return np.stack([t, b, s], axis=1)

class BCDatasetContinuous(BaseDataset):
    """Dataset for Continuous Low-Level Control"""
    def __init__(
        self,
        npz_path,
        one_hot_presence=True, 
        include_traffic_signs=False,
        num_classes=4
    ):
        super().__init__(
            npz_path,
            one_hot_presence=one_hot_presence, 
            include_traffic_signs=include_traffic_signs,
            num_classes=num_classes
        )

        # 1. Handle unified actions ONLY if it's explicitly 3 columns wide
        if "actions" in self.data and len(self.data["actions"].shape) > 1 and self.data["actions"].shape[1] == 3:
            actions_array = self.data["actions"].astype(np.float32)
            throttle = actions_array[:, 0]
            brake = actions_array[:, 1]
            steer = actions_array[:, 2]
            
        # 2. Otherwise, dynamically find the split keys (handles both BC targets and RL obs)
        else:
            def get_action_array(target_key, obs_key):
                if target_key in self.data: return self.data[target_key]
                if obs_key in self.data: return self.data[obs_key]
                raise KeyError(f"Could not find continuous action key: {target_key} or {obs_key}")

            throttle = get_action_array("target_throttle", "obs_throttle").astype(np.float32)
            brake = get_action_array("target_brake", "obs_brake").astype(np.float32)
            steer = get_action_array("target_steering_angle", "obs_steering_angle").astype(np.float32)
            

        steer = np.nan_to_num(steer, nan=0.0, posinf=0.0, neginf=0.0)
        steer = np.clip(steer, -1.0, 1.0)

        targets = np.column_stack([throttle, brake, steer])
        targets = _process_actions_np(targets)
        self.targets = targets.astype(np.float32)

        
        # =========================================================
        # Continuous Undersampling (Feature Flag)
        # =========================================================
        if bc_config.USE_CONTINUOUS_UNDERSAMPLING:
            steer_angles = np.abs(self.targets[:, 2])
            is_straight = steer_angles < bc_config.UNDERSAMPLING_THRESHOLD
            random_probs = np.random.rand(len(steer_angles))
            keep_straight = is_straight & (random_probs > bc_config.UNDERSAMPLING_PROBABILITY)
            keep_turned = ~is_straight
            keep_mask = keep_straight | keep_turned
            self.presence = self.presence[keep_mask]
            self.speed_x = self.speed_x[keep_mask]
            self.speed_y = self.speed_y[keep_mask]
            self.scalars = self.scalars[keep_mask]
            self.targets = self.targets[keep_mask]
            dropped_count = len(keep_mask) - keep_mask.sum()
            print(f"[Dataset] Continuous Undersampling is ON!")
            print(f"[Dataset] Dropped {dropped_count} straight-driving samples out of {len(keep_mask)}.")





    def __getitem__(self, idx):
        grid = self._get_processed_grid(idx).float()
        scalars = torch.from_numpy(self.scalars[idx])
        target = torch.from_numpy(self.targets[idx])

        return grid, scalars, target