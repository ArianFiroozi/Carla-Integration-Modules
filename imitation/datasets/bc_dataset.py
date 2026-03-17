import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        
        
        # GRID
        self.presence = data["obs_presence"].astype(np.int64)

        self.num_classes = max(num_classes, int(self.presence.max()) + 1)


        # SCALARS
        scalars = []

        def add_scalar(key):
            if key in data.files:
                arr = data[key].astype(np.float32)
                # Ensure it's a 2D column vector (N, 1)
                if arr.ndim == 1:
                    arr = arr[:, None]
                scalars.append(arr)

        add_scalar("obs_lane_angle")
        add_scalar("obs_ego_in_lane_position_x")
        add_scalar("obs_ego_speed_x")
        add_scalar("obs_ego_speed_y")

        if include_traffic_signs:
            add_scalar("obs_traffic_signs")

        if len(scalars) == 0:
            self.scalars = np.zeros((self.presence.shape[0], 1), dtype=np.float32)
        else:
            self.scalars = np.concatenate(scalars, axis=1)
            # if self.scalars.shape[1] >= 4:
            #     # TODO: THIS NORMALIZATION IS NOT DYNAMIC
            #     self.scalars[:, 0] = self.scalars[:, 0] / np.pi   # lane_angle -> [-1,1]
            #     self.scalars[:, 1] = self.scalars[:, 1] / 2.0     # lane_pos -> [-1,1]
            #     self.scalars[:, 2] = self.scalars[:, 2] / 40.0    # speed_x -> [0,1]
            #     self.scalars[:, 3] = self.scalars[:, 3] / 10.0    # speed_y -> ~[-1,1]

        # Normalize only if NOT using one-hot
        if not self.one_hot_presence:
            maxv = float(np.max(self.presence))
            if maxv > 0:
                self.presence = self.presence.astype(np.float32) / maxv
            else:
                self.presence = self.presence.astype(np.float32)

        self.data = data

    def process_grid(self, grid):
        if self.one_hot_presence:
            grid_tensor = torch.from_numpy(grid.astype(np.int64))
            grid_tensor = F.one_hot(grid_tensor, num_classes=self.num_classes)
            return grid_tensor.permute(2, 0, 1).float()
        else:
            grid_tensor = torch.from_numpy(grid.astype(np.float32))
            
            return grid_tensor.unsqueeze(0)

    def __len__(self):
        return self.presence.shape[0]


class BCDataset(BaseDataset):
    """Dataset for Discrete High-Level Actions"""
    def __init__(
        self,
        npz_path,
        one_hot_presence=False,
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
        grid = self.process_grid(self.presence[idx])
        scalars = torch.from_numpy(self.scalars[idx])
        action = torch.from_numpy(self.actions[idx])

        return grid, scalars, action


class BCDatasetContinuous(BaseDataset):
    """Dataset for Continuous Low-Level Control"""
    def __init__(
        self,
        npz_path,
        one_hot_presence=False,
        include_traffic_signs=False,
        num_classes=4
    ):
        super().__init__(
            npz_path,
            one_hot_presence=one_hot_presence,
            include_traffic_signs=include_traffic_signs,
            num_classes=num_classes
        )

        throttle = self.data["target_throttle"].astype(np.float32)
        brake = self.data["target_brake"].astype(np.float32)
        steer = self.data["target_steering_angle"].astype(np.float32)

        steer = np.nan_to_num(steer, nan=0.0, posinf=0.0, neginf=0.0)
        steer = np.clip(steer, -1.0, 1.0)

        targets = np.column_stack([throttle, brake, steer])

        targets[:, 0] = np.clip(targets[:, 0], 0.0, 1.0)
        targets[:, 1] = np.clip(targets[:, 1], 0.0, 1.0)

        self.targets = targets.astype(np.float32)

    def __getitem__(self, idx):
        grid = self.process_grid(self.presence[idx])
        scalars = torch.from_numpy(self.scalars[idx])
        target = torch.from_numpy(self.targets[idx])

        return grid, scalars, target