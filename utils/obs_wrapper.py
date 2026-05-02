
import numpy as np
import torch
from collections import deque
from config import bc_config

ACTION_MODE = bc_config.ACTION_MODE
SIMPLIFIED_ACTION_SPACE = bc_config.SIMPLIFY_ACTIONS

if SIMPLIFIED_ACTION_SPACE:
    speed_map = bc_config.SIMPLIFY_SPEED_MAP
    turn_map = bc_config.SIMPLIFY_TURN_MAP
else:
    speed_map = bc_config.SPEED_MAP
    turn_map = bc_config.TURN_MAP


def _normalize_value(v, key, norm_stats):
    v = np.asarray(v)

    if bc_config.SCALING_METHOD == "min_max":
        if key in norm_stats:
            s_min = norm_stats[key]["min"]
            s_max = norm_stats[key]["max"]
            if s_max - s_min > 1e-6:
                norm_v = 2.0 * (v - s_min) / (s_max - s_min) - 1.0
                return np.clip(norm_v, -1.0, 1.0)
        return np.zeros_like(v)

    elif bc_config.SCALING_METHOD == "z_score":
        if key in norm_stats:
            mean = norm_stats[key]["mean"]
            std = norm_stats[key]["std"]
            if std > 1e-6:
                return (v - mean) / std
        return np.zeros_like(v)

    elif bc_config.SCALING_METHOD == "fixed" and "speed" in key:
        return v / bc_config.MAX_SPEED

    return v if "speed" in key else np.zeros_like(v)


def wrap_angle_pi(angle):
    angle = np.asarray(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_spatial_distances(grid_2d):
    MAX_FRONT = 10.0
    MAX_BACK = 10.0
    MAX_SIDE = 5.0

    dist_front = MAX_FRONT
    dist_back  = MAX_BACK
    dist_right = MAX_SIDE
    dist_left  = MAX_SIDE

    ego_positions = np.argwhere(grid_2d == 3)
    if len(ego_positions) == 0:
        return np.array([dist_front, dist_back, dist_left, dist_right], dtype=np.float32)

    ego_y, ego_x = ego_positions[0]

    column_in_front = grid_2d[ego_y + 1:, ego_x]
    obstacles_ahead = np.argwhere((column_in_front == 1) | (column_in_front == 2))
    if len(obstacles_ahead) > 0:
        dist_front = float(obstacles_ahead[0][0] + 1)

    column_behind = grid_2d[:ego_y, ego_x][::-1]
    obstacles_behind = np.argwhere((column_behind == 1) | (column_behind == 2))
    if len(obstacles_behind) > 0:
        dist_back = float(obstacles_behind[0][0] + 1)

    row_to_left = grid_2d[ego_y, :ego_x][::-1]
    obstacles_left = np.argwhere((row_to_left == 1) | (row_to_left == 2))
    if len(obstacles_left) > 0:
        dist_left = float(obstacles_left[0][0] + 1)

    row_to_right = grid_2d[ego_y, ego_x + 1:]
    obstacles_right = np.argwhere((row_to_right == 1) | (row_to_right == 2))
    if len(obstacles_right) > 0:
        dist_right = float(obstacles_right[0][0] + 1)

    return np.array([dist_front, dist_back, dist_left, dist_right], dtype=np.float32)


class ObsHistory:
    def __init__(self, window_size=3, one_hot=True, norm_stats=None):
        self.window_size = window_size
        self.one_hot = one_hot
        self.norm_stats = norm_stats or {}

        self.presence = deque(maxlen=window_size)
        self.speed_x = deque(maxlen=window_size)
        self.speed_y = deque(maxlen=window_size)

    def reset(self):
        self.presence.clear()
        self.speed_x.clear()
        self.speed_y.clear()

    def update(self, obs):
        p = obs["presence"]
        sx = obs.get("speed_x", np.zeros_like(p))
        sy = obs.get("speed_y", np.zeros_like(p))

        if torch.is_tensor(p):  p  = p.cpu().numpy()
        if torch.is_tensor(sx): sx = sx.cpu().numpy()
        if torch.is_tensor(sy): sy = sy.cpu().numpy()

        if len(self.presence) == 0:
            for _ in range(self.window_size):
                self.presence.append(p)
                self.speed_x.append(sx)
                self.speed_y.append(sy)
        else:
            self.presence.append(p)
            self.speed_x.append(sx)
            self.speed_y.append(sy)

    def get_grid(self):
        frames = []
        for i in range(self.window_size):
            p = self.presence[i]
            sx = _normalize_value(self.speed_x[i], "obs_speed_x", self.norm_stats)
            sy = _normalize_value(self.speed_y[i], "obs_speed_y", self.norm_stats)

            if self.one_hot:
                mask_v = (p == 1).astype(np.float32)
                mask_w = (p == 2).astype(np.float32)
                mask_e = (p == 3).astype(np.float32)
                frame_stack = np.stack([mask_v, mask_w, mask_e, sx, sy], axis=0)
            else:
                p_norm = p.astype(np.float32)
                frame_stack = np.stack([p_norm, sx, sy], axis=0)

            frames.append(frame_stack)

        return np.concatenate(frames, axis=0).astype(np.float32)


class CarlaObsWrapper:
    """
    Reusable wrapper for:
    - grid/scalar preprocessing
    - history stacking
    - normalization
    - spatial features
    - action mapping
    - continuous post-processing
    """
    def __init__(self, norm_stats, device, action_mode=None):
        self.norm_stats = norm_stats or {}
        self.device = device
        self.action_mode = action_mode or bc_config.ACTION_MODE

        self.history = ObsHistory(
            one_hot=bc_config.USE_ONE_HOT_GRID,
            window_size=bc_config.WINDOW_SIZE,
            norm_stats=self.norm_stats
        )
        self.prev_steer = 0.0

    def reset(self):
        self.history.reset()
        self.prev_steer = 0.0

    def preprocess(self, obs):
        # fix presence grid shape
        presence = np.array(obs["presence"])
        if presence.ndim == 1:
            presence = presence.reshape(25, 11)
        elif presence.ndim == 3:
            presence = presence.squeeze()

        # fix ego encoding
        presence[presence == 9] = 3
        obs["presence"] = presence

        # spatial features
        dist_front, dist_back, dist_left, dist_right = 10.0, 10.0, 5.0, 5.0
        if getattr(bc_config, "USE_SPATIAL_FEATURES", False):
            dists = compute_spatial_distances(obs["presence"])
            dist_front = _normalize_value(dists[0], "obs_dist_front", self.norm_stats)
            dist_back  = _normalize_value(dists[1], "obs_dist_back", self.norm_stats)
            dist_left  = _normalize_value(dists[2], "obs_dist_left", self.norm_stats)
            dist_right = _normalize_value(dists[3], "obs_dist_right", self.norm_stats)

        # update history
        self.history.update(obs)
        grid = self.history.get_grid()

        # scalars
        raw_lane_angle = wrap_angle_pi(obs["lane_angle"][0])
        lane_angle = _normalize_value(raw_lane_angle, "obs_lane_angle", self.norm_stats)
        lane_pos   = _normalize_value(obs["ego_in_lane_position_x"][0], "obs_ego_in_lane_position_x", self.norm_stats)
        speed_x    = _normalize_value(obs["ego_speed_x"][0], "obs_ego_speed_x", self.norm_stats)
        speed_y    = _normalize_value(obs["ego_speed_y"][0], "obs_ego_speed_y", self.norm_stats)

        if getattr(bc_config, "USE_SPATIAL_FEATURES", False):
            scalars = np.array([lane_angle, lane_pos, speed_x, speed_y,
                                dist_front, dist_back, dist_left, dist_right], dtype=np.float32)
        else:
            scalars = np.array([lane_angle, lane_pos, speed_x, speed_y], dtype=np.float32)

        return grid, scalars

    def to_tensor(self, grid, scalars):
        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
        scal_t = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(self.device)
        return grid_t, scal_t

    def map_action_for_env(self, action):
        speed, turn = action
        if SIMPLIFIED_ACTION_SPACE:
            if speed == 3: speed = 4
            if turn == 2:  turn = 3
        return [speed, turn]

    def process_continuous_output(self, out):
        throttle = float(np.clip(out[0], 0.0, 1.0))
        brake    = float(np.clip(out[1], 0.0, 1.0))
        steer    = float(np.clip(out[2], -1.0, 1.0))

        if throttle < 0.13 and throttle > 0.05:
            throttle = 0.13

        if brake > 0.1:
            throttle = 0.0
        else:
            brake = 0.0

        if bc_config.SMOOTH_STEERING:
            steer = 0.7 * self.prev_steer + 0.3 * steer
            self.prev_steer = steer
            steer = np.clip(steer, -1.0, 1.0)

        return [throttle, brake, steer]
