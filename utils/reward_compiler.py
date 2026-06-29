import torch
import numpy as np

def compile_reward(data, cfg, mode="info", is_tensor=False):
    """
    Unified Reward Compiler.
    - mode="info": Uses rich 3D vectors from CARLA environment.
    - mode="obs": Uses 2D trigonometry approximations from offline datasets.
    """
    def clip(val, min_v, max_v):
        return torch.clamp(val, min_v, max_v) if is_tensor else np.clip(val, min_v, max_v)
            
    def exp(val):
        return torch.exp(val) if is_tensor else np.exp(val)

    def sqrt(val):
        return torch.sqrt(val) if is_tensor else np.sqrt(val)
        
    def cos(val):
        return torch.cos(val) if is_tensor else np.cos(val)
        
    def abs_val(val):
        return torch.abs(val) if is_tensor else np.abs(val)

    # =========================================================================
    # ENGINE CALCULATIONS (Mode Dependent)
    # =========================================================================
    if mode == "info":
        speed_ms = sqrt(data['velocity_x']**2 + data['velocity_y']**2 + data['velocity_z']**2)
        
        # Progress
        raw_progress = (data['velocity_x'] * data['road_forward_x']) + (data['velocity_y'] * data['road_forward_y'])
        progress_score = clip(raw_progress / cfg.TARGET_SPEED_MS, 0.0, 1.0) * cfg.WEIGHT_PROGRESS
        
        # Alignment
        car_vec_norm_x = data['car_forward_x'] / (sqrt(data['car_forward_x']**2 + data['car_forward_y']**2) + 1e-6)
        car_vec_norm_y = data['car_forward_y'] / (sqrt(data['car_forward_x']**2 + data['car_forward_y']**2) + 1e-6)
        raw_heading_dot = (car_vec_norm_x * data['road_forward_x']) + (car_vec_norm_y * data['road_forward_y'])
        
        dx = data['vehicle_loc_x'] - data['lane_center_x']
        dy = data['vehicle_loc_y'] - data['lane_center_y']
        raw_lane_distance = sqrt(dx**2 + dy**2)
        
        centering_score = exp(-cfg.LANE_ALPHA * raw_lane_distance) * cfg.WEIGHT_CENTERING
        heading_score = raw_heading_dot * cfg.WEIGHT_HEADING
        
        # Physics Flags
        v_dot_fwd = (data['velocity_x'] * data['car_forward_x']) + \
                    (data['velocity_y'] * data['car_forward_y']) + \
                    (data['velocity_z'] * data['car_forward_z'])
                    
        is_pedal_overlap_raw = data['is_pedal_overlap']
        is_reversing_raw = (v_dot_fwd < -0.5)
        is_lane_invaded_raw = data['is_lane_invaded']
        is_terminal_raw = data['is_terminal_crash']

    elif mode == "obs":
        speed_ms = sqrt(data['obs_ego_speed_x']**2 + data['obs_ego_speed_y']**2)
        
        # Progress (Trig Approximation)
        progress_velocity = speed_ms * cos(data['obs_lane_angle'])
        progress_score = clip(progress_velocity / cfg.TARGET_SPEED_MS, 0.0, 1.0) * cfg.WEIGHT_PROGRESS
        
        # Alignment
        lane_distance = abs_val(data['obs_ego_in_lane_position_x'])
        centering_score = exp(-cfg.LANE_ALPHA * lane_distance) * cfg.WEIGHT_CENTERING
        heading_score = cos(data['obs_lane_angle']) * cfg.WEIGHT_HEADING
        
        # Physics Flags
        is_pedal_overlap_raw = (data['obs_throttle'] > 0.1) & (data['obs_brake'] > 0.1) if is_tensor else (float(data['obs_throttle']) > 0.1 and float(data['obs_brake']) > 0.1)
        is_reversing_raw = data['obs_reverse']
        is_lane_invaded_raw = 0.0  # We don't have lane invasion sensors in the old dataset, default to safe
        is_terminal_raw = data['terminated']

    else:
        raise ValueError("Invalid mode. Choose 'info' or 'obs'.")

    # =========================================================================
    # UNIVERSAL PENALTIES & SUMMATION (Mode Independent)
    # =========================================================================
    
    # Smoothness (Both modes now provide these keys!)
    steer_penalty = -(data['steer_change'] * cfg.PENALTY_STEER_DELTA)
    throttle_penalty = -(data['throttle_change'] * cfg.PENALTY_THROTTLE_DELTA)

    # Cast flags safely
    if is_tensor:
        pedal_flag = is_pedal_overlap_raw.float()
        is_rolling = is_reversing_raw.float()
        is_stalling = (speed_ms < cfg.STALL_SPEED_THRESHOLD).float()
        crash_mask = is_terminal_raw.float()
        lane_flag = is_lane_invaded_raw.float() if isinstance(is_lane_invaded_raw, torch.Tensor) else float(is_lane_invaded_raw)
    else:
        pedal_flag = float(is_pedal_overlap_raw)
        is_rolling = float(is_reversing_raw)
        is_stalling = 1.0 if float(speed_ms) < cfg.STALL_SPEED_THRESHOLD else 0.0
        crash_mask = float(is_terminal_raw)
        lane_flag = float(is_lane_invaded_raw)

    pedal_penalty = pedal_flag * cfg.PENALTY_PEDAL_OVERLAP
    rolling_penalty = is_rolling * cfg.PENALTY_ROLLING_BACKWARD
    lane_penalty = lane_flag * cfg.PENALTY_LANE_INVASION
    stall_penalty = is_stalling * cfg.PENALTY_STALLING

    total_reward = (progress_score + centering_score + heading_score + 
                    steer_penalty + throttle_penalty + 
                    pedal_penalty + rolling_penalty + lane_penalty + stall_penalty)
                    
    final_reward = (crash_mask * cfg.PENALTY_TERMINAL_CRASH) + ((1.0 - crash_mask) * total_reward)

    metrics = {}
    if not is_tensor:
        metrics = {
            'reward_progress': float(progress_score),
            'reward_centering': float(centering_score),
            'reward_heading': float(heading_score),
            'penalty_steer': float(steer_penalty),
            'penalty_throttle': float(throttle_penalty),
            'penalty_pedals': float(pedal_penalty),
            'penalty_rolling': float(rolling_penalty),
            'penalty_lane': float(lane_penalty),
            'penalty_stall': float(stall_penalty),
            'terminal_crash': float(crash_mask * cfg.PENALTY_TERMINAL_CRASH)
        }
        
    scaled_final_reward = final_reward / cfg.REWARD_SCALE_FACTOR
    return float(scaled_final_reward) if not is_tensor else scaled_final_reward, metrics