import torch
import numpy as np

def compile_reward(info, cfg, is_tensor=False):
    """
    Takes a dictionary of RAW physical components, reconstructs the math, 
    and compiles them into a single reward.
    Returns: (final_reward, metrics_breakdown_dict)
    """
    def clip(val, min_v, max_v):
        return torch.clamp(val, min_v, max_v) if is_tensor else np.clip(val, min_v, max_v)
            
    def exp(val):
        return torch.exp(val) if is_tensor else np.exp(val)

    def sqrt(val):
        return torch.sqrt(val) if is_tensor else np.sqrt(val)

    # Reconstruct Vectors and Magnitudes
    speed_ms = sqrt(info['velocity_x']**2 + info['velocity_y']**2 + info['velocity_z']**2)
    
    # Dot Products for Progress and Heading
    raw_progress = (info['velocity_x'] * info['road_forward_x']) + (info['velocity_y'] * info['road_forward_y'])
    
    car_vec_norm_x = info['car_forward_x'] / (sqrt(info['car_forward_x']**2 + info['car_forward_y']**2) + 1e-6)
    car_vec_norm_y = info['car_forward_y'] / (sqrt(info['car_forward_x']**2 + info['car_forward_y']**2) + 1e-6)
    
    raw_heading_dot = (car_vec_norm_x * info['road_forward_x']) + (car_vec_norm_y * info['road_forward_y'])
    
    # Distance calculation
    dx = info['vehicle_loc_x'] - info['lane_center_x']
    dy = info['vehicle_loc_y'] - info['lane_center_y']
    raw_lane_distance = sqrt(dx**2 + dy**2)
    
    # 3D Dot product for rolling backward check
    v_dot_fwd = (info['velocity_x'] * info['car_forward_x']) + \
                (info['velocity_y'] * info['car_forward_y']) + \
                (info['velocity_z'] * info['car_forward_z'])

    # 1. Progress Engine
    progress_score = clip(raw_progress / cfg.TARGET_SPEED_MS, 0.0, 1.0) * cfg.WEIGHT_PROGRESS
    
    # 2. Alignment Engine
    centering_score = exp(-cfg.LANE_ALPHA * raw_lane_distance) * cfg.WEIGHT_CENTERING
    heading_score = raw_heading_dot * cfg.WEIGHT_HEADING
    
    # 3. Smoothness Penalties
    steer_penalty = -(info['steer_change'] * cfg.PENALTY_STEER_DELTA)
    throttle_penalty = -(info['throttle_change'] * cfg.PENALTY_THROTTLE_DELTA)
    
    # 4. Behavioral & Safety Penalties
    # Cast boolean/int flags to float to prevent PyTorch type mismatch errors
    if is_tensor:
        pedal_flag = info['is_pedal_overlap'].float()
        is_rolling = (v_dot_fwd < -0.5).float()
        is_stalling = (speed_ms < cfg.STALL_SPEED_THRESHOLD).float()
        crash_mask = info['is_terminal_crash'].float()
        lane_flag = info['is_lane_invaded'].float()
    else:
        pedal_flag = float(info['is_pedal_overlap'])
        is_rolling = 1.0 if v_dot_fwd < -0.5 else 0.0
        is_stalling = 1.0 if speed_ms < cfg.STALL_SPEED_THRESHOLD else 0.0
        crash_mask = float(info['is_terminal_crash'])
        lane_flag = float(info['is_lane_invaded'])

    pedal_penalty = pedal_flag * cfg.PENALTY_PEDAL_OVERLAP
    rolling_penalty = is_rolling * cfg.PENALTY_ROLLING_BACKWARD
    lane_penalty = lane_flag * cfg.PENALTY_LANE_INVASION
    stall_penalty = is_stalling * cfg.PENALTY_STALLING

    # Sum everything up
    total_reward = (progress_score + centering_score + heading_score + 
                    steer_penalty + throttle_penalty + 
                    pedal_penalty + rolling_penalty + lane_penalty + stall_penalty)
                    
    # 0. Terminal Override (Crash)
    final_reward = (crash_mask * cfg.PENALTY_TERMINAL_CRASH) + ((1.0 - crash_mask) * total_reward)

    # THE BREAKDOWN METRICS
    # We only need the breakdown dictionary during environment stepping (is_tensor=False)
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
        final_reward = float(final_reward)

    return final_reward, metrics