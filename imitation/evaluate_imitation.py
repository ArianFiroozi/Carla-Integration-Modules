import time
from collections import Counter, deque
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from CarlaEnv.env import CarlaEnv
from agents.bc.imitation_policy import ImitationPolicy
from config import bc_config
import datetime
from torch.utils.tensorboard import SummaryWriter 
import carla
import cv2

ACTION_MODE = bc_config.ACTION_MODE
DEVICE = bc_config.DEVICE
SIMPLIFIED_ACTION_SPACE = bc_config.SIMPLIFY_ACTIONS
DEBUG_PRINT_STEPS = bc_config.DEBUG_PRINT_STEPS

if SIMPLIFIED_ACTION_SPACE:
    speed_map = bc_config.SIMPLIFY_SPEED_MAP
    turn_map = bc_config.SIMPLIFY_TURN_MAP
else:
    speed_map = bc_config.SPEED_MAP
    turn_map = bc_config.TURN_MAP


def _normalize_value(v, key, norm_stats):
    """
    Unified normalization function that mirrors the dataset's logic.
    This is now used for both grid speeds and scalars.
    """
    # This check ensures the function can handle both single float values and numpy arrays
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
    
    # Default for 'fixed' mode scalars, or if stats are missing
    return v if "speed" in key else np.zeros_like(v)

def wrap_angle_pi(angle):
    """Wrap angle to [-pi, pi]. Works for scalar or numpy array."""
    angle = np.asarray(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi



class ObsHistory:
    """Maintains a rolling window of grid observations."""

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

        if torch.is_tensor(p): p = p.cpu().numpy()
        if torch.is_tensor(sx): sx = sx.cpu().numpy()
        if torch.is_tensor(sy): sy = sy.cpu().numpy()
        # If history empty, fill with first frame
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

        grid = np.concatenate(frames, axis=0).astype(np.float32)
        return grid


def create_video_recorder(env, save_path, width=640, height=360, fps=20):
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")

    cam_transform = carla.Transform(
        carla.Location(x=-6, z=3),
        carla.Rotation(pitch=-15)
    )

    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

    video = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame = array[:, :, :3]   # remove alpha
        frame = frame[:, :, ::-1] # BGRA → BGR
        video.write(frame)

    camera.listen(callback)

    return camera, video


def compute_spatial_distances(grid_2d):
    """
    Computes the distance to the nearest obstacle in 4 directions (front, back, left, right).
    """
    MAX_FRONT = 10.0
    MAX_BACK = 10.0
    MAX_SIDE = 5.0

    dist_front = MAX_FRONT
    dist_back  = MAX_BACK
    dist_right = MAX_SIDE
    dist_left  = MAX_SIDE

    # Find the ego vehicle's position (looking for 3, as we converted 9 to 3)
    ego_positions = np.argwhere(grid_2d == 3)
    
    if len(ego_positions) == 0:
        # Return default max distances if ego is not found
        return np.array([dist_front, dist_back, dist_left, dist_right], dtype=np.float32)

    ego_y, ego_x = ego_positions[0]

    # Front
    column_in_front = grid_2d[ego_y + 1:, ego_x]
    obstacles_ahead = np.argwhere((column_in_front == 1) | (column_in_front == 2))
    if len(obstacles_ahead) > 0:
        closest_obs_y = obstacles_ahead[0][0]
        dist_front = float(closest_obs_y + 1)

    # Back
    column_behind = grid_2d[:ego_y, ego_x][::-1]
    obstacles_behind = np.argwhere((column_behind == 1) | (column_behind == 2))
    if len(obstacles_behind) > 0:
        closest_obs_y = obstacles_behind[0][0]
        dist_back = float(closest_obs_y + 1)

    # Left
    row_to_left = grid_2d[ego_y, :ego_x][::-1]
    obstacles_left = np.argwhere((row_to_left == 1) | (row_to_left == 2))
    if len(obstacles_left) > 0:
        closest_obs_x = obstacles_left[0][0]
        dist_left = float(closest_obs_x + 1)

    # Right
    row_to_right = grid_2d[ego_y, ego_x + 1:]
    obstacles_right = np.argwhere((row_to_right == 1) | (row_to_right == 2))
    if len(obstacles_right) > 0:
        closest_obs_x = obstacles_right[0][0]
        dist_right = float(closest_obs_x + 1)
        
    return np.array([dist_front, dist_back, dist_left, dist_right], dtype=np.float32)


def extract_grid_and_scalars(obs, history: ObsHistory, norm_stats):
    # Fix the shape of the presence grid if the env flattens it
    presence = np.array(obs["presence"])
    if presence.ndim == 1:
        presence = presence.reshape(25, 11)
    elif presence.ndim == 3:
        presence = presence.squeeze()
        
    # Fix the ego vehicle encoding (Env outputs 9, model expects 3)
    presence[presence == 9] = 3
    obs["presence"] = presence # Update the dict so history buffer gets the correct grid

    # Initialize with default values
    dist_front, dist_back, dist_left, dist_right = 10.0, 10.0, 5.0, 5.0

    if getattr(bc_config, "USE_SPATIAL_FEATURES", False):
        dists = compute_spatial_distances(obs["presence"])
        dist_front = dists[0]
        dist_back  = dists[1]
        dist_left  = dists[2]
        dist_right = dists[3]
        
        dist_front = _normalize_value(dist_front, "obs_dist_front", norm_stats)
        dist_back  = _normalize_value(dist_back, "obs_dist_back", norm_stats)
        dist_left  = _normalize_value(dist_left, "obs_dist_left", norm_stats)
        dist_right = _normalize_value(dist_right, "obs_dist_right", norm_stats)
    
    # Update history and get stacked grid
    history.update(obs)
    grid = history.get_grid()
    
    # Process scalars
    raw_lane_angle = wrap_angle_pi(obs["lane_angle"][0])
    lane_angle = _normalize_value(raw_lane_angle, "obs_lane_angle", norm_stats)
    lane_pos = _normalize_value(obs["ego_in_lane_position_x"][0], "obs_ego_in_lane_position_x", norm_stats)
    speed_x = _normalize_value(obs["ego_speed_x"][0], "obs_ego_speed_x", norm_stats)
    speed_y = _normalize_value(obs["ego_speed_y"][0], "obs_ego_speed_y", norm_stats)
    
    if getattr(bc_config, "USE_SPATIAL_FEATURES", False):
        scalars = np.array([lane_angle, lane_pos, speed_x, speed_y, dist_front, dist_back, dist_left, dist_right], dtype=np.float32)
    else:
        scalars = np.array([lane_angle, lane_pos, speed_x, speed_y], dtype=np.float32)
        
    return grid, scalars


def map_action_for_env(action):
    """
    Convert (speed_idx, turn_idx) from the discrete head into
    integer action indices expected by CarlaEnv.
    """
    
    speed, turn = action

    if SIMPLIFIED_ACTION_SPACE:
        if speed == 3: speed = 4
        if turn == 2: turn = 3

    return [speed, turn]


prev_steer = 0.0
def process_continuous_output(out):
    """
    model output → [throttle, brake, steer]
    out: np.array of shape (3,)
    """
    global prev_steer

    throttle = float(np.clip(out[0], 0.0, 1.0))
    brake = float(np.clip(out[1], 0.0, 1.0))
    steer = float(np.clip(out[2], -1.0, 1.0))
    
    if throttle < 0.13 and throttle > throttle > 0.05:
        throttle = 0.13
    
    
    # prevent throttle+brake conflict
    if brake > 0.1:
        throttle = 0.0
    else:
        brake = 0.0

    if bc_config.SMOOTH_STEERING:
        steer = 0.7 * prev_steer + 0.3 * steer
        prev_steer = steer
        steer = np.clip(steer, -1.0, 1.0)
        
    return [throttle, brake, steer]


debug_counter = 0

def predict_action(policy, obs, history, norm_stats):
    """
    Run one policy step.
    obs is assumed to be (grid, scalars), as returned by CarlaEnv.
    Returns:
        env_action: list usable by env.step(...)
        action_log: (speed, turn) for discrete, or None for continuous
    """
    global debug_counter

    grid, scalars = extract_grid_and_scalars(obs, history, norm_stats)
    grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    scalars = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if ACTION_MODE == "discrete":
            logits_speed, logits_turn = policy(grid, scalars)
            speed = torch.argmax(logits_speed, dim=1).item()
            turn = torch.argmax(logits_turn, dim=1).item()
            env_action = map_action_for_env((speed, turn))
            if debug_counter < DEBUG_PRINT_STEPS:
                print(
                    f"lane_angle={obs['lane_angle'][0]:.3f}, "
                    f"lane_pos={obs['ego_in_lane_position_x'][0]:.3f}, "
                    f"vx={obs['ego_speed_x'][0]:.3f}, "
                    f"vy={obs['ego_speed_y'][0]:.3f}"
                )
            debug_counter += 1
            return env_action, (speed, turn)

        else:
            pred = policy(grid, scalars)
            if isinstance(pred, tuple):
                mean, std = pred
                action = mean.cpu().numpy()[0]
                uncertainty = std.cpu().numpy()[0]
            else:
                action = pred.cpu().numpy()[0]
                uncertainty = None

            if debug_counter < DEBUG_PRINT_STEPS:
                print(
                    f"[DEBUG] raw model output: "
                    f"throttle={action[0]:.3f}, brake={action[1]:.3f}, steer={action[2]:.3f}"
                )
                if uncertainty is not None:
                    print(f"σ_throttle={uncertainty[0]:.3f}, σ_brake={uncertainty[1]:.3f}, σ_steer={uncertainty[2]:.3f}")
            env_action = process_continuous_output(action)

            if debug_counter < DEBUG_PRINT_STEPS:
                print(
                    f"[DEBUG] env action: "
                    f"throttle={env_action[0]:.3f}, brake={env_action[1]:.3f}, steer={env_action[2]:.3f}"
                )
                print(
                    f"lane_angle={obs['lane_angle'][0]:.3f}, "
                    f"lane_pos={obs['ego_in_lane_position_x'][0]:.3f}, "
                    f"vx={obs['ego_speed_x'][0]:.3f}, "
                    f"vy={obs['ego_speed_y'][0]:.3f}"
                )

                print("-" * 40)

            debug_counter += 1

            return env_action, None


def update_spectator(env):
    world = env.world
    ego_vehicle = env.ego_vehicle
    if ego_vehicle is None:
        return

    spectator = world.get_spectator()
    tr = ego_vehicle.get_transform()
    forward = tr.get_forward_vector()

    cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
    cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))
    

    
def run_episode(env, policy, norm_stats, max_steps=2000, render_log_every=200, video_path=None):
    obs, _ = env.reset()
    history = ObsHistory(one_hot=bc_config.USE_ONE_HOT_GRID, window_size= bc_config.WINDOW_SIZE, norm_stats=norm_stats)
    camera = None
    video = None
    if video_path is not None:
        camera, video = create_video_recorder(env, video_path)  
    rewards = []
    action_counts = Counter()
    terminated_flag = False
    truncated_flag = False

    t0 = time.time()

    for t in range(max_steps):
        env_action, action_log = predict_action(policy, obs, history, norm_stats)
        obs, reward, terminated, truncated, info = env.step(env_action)

        if action_log is not None:
            action_counts[action_log] += 1
            
        update_spectator(env)
        
        rewards.append(float(reward))
        if (t + 1) % render_log_every == 0:
            elapsed = time.time() - t0
            fps = (t + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[t={t+1}] mean_reward(last {render_log_every})="
                f"{np.mean(rewards[-render_log_every:]):.2f} fps={fps:.1f}"
            )
        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break

    ep_len = len(rewards)
    if camera is not None:
        camera.stop()
        camera.destroy()

    if video is not None:
        video.release()

    return {
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "length": ep_len,
        "end_reason": "terminated" if terminated_flag else ("truncated" if truncated_flag else "max_steps"),
        "action_counts": action_counts,
    }


def get_latest_experiment(root):
    folders = [f for f in Path(root).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No experiment folders found.")
    return sorted(folders)[-1]

def get_best_or_last_checkpoint(model_dir):
    best = model_dir / "best_model.pt"
    if best.exists():
        return best
    ckpts = list(model_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    ckpts_sorted = sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts_sorted[-1]

def resolve_paths(args):
    # direct model override
    if args.model_path:
        model_path = Path(args.model_path)
        exp_dir = model_path.parents[1]
        config_path = exp_dir / "config.json"
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        return model_path, config_path, eval_dir

    root = Path(args.experiments_root)
    exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
    model_path = get_best_or_last_checkpoint(exp_dir / "models")
    config_path = exp_dir / "config.json"
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    return model_path, config_path, eval_dir

def load_policy_from_checkpoint(model_path, config_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    mode = ckpt["mode"]
    scalar_dim = ckpt["scalar_dim"]
    grid_channels = ckpt["grid_channels"]
    kwargs = {
    "grid_channels": grid_channels,
    "scalar_dim": scalar_dim,
    "cnn_channels": bc_config.CNN_CHANNELS,
    "kernel_sizes": bc_config.KERNEL_SIZES,
    "head_n_mlp_layers": bc_config.HEAD_N_MLP_LAYERS,
    "head_mlp_hidden_size": bc_config.HEAD_MLP_HIDDEN_SIZE,
    "scalar_n_mlp_layers": bc_config.SCALAR_N_MLP_LAYERS,
    "scalar_mlp_hidden_size": bc_config.SCALAR_MLP_HIDDEN_SIZE,
    "latent_dim": bc_config.LATENT_DIM,
    "decoupled": bc_config.IS_DECOUPLED 
    }

    if mode == "discrete":
        n_speed = ckpt["n_speed"]
        n_turn = ckpt["n_turn"]
        policy = ImitationPolicy(
            mode="discrete",
            n_speed=n_speed,
            n_turn=n_turn,
            **kwargs
        ).to(DEVICE)
    else:
        policy = ImitationPolicy(
            mode="continuous",
            is_gaussian=bc_config.IS_GAUSSIAN,
            **kwargs
        ).to(DEVICE)

    expected_channels = bc_config.WINDOW_SIZE * (5 if bc_config.USE_ONE_HOT_GRID else 3)

    assert expected_channels == grid_channels, \
        f"Grid channel mismatch! model={grid_channels}, expected={expected_channels}"

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    print("\nLoaded model:")
    print("  mode:", mode)
    print("  grid_channels:", grid_channels)
    print("  scalar_dim:", scalar_dim)
    if mode == "discrete":
        print("  n_speed:", n_speed)
        print("  n_turn:", n_turn)
    return policy, cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=bc_config.CARLA_MAP_PATH)
    parser.add_argument("--episodes", type=int, default=bc_config.EVAL_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=bc_config.EVAL_MAX_STEPS)
    parser.add_argument("--mode", choices=["discrete", "continuous"], default=bc_config.ACTION_MODE)
    parser.add_argument("--device", default=bc_config.DEVICE)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--experiments_root", type=str, default=bc_config.BC_EXPERIMENT_FOLDER)

    args = parser.parse_args()

    global ACTION_MODE, DEVICE
    ACTION_MODE = args.mode
    DEVICE = args.device

    # Load model + config.json
    model_path, config_path, eval_dir = resolve_paths(args)
    print("Using model:", model_path)
    print("Using config:", config_path)

    policy, cfg = load_policy_from_checkpoint(model_path, config_path)

    norm_stats = cfg.get("dataset_meta", {}).get("normalization_stats", {})
    if not norm_stats:
        print("\n[WARN] No normalization stats found in config.json! Evaluation might fail or behave poorly.\n")
    # create timestamped eval run folder
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_eval_dir = eval_dir / run_stamp
    current_eval_dir.mkdir(exist_ok=True)
    print("Saving eval logs to:", current_eval_dir)
    
    # TensorBoard writer for this eval run
    tb_dir = current_eval_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(str(tb_dir))

    env = CarlaEnv(
        map_path=args.map,
        walkers_count=bc_config.CARLA_WALKERS,
        vehicles_count=bc_config.CARLA_VEHICLES,
        max_steps=args.max_steps,
        init_speed=bc_config.CARLA_INIT_SPEED,
        action_mode=ACTION_MODE,
        random_ego_spawn= bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn= bc_config.RANDOM_VEHICLE_START_POS
    )
    print("Action mode:", ACTION_MODE)
    print("Device:", DEVICE)

    all_returns = []
    all_lengths = []
    end_reasons = Counter()
    global_action_counts = Counter()
    overall_t0 = time.time()
    num_episodes= args.episodes
    try:
        for ep in range(num_episodes):
            print(f"\n=== Episode {ep+1}/{num_episodes} ===")
            video_path = current_eval_dir / f"episode_{ep+1:03d}.mp4"

            result = run_episode(
                env,
                policy,
                norm_stats=norm_stats,
                max_steps=args.max_steps,
                video_path=video_path
            )


            episode_path = current_eval_dir / f"episode_{ep+1:03d}.json"
            with open(episode_path, "w") as f:
                json.dump({
                    "return": result["return"],
                    "mean_reward": result["mean_reward"],
                    "length": result["length"],
                    "end_reason": result["end_reason"],
                    "action_counts": { f"{k[0]}_{k[1]}": v for k, v in result["action_counts"].items() } }, f, indent=2)

            all_returns.append(result["return"])
            all_lengths.append(result["length"])
            end_reasons[result["end_reason"]] += 1
            global_action_counts.update(result["action_counts"])

            print(
                f"Episode {ep+1}: "
                f"return={result['return']:.2f}, "
                f"mean_reward={result['mean_reward']:.3f}, "
                f"length={result['length']}, "
                f"end={result['end_reason']}"
            )
            # TensorBoard episode metrics
            tb_writer.add_scalar("eval/episode_return", result["return"], ep + 1)
            tb_writer.add_scalar("eval/episode_length", result["length"], ep + 1)

        total_time = time.time() - overall_t0

        print("\n===== Summary =====")
        print(f"Episodes: {num_episodes}")
        print(f"Avg return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
        print(f"Avg length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
        print(f"End reasons: {dict(end_reasons)}")
        print(f"Wall time: {total_time:.1f}s")

        if ACTION_MODE == "discrete":
            print("\nTop 10 actions overall:")
            for (speed, turn), v in global_action_counts.most_common(10):
                speed_name = speed_map.get(speed, f"speed_{speed}")
                turn_name = turn_map.get(turn, f"turn_{turn}")
                print(f"{speed_name} | {turn_name} : {v}")


        summary = {
            "episodes": num_episodes,
            "avg_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "avg_length": float(np.mean(all_lengths)),
            "std_length": float(np.std(all_lengths)),
            "end_reasons": dict(end_reasons),
            "wall_time_sec": total_time,
            "action_mode": ACTION_MODE,
            "smooth_steering": bc_config.SMOOTH_STEERING,
            "number of cars": bc_config.CARLA_VEHICLES,
        }

        if ACTION_MODE == "discrete":
            summary["global_action_counts"] = {
                str(k): v for k, v in global_action_counts.items()
            }

        summary_path = current_eval_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # TensorBoard summary metrics
        tb_writer.add_scalar("eval/avg_return", np.mean(all_returns), 0)
        tb_writer.add_scalar("eval/avg_length", np.mean(all_lengths), 0)
        tb_writer.add_scalar("eval/carla_vehicles", bc_config.CARLA_VEHICLES, 0)
        tb_writer.add_scalar("eval/smooth_steering", float(bc_config.SMOOTH_STEERING),0)
        tb_writer.flush()
        tb_writer.close()
        

        print("Saved summary to:", summary_path)
    
    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
