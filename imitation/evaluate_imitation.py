import time
from collections import Counter
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

from utils.obs_wrapper import CarlaObsWrapper, speed_map, turn_map

ACTION_MODE = bc_config.ACTION_MODE
DEVICE = bc_config.DEVICE
DEBUG_PRINT_STEPS = bc_config.DEBUG_PRINT_STEPS


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


debug_counter = 0

def predict_action(policy, obs, wrapper: CarlaObsWrapper):
    global debug_counter

    grid, scalars = wrapper.preprocess(obs)
    grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

    with torch.no_grad():
        if ACTION_MODE == "discrete":
            logits_speed, logits_turn = policy(grid_t, scalars_t)
            speed = torch.argmax(logits_speed, dim=1).item()
            turn  = torch.argmax(logits_turn, dim=1).item()
            env_action = wrapper.map_action_for_env((speed, turn))

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
            pred = policy(grid_t, scalars_t)
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

            env_action = action

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


def run_episode(env, policy, wrapper, max_steps=2000, render_log_every=200, video_path=None):
    obs, _ = env.reset()
    wrapper.reset()

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
        env_action, action_log = predict_action(policy, obs, wrapper)
        obs, reward, terminated, truncated, info = env.step(env_action)

        if action_log is not None:
            action_counts[action_log] += 1

        update_spectator(env)
        rewards.append(float(reward))

        if (t + 1) % render_log_every == 0:
            elapsed = time.time() - t0
            fps = (t + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[t={t+1}] mean_reward(last {render_log_every})={np.mean(rewards[-render_log_every:]):.2f} fps={fps:.1f}")

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

    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=DEVICE, action_mode=ACTION_MODE)

    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_eval_dir = eval_dir / run_stamp
    current_eval_dir.mkdir(exist_ok=True)
    print("Saving eval logs to:", current_eval_dir)

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
        random_ego_spawn=bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=bc_config.RANDOM_VEHICLE_START_POS
    )
    print("Action mode:", ACTION_MODE)
    print("Device:", DEVICE)

    all_returns = []
    all_lengths = []
    end_reasons = Counter()
    global_action_counts = Counter()
    overall_t0 = time.time()
    num_episodes = args.episodes

    try:
        for ep in range(num_episodes):
            print(f"\n=== Episode {ep+1}/{num_episodes} ===")
            video_path = current_eval_dir / f"episode_{ep+1:03d}.mp4"
            if not bc_config.RECORD_BC_EVAL_VID:
                video_path = None

            result = run_episode(
                env,
                policy,
                wrapper=wrapper,
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
                    "action_counts": { f"{k[0]}_{k[1]}": v for k, v in result["action_counts"].items() }
                }, f, indent=2)

            all_returns.append(result["return"])
            all_lengths.append(result["length"])
            end_reasons[result["end_reason"]] += 1
            global_action_counts.update(result["action_counts"])

            print(
                f"Episode {ep+1}: return={result['return']:.2f}, "
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
            summary["global_action_counts"] = {str(k): v for k, v in global_action_counts.items()}

        summary_path = current_eval_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        tb_writer.add_scalar("eval/avg_return", np.mean(all_returns), 0)
        tb_writer.add_scalar("eval/avg_length", np.mean(all_lengths), 0)
        tb_writer.add_scalar("eval/carla_vehicles", bc_config.CARLA_VEHICLES, 0)
        tb_writer.add_scalar("eval/smooth_steering", float(bc_config.SMOOTH_STEERING), 0)
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
