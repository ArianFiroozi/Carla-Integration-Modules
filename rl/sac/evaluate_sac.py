# File: rl/sac/eval_sac.py

import time
from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from CarlaEnv.env import CarlaEnv
from agents.sac.sac_agent import SACAgent
from config import sac_config as cfg
from config import bc_config
import datetime
from torch.utils.tensorboard import SummaryWriter
import carla
import cv2
from config import general_config
from utils.reward_compiler import compile_reward
from utils.obs_wrapper import CarlaObsWrapper
from utils.seed_utils import seed_everything
import random
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

DEVICE = cfg.DEVICE


def load_norm_stats_from_bc_checkpoint():
    """Try to find normalization stats from the BC experiment folder."""
    try:
        bc_ckpt = Path(cfg.BC_CHECKPOINT_PATH)
        exp_dir = bc_ckpt.parents[1]
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            stats = data.get("dataset_meta", {}).get("normalization_stats", {})
            return stats
    except Exception:
        pass
    return {}


def get_latest_experiment(root):
    """Find the latest experiment directory."""
    folders = [f for f in Path(root).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No experiment folders found.")
    return sorted(folders)[-1]


def get_best_or_last_checkpoint(model_dir):
    """Get best model or latest checkpoint."""
    best = model_dir / "best_model.pt"
    if best.exists():
        return best, "best_model"
    
    ckpts = list(model_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    ckpts_sorted = sorted(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts_sorted[-1], ckpts_sorted[-1].stem


def resolve_paths(args):
    """Resolve model path and experiment directory."""
    if args.model_path:
        model_path = Path(args.model_path)
        exp_dir = model_path.parents[1]
        config_path = exp_dir / "config.json"
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        return model_path, config_path, eval_dir

    root = Path(args.experiments_root)
    exp_dir = root / args.exp_id if args.exp_id else get_latest_experiment(root)
    model_path, _ = get_best_or_last_checkpoint(exp_dir / "models")
    config_path = exp_dir / "config.json"
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    return model_path, config_path, eval_dir


# FORCE JPG FRAME DUMPING — bypass cv2.VideoWriter entirely.
# On the lab server, cv2.VideoWriter reports isOpened()=True for mp4v and then SEGFAULTS
# natively on the first write() (confirmed 2026-07-05: "[VIDEO] Recording ... mp4v" prints,
# process dies silently — the crash is in OpenCV's videoio/ffmpeg DLL, below Python, so no
# try/except can catch it). JPG encoding uses a different native path (imgcodecs) and works.
# Frames land in <video_name>_frames/ next to the intended video path; stitch with ffmpeg
# (the exact command is printed at record time). Set False to re-try real video encoding
# (e.g. after `pip install --force-reinstall opencv-python` on the server).
FORCE_JPG_FRAMES = True


class RobustVideoRecorder:
    """
    Crash-proof video sink for CARLA camera callbacks.

    CARLA runs sensor callbacks on a background thread: any exception or native fault there
    (unsupported codec, frame-size mismatch, non-contiguous buffer) kills the whole process
    SILENTLY — no traceback, straight back to the prompt. This recorder therefore:
      - initializes the cv2.VideoWriter LAZILY from the first real frame, so the writer's
        dimensions always match what the camera actually delivers;
      - tries a ladder of codecs (mp4v -> XVID -> MJPG) and trusts none until isOpened();
      - writes contiguous BGR arrays only;
      - never lets an exception escape: on any failure it degrades to dumping JPG frames
        that can be stitched with ffmpeg later.
    """

    CODEC_LADDER = [("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]

    def __init__(self, save_path, fps=20, force_frames=None):
        self.save_path = Path(save_path)
        self.fps = fps
        self.force_frames = FORCE_JPG_FRAMES if force_frames is None else force_frames
        self.writer = None
        self.frames_dir = None   # set when in JPG-fallback mode
        self.frame_idx = 0
        self.closed = False
        self.final_path = None

    def _open_writer(self, w, h):
        for fourcc, ext in self.CODEC_LADDER:
            path = self.save_path.with_suffix(ext)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, (w, h))
            if writer.isOpened():
                print(f"[VIDEO] Recording {w}x{h}@{self.fps} with codec {fourcc} -> {path.name}")
                self.final_path = path
                return writer
            writer.release()
            try:
                if path.exists():
                    path.unlink()   # remove the 0-byte stub a failed writer leaves behind
            except OSError:
                pass
            print(f"[VIDEO WARN] Codec {fourcc} not available on this machine, trying next...")
        return None

    def _fallback_to_frames(self):
        self.frames_dir = self.save_path.parent / (self.save_path.stem + "_frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        reason = "FORCE_JPG_FRAMES=True (cv2.VideoWriter bypassed)" if self.force_frames \
                 else "no working video codec"
        print(f"[VIDEO] Dumping JPG frames ({reason}) -> {self.frames_dir}")
        print(f"[VIDEO] Stitch later with:\n"
              f"  ffmpeg -framerate {self.fps} -i \"{self.frames_dir}\\frame_%06d.jpg\" "
              f"-c:v libx264 -pix_fmt yuv420p \"{self.save_path.with_suffix('.mp4')}\"")

    def write(self, frame_bgr):
        if self.closed:
            return
        try:
            # cv2's native writer can crash on reversed-stride/non-contiguous views
            frame_bgr = np.ascontiguousarray(frame_bgr)
            if self.writer is None and self.frames_dir is None:
                if self.force_frames:
                    # cv2.VideoWriter is NEVER touched in this mode (native segfault on
                    # this server's videoio backend) — straight to JPG dumping.
                    self._fallback_to_frames()
                else:
                    h, w = frame_bgr.shape[:2]
                    self.writer = self._open_writer(w, h)
                    if self.writer is None:
                        self._fallback_to_frames()
            if self.writer is not None:
                self.writer.write(frame_bgr)
            else:
                cv2.imwrite(str(self.frames_dir / f"frame_{self.frame_idx:06d}.jpg"), frame_bgr)
            self.frame_idx += 1
        except Exception as e:
            # NEVER raise out of a CARLA sensor callback thread — degrade instead.
            print(f"[VIDEO WARN] Frame write failed ({e}); switching to JPG frame fallback.")
            try:
                if self.writer is not None:
                    self.writer.release()
            except Exception:
                pass
            self.writer = None
            if self.frames_dir is None:
                try:
                    self._fallback_to_frames()
                except Exception as e2:
                    print(f"[VIDEO WARN] JPG fallback also failed ({e2}); recording disabled.")
                    self.closed = True

    def release(self):
        self.closed = True
        if self.writer is not None:
            try:
                self.writer.release()
                print(f"[VIDEO] Saved {self.frame_idx} frames -> {self.final_path}")
            except Exception as e:
                print(f"[VIDEO WARN] Writer release failed: {e}")
            self.writer = None
        elif self.frames_dir is not None:
            print(f"[VIDEO] Dumped {self.frame_idx} JPG frames -> {self.frames_dir}")


def create_third_person_camera(env, save_path, width=640, height=360, fps=20):
    """
    Create a third-person chase camera attached to the ego vehicle.
    Camera positioned behind and above the vehicle.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    # Camera settings
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("enable_postprocess_effects", "True")

    # Position camera behind and above the vehicle
    # x=-6 (6 meters behind), z=3 (3 meters above ground)
    # pitch=-15 (slightly looking down)
    cam_transform = carla.Transform(
        carla.Location(x=-6, z=3),
        carla.Rotation(pitch=-15)
    )

    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

    recorder = RobustVideoRecorder(save_path, fps=fps)

    def callback(image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3]   # CARLA raw_data is BGRA; dropping alpha yields BGR,
                                      # which is exactly cv2's expected order. (The old code
                                      # additionally reversed channels — that produced RGB,
                                      # i.e. color-swapped videos, AND a non-contiguous view.)
            recorder.write(frame)
        except Exception as e:
            print(f"[VIDEO WARN] Camera callback error: {e}")

    camera.listen(callback)
    return camera, recorder


def create_top_down_camera(env, save_path, width=640, height=360, fps=20):
    """
    Create a top-down camera for a bird's eye view.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")

    # Position camera directly above the vehicle
    cam_transform = carla.Transform(
        carla.Location(z=20),  # 20 meters above
        carla.Rotation(pitch=-90)  # Looking straight down
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
        frame = array[:, :, :3]
        frame = frame[:, :, ::-1]
        video.write(frame)

    camera.listen(callback)
    return camera, video


def update_spectator(env):
    """
    Update the spectator (free camera in CARLA window) to follow the vehicle.
    """
    world = env.world
    ego_vehicle = env.ego_vehicle
    if ego_vehicle is None:
        return

    spectator = world.get_spectator()
    tr = ego_vehicle.get_transform()
    forward = tr.get_forward_vector()

    # Position spectator behind and above
    cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
    cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))


def run_eval_episode(env, agent, wrapper, max_steps, record_video=False, video_path=None, update_spectator_flag=True, seed=None):
    """Run a single evaluation episode.
    If seed is provided, pass it to env.reset for deterministic initial state.
    """
    obs, _ = env.reset(seed=seed)
    wrapper.reset()

    # Setup camera and video recording if requested
    camera = None
    video = None
    if record_video and video_path is not None:
        camera, video = create_third_person_camera(env, video_path)

    rewards = []
    action_history = []
    terminated_flag = False
    truncated_flag = False

    for t in range(max_steps):
        grid, scalars = wrapper.preprocess(obs)
        grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

        # Use deterministic actions for evaluation
        action = agent.select_action(grid_t, scalars_t, evaluate=True)[0]
        
        action_history.append({
            "step": t,
            "raw_action": action.tolist() if hasattr(action, 'tolist') else list(action),
            "env_action": action if isinstance(action, list) else action.tolist()
        })

        obs, raw_reward, terminated, truncated, info = env.step(action)
        reward, _ = compile_reward(info, general_config, is_tensor=False)
        rewards.append(float(reward))

        # Update spectator camera in CARLA window
        if update_spectator_flag:
            update_spectator(env)

        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break

    # Clean up camera
    if camera is not None:
        camera.stop()
        camera.destroy()
    if video is not None:
        video.release()

    return {
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "length": len(rewards),
        "end_reason": "terminated" if terminated_flag else ("truncated" if truncated_flag else "max_steps"),
        "rewards": rewards,
        "actions": action_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC agent")
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--exp_id", type=str, default=None, help="Specific experiment ID to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific model checkpoint")
    parser.add_argument("--experiments_root", type=str, default=str(cfg.SAVE_DIR))
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    parser.add_argument("--record", action="store_true", default=cfg.RECORD_SAC_EVAL_VID,help="Record video of evaluation episodes")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable video recording AND the camera sensor entirely, overriding "
                             "config RECORD_SAC_EVAL_VID. Use on machines where spawning an RGB "
                             "camera segfaults (e.g. remote sessions without a rendering context) "
                             "— metrics-only headless eval, same conditions training ran under.")
    parser.add_argument("--no-spectator", action="store_true", help="Don't update spectator camera")
    parser.add_argument("--watch", action="store_true", help="Render the CARLA window so you can watch the agent (turns off headless no_rendering)")
    parser.add_argument("--compare_with", type=str, default=None, help="Path to a second model checkpoint for paired comparison (Wilcoxon test).")

<<<<<<< HEAD
=======
    seed_everything(bc_config.GLOBAL_SEED)

>>>>>>> b839bdd08d0540886b8073421df83ef8934ad480
    args = parser.parse_args()

    # --no-record beats the config default (argparse can't turn off a default=True store_true)
    if args.no_record:
        args.record = False

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + config
    model_path, config_path, eval_dir = resolve_paths(args)
    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")

    # Load normalization stats
    norm_stats = load_norm_stats_from_bc_checkpoint()
    if not norm_stats:
        print("[WARN] No normalization stats found. Using empty stats.")

    # Create agent and load weights
    agent = SACAgent(device=device)
    agent.load(model_path)
    agent.actor.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    # Create wrapper
    wrapper = CarlaObsWrapper(
        norm_stats=norm_stats, 
        device=device, 
        action_mode="continuous"
    )

    # Create evaluation directory
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_eval_dir = eval_dir / run_stamp
    current_eval_dir.mkdir(exist_ok=True)
    if args.record:
        (current_eval_dir / "videos").mkdir(exist_ok=True)
    print(f"Saving eval logs to: {current_eval_dir}")

    # TensorBoard
    tb_dir = current_eval_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(str(tb_dir))

    # Create environment
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=cfg.CARLA_WALKERS,
        vehicles_count=cfg.CARLA_VEHICLES,
        max_steps=args.max_steps,
        init_speed=cfg.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=cfg.RANDOM_EGO_START_POS,
        random_vehicle_spawn=cfg.RANDOM_VEHICLE_START_POS,
        no_rendering=not (args.watch or args.record),  # watching/recording needs the server to render
    )

    # ---- Generate deterministic episode seeds ----
    master_seed = args.seed
    rng = random.Random(master_seed)
    num_episodes = args.episodes
    episode_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_episodes)]

    all_returns = []
    all_lengths = []
    end_reasons = Counter()
    overall_t0 = time.time()

    print(f"\n{'='*50}")
    print(f"Starting evaluation: {num_episodes} episodes")
    if args.record:
        print("Video recording: ENABLED")
    print(f"{'='*50}\n")

    try:
        for ep in range(num_episodes):
            print(f"=== Episode {ep+1}/{num_episodes} ===")
            
            # Setup video path if recording
            video_path = None
            if args.record:
                video_path = current_eval_dir / "videos" / f"episode_{ep+1:03d}.mp4"
            
            result = run_eval_episode(
                env=env,
                agent=agent,
                wrapper=wrapper,
                max_steps=args.max_steps,
                record_video=args.record,
                video_path=video_path,
                update_spectator_flag=not args.no_spectator,
                seed=episode_seeds[ep],
            )

            # Save episode results
            episode_path = current_eval_dir / f"episode_{ep+1:03d}.json"
            with open(episode_path, "w") as f:
                json.dump({
                    "return": result["return"],
                    "mean_reward": result["mean_reward"],
                    "length": result["length"],
                    "end_reason": result["end_reason"],
                }, f, indent=2)

            all_returns.append(result["return"])
            all_lengths.append(result["length"])
            end_reasons[result["end_reason"]] += 1

            print(
                f"Episode {ep+1}: return={result['return']:.2f}, "
                f"mean_reward={result['mean_reward']:.3f}, "
                f"length={result['length']}, "
                f"end={result['end_reason']}"
            )
            if args.record:
                print(f"  Video saved: {video_path}")

            # TensorBoard episode metrics
            tb_writer.add_scalar("eval/episode_return", result["return"], ep + 1)
            tb_writer.add_scalar("eval/episode_length", result["length"], ep + 1)

        total_time = time.time() - overall_t0

        # Summary statistics
        primary_returns = np.array(all_returns)
        primary_lengths = np.array(all_lengths)
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Episodes: {num_episodes}")
        print(f"Avg return: {primary_returns.mean():.2f} ± {primary_returns.std():.2f}")
        print(f"Min return: {primary_returns.min():.2f}")
        print(f"Max return: {primary_returns.max():.2f}")
        print(f"Avg length: {primary_lengths.mean():.1f} ± {primary_lengths.std():.1f}")
        print(f"End reasons: {dict(end_reasons)}")
        print(f"Wall time: {total_time:.1f}s")
        print(f"Model: {model_path}")

        # Save summary
        summary = {
            "model_path": str(model_path),
            "episodes": num_episodes,
            "avg_return": float(primary_returns.mean()),
            "std_return": float(primary_returns.std()),
            "min_return": float(primary_returns.min()),
            "max_return": float(primary_returns.max()),
            "avg_length": float(primary_lengths.mean()),
            "std_length": float(primary_lengths.std()),
            "end_reasons": dict(end_reasons),
            "wall_time_sec": total_time,
            "carla_vehicles": cfg.CARLA_VEHICLES,
            "carla_walkers": cfg.CARLA_WALKERS,
            "video_recorded": args.record,
        }

        summary_path = current_eval_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # TensorBoard summary metrics
        tb_writer.add_scalar("eval/avg_return", primary_returns.mean(), 0)
        tb_writer.add_scalar("eval/std_return", primary_returns.std(), 0)
        tb_writer.add_scalar("eval/avg_length", primary_lengths.mean(), 0)
        
        tb_writer.flush()
        tb_writer.close()

        print(f"\nSaved summary to: {summary_path}")

    finally:
        pass  # We'll close env later after possible comparison run

    # ---------- Comparison Model (if requested) ----------
    if args.compare_with:
        print("\n" + "="*50)
        print("COMPARISON MODE")
        print(f"Loading comparison model from: {args.compare_with}")
        compare_model_path = Path(args.compare_with)
        if not compare_model_path.exists():
            print(f"[ERROR] Comparison model not found at {compare_model_path}")
        else:
            # Load second agent (no tensorboard logs, no videos, no per-episode saves)
            compare_agent = SACAgent(device=device)
            compare_agent.load(compare_model_path)
            compare_agent.actor.eval()

            compare_returns = []
            compare_lengths = []

            t0_compare = time.time()
            for ep in range(num_episodes):
                print(f"Comparison Episode {ep+1}/{num_episodes}")
                # Run same episode seed, no video (unless you want, but we skip)
                result_cmp = run_eval_episode(
                    env=env,
                    agent=compare_agent,
                    wrapper=wrapper,
                    max_steps=args.max_steps,
                    record_video=False,           # no video for comparison
                    video_path=None,
                    update_spectator_flag=not args.no_spectator,
                    seed=episode_seeds[ep],
                )
                compare_returns.append(result_cmp["return"])
                compare_lengths.append(result_cmp["length"])

            compare_time = time.time() - t0_compare
            compare_returns = np.array(compare_returns)
            compare_lengths = np.array(compare_lengths)

            # Wilcoxon signed-rank test on returns and lengths
            stat_return, p_return = scipy_stats.wilcoxon(primary_returns, compare_returns)
            stat_length, p_length = scipy_stats.wilcoxon(primary_lengths, compare_lengths)

            print("\n--- Comparison Results ---")
            print(f"Primary model   : mean return={primary_returns.mean():.2f} ± {primary_returns.std():.2f}, mean length={primary_lengths.mean():.1f} ± {primary_lengths.std():.1f}")
            print(f"Comparison model: mean return={compare_returns.mean():.2f} ± {compare_returns.std():.2f}, mean length={compare_lengths.mean():.1f} ± {compare_lengths.std():.1f}")
            print(f"Mean difference (primary - comparison): return={primary_returns.mean()-compare_returns.mean():.2f}, length={primary_lengths.mean()-compare_lengths.mean():.1f}")
            print(f"Wilcoxon test on returns: statistic={stat_return:.2f}, p-value={p_return:.4f}")
            print(f"Wilcoxon test on lengths: statistic={stat_length:.2f}, p-value={p_length:.4f}")

            # Plot distributions side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(primary_returns, alpha=0.7, label="Primary", bins=max(10, num_episodes//3), color="steelblue")
            axes[0].hist(compare_returns, alpha=0.7, label="Comparison", bins=max(10, num_episodes//3), color="darkorange")
            axes[0].axvline(primary_returns.mean(), color="steelblue", linestyle="--", linewidth=1.5)
            axes[0].axvline(compare_returns.mean(), color="darkorange", linestyle="--", linewidth=1.5)
            axes[0].set_xlabel("Return")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(f"Returns distribution\n(Wilcoxon p={p_return:.3f})")
            axes[0].legend()

            axes[1].hist(primary_lengths, alpha=0.7, label="Primary", bins=max(10, num_episodes//3), color="steelblue")
            axes[1].hist(compare_lengths, alpha=0.7, label="Comparison", bins=max(10, num_episodes//3), color="darkorange")
            axes[1].axvline(primary_lengths.mean(), color="steelblue", linestyle="--", linewidth=1.5)
            axes[1].axvline(compare_lengths.mean(), color="darkorange", linestyle="--", linewidth=1.5)
            axes[1].set_xlabel("Length")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title(f"Lengths distribution\n(Wilcoxon p={p_length:.3f})")
            axes[1].legend()

            plt.tight_layout()
            plot_path = current_eval_dir / "comparison_plot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Comparison plot saved to: {plot_path}")

            # Also save a comparison JSON
            comparison_summary = {
                "primary_model": str(model_path),
                "comparison_model": str(compare_model_path),
                "primary_return_mean": float(primary_returns.mean()),
                "primary_return_std": float(primary_returns.std()),
                "comparison_return_mean": float(compare_returns.mean()),
                "comparison_return_std": float(compare_returns.std()),
                "primary_length_mean": float(primary_lengths.mean()),
                "primary_length_std": float(primary_lengths.std()),
                "comparison_length_mean": float(compare_lengths.mean()),
                "comparison_length_std": float(compare_lengths.std()),
                "wilcoxon_return_pvalue": float(p_return),
                "wilcoxon_length_pvalue": float(p_length),
                "episodes": num_episodes,
            }
            comparison_json_path = current_eval_dir / "comparison_summary.json"
            with open(comparison_json_path, "w") as f:
                json.dump(comparison_summary, f, indent=2)

    # Final cleanup
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()