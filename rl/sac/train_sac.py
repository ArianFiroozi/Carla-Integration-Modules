import os
import time
import json
import random
import argparse
import datetime
from pathlib import Path
from utils.seed_utils import seed_everything
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from CarlaEnv.env import CarlaEnv
from agents.sac.sac_agent import SACAgent
from rl.sac.replay_buffer import SACReplayBuffer
from utils.obs_wrapper import CarlaObsWrapper
from config import sac_config as cfg
from config import bc_config  # for wrapper settings

 
# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------


def load_norm_stats_from_bc_checkpoint():
    """
    Try to find normalization stats from the BC experiment folder.
    This is optional. If not found, wrapper will use empty stats.
    """
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


def make_experiment_dir():
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(cfg.SAVE_DIR) / run_stamp
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "tb").mkdir(exist_ok=True)
    return exp_dir


def save_config(exp_dir):
    config_path = exp_dir / "config.json"
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items() if k.isupper()}
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)


# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------
def run_eval_episode(env, agent, wrapper, max_steps):
    obs, _ = env.reset()
    wrapper.reset()

    rewards = []
    terminated_flag = False
    truncated_flag = False

    for t in range(max_steps):
        grid, scalars = wrapper.preprocess(obs)
        grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

        action = agent.select_action(grid_t, scalars_t, evaluate=True)[0]
        env_action = wrapper.process_continuous_output(action)

        obs, reward, terminated, truncated, info = env.step(env_action)
        rewards.append(float(reward))

        if terminated or truncated:
            terminated_flag = terminated
            truncated_flag = truncated
            break

    return {
        "return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "length": len(rewards),
        "end_reason": "terminated" if terminated_flag else ("truncated" if truncated_flag else "max_steps"),
    }


def evaluate(agent, env, wrapper, episodes, max_steps):
    all_returns = []
    all_lengths = []
    end_reasons = {}

    for ep in range(episodes):
        result = run_eval_episode(env, agent, wrapper, max_steps)
        all_returns.append(result["return"])
        all_lengths.append(result["length"])
        end_reasons[result["end_reason"]] = end_reasons.get(result["end_reason"], 0) + 1

    return {
        "avg_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "std_return": float(np.std(all_returns)) if all_returns else 0.0,
        "avg_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "end_reasons": end_reasons,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    args = parser.parse_args()

    # --------------------------- Assumptions ---------------------------
    # We assume CONTINUOUS ACTIONS ONLY (no discrete mode).
    # We use CarlaObsWrapper for preprocessing.
    # -------------------------------------------------------------------

    seed_everything(args.seed)

    exp_dir = make_experiment_dir()
    save_config(exp_dir)
    tb_writer = SummaryWriter(str(exp_dir / "tb"))

    print("Experiment dir:", exp_dir)
    print("Device:", args.device)

    # Load norm stats (optional)
    norm_stats = load_norm_stats_from_bc_checkpoint()
    if not norm_stats:
        print("[WARN] No normalization stats found. Wrapper will use empty stats.")

    # Create environment
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=cfg.CARLA_WALKERS,
        vehicles_count=cfg.CARLA_VEHICLES,
        max_steps=args.max_steps,
        init_speed=cfg.CARLA_INIT_SPEED,
        action_mode="continuous",
        random_ego_spawn=cfg.RANDOM_EGO_START_POS,
        random_vehicle_spawn=cfg.RANDOM_VEHICLE_START_POS
    )

    # Wrapper + Agent + Replay Buffer
    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=args.device, action_mode="continuous")
    agent = SACAgent(device=args.device)
    replay_buffer = SACReplayBuffer(capacity=cfg.REPLAY_BUFFER_SIZE, device=args.device)


    ## TODO: mehdi , here we load the saved bc, you might need to check or even fix how we "save" and load the model 
    ## begin
    
# =========================================================
    # Load BC Weights (Warm Start)
    # =========================================================
    if cfg.LOAD_BC_WEIGHTS and Path(cfg.BC_CHECKPOINT_PATH).exists():
        print(f"Loading BC weights from: {cfg.BC_CHECKPOINT_PATH}")
        # strict=False allows ignoring keys that don't match perfectly (e.g., log_std which is not in deterministic BC)
        agent.load_actor_from_bc(cfg.BC_CHECKPOINT_PATH, strict=False)
    elif cfg.LOAD_BC_WEIGHTS:
        print(f"[WARN] BC checkpoint not found at: {cfg.BC_CHECKPOINT_PATH}")
    # =========================================================
        


    total_steps = 0
    episode = 0
    best_eval_return = -1e9

    obs, _ = env.reset()
    wrapper.reset()

    episode_reward = 0.0
    episode_len = 0
    episode_start = time.time()

    # print("obs" , obs)
    try:
        while total_steps < cfg.MAX_TRAIN_STEPS:
            # Preprocess obs
            grid, scalars = wrapper.preprocess(obs)
            grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

            # Action selection
            if cfg.USE_RANDOM_POLICY_WARMUP and total_steps < cfg.WARMUP_STEPS:
                # random action in env bounds
                action = np.random.uniform(low=cfg.ACTION_LOW, high=cfg.ACTION_HIGH, size=(cfg.ACTION_DIM,))
            else:
                action = agent.select_action(grid_t, scalars_t, evaluate=True)[0]
                # print("actions", action)
                # print("scalars", scalars_t)
                # print("grid", grid_t)




            # env_action = wrapper.process_continuous_output(action)

            # Env step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Preprocess next obs
            next_grid, next_scalars = wrapper.preprocess(next_obs)

            # Store transition (store ENV action, because that is what executed)
            replay_buffer.add(
                grid_obs=grid,
                scalar_obs=scalars,
                action= action,
                reward=reward,
                next_grid_obs=next_grid,
                next_scalar_obs=next_scalars,
                done=done
            )

            obs = next_obs
            episode_reward += float(reward)
            episode_len += 1
            total_steps += 1

            # Updates
            if total_steps >= cfg.UPDATE_AFTER and len(replay_buffer) >= cfg.BATCH_SIZE:
                if total_steps % cfg.UPDATE_EVERY == 0:
                    for _ in range(cfg.GRADIENT_UPDATES):
                        losses = agent.update(replay_buffer)
                        tb_writer.add_scalar("train/critic_loss", losses["critic_loss"], total_steps)
                        tb_writer.add_scalar("train/actor_loss", losses["actor_loss"], total_steps)
                        tb_writer.add_scalar("train/alpha_loss", losses["alpha_loss"], total_steps)
                        tb_writer.add_scalar("train/alpha", losses["alpha"], total_steps)

            # End of episode
            if done or episode_len >= args.max_steps:
                ep_time = time.time() - episode_start
                print(f"[Episode {episode+1}] return={episode_reward:.2f} len={episode_len} time={ep_time:.1f}s")
                tb_writer.add_scalar("train/episode_return", episode_reward, episode + 1)
                tb_writer.add_scalar("train/episode_length", episode_len, episode + 1)

                obs, _ = env.reset()
                wrapper.reset()

                episode += 1
                episode_reward = 0.0
                episode_len = 0
                episode_start = time.time()

            # Evaluation
            if total_steps % cfg.EVAL_INTERVAL == 0:
                eval_result = evaluate(agent, env, wrapper, cfg.EVAL_EPISODES, args.max_steps)
                print(f"[EVAL @ step {total_steps}] avg_return={eval_result['avg_return']:.2f}")
                tb_writer.add_scalar("eval/avg_return", eval_result["avg_return"], total_steps)
                tb_writer.add_scalar("eval/avg_length", eval_result["avg_length"], total_steps)

                # Save best model
                if eval_result["avg_return"] > best_eval_return:
                    best_eval_return = eval_result["avg_return"]
                    best_path = exp_dir / "models" / "best_model.pt"
                    agent.save(best_path)
                    print("Saved best model:", best_path)

            # Checkpoint
            if total_steps % cfg.CHECKPOINT_INTERVAL == 0:
                ckpt_path = exp_dir / "models" / f"checkpoint_step_{total_steps}.pt"
                agent.save(ckpt_path)
                print("Saved checkpoint:", ckpt_path)

        print("Training finished.")

    finally:
        tb_writer.flush()
        tb_writer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
