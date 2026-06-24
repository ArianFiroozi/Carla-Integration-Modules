import os
import time
import json
import random
import argparse
import datetime
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.seed_utils import seed_everything
from config import general_config
from utils.reward_compiler import compile_reward
from CarlaEnv.env import CarlaEnv

# AWAC Specific Imports
from agents.awac.awac_agent import AWACAgent
from config import awac_config as cfg
from utils.replay_buffer import SACReplayBuffer
from utils.obs_wrapper import CarlaObsWrapper
from config import bc_config

# -------------------------------------------------------------
# Helpers & Checkpointing
# -------------------------------------------------------------

def load_norm_stats_from_bc_checkpoint():
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

def make_experiment_dir(resume_dir=None):
    if resume_dir is not None:
        exp_dir = Path(resume_dir)
        if not exp_dir.exists():
            raise ValueError(f"Resume directory does not exist: {exp_dir}")
        print(f"Resuming experiment from: {exp_dir}")
        return exp_dir
    else:
        run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = Path(cfg.SAVE_DIR) / f"{run_stamp}_awac"
        (exp_dir / "models").mkdir(parents=True, exist_ok=True)
        (exp_dir / "tb").mkdir(exist_ok=True)
        return exp_dir

def save_config(exp_dir):
    config_path = exp_dir / "config.json"
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items() if k.isupper()}
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

def find_latest_checkpoint(exp_dir):
    models_dir = Path(exp_dir) / "models"
    if not models_dir.exists(): return None
    checkpoints = list(models_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints[-1]

def find_checkpoint_state(exp_dir):
    models_dir = Path(exp_dir) / "models"
    if not models_dir.exists(): return None
    state_files = list(models_dir.glob("checkpoint_state_*.pkl"))
    if not state_files: return None
    state_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return state_files[-1]

def save_full_checkpoint(exp_dir, step, agent, replay_buffer=None, optimizer_states=None, extra_info=None):
    models_dir = exp_dir / "models"
    model_path = models_dir / f"checkpoint_step_{step}.pt"
    agent.save(model_path)
    
    state = {
        'step': step,
        'optimizer_states': optimizer_states,
        'extra_info': extra_info,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
    }
    
    if replay_buffer is not None:
        state['replay_buffer'] = replay_buffer
        print("  Including replay buffer in this checkpoint")
    
    state_path = models_dir / f"checkpoint_state_{step}.pkl"
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
    
    size_mb = state_path.stat().st_size / (1024*1024)
    print(f"Saved AWAC checkpoint at step {step} ({size_mb:.0f}MB)")
    return model_path, state_path

def load_full_checkpoint(exp_dir, device):
    latest_model = find_latest_checkpoint(exp_dir)
    latest_state = find_checkpoint_state(exp_dir)
    if latest_model is None or latest_state is None:
        return None
    with open(latest_state, 'rb') as f:
        checkpoint_state = pickle.load(f)
    return {'model_path': latest_model, 'checkpoint_state': checkpoint_state}

def cleanup_old_checkpoints(exp_dir, keep=3):
    models_dir = Path(exp_dir) / "models"
    ckpts = sorted(models_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    states = sorted(models_dir.glob("checkpoint_state_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    to_keep = {models_dir / "best_model.pt"}
    for f in ckpts[-keep:] + states[-keep:]: to_keep.add(f)
    for pattern in ["checkpoint_step_*", "checkpoint_state_*"]:
        for f in models_dir.glob(pattern):
            if f not in to_keep:
                f.unlink()

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
        obs, reward, terminated, truncated, info = env.step(action)
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
    all_returns, all_lengths, end_reasons = [], [], {}
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

# -------------------------------------------------------------
# TensorBoard Logging
# -------------------------------------------------------------
def log_config_to_tensorboard(tb_writer, config_dict):
    clean_config = {k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items() if k.isupper() and not k.startswith('_')}
    config_str = json.dumps(clean_config, indent=2, default=str)
    tb_writer.add_text("Config/Hyperparameters", f"```json\n{config_str}\n```", 0)
    for k, v in clean_config.items():
        if isinstance(v, (int, float, bool)): tb_writer.add_scalar(f"Config/{k}", float(v), 0)
        else: tb_writer.add_text(f"Config/{k}", str(v), 0)

def log_network_info(tb_writer, agent):
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    total_params = actor_params + critic_params
    tb_writer.add_scalar("Network/total_params", total_params, 0)
    print(f"[TB] AWAC Network info logged ({total_params:,} total params)")

def log_policy_stats(tb_writer, agent, grid_t, scalars_t, step):
    with torch.no_grad():
        mean, log_std = agent.actor.forward(grid_t, scalars_t)
        std = log_std.exp()
        tb_writer.add_scalar("policy/log_std_mean", log_std.mean().item(), step)
        actions, logp, _ = agent.actor.sample(grid_t, scalars_t)
        for i, name in enumerate(["throttle", "brake", "steer"]):
            tb_writer.add_scalar(f"policy/action_mean_{name}", actions[:, i].mean().item(), step)
            tb_writer.add_scalar(f"policy/action_std_{name}", actions[:, i].std(unbiased=False).item(), step)

def log_critic_stats(tb_writer, agent, grid_t, scalars_t, actions_t, step):
    with torch.no_grad():
        q1, q2 = agent.critic(grid_t, scalars_t, actions_t)
        tb_writer.add_scalar("critic/q1_mean", q1.mean().item(), step)
        tb_writer.add_scalar("critic/q2_mean", q2.mean().item(), step)

# -------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    parser.add_argument("--resume", action="store_true", default=getattr(cfg, "RESUME_CHECKPOINT", False))
    parser.add_argument("--resume-dir", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.resume and args.resume_dir:
        exp_dir = make_experiment_dir(resume_dir=args.resume_dir)
    else:
        exp_dir = make_experiment_dir()
        save_config(exp_dir)
    
    tb_writer = SummaryWriter(str(exp_dir / "tb"))
    log_config_to_tensorboard(tb_writer, cfg.__dict__)
    
    norm_stats = load_norm_stats_from_bc_checkpoint()
    
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

    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=args.device, action_mode="continuous")
    agent = AWACAgent(device=args.device)
    log_network_info(tb_writer, agent)
    
    total_steps = 0
    episode = 0
    best_eval_return = -1e9
    
    checkpoint_data = load_full_checkpoint(exp_dir, args.device) if args.resume else None
        
    if checkpoint_data is not None:
        agent.load(checkpoint_data['model_path'])
        state = checkpoint_data['checkpoint_state']
        total_steps = state['step']
        
        replay_buffer = state.get('replay_buffer')
        if replay_buffer is None:
            replay_buffer = SACReplayBuffer(capacity=cfg.REPLAY_BUFFER_SIZE, device=args.device)
            if cfg.PRELOAD_EXPERT_DATA: replay_buffer.load_offline_dataset(cfg.COMPILED_DATASET_PATH)
            
        opt_states = state.get('optimizer_states', {})
        if 'actor_opt' in opt_states: agent.actor_opt.load_state_dict(opt_states['actor_opt'])
        if 'critic_opt' in opt_states: agent.critic_opt.load_state_dict(opt_states['critic_opt'])
        
        extra_info = state.get('extra_info', {})
        episode = extra_info.get('episode', 0)
        best_eval_return = extra_info.get('best_eval_return', -1e9)
        print(f"Resumed training from step {total_steps}")
    else:
        print("Starting fresh AWAC run.")
        replay_buffer = SACReplayBuffer(capacity=cfg.REPLAY_BUFFER_SIZE, device=args.device)
        if cfg.PRELOAD_EXPERT_DATA:
            replay_buffer.load_offline_dataset(cfg.COMPILED_DATASET_PATH)
        
        if cfg.LOAD_BC_WEIGHTS and Path(cfg.BC_CHECKPOINT_PATH).exists():
            agent.load_actor_from_bc(cfg.BC_CHECKPOINT_PATH, strict=False)

    try:
        # =========================================================
        # PHASE 1: PURE OFFLINE PRETRAINING
        # =========================================================
        offline_steps = getattr(cfg, "OFFLINE_PRETRAIN_STEPS", 0)
        if total_steps < offline_steps:
            print(f"Starting Phase 1: Offline Pretraining for {offline_steps - total_steps} steps...")
            while total_steps < offline_steps:
                if len(replay_buffer) < cfg.BATCH_SIZE:
                    print("Buffer doesn't have enough samples for offline pretraining. Aborting Phase 1.")
                    break
                
                # AWAC Aggregator
                agg = {"critic_loss": 0.0, "actor_loss": 0.0, "awac_weight": 0.0, "awac_adv": 0.0,
                       "critic_grad_norm": 0.0, "actor_grad_norm": 0.0}
                
                grad_updates = getattr(cfg, "GRADIENT_UPDATES", 1)
                for _ in range(grad_updates):
                    losses = agent.update(replay_buffer)
                    for k in agg: agg[k] += losses.get(k, 0)
                for k in agg: agg[k] /= grad_updates

                total_steps += 1

                # Log
                tb_writer.add_scalar("train_offline/critic_loss", agg["critic_loss"], total_steps)
                tb_writer.add_scalar("train_offline/actor_loss",  agg["actor_loss"],  total_steps)
                tb_writer.add_scalar("train_offline/awac_weight", agg["awac_weight"], total_steps)
                tb_writer.add_scalar("train_offline/awac_adv",    agg["awac_adv"],    total_steps)
                tb_writer.add_scalar("grad_norms_offline/critic", agg["critic_grad_norm"], total_steps)
                tb_writer.add_scalar("grad_norms_offline/actor",  agg["actor_grad_norm"],  total_steps)

                if total_steps % getattr(cfg, "LOG_EVERY", 1000) == 0:
                    g, s, a, r, _, _, _ = replay_buffer.sample(cfg.BATCH_SIZE)
                    log_policy_stats(tb_writer, agent, g, s, total_steps)
                    log_critic_stats(tb_writer, agent, g, s, a, total_steps)

                if total_steps % getattr(cfg, "EVAL_INTERVAL", 10000) == 0:
                    eval_result = evaluate(agent, env, wrapper, getattr(cfg, "EVAL_EPISODES", 5), args.max_steps)
                    print(f"[EVAL @ step {total_steps}] avg_return={eval_result['avg_return']:.2f}")
                    tb_writer.add_scalar("eval_offline/avg_return", eval_result["avg_return"], total_steps)
                    
                    if eval_result["avg_return"] > best_eval_return:
                        best_eval_return = eval_result["avg_return"]
                        agent.save(exp_dir / "models" / "best_model.pt")

                if total_steps % getattr(cfg, "CHECKPOINT_INTERVAL", 25000) == 0:
                    opt_states = {'actor_opt': agent.actor_opt.state_dict(), 'critic_opt': agent.critic_opt.state_dict()}
                    extra_info = {'episode': episode, 'best_eval_return': best_eval_return}
                    save_buffer = (total_steps % (cfg.CHECKPOINT_INTERVAL * getattr(cfg, "SAVE_BUFFER_EVERY", 4)) == 0)
                    
                    save_full_checkpoint(exp_dir, total_steps, agent, replay_buffer if save_buffer else None, opt_states, extra_info)
                    cleanup_old_checkpoints(exp_dir, keep=getattr(cfg, "KEEP_CHECKPOINTS", 3))

        # =========================================================
        # PHASE 2: ONLINE FINE-TUNING
        # =========================================================
        if total_steps < cfg.MAX_TRAIN_STEPS:
            print("Starting Phase 2: Online Fine-Tuning...")
            obs, _ = env.reset()
            wrapper.reset()
            episode_reward, episode_len, episode_start = 0.0, 0, time.time()

            while total_steps < cfg.MAX_TRAIN_STEPS:
                grid, scalars = wrapper.preprocess(obs)
                grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

                # Epsilon-greedy exploration or deterministic based on warmup
                if getattr(cfg, "USE_RANDOM_POLICY_WARMUP", False) and total_steps < getattr(cfg, "WARMUP_STEPS", 5000):
                    raw_action = np.random.uniform(low=cfg.ACTION_LOW, high=cfg.ACTION_HIGH, size=(cfg.ACTION_DIM,))
                else:
                    raw_action = agent.select_action(grid_t, scalars_t, evaluate=False)[0]

                next_obs, raw_reward, terminated, truncated, info = env.step(raw_action)
                done = terminated or truncated
                reward, _ = compile_reward(info, general_config, is_tensor=False)
                next_grid, next_scalars = wrapper.preprocess(next_obs)

                replay_buffer.add(grid, scalars, raw_action, reward, next_grid, next_scalars, done)
                obs = next_obs
                episode_reward += float(reward)
                episode_len += 1
                total_steps += 1

                # Updates   
                update_after = getattr(cfg, "UPDATE_AFTER", 1000)
                if total_steps >= update_after and len(replay_buffer) >= cfg.BATCH_SIZE:
                    
                    # AWAC Aggregator
                    agg = {"critic_loss": 0.0, "actor_loss": 0.0, "awac_weight": 0.0, "awac_adv": 0.0,
                           "critic_grad_norm": 0.0, "actor_grad_norm": 0.0}
                    
                    grad_updates = getattr(cfg, "GRADIENT_UPDATES", 1)
                    for _ in range(grad_updates):
                        losses = agent.update(replay_buffer)
                        for k in agg: agg[k] += losses.get(k, 0)
                    for k in agg: agg[k] /= grad_updates

                    # Log
                    tb_writer.add_scalar("train/critic_loss", agg["critic_loss"], total_steps)
                    tb_writer.add_scalar("train/actor_loss",  agg["actor_loss"],  total_steps)
                    tb_writer.add_scalar("train/awac_weight", agg["awac_weight"], total_steps)
                    tb_writer.add_scalar("train/awac_adv",    agg["awac_adv"],    total_steps)
                    tb_writer.add_scalar("grad_norms/critic", agg["critic_grad_norm"], total_steps)
                    tb_writer.add_scalar("grad_norms/actor",  agg["actor_grad_norm"],  total_steps)

                    if total_steps % getattr(cfg, "LOG_EVERY", 1000) == 0:
                        g, s, a, r, _, _, _ = replay_buffer.sample(cfg.BATCH_SIZE)
                        log_policy_stats(tb_writer, agent, g, s, total_steps)
                        log_critic_stats(tb_writer, agent, g, s, a, total_steps)

                if done or episode_len >= args.max_steps:
                    print(f"[Episode {episode+1}] return={episode_reward:.2f} len={episode_len} time={time.time()-episode_start:.1f}s")
                    tb_writer.add_scalar("train/episode_return", episode_reward, episode + 1)
                    tb_writer.add_scalar("train/episode_length", episode_len, episode + 1)
                    
                    obs, _ = env.reset()
                    wrapper.reset()
                    episode += 1
                    episode_reward, episode_len, episode_start = 0.0, 0, time.time()

                if total_steps % getattr(cfg, "EVAL_INTERVAL", 10000) == 0:
                    eval_result = evaluate(agent, env, wrapper, getattr(cfg, "EVAL_EPISODES", 5), args.max_steps)
                    print(f"[EVAL @ step {total_steps}] avg_return={eval_result['avg_return']:.2f}")
                    tb_writer.add_scalar("eval/avg_return", eval_result["avg_return"], total_steps)
                    
                    if eval_result["avg_return"] > best_eval_return:
                        best_eval_return = eval_result["avg_return"]
                        agent.save(exp_dir / "models" / "best_model.pt")

                if total_steps % getattr(cfg, "CHECKPOINT_INTERVAL", 25000) == 0:
                    opt_states = {'actor_opt': agent.actor_opt.state_dict(), 'critic_opt': agent.critic_opt.state_dict()}
                    extra_info = {'episode': episode, 'best_eval_return': best_eval_return}
                    save_buffer = (total_steps % (cfg.CHECKPOINT_INTERVAL * getattr(cfg, "SAVE_BUFFER_EVERY", 4)) == 0)
                    
                    save_full_checkpoint(exp_dir, total_steps, agent, replay_buffer if save_buffer else None, opt_states, extra_info)
                    cleanup_old_checkpoints(exp_dir, keep=getattr(cfg, "KEEP_CHECKPOINTS", 3))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving final checkpoint...")
        save_full_checkpoint(exp_dir, total_steps, agent, replay_buffer, 
                             {'actor_opt': agent.actor_opt.state_dict(), 'critic_opt': agent.critic_opt.state_dict()}, 
                             {'episode': episode, 'best_eval_return': best_eval_return})
    finally:
        tb_writer.close()
        env.close()

if __name__ == "__main__":
    main()