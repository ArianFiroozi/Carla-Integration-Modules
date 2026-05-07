import os
import time
import json
import random
import argparse
import datetime
import pickle
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


def make_experiment_dir(resume_dir=None):
    """Create new experiment dir or use existing one for resume."""
    if resume_dir is not None:
        exp_dir = Path(resume_dir)
        if not exp_dir.exists():
            raise ValueError(f"Resume directory does not exist: {exp_dir}")
        print(f"Resuming experiment from: {exp_dir}")
        return exp_dir
    else:
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


def find_latest_checkpoint(exp_dir):
    """Find the latest checkpoint in the experiment directory."""
    models_dir = Path(exp_dir) / "models"
    if not models_dir.exists():
        return None
    
    checkpoints = list(models_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints[-1]


def find_checkpoint_state(exp_dir):
    """Find the latest checkpoint state file."""
    models_dir = Path(exp_dir) / "models"
    if not models_dir.exists():
        return None
    
    state_files = list(models_dir.glob("checkpoint_state_*.pkl"))
    print(f"Looking for state files in: {models_dir}")
    print(f"Found state files: {state_files}")
    
    if not state_files:
        return None
    state_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return state_files[-1]


def save_full_checkpoint(exp_dir, step, agent, replay_buffer, optimizer_states, extra_info):
    """Save a comprehensive checkpoint including model, buffer, and training state."""
    models_dir = exp_dir / "models"
    
    # Save model checkpoint
    model_path = models_dir / f"checkpoint_step_{step}.pt"
    agent.save(model_path)
    
    # Save replay buffer and training state
    state = {
        'step': step,
        'replay_buffer': replay_buffer,
        'optimizer_states': optimizer_states,
        'extra_info': extra_info,
        'log_alpha': agent.log_alpha.detach().cpu() if agent.auto_entropy else None,  # FIX #2: Save log_alpha
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
    }
    
    state_path = models_dir / f"checkpoint_state_{step}.pkl"
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Saved full checkpoint at step {step}: {model_path}")
    return model_path, state_path


def load_full_checkpoint(exp_dir, device):
    """Load the latest checkpoint including model, buffer, and training state."""
    latest_model = find_latest_checkpoint(exp_dir)
    latest_state = find_checkpoint_state(exp_dir)
    
    if latest_model is None or latest_state is None:
        print("No complete checkpoint found.")
        return None
    
    print(f"Loading model from: {latest_model}")
    print(f"Loading state from: {latest_state}")
    
    # Load state
    with open(latest_state, 'rb') as f:
        checkpoint_state = pickle.load(f)
    
    return {
        'model_path': latest_model,
        'checkpoint_state': checkpoint_state
    }


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






# ============================================================
# Enhanced TensorBoard Logging
# ============================================================

def log_config_to_tensorboard(tb_writer, config_dict):
    """
    Log all config parameters to TensorBoard as both text and scalars.
    """
    # Full config as formatted JSON in Text tab
    clean_config = {k: str(v) if isinstance(v, Path) else v 
                    for k, v in config_dict.items() 
                    if k.isupper() and not k.startswith('_')}
    
    config_str = json.dumps(clean_config, indent=2, default=str)
    tb_writer.add_text("Config/Hyperparameters", f"```json\n{config_str}\n```", 0)
    
    # Individual parameters as scalars or text
    for k, v in clean_config.items():
        if isinstance(v, (int, float, bool)):
            tb_writer.add_scalar(f"Config/{k}", float(v), 0)
        else:
            tb_writer.add_text(f"Config/{k}", str(v), 0)
    
    print("[TB] Config logged to TensorBoard")


def log_network_info(tb_writer, agent):
    """
    Log network architecture details.
    """
    # Count parameters
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    total_params = actor_params + critic_params
    
    info_text = (
        f"Actor parameters: {actor_params:,}\n"
        f"Critic parameters: {critic_params:,}\n"
        f"Total parameters: {total_params:,}\n"
        f"Grid channels: {cfg.GRID_CHANNELS}\n"
        f"Scalar dim: {cfg.SCALAR_DIM}\n"
        f"Latent dim: {cfg.LATENT_DIM}\n"
        f"Action dim: {cfg.ACTION_DIM}\n"
        f"CNN channels: {cfg.CNN_CHANNELS}\n"
        f"Kernel sizes: {cfg.KERNEL_SIZES}"
    )
    tb_writer.add_text("Network/Architecture", info_text, 0)
    tb_writer.add_scalar("Network/actor_params", actor_params, 0)
    tb_writer.add_scalar("Network/critic_params", critic_params, 0)
    tb_writer.add_scalar("Network/total_params", total_params, 0)
    
    print(f"[TB] Network info logged ({total_params:,} total params)")


def log_policy_stats(tb_writer, agent, grid_t, scalars_t, step):
    """
    Log detailed policy statistics: log_std, action distribution, etc.
    """
    with torch.no_grad():
        mean, log_std = agent.actor.forward(grid_t, scalars_t)
        std = log_std.exp()
        
        # Log log_std statistics
        tb_writer.add_scalar("policy/log_std_mean", log_std.mean().item(), step)
        tb_writer.add_scalar("policy/log_std_max", log_std.max().item(), step)
        tb_writer.add_scalar("policy/log_std_min", log_std.min().item(), step)
        
        # Log std statistics
        tb_writer.add_scalar("policy/std_mean", std.mean().item(), step)
        tb_writer.add_scalar("policy/std_max", std.max().item(), step)
        
        # Per-dimension log_std
        for i, name in enumerate(["throttle", "brake", "steer"]):
            tb_writer.add_scalar(f"policy/log_std_{name}", log_std[0, i].item(), step)
            tb_writer.add_scalar(f"policy/mean_{name}", mean[0, i].item(), step)
        
        # Sample actions and log their statistics
        actions, logp, mean_actions = agent.actor.sample(grid_t, scalars_t)
        
        tb_writer.add_scalar("policy/action_mean_throttle", actions[:, 0].mean().item(), step)
        tb_writer.add_scalar("policy/action_mean_brake", actions[:, 1].mean().item(), step)
        tb_writer.add_scalar("policy/action_mean_steer", actions[:, 2].mean().item(), step)
        
        tb_writer.add_scalar("policy/action_std_throttle", actions[:, 0].std().item(), step)
        tb_writer.add_scalar("policy/action_std_brake", actions[:, 1].std().item(), step)
        tb_writer.add_scalar("policy/action_std_steer", actions[:, 2].std().item(), step)
        
        # Log deterministic action (for comparison)
        tb_writer.add_scalar("policy/det_action_throttle", mean_actions[:, 0].mean().item(), step)
        tb_writer.add_scalar("policy/det_action_brake", mean_actions[:, 1].mean().item(), step)
        tb_writer.add_scalar("policy/det_action_steer", mean_actions[:, 2].mean().item(), step)
        
        # Log entropy of current policy
        tb_writer.add_scalar("policy/entropy", -logp.mean().item(), step)


def log_critic_stats(tb_writer, agent, grid_t, scalars_t, actions_t, step):
    """
    Log critic Q-value statistics.
    """
    with torch.no_grad():
        q1, q2 = agent.critic(grid_t, scalars_t, actions_t)
        
        tb_writer.add_scalar("critic/q1_mean", q1.mean().item(), step)
        tb_writer.add_scalar("critic/q1_std", q1.std().item(), step)
        tb_writer.add_scalar("critic/q1_min", q1.min().item(), step)
        tb_writer.add_scalar("critic/q1_max", q1.max().item(), step)
        
        tb_writer.add_scalar("critic/q2_mean", q2.mean().item(), step)
        tb_writer.add_scalar("critic/q2_std", q2.std().item(), step)
        
        tb_writer.add_scalar("critic/q_diff_mean", (q1 - q2).abs().mean().item(), step)


def log_replay_buffer_stats(tb_writer, replay_buffer, step):
    """
    Log replay buffer statistics.
    """
    if len(replay_buffer) > 0:
        # Sample a batch to get reward statistics
        _, _, _, rewards, _, _, _ = replay_buffer.sample(min(1000, len(replay_buffer)))
        
        tb_writer.add_scalar("replay/buffer_size", len(replay_buffer), step)
        tb_writer.add_scalar("replay/reward_mean", rewards.mean().item(), step)
        tb_writer.add_scalar("replay/reward_std", rewards.std().item(), step)
        tb_writer.add_scalar("replay/reward_min", rewards.min().item(), step)
        tb_writer.add_scalar("replay/reward_max", rewards.max().item(), step)


def log_training_diagnostics(tb_writer, agent, replay_buffer, grid_t, scalars_t, 
                             actions_t, losses, step, episode, episode_reward):
    """
    Comprehensive logging called periodically during training.
    """
    # Only log detailed stats every N steps to avoid overhead
    LOG_DETAIL_EVERY = 1000  # Adjust based on your preference
    
    if step % LOG_DETAIL_EVERY == 0:
        log_policy_stats(tb_writer, agent, grid_t, scalars_t, step)
        log_critic_stats(tb_writer, agent, grid_t, scalars_t, actions_t, step)
        log_replay_buffer_stats(tb_writer, replay_buffer, step)
    
    # Always log these (lightweight)
    tb_writer.add_scalar("train/critic_loss", losses["critic_loss"], step)
    tb_writer.add_scalar("train/actor_loss", losses["actor_loss"], step)
    tb_writer.add_scalar("train/alpha_loss", losses["alpha_loss"], step)
    tb_writer.add_scalar("train/alpha", losses["alpha"], step)
    
    # Log learning rates
    tb_writer.add_scalar("train/actor_lr", agent.actor_opt.param_groups[0]['lr'], step)
    tb_writer.add_scalar("train/critic_lr", agent.critic_opt.param_groups[0]['lr'], step)
    if agent.auto_entropy:
        tb_writer.add_scalar("train/alpha_lr", agent.alpha_opt.param_groups[0]['lr'], step)
    
    # Gradient norms (if you want to add - requires accessing after backward, before step)
    # This would need to be added inside agent.update() itself






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=cfg.CARLA_MAP_PATH)
    parser.add_argument("--max-steps", type=int, default=cfg.CARLA_MAX_STEPS)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    parser.add_argument("--resume", action="store_true",default=cfg.RESUME_CHECKPOINT, help="Resume from latest checkpoint")
    parser.add_argument("--resume-dir", type=str, default=None, help="Specific experiment directory to resume from")
    args = parser.parse_args()

    # --------------------------- Assumptions ---------------------------
    # We assume CONTINUOUS ACTIONS ONLY (no discrete mode).
    # We use CarlaObsWrapper for preprocessing.
    # -------------------------------------------------------------------

    seed_everything(args.seed)

    # Determine experiment directory
    if args.resume and args.resume_dir:
        exp_dir = make_experiment_dir(resume_dir=args.resume_dir)
    elif args.resume:
        # Try to find the latest experiment directory
        save_dir = Path(cfg.SAVE_DIR)
        exp_dirs = sorted([d for d in save_dir.iterdir() if d.is_dir()], reverse=True)
        if exp_dirs:
            exp_dir = make_experiment_dir(resume_dir=str(exp_dirs[0]))
        else:
            print("No experiment directories found to resume from. Starting new experiment.")
            exp_dir = make_experiment_dir()
            save_config(exp_dir)
    else:
        exp_dir = make_experiment_dir()
        save_config(exp_dir)
    
    tb_writer = SummaryWriter(str(exp_dir / "tb"))
    log_config_to_tensorboard(tb_writer, cfg.__dict__)
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
    log_network_info(tb_writer, agent)
    # Initialize training state variables
    total_steps = 0
    episode = 0
    best_eval_return = -1e9
    
    # Handle resume vs fresh start
    if args.resume:
        checkpoint_data = load_full_checkpoint(exp_dir, args.device)
        
        if checkpoint_data is not None:
            # Load model
            agent.load(checkpoint_data['model_path'])
            
            # Load state
            state = checkpoint_data['checkpoint_state']
            total_steps = state['step']
            replay_buffer = state['replay_buffer']
            
    
            # Restore optimizer states
            opt_states = state.get('optimizer_states', {})
            if 'actor_opt' in opt_states:
                agent.actor_opt.load_state_dict(opt_states['actor_opt'])
                print("Restored actor optimizer state")
            if 'critic_opt' in opt_states:
                agent.critic_opt.load_state_dict(opt_states['critic_opt'])
                print("Restored critic optimizer state")
            if agent.auto_entropy and 'alpha_opt' in opt_states:
                agent.alpha_opt.load_state_dict(opt_states['alpha_opt'])
                print("Restored alpha optimizer state")
            
            # Restore log_alpha (SAC entropy temperature) 
            if agent.auto_entropy and state.get('log_alpha') is not None:
                agent.log_alpha.data.copy_(state['log_alpha'].to(agent.device))
                print(f"Restored log_alpha: {agent.log_alpha.exp().item():.4f}")
            
            # Restore episode info if available
            extra_info = state.get('extra_info', {})
            episode = extra_info.get('episode', 0)
            best_eval_return = extra_info.get('best_eval_return', -1e9)
            
            # Restore RNG states
            rng_state = state.get('rng_state', {})
            if rng_state:
                random.setstate(rng_state['python'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch'])
                if rng_state['torch_cuda'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng_state['torch_cuda'])
            
            print(f"Resumed training from step {total_steps}, episode {episode}")
        else:
            print("No checkpoint found to resume. Starting fresh.")
            replay_buffer = SACReplayBuffer(capacity=cfg.REPLAY_BUFFER_SIZE, device=args.device)
            
            # Load BC weights if configured
            if cfg.LOAD_BC_WEIGHTS and Path(cfg.BC_CHECKPOINT_PATH).exists():
                print(f"Loading BC weights from: {cfg.BC_CHECKPOINT_PATH}")
                agent.load_actor_from_bc(cfg.BC_CHECKPOINT_PATH, strict=False)
            elif cfg.LOAD_BC_WEIGHTS:
                print(f"[WARN] BC checkpoint not found at: {cfg.BC_CHECKPOINT_PATH}")
    else:
        # Fresh start
        replay_buffer = SACReplayBuffer(capacity=cfg.REPLAY_BUFFER_SIZE, device=args.device)
        
        # Load BC weights if configured
        if cfg.LOAD_BC_WEIGHTS and Path(cfg.BC_CHECKPOINT_PATH).exists():
            print(f"Loading BC weights from: {cfg.BC_CHECKPOINT_PATH}")
            agent.load_actor_from_bc(cfg.BC_CHECKPOINT_PATH, strict=False)
        elif cfg.LOAD_BC_WEIGHTS:
            print(f"[WARN] BC checkpoint not found at: {cfg.BC_CHECKPOINT_PATH}")
    
    # Continue training if not finished
    if total_steps >= cfg.MAX_TRAIN_STEPS:
        print(f"Training already completed! Total steps: {total_steps} >= {cfg.MAX_TRAIN_STEPS}")
        tb_writer.close()
        env.close()
        return

    obs, _ = env.reset()
    wrapper.reset()

    episode_reward = 0.0
    episode_len = 0
    episode_start = time.time()

    try:
        while total_steps < cfg.MAX_TRAIN_STEPS:
            # Preprocess obs
            grid, scalars = wrapper.preprocess(obs)
            grid_t, scalars_t = wrapper.to_tensor(grid, scalars)

            # Action selection
            if cfg.USE_RANDOM_POLICY_WARMUP and total_steps < cfg.WARMUP_STEPS:
                # random action in env bounds
                raw_action = np.random.uniform(low=cfg.ACTION_LOW, high=cfg.ACTION_HIGH, size=(cfg.ACTION_DIM,))
            else:
                raw_action = agent.select_action(grid_t, scalars_t, evaluate=False)[0]

            raw_action = np.asarray(raw_action, dtype=np.float32)
            

            # Env step
            next_obs, reward, terminated, truncated, info = env.step(raw_action)
            done = terminated or truncated

            # Preprocess next obs
            next_grid, next_scalars = wrapper.preprocess(next_obs)

            # Store transition (store ENV action, because that is what executed)
            replay_buffer.add(
                grid_obs=grid,
                scalar_obs=scalars,
                action=raw_action,
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
                        
                        
                        tb_writer.add_scalar("grad_norms/critic", losses.get("critic_grad_norm", 0), total_steps)
                        tb_writer.add_scalar("grad_norms/actor", losses.get("actor_grad_norm", 0), total_steps)
                        tb_writer.add_scalar("grad_norms/alpha", losses.get("alpha_grad_norm", 0), total_steps)
                        
                        if total_steps % cfg.LOG_EVERY == 0:
                            log_policy_stats(tb_writer, agent, grid_t, scalars_t, total_steps)
                            log_replay_buffer_stats(tb_writer, replay_buffer, total_steps)

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
                # Collect optimizer states
                optimizer_states = {
                    'actor_opt': agent.actor_opt.state_dict(),
                    'critic_opt': agent.critic_opt.state_dict(),
                }
                if agent.auto_entropy:
                    optimizer_states['alpha_opt'] = agent.alpha_opt.state_dict()
                
                # Collect extra info for resume
                extra_info = {
                    'episode': episode,
                    'best_eval_return': best_eval_return,
                    'episode_reward': episode_reward,
                    'episode_len': episode_len,
                }
                
                save_full_checkpoint(
                    exp_dir=exp_dir,
                    step=total_steps,
                    agent=agent,
                    replay_buffer=replay_buffer,
                    optimizer_states=optimizer_states,
                    extra_info=extra_info
                )

        print("Training finished.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        # Save checkpoint on interrupt
        optimizer_states = {
            'actor_opt': agent.actor_opt.state_dict(),
            'critic_opt': agent.critic_opt.state_dict(),
        }
        if agent.auto_entropy:
            optimizer_states['alpha_opt'] = agent.alpha_opt.state_dict()
        
        extra_info = {
            'episode': episode,
            'best_eval_return': best_eval_return,
            'episode_reward': episode_reward,
            'episode_len': episode_len,
        }
        
        save_full_checkpoint(
            exp_dir=exp_dir,
            step=total_steps,
            agent=agent,
            replay_buffer=replay_buffer,
            optimizer_states=optimizer_states,
            extra_info=extra_info
        )
        print("Final checkpoint saved.")

    finally:
        tb_writer.flush()
        tb_writer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()