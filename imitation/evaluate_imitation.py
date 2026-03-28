import time
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import torch

from CarlaEnv.env import CarlaEnv
from .models.imitation_policy import ImitationPolicy

from . import config


ACTION_MODE = config.ACTION_MODE
SIMPLIFIED_ACTION_SPACE = config.SIMPLIFY_ACTIONS
DEVICE = config.DEVICE

CONTINUOUS_MODEL = config.CONTINUOUS_MODEL_PATH
DISCRETE_MODEL = config.DISCRETE_MODEL_PATH

DEBUG_PRINT_STEPS = config.DEBUG_PRINT_STEPS



debug_counter = 0


if SIMPLIFIED_ACTION_SPACE:
    speed_map = config.SIMPLIFY_SPEED_MAP

    turn_map = config.SIMPLIFY_TURN_MAP
else:
    speed_map = config.SPEED_MAP

    turn_map = config.TURN_MAP
    
    
def extract_grid_and_scalars(obs):

    presence = obs["presence"]

    if torch.is_tensor(presence):
        presence = presence.cpu().numpy()

    # TODO: this is  not dynamic
    lane_angle = obs["lane_angle"][0] / np.pi
    lane_pos = obs["ego_in_lane_position_x"][0] / 2.0
    speed_x = np.clip(obs["ego_speed_x"][0], -1, 15) / 15.0
    speed_y = np.clip(obs["ego_speed_y"][0], -2, 2) / 2.0
    
    grid = presence[None, :, :]
    
    scalars = np.array([
        lane_angle,
        lane_pos,
        speed_x,
        speed_y,
    ], dtype=np.float32)

    return grid, scalars

def map_action_for_env(action):
    """
    Convert (speed_idx, turn_idx) from the discrete head into
    integer action indices expected by CarlaEnv.
    """
    speed, turn = action

    if SIMPLIFIED_ACTION_SPACE:
        # model only knows [0,1,2,3] but env expects [0,1,2,4] for speed
        if speed == 3:
            speed = 4
        # model only knows [0,1,2] but env expects [0,1,3] for turn
        if turn == 2:
            turn = 3

    return [speed, turn]


prev_steer = 0.0

def process_continuous_output(out):
    """
    model output → [throttle, brake, steer]
    out: np.array of shape (3,)
    """
    global prev_steer
    throttle = max(float(np.clip(out[0], 0.0, 1.0)),0.13)
    brake = float(np.clip(out[1], 0.0, 1.0))
    steer = float(np.clip(out[2], -1.0, 1.0))

    # prevent throttle+brake conflict
    if brake > 0.05:
        throttle = 0.0
    else:
        brake= 0.0
    
    if config.SMOOTH_STEERING:  
        steer = 0.7 * prev_steer + 0.3 * steer
        prev_steer = steer
        steer = np.clip(steer, -1.0, 1.0)


    return [throttle, brake, steer]


def predict_action(policy, obs):
    """
    Run one policy step.
    obs is assumed to be (grid, scalars), as returned by CarlaEnv.
    Returns:
        env_action: list usable by env.step(...)
        action_log: (speed, turn) for discrete, or None for continuous
    """
    
    global debug_counter

    grid, scalars = extract_grid_and_scalars(obs)

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
                
            debug_counter+=1
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



def run_episode(env, policy, max_steps=2000, render_log_every=200):
    obs, _ = env.reset()


    action_counts = Counter()
    rewards = []

    terminated_flag = False
    truncated_flag = False

    t0 = time.time()

    for t in range(max_steps):
        env_action, action_log = predict_action(policy, obs)

        obs, reward, terminated, truncated, info = env.step(env_action)

        if action_log is not None:
            action_counts[action_log] += 1

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
    ep_return = float(np.sum(rewards)) if rewards else 0.0
    ep_mean_reward = float(np.mean(rewards)) if rewards else 0.0

    if terminated_flag:
        end_reason = "terminated"
    elif truncated_flag:
        end_reason = "truncated"
    else:
        end_reason = "max_steps"

    return {
        "return": ep_return,
        "mean_reward": ep_mean_reward,
        "length": ep_len,
        "end_reason": end_reason,
        "action_counts": action_counts,
    }




def load_policy():
    """
    Load an ImitationPolicy from a checkpoint file produced by train_bc.py.
    """
    ckpt_path = DISCRETE_MODEL if ACTION_MODE == "discrete" else CONTINUOUS_MODEL

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    mode = ckpt["mode"]
    scalar_dim = ckpt["scalar_dim"]
    grid_channels = ckpt["grid_channels"]

    if mode == "discrete":
        n_speed = ckpt["n_speed"]
        n_turn = ckpt["n_turn"]
        policy = ImitationPolicy(
            mode="discrete",
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            n_speed=n_speed,
            n_turn=n_turn,
        ).to(DEVICE)
    else:
        is_gaussian = config.IS_GAUSSIAN

        policy = ImitationPolicy(
            mode="continuous",
            is_gaussian=is_gaussian,
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
        ).to(DEVICE)

    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    print("\nLoaded model:")
    print("  mode:", mode)
    print("  grid_channels:", grid_channels)
    print("  scalar_dim:", scalar_dim)
    if mode == "discrete":
        print("  n_speed:", n_speed)
        print("  n_turn:", n_turn)

    return policy




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--map", type=str, default=config.CARLA_MAP_PATH)
    parser.add_argument("--episodes", type=int, default=config.EVAL_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)

    parser.add_argument("--mode", choices=["discrete","continuous"], default=config.ACTION_MODE)
    parser.add_argument("--device", default=config.DEVICE)

    
    args = parser.parse_args()
    
    global ACTION_MODE
    global DEVICE
    ACTION_MODE = args.mode
    DEVICE = args.device

    map_path = args.map
    num_episodes = args.episodes
    max_steps = args.max_steps

    env = CarlaEnv(
        map_path=map_path,
        walkers_count=config.CARLA_WALKERS,
        vehicles_count=config.CARLA_VEHICLES,
        max_steps=max_steps,
        init_speed=config.CARLA_INIT_SPEED,
        action_mode=ACTION_MODE,
    )


    policy = load_policy()

    print("Action mode:", ACTION_MODE)
    print("Device:", DEVICE)

    all_returns = []
    all_lengths = []
    end_reasons = Counter()
    global_action_counts = Counter()

    overall_t0 = time.time()

    try:
        for ep in range(num_episodes):
            print(f"\n=== Episode {ep+1}/{num_episodes} ===")

            result = run_episode(env, policy, max_steps=max_steps)

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

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
