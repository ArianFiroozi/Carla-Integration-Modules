import time
import numpy as np
from collections import Counter

from carla_env.env import CarlaEnv
from imitation.bc_policy import BCPolicy, BCPolicyConfig


def main():
    map_path = r"C:\carla\Carla-Integration-Modules\LoadOpenDrive2\harder.xodr"

    env = CarlaEnv(map_path=map_path, walkers_count=0, vehicles_count=0, max_steps=2000, init_speed=0)

    policy = BCPolicy(
        ckpt_path="bc_cnn.pt",
        device="auto",
        cfg=BCPolicyConfig(presence_max=9.0, deterministic=True),
    )

    obs, _ = env.reset()

    action_counts = Counter()
    rewards = []
    t0 = time.time()

    for t in range(2000):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action.tolist())

        action_counts[tuple(action.tolist())] += 1
        rewards.append(float(reward))

        if (t + 1) % 200 == 0:
            fps = (t + 1) / max(1e-6, (time.time() - t0))
            print(f"[t={t+1}] mean_reward={np.mean(rewards[-200:]):.2f} fps={fps:.1f}")

        if terminated or truncated:
            print("DONE:", "terminated" if terminated else "truncated", "at t=", t)
            break

    print("Action distribution (top 10):")
    for k, v in action_counts.most_common(10):
        print(k, v)

    print("Episode mean reward:", np.mean(rewards) if rewards else None)


if __name__ == "__main__":
    main()