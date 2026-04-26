import numpy as np

class DummyCarlaEnv:
    """
    محیط فیک برای تستِ کدهای RL بدون نیاز به CARLA
    """
    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        obs = {
            "grid": np.random.uniform(-1, 1, size=(5, 25, 11)).astype(np.float32),
            "scalars": np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
        }
        return obs

    def step(self, action):
        self.current_step += 1
        obs = {
            "grid": np.random.uniform(-1, 1, size=(5, 25, 11)).astype(np.float32),
            "scalars": np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
        }
        reward = float(np.random.randn())
        done = self.current_step >= self.max_steps
        info = {}
        return obs, reward, done, info