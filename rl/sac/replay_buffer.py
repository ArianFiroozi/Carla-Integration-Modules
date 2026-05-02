import numpy as np
import torch


class SACReplayBuffer:
    """
    Replay buffer for SAC.
    Stores:
      - grid_obs: (C, H, W)  e.g. (5, 25, 11)
      - scalar_obs: (S,)     e.g. (8,)
      - action: (A,)         e.g. (3,)
      - reward: float
      - next_grid_obs
      - next_scalar_obs
      - done: float (1.0 if terminal, else 0.0)
    """

    def __init__(self, capacity, device="cpu"):
        self.capacity = int(capacity)
        self.device = device

        self.ptr = 0
        self.size = 0

        # buffers will be initialized after first add (lazy init)
        self.grid_obs = None
        self.scalar_obs = None
        self.actions = None
        self.rewards = None
        self.next_grid_obs = None
        self.next_scalar_obs = None
        self.dones = None

    def _lazy_init(self, grid_obs, scalar_obs, action):
        grid_shape = grid_obs.shape  # (C, H, W)
        scalar_dim = scalar_obs.shape[0]
        action_dim = action.shape[0]

        self.grid_obs = np.zeros((self.capacity, *grid_shape), dtype=np.float32)
        self.scalar_obs = np.zeros((self.capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_grid_obs = np.zeros((self.capacity, *grid_shape), dtype=np.float32)
        self.next_scalar_obs = np.zeros((self.capacity, scalar_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, grid_obs, scalar_obs, action, reward, next_grid_obs, next_scalar_obs, done):
        # ensure numpy arrays
        grid_obs = np.asarray(grid_obs, dtype=np.float32)
        scalar_obs = np.asarray(scalar_obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_grid_obs = np.asarray(next_grid_obs, dtype=np.float32)
        next_scalar_obs = np.asarray(next_scalar_obs, dtype=np.float32)

        if self.grid_obs is None:
            self._lazy_init(grid_obs, scalar_obs, action)

        idx = self.ptr
        self.grid_obs[idx] = grid_obs
        self.scalar_obs[idx] = scalar_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_grid_obs[idx] = next_grid_obs
        self.next_scalar_obs[idx] = next_scalar_obs
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        assert self.size > 0, "Replay buffer is empty!"
        batch_size = int(batch_size)
        idxs = np.random.randint(0, self.size, size=batch_size)

        grid_obs = torch.tensor(self.grid_obs[idxs], dtype=torch.float32, device=self.device)
        scalar_obs = torch.tensor(self.scalar_obs[idxs], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[idxs], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        next_grid_obs = torch.tensor(self.next_grid_obs[idxs], dtype=torch.float32, device=self.device)
        next_scalar_obs = torch.tensor(self.next_scalar_obs[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device)

        return grid_obs, scalar_obs, actions, rewards, next_grid_obs, next_scalar_obs, dones

    def __len__(self):
        return self.size
