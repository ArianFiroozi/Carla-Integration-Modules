import numpy as np
import torch
from pathlib import Path
from imitation.datasets.bc_dataset import BCDatasetContinuous
class SACReplayBuffer:
    """
    Dual-Source Replay Buffer for SAC.
    Maintains a permanent offline expert dataset partition and a 
    circular online exploration partition to prevent data dilution.
    """
    def __init__(self, capacity, device="cpu", expert_ratio=0.5):
        self.capacity = int(capacity)
        self.device = device
        self.expert_ratio = expert_ratio

        # Online Buffer Trackers (Circular FIFO)
        self.ptr = 0
        self.online_size = 0
        self.total_size = 0

        # Lazily initialized arrays for Online Exploration Data
        self.grid_obs = None
        self.scalar_obs = None
        self.actions = None
        self.rewards = None
        self.next_grid_obs = None
        self.next_scalar_obs = None
        self.dones = None

        # Permanent Expert Dataset Arrays
        self.has_expert = False
        self.expert_size = 0
        self.exp_grid_obs = None
        self.exp_scalar_obs = None
        self.exp_actions = None
        self.exp_rewards = None
        self.exp_next_grid_obs = None
        self.exp_scalar_obs_next = None
        self.exp_dones = None

    def _lazy_init(self, grid_shape, scalar_dim, action_dim):
        self.grid_obs = np.zeros((self.capacity, *grid_shape), dtype=np.float32)
        self.scalar_obs = np.zeros((self.capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_grid_obs = np.zeros((self.capacity, *grid_shape), dtype=np.float32)
        self.next_scalar_obs = np.zeros((self.capacity, scalar_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def load_offline_dataset(self, npz_path):
        """Loads and locks compiled expert data using the BCDataset loader."""
        print(f"[BUFFER] Loading permanent expert data from: {npz_path}")
        
        # Instantiate the dataset to handle all one-hot encoding, normalization, and bounds
        expert_dataset = BCDatasetContinuous(str(npz_path), one_hot_presence=True)
        
        self.expert_size = len(expert_dataset)
        
        # Load the raw data to extract rewards and next states
        data = np.load(npz_path, allow_pickle=True)
        dones = data["dones"].astype(np.float32).reshape(-1, 1) if "dones" in data else data["terminated"].astype(np.float32).reshape(-1, 1)
        rewards = data["rewards"].astype(np.float32).reshape(-1, 1)

        print("[BUFFER] Extracting and formatting expert tensors...")
        
        # Pre-allocate numpy arrays to hold the formatted data
        # We grab the first sample to get the exact formatted dimensions
        grid0, scalar0, action0 = expert_dataset[0]
        
        self.exp_grid_obs = np.zeros((self.expert_size, *grid0.shape), dtype=np.float32)
        self.exp_scalar_obs = np.zeros((self.expert_size, scalar0.shape[0]), dtype=np.float32)
        self.exp_actions = np.zeros((self.expert_size, action0.shape[0]), dtype=np.float32)
        self.exp_rewards = rewards
        self.exp_dones = dones
        
        # We need next states for SAC. 
        self.exp_next_grid_obs = np.zeros_like(self.exp_grid_obs)
        self.exp_scalar_obs_next = np.zeros_like(self.exp_scalar_obs)

        # Populate the arrays
        for i in range(self.expert_size):
            g, s, a = expert_dataset[i]
            self.exp_grid_obs[i] = g.numpy()
            self.exp_scalar_obs[i] = s.numpy()
            self.exp_actions[i] = a.numpy()
            
            # The next state is simply the observation at index i+1
            # (The final step will just duplicate its current state as next state, which is fine since done=1)
            next_idx = min(i + 1, self.expert_size - 1)
            g_next, s_next, _ = expert_dataset[next_idx]
            self.exp_next_grid_obs[i] = g_next.numpy()
            self.exp_scalar_obs_next[i] = s_next.numpy()

        self.has_expert = True
        self.total_size = self.online_size + self.expert_size
        print(f"[BUFFER] Locked {self.expert_size} permanent expert transitions in buffer.")
        
        
        
        
        
    def add(self, grid_obs, scalar_obs, action, reward, next_grid_obs, next_scalar_obs, done):
        """Appends a single transition into the circular online partition."""
        grid_obs = np.asarray(grid_obs, dtype=np.float32)
        scalar_obs = np.asarray(scalar_obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_grid_obs = np.asarray(next_grid_obs, dtype=np.float32)
        next_scalar_obs = np.asarray(next_scalar_obs, dtype=np.float32)

        if self.grid_obs is None:
            self._lazy_init(grid_obs.shape, scalar_obs.shape[0], action.shape[0])

        idx = self.ptr
        self.grid_obs[idx] = grid_obs
        self.scalar_obs[idx] = scalar_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_grid_obs[idx] = next_grid_obs
        self.next_scalar_obs[idx] = next_scalar_obs
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.online_size = min(self.online_size + 1, self.capacity)
        self.total_size = self.online_size + self.expert_size

    def sample(self, batch_size):
        assert self.total_size > 0, "Replay buffer is empty!"
        batch_size = int(batch_size)

        if self.has_expert and self.online_size > 0:
            # Determine split bounds
            n_expert = int(batch_size * self.expert_ratio)
            n_online = batch_size - n_expert

            exp_idxs = np.random.randint(0, self.expert_size, size=n_expert)
            on_idxs = np.random.randint(0, self.online_size, size=n_online)

            # Reconstruct batch arrays natively via stacked lookups
            g = np.concatenate([self.exp_grid_obs[exp_idxs], self.grid_obs[on_idxs]], axis=0)
            s = np.concatenate([self.exp_scalar_obs[exp_idxs], self.scalar_obs[on_idxs]], axis=0)
            a = np.concatenate([self.exp_actions[exp_idxs], self.actions[on_idxs]], axis=0)
            r = np.concatenate([self.exp_rewards[exp_idxs], self.rewards[on_idxs]], axis=0)
            g_next = np.concatenate([self.exp_next_grid_obs[exp_idxs], self.next_grid_obs[on_idxs]], axis=0)
            s_next = np.concatenate([self.exp_scalar_obs_next[exp_idxs], self.next_scalar_obs[on_idxs]], axis=0)
            d = np.concatenate([self.exp_dones[exp_idxs], self.dones[on_idxs]], axis=0)

        elif self.has_expert:
            # Replay buffer has only expert data (e.g., during startup warmup steps)
            idxs = np.random.randint(0, self.expert_size, size=batch_size)
            g, s, a, r = self.exp_grid_obs[idxs], self.exp_scalar_obs[idxs], self.exp_actions[idxs], self.exp_rewards[idxs]
            g_next, s_next, d = self.exp_next_grid_obs[idxs], self.exp_scalar_obs_next[idxs], self.exp_dones[idxs]
        else:
            # Replay buffer has only online exploration data
            idxs = np.random.randint(0, self.online_size, size=batch_size)
            g, s, a, r = self.grid_obs[idxs], self.scalar_obs[idxs], self.actions[idxs], self.rewards[idxs]
            g_next, s_next, d = self.next_grid_obs[idxs], self.next_scalar_obs[idxs], self.dones[idxs]

        # Push uniform float matrices out as device-resident tensors
        return (
            torch.tensor(g, dtype=torch.float32, device=self.device),
            torch.tensor(s, dtype=torch.float32, device=self.device),
            torch.tensor(a, dtype=torch.float32, device=self.device),
            torch.tensor(r, dtype=torch.float32, device=self.device),
            torch.tensor(g_next, dtype=torch.float32, device=self.device),
            torch.tensor(s_next, dtype=torch.float32, device=self.device),
            torch.tensor(d, dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return self.total_size