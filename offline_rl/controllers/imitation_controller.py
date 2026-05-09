import time
import json
import numpy as np
import torch
from pathlib import Path
from config import offline_rl_config, bc_config
from agents.bc.imitation_policy import ImitationPolicy
from utils.obs_wrapper import CarlaObsWrapper


class ImitationController:
    def __init__(self, env, model_path, record_dir="offline_demos", base_name="offline_data",
                 max_steps=2000, epsilon=0.1, device="cuda"):
        self.env = env
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.max_steps = max_steps
        self.device = device
        self.epsilon = epsilon
        self.ep_idx = 0

        # Load model using the same logic as BC evaluation
        self.policy = self._load_policy_from_checkpoint(model_path)
        
        # Extract normalization stats from the BC experiment config
        norm_stats = self._load_norm_stats_from_config(model_path)
        self.wrapper = CarlaObsWrapper(
            norm_stats=norm_stats,
            device=self.device,
            action_mode="continuous"
        )

        # Action space bounds for random exploration
        # throttle ∈ [0,1], brake ∈ [0,1], steer ∈ [-1,1]
        self.action_low = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def _load_policy_from_checkpoint(self, model_path):
        """
        Load the BC policy using the same logic as evaluate_bc.py.
        Reads checkpoint metadata (grid_channels, scalar_dim, mode) to construct the correct architecture.
        """
        model_path = Path(model_path)
        print(f"[INFO] Loading BC model from {model_path}")
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract metadata from checkpoint
        mode = ckpt.get("mode", "continuous")
        grid_channels = ckpt.get("grid_channels", offline_rl_config.GRID_CHANNELS)
        scalar_dim = ckpt.get("scalar_dim", offline_rl_config.SCALAR_DIM)
        
        # Build policy with correct architecture
        policy_kwargs = {
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
            policy = ImitationPolicy(
                mode="discrete",
                n_speed=ckpt["n_speed"],
                n_turn=ckpt["n_turn"],
                **policy_kwargs
            ).to(self.device)
        else:
            policy = ImitationPolicy(
                mode="continuous",
                is_gaussian=False,  # We only need deterministic output for data collection
                **policy_kwargs
            ).to(self.device)
        
        # Verify grid channel consistency
        expected_channels = bc_config.WINDOW_SIZE * (5 if bc_config.USE_ONE_HOT_GRID else 3)
        assert expected_channels == grid_channels, \
            f"Grid channel mismatch! model={grid_channels}, expected={expected_channels}"
        
        # Load weights
        if "model_state_dict" in ckpt:
            policy.load_state_dict(ckpt["model_state_dict"])
        else:
            policy.load_state_dict(ckpt)
        
        policy.eval()
        
        print(f"  mode: {mode}")
        print(f"  grid_channels: {grid_channels}")
        print(f"  scalar_dim: {scalar_dim}")
        
        return policy

    def _load_norm_stats_from_config(self, model_path):
        """Extract normalization stats from the BC experiment's config.json."""
        try:
            exp_dir = Path(model_path).parents[1]
            config_path = exp_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    data = json.load(f)
                stats = data.get("dataset_meta", {}).get("normalization_stats", {})
                if stats:
                    print(f"[INFO] Loaded normalization stats from {config_path}")
                    return stats
        except Exception:
            pass
        
        print("[WARN] No normalization stats found. Using empty stats.")
        return {}

    def _prepare_tensors(self, obs):
        """Use CarlaObsWrapper to convert raw observation into grid and scalars tensors."""
        grid, scalars = self.wrapper.preprocess(obs)
        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
        scal_t = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(self.device)
        return grid_t, scal_t

    def _get_action(self, grid_t, scal_t):
        """
        Sample an action:
        - With probability epsilon: uniformly random in valid bounds
        - Otherwise: BC deterministic action (clipped to bounds)
        """
        if np.random.random() < self.epsilon:
            # Random action in environment bounds
            action = np.random.uniform(self.action_low, self.action_high).astype(np.float32)
        else:
            with torch.no_grad():
                # BC policy forward pass, shape [1, 3] -> [throttle, brake, steer]
                raw = self.policy(grid_t, scal_t)
                # Handle both tuple output (mean, std) and direct output
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw = raw.squeeze(0).cpu().numpy()
            
            # Clip to valid ranges
            action = np.clip(raw, self.action_low, self.action_high)
        
        return action

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            self.wrapper.reset()
            steps = []
            done = False
            t = 0

            print(f"--- Starting Offline Data Collection Episode {self.ep_idx} ---")

            while not done and t < self.max_steps:
                grid_t, scal_t = self._prepare_tensors(obs)
                action = self._get_action(grid_t, scal_t)  # [throttle, brake, steer]

                # Step environment with raw action (env applies _process_action internally)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store raw observation and raw action
                obs_np = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
                steps.append({
                    "obs": obs_np,
                    "action": action,          # raw, before any post-processing
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": bool(done)
                })

                obs = next_obs
                t += 1

            if len(steps) > 0:
                self._save_episode(steps)
            self.ep_idx += 1

    def _save_episode(self, steps):
        """Save a single episode as a compressed .npz file."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = self.record_dir / f"{self.base_name}_{ts}_ep{self.ep_idx:05d}.npz"

        obs_keys = list(steps[0]["obs"].keys())

        arrays = {}
        for key in obs_keys:
            arrays[f"obs_{key}"] = np.stack([s["obs"][key] for s in steps], axis=0).astype(np.float32)

        arrays["actions"]     = np.stack([s["action"] for s in steps], axis=0).astype(np.float32)
        arrays["rewards"]     = np.array([s["reward"] for s in steps], dtype=np.float32)
        arrays["terminated"]  = np.array([s["terminated"] for s in steps], dtype=bool)
        arrays["truncated"]   = np.array([s["truncated"] for s in steps], dtype=bool)
        arrays["done"]        = np.array([s["done"] for s in steps], dtype=bool)

        np.savez_compressed(out_file, **arrays)
        print(f"[SAVED] {out_file} | Steps: {len(steps)}")