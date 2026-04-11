import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import argparse
import json
import time
from . import config
from .utils.viz import *
from .utils.stats import *

ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = Path(__file__).resolve().parent

DEMO_DIRS = config.DEMO_LIST
OUT_PATH = config.DATASET_PATH
RNG_SEED = config.BUILD_RNG_SEED

# Window Size configuration 
WINDOW_SIZE = config.WINDOW_SIZE

# Termination Filtering
DROP_TERMINATED = config.DROP_TERMINATED
DROP_LAST_N_BEFORE_TERMINATION = config.DROP_LAST_N_BEFORE_TERMINATION

# Idle / Silence Filtering
FILTER_IDLE_FRAMES = config.FILTER_IDLE_FRAMES
IDLE_FILTER_MODE = config.IDLE_FILTER_MODE
IDLE_SPEED_THRESHOLD = config.IDLE_SPEED_THRESHOLD
IDLE_THROTTLE_THRESHOLD = config.IDLE_THROTTLE_THRESHOLD
IDLE_BRAKE_THRESHOLD = config.IDLE_BRAKE_THRESHOLD

# Action Maps & Discrete
SPEED_MAP = config.SPEED_MAP
TURN_MAP = config.TURN_MAP
JOINT_KEEP_PROBS = config.JOINT_KEEP_PROBS


# Observation Bounds
OBS_BOUNDS = config.OBS_BOUNDS

# Optional simplification
SIMPLIFY_ACTIONS = config.SIMPLIFY_ACTIONS
REMOVE_REVERSE = config.REMOVE_REVERSE
REMOVE_NO_TURN = config.REMOVE_NO_TURN
 
# Mirror augmentation
MIRROR_DATASET = config.MIRROR_DATASET
MIRROR_STEERING_THRESHOLD = config.MIRROR_STEERING_THRESHOLD


# Target keys mapping
TARGET_RENAME_KEYS = ["obs_throttle", "obs_brake", "obs_steering_angle", "obs_reverse"]
GRID_KEYS = ["obs_presence", "obs_speed_x", "obs_speed_y"]


def init_stats():
    """Initializes and returns a clean dictionary for tracking dataset statistics."""
    return {
        "total_frames": 0,
        "obs_violations": Counter(),
        "obs_violation_frames": 0,
        "idle_frames_trimmed": 0,
        "terminated_dropped": 0,
        "truncated_dropped": 0,
        "pre_termination_dropped": 0,
        "sampling_dropped": 0, 
        "kept": 0,
    }

def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if hasattr(obj, "__int__") and not isinstance(obj, bool):
        return int(obj)
    return obj

def save_dataset_meta(stats, obs_keys, obs_shapes, total_kept, files, mode):
    """Save dataset metadata (.meta.json) next to the .npz file."""
    clean_stats = to_plain(stats)
    pipeline_config = {
        "mode": mode,
        "window_size": WINDOW_SIZE,
        "simplify_actions": SIMPLIFY_ACTIONS,
        "remove_reverse": REMOVE_REVERSE,
        "remove_no_turn": REMOVE_NO_TURN,
        "idle_trim_enabled": FILTER_IDLE_FRAMES,
        "idle_speed_threshold": IDLE_SPEED_THRESHOLD,
        "idle_throttle_threshold": IDLE_THROTTLE_THRESHOLD,
        "mirror_enabled": MIRROR_DATASET,
        "mirror_steering_threshold": MIRROR_STEERING_THRESHOLD,
    }


    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(OUT_PATH.resolve().relative_to(PROJECT_ROOT)),
        "source_files": [str(Path(f).resolve().relative_to(PROJECT_ROOT)) for f in files],
        "total_samples": int(total_kept),
        "observation_keys": list(obs_keys),
        "observation_shapes": {k: list(v) for k, v in obs_shapes.items()},
        "stats": clean_stats,
        "pipeline_config": pipeline_config,
    }

    meta_path = OUT_PATH.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] saved → {meta_path}")

def gather_demo_files(dirs):
    """Collects all .npz demo files from given directories."""
    all_files = []
    for d in dirs:
        path = Path(d)

        if not path.exists():
            print(f"[WARN] Directory does not exist: {path}")
            continue

        found = sorted(path.glob("*.npz"))
        print(f"[DATA] Found {len(found)} demos in {path.resolve()}")
        all_files.extend(found)

    assert len(all_files) > 0, "No demo files found in any provided directory."
    print(f"[DATA] Total demos collected: {len(all_files)}\n")

    return sorted(all_files)




# ==============================================================================
# 1. FILTERING LOGIC
# ==============================================================================

def get_idle_trim_mask(d):
    speed = np.abs(d["obs_ego_speed_x"].reshape(-1))
    throttle = d["obs_throttle"].reshape(-1)
    brake = d["obs_brake"].reshape(-1)
    
    is_idle = (speed < IDLE_SPEED_THRESHOLD) & \
              (throttle < IDLE_THROTTLE_THRESHOLD) & \
              (brake < IDLE_BRAKE_THRESHOLD)
              
    trim_mask = np.zeros(len(speed), dtype=bool)
    if IDLE_FILTER_MODE == "all":
        trim_mask = is_idle
    elif IDLE_FILTER_MODE == "start":
        non_idle_idx = np.where(~is_idle)[0]
        if len(non_idle_idx) > 0:
            trim_mask[:non_idle_idx[0]] = True
        else:
            trim_mask[:] = True
    else:
        raise ValueError(f"Unknown IDLE_FILTER_MODE: {IDLE_FILTER_MODE}")       
            
    return trim_mask

def validate_observations(d, stats):
    T = next(iter(d.values())).shape[0]
    valid_mask = np.ones(T, dtype=bool)
    for obs_key, bounds in OBS_BOUNDS.items():
        if obs_key not in d: continue
        x_flat = d[obs_key].reshape(T, -1)
        violated_any = ((x_flat < bounds["low"]) | (x_flat > bounds["high"])).any(axis=1)
        if violated_any.sum() > 0:
            stats["obs_violations"][obs_key] += int(violated_any.sum())
            valid_mask &= ~violated_any
    stats["obs_violation_frames"] += int((~valid_mask).sum())
    return valid_mask

def should_keep_discrete(action, rng):
    speed, turn = map(int, action)
    return rng.random() < JOINT_KEEP_PROBS.get((speed, turn), 1.0)

def compute_episode_mask(d, stats, mode, rng):
    """Evaluates all filters on the raw episode. Returns boolean mask of shape (T,)."""
    T = d["actions"].shape[0]
    stats["total_frames"] += T
    mask = validate_observations(d, stats)

    if FILTER_IDLE_FRAMES:
        idle_trim_mask = get_idle_trim_mask(d)
        stats["idle_frames_trimmed"] += int(idle_trim_mask.sum())
        mask[idle_trim_mask] = False

    terminated = d["terminated"].astype(bool)
    truncated = d["truncated"].astype(bool)
    term_idx = int(np.argmax(terminated)) if terminated.any() else None

    if DROP_TERMINATED:
        stats["terminated_dropped"] += np.sum(terminated)
        stats["truncated_dropped"] += np.sum(truncated)
        mask &= ~(terminated | truncated)

    if term_idx is not None and DROP_LAST_N_BEFORE_TERMINATION > 0:
        start = max(0, term_idx - DROP_LAST_N_BEFORE_TERMINATION)
        stats["pre_termination_dropped"] += np.sum(mask[start:term_idx + 1])
        mask[start:term_idx + 1] = False

    if mode == "discrete":
        actions = d["actions"]
        for t in range(T):
            if mask[t]:
                if not should_keep_discrete(actions[t], rng):
                    mask[t] = False
                    stats["sampling_dropped"] += 1

    return mask

# ==============================================================================
# 2. DATASET AUGMENTATION & FORMATTING
# ==============================================================================

def remap_presence_grid(out_obs, mapping=None, verify=True):
    """Remaps presence IDs across ALL time steps seamlessly."""
    if mapping is None:
        mapping = {0: 0, 1: 1, 2: 2, 9: 3}

    grid = out_obs["obs_presence"]

    if verify:
        values, counts = np.unique(grid, return_counts=True)
        print("\n[PRESENCE] BEFORE remapping:")
        for v, c in zip(values, counts):
            print(f"value {v}: {c}")

    remapped = np.zeros_like(grid, dtype=np.int32)
    for old, new in mapping.items():
        remapped[grid == old] = new

    out_obs["obs_presence"] = remapped.astype(np.float32)

    if verify:
        values, counts = np.unique(remapped, return_counts=True)
        print("\n[PRESENCE] AFTER remapping:")
        for v, c in zip(values, counts):
            print(f"value {v}: {c}")

    return out_obs


def apply_mirror_augmentation(out_obs, out_actions, mode):
    """
    Mirrors dataset. Flips grids horizontally and negates lateral/steering variables.
    Works perfectly across temporal windows of shape [Batch, Window, H, W].
    """
    if "obs_steering_angle" not in out_obs:
        return out_obs, out_actions

    if mode == "continuous":
        steer = out_obs["obs_steering_angle"].reshape(-1)
        mirror_mask = np.abs(steer) > MIRROR_STEERING_THRESHOLD
    else:
        left_idx = TURN_MAP.get("left", 0) if isinstance(TURN_MAP, dict) else 0
        right_idx = TURN_MAP.get("right", 1) if isinstance(TURN_MAP, dict) else 1
        turn = out_actions[:, 1]
        mirror_mask = (turn == left_idx) | (turn == right_idx)

    m = int(mirror_mask.sum())
    if m == 0: return out_obs, out_actions

    obs_m = {k: v[mirror_mask].copy() for k, v in out_obs.items()}
    act_m = out_actions[mirror_mask].copy()

    # Mirror Actions
    if mode == "discrete":
        act_m[act_m[:, 1] == left_idx, 1] = right_idx
        act_m[act_m[:, 1] == right_idx, 1] = left_idx

    # Mirror Observations
    for k, v in obs_m.items():
        # Negate lateral scalars
        if k in ("obs_steering_angle", "obs_lane_angle", "obs_ego_in_lane_position_x"):
            obs_m[k] = -v
        # Flip grids horizontally (axis=-1 is width)
        elif k in GRID_KEYS:
            flipped = np.flip(v, axis=-1)
            # Lateral speed left becomes right, so it must ALSO be negated
            if k == "obs_speed_y":
                flipped = -flipped
            obs_m[k] = flipped

    out_obs_aug = {k: np.concatenate([out_obs[k], obs_m[k]], axis=0) for k in out_obs}
    out_actions_aug = np.concatenate([out_actions, act_m], axis=0)
    
    print(f"[MIRROR] Added {m} mirrored samples (dataset size -> {len(out_actions_aug)})")
    return out_obs_aug, out_actions_aug

def simplify_actions(out_obs, out_actions):
    """Simplifies the discrete action space by optionally removing Reverse / No Turn."""
    speed, turn = out_actions[:, 0], out_actions[:, 1]
    keep_mask = np.ones(len(out_actions), dtype=bool)

    if REMOVE_REVERSE: keep_mask &= (speed != 3)
    if REMOVE_NO_TURN: keep_mask &= (turn != 2)

    new_act = out_actions[keep_mask].copy()
    new_obs = {k: v[keep_mask] for k, v in out_obs.items()}

    for col in [0, 1]:
        unique_vals = sorted(np.unique(new_act[:, col]))
        mapping = {old: new for new, old in enumerate(unique_vals)}
        for old, new in mapping.items():
            new_act[new_act[:, col] == old, col] = new

    return new_obs, new_act

# ==============================================================================
# 3. PIPELINE PASSES (Memory-efficient Pre-allocation)
# ==============================================================================

def pass_1_compute_masks(files, stats, mode, rng):
    """Calculates exactly how many valid windows each file yields."""
    keep_masks = []
    total_kept = 0
    obs_keys = None
    obs_shapes = {}

    for f in files:
        d = np.load(f, allow_pickle=True)
        if obs_keys is None:
            obs_keys = [k for k in d.files if k.startswith("obs_")]
            for k in obs_keys:
                obs_shapes[k] = (1,) if d[k].ndim == 1 else d[k].shape[1:]

        mask = compute_episode_mask(d, stats, mode, rng)
        
        # A frame is only a valid TARGET if we have enough physical history for it
        # i.e., index >= WINDOW_SIZE - 1
        valid_count = np.sum(mask[WINDOW_SIZE - 1:])
        total_kept += int(valid_count)
        keep_masks.append(mask)

    stats["kept"] = total_kept
    return keep_masks, total_kept, obs_keys, obs_shapes

def pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes):
    """Fills pre-allocated temporal arrays seamlessly directly from raw history."""
    out_obs = {}
    
    # Pre-allocate
    for k in obs_keys:
        if k in GRID_KEYS:
            H, W = obs_shapes[k]
            out_obs[k] = np.empty((total_kept, WINDOW_SIZE, H, W), dtype=np.float32)
        else:
            out_obs[k] = np.empty((total_kept, *obs_shapes[k]), dtype=np.float32)
            
    out_actions = np.empty((total_kept, 2), dtype=np.int64)

    idx = 0
    for f, mask in zip(files, keep_masks):
        d = np.load(f, allow_pickle=True)
        
        # Find valid targets that have enough history
        valid_indices = np.where(mask)[0]
        valid_indices = valid_indices[valid_indices >= (WINDOW_SIZE - 1)]

        for i in valid_indices:
            # 1. Fill Temporal Grids
            for k in GRID_KEYS:
                # Want chronological order: (t-2, t-1, t)
                out_obs[k][idx] = d[k][i-WINDOW_SIZE+1:i+1]


            # 2. Fill Scalars (Target frame t only)
            for k in obs_keys:
                if k not in GRID_KEYS:
                    arr = d[k][i]
                    if arr.ndim == 0:
                        arr = np.array([arr], dtype=np.float32)
                    out_obs[k][idx] = arr.astype(np.float32)

            out_actions[idx] = d["actions"][i].astype(np.int64)
            idx += 1

    return out_obs, out_actions

# ==============================================================================
# 4. MAIN ORCHESTRATOR
# ==============================================================================

def run_pipeline(mode, visualize=False):
    print(f"[{mode.upper()}] Starting dataset pipeline with Window Size {WINDOW_SIZE}...")
    rng = np.random.default_rng(RNG_SEED)
    stats = init_stats()
    files = gather_demo_files(DEMO_DIRS)

    # --- PASS 1 ---
    print(f"[{mode.upper()}] Computing filtering masks...")
    keep_masks, total_kept, obs_keys, obs_shapes = pass_1_compute_masks(files, stats, mode, rng)
    assert total_kept > 0, "No samples kept! Check filtering/sampling rules."
    print(f"[{mode.upper()}] Total windows kept for dataset: {total_kept}")

    # --- PASS 2 ---
    print(f"[{mode.upper()}] Building temporal windows...")
    out_obs, out_actions = pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes)
    print_minmax_summary(out_obs)
    # --- Post Processing ---
    out_obs = remap_presence_grid(out_obs)

    if MIRROR_DATASET:
        out_obs, out_actions = apply_mirror_augmentation(out_obs, out_actions, mode)

    if mode == "discrete" and SIMPLIFY_ACTIONS:
        out_obs, out_actions = simplify_actions(out_obs, out_actions)


    print_trim_statistics(stats)
    if mode == "discrete":
        print_distribution_summary(out_actions, files)
        if visualize:
            visualize_discrete(out_actions, out_obs)
            pass
    else:
        print_obs_continuous_stats(out_obs)
        if visualize:
            visualize_continuous(out_obs)
            pass

    # --- Rename targets & Save ---
    for k in TARGET_RENAME_KEYS:
        if k in out_obs:
            out_obs[f"target_{k[4:]}"] = out_obs.pop(k)

    save_dict = {**out_obs, "actions": out_actions}
    np.savez_compressed(OUT_PATH, **save_dict)
    save_dataset_meta(stats, obs_keys, obs_shapes, total_kept, files, mode=mode)
    print(f"[{mode.upper()}] ✅ Successfully saved to: {OUT_PATH.resolve()}")





# ==============================================================================
# 5. VIZ
# ==============================================================================



def compute_action_stats_from_dataset(out_actions):
    speed = out_actions[:, 0]
    turn = out_actions[:, 1]

    speed_counts = np.bincount(speed, minlength=5)
    turn_counts = np.bincount(turn, minlength=4)

    joint_counts = Counter(map(tuple, out_actions))

    stats = {
        "speed_counts": speed_counts,
        "turn_counts": turn_counts,
        "joint_counts": joint_counts
    }

    return stats


def visualize_discrete(out_actions, out_obs):
    """Discrete visualization entry point."""

    stats = compute_action_stats_from_dataset(out_actions)

    features = {k: v.reshape(-1) for k, v in out_obs.items()}

    plot_discrete_actions(stats)
    plot_joint_heatmap(stats["joint_counts"])
    plot_feature_distributions(features)


def visualize_continuous(out_obs):
    """Continuous visualization entry point."""

    features = {k: v.reshape(-1) for k, v in out_obs.items()}

    plot_continuous_deltas(features)
    plot_continuous_2d_relationships(features)
    plot_feature_distributions(features)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--action-mode", type=str, default=config.ACTION_MODE, choices=["discrete", "continuous"])
    args = parser.parse_args()

    if args.out_path:
        OUT_PATH = Path(args.out_path)

    run_pipeline(mode=args.action_mode, visualize=args.visualize or config.BUILD_VISUALIZE)
