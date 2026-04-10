import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import argparse
from . import config
from .utils.viz import *
from .utils.stats import *
import json
import time

ROOT = Path(__file__).resolve().parents[0]

DEMO_DIRS = config.DEMO_LIST
OUT_PATH = config.DATASET_PATH
RNG_SEED = config.BUILD_RNG_SEED

# Termination Filtering
DROP_TERMINATED = config.DROP_TERMINATED
DROP_LAST_N_BEFORE_TERMINATION = config.DROP_LAST_N_BEFORE_TERMINATION

# Idle / Silence Filtering
FILTER_IDLE_FRAMES = config.FILTER_IDLE_FRAMES
IDLE_FILTER_MODE = config.IDLE_FILTER_MODE
IDLE_SPEED_THRESHOLD = config.IDLE_SPEED_THRESHOLD
IDLE_THROTTLE_THRESHOLD = config.IDLE_THROTTLE_THRESHOLD
IDLE_BRAKE_THRESHOLD = config.IDLE_BRAKE_THRESHOLD

# Action Maps (for discrete)
SPEED_MAP = config.SPEED_MAP
TURN_MAP = config.TURN_MAP

# Discrete sampling config
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
        "sampling_dropped": 0,  # used only in discrete
        "kept": 0,
    }

PROJECT_ROOT = Path(__file__).resolve().parent

def to_plain(obj):
    """Recursively convert numpy/scalar types to json-friendly Python types."""
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
    """
    Save dataset metadata (.meta.json) next to the .npz file.

    Args:
        stats: dict of dataset statistics
        obs_keys: observation keys list
        obs_shapes: dict of their shapes
        total_kept: number of kept frames
        files: list of demo file paths
        mode: 'continuous' or 'discrete'
    """

    clean_stats = to_plain(stats)

    pipeline_config = {
        "mode": mode,
        "downsample": getattr(config, "DOWN_SAMPLE", None),
        "simplify_actions": getattr(config, "SIMPLIFY_ACTIONS", None),
        "remove_reverse": getattr(config, "REMOVE_REVERSE", None),
        "remove_no_turn": getattr(config, "REMOVE_NO_TURN", None),
        "idle_speed_threshold": getattr(config, "IDLE_SPEED_THRESHOLD", None),
        "idle_throttle_threshold": getattr(config, "IDLE_THROTTLE_THRESHOLD", None),
        "idle_trim_enabled": config.FILTER_IDLE_FRAMES,
        "mirror_enabled": getattr(config, "MIRROR_DATASET", False),
        "mirror_steering_threshold": getattr(config, "MIRROR_STEERING_THRESHOLD", None),
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






def get_idle_trim_mask(d):
    """
    Returns a boolean mask of frames to trim based on idle conditions:
    speed < threshold AND throttle < threshold AND brake < threshold.
    """
    speed = np.abs(d["obs_ego_speed_x"].reshape(-1))
    throttle = d["obs_throttle"].reshape(-1)
    brake = d["obs_brake"].reshape(-1)
    
    is_idle = (speed < IDLE_SPEED_THRESHOLD) & \
              (throttle < IDLE_THROTTLE_THRESHOLD) & \
              (brake < IDLE_BRAKE_THRESHOLD)
              
    T = len(speed)
    trim_mask = np.zeros(T, dtype=bool)
    
    if IDLE_FILTER_MODE == "all":
        trim_mask = is_idle
    elif IDLE_FILTER_MODE == "start":
        non_idle_idx = np.where(~is_idle)[0]
        if len(non_idle_idx) > 0:
            first_active = non_idle_idx[0]
            trim_mask[:first_active] = True
        else:
            trim_mask[:] = True
    else:
        raise ValueError(f"Unknown IDLE_FILTER_MODE: {IDLE_FILTER_MODE}")
        
    return trim_mask


def validate_observations(d, stats):
    """Validates observations against defined bounds. Updates stats in-place."""
    T = next(iter(d.values())).shape[0]
    valid_mask = np.ones(T, dtype=bool)

    for obs_key, bounds in OBS_BOUNDS.items():
        if obs_key not in d:
            continue
        
        x_flat = d[obs_key].reshape(T, -1)
        violated = (x_flat < bounds["low"]) | (x_flat > bounds["high"])
        violated_any = violated.any(axis=1)
        
        count = int(violated_any.sum())
        if count > 0:
            stats["obs_violations"][obs_key] += count
            valid_mask &= ~violated_any

    stats["obs_violation_frames"] += int((~valid_mask).sum())
    return valid_mask


def remap_presence_grid(out_obs, mapping=None, verify=True):
    """Remaps categorical IDs in obs_presence to contiguous integers."""
    if mapping is None:
        mapping = {0: 0, 1: 1, 2: 2, 9: 3}

    grid = out_obs["obs_presence"]

    if verify:
        values, counts = np.unique(grid, return_counts=True)
        print("\nPresence grid BEFORE remapping:")
        for v, c in zip(values, counts):
            print(f"value {v}: {c}")

    remapped = np.zeros_like(grid, dtype=np.int32)
    for old, new in mapping.items():
        remapped[grid == old] = new

    out_obs["obs_presence"] = remapped.astype(np.float32)

    if verify:
        values, counts = np.unique(remapped, return_counts=True)
        print("\nPresence grid AFTER remapping:")
        for v, c in zip(values, counts):
            print(f"value {v}: {c}")

    return out_obs



def apply_mirror_augmentation_continuous(out_obs, threshold=MIRROR_STEERING_THRESHOLD):
    """
    Duplicates samples by mirroring lane-relative obs + steering.
    Assumes steering signal is in obs_steering_angle (pre-rename).
    Returns a NEW out_obs dict with 2x samples (or close, depending on threshold).
    """
    if "obs_steering_angle" not in out_obs:
        print("[MIRROR] obs_steering_angle not found -> skipping mirror augmentation.")
        return out_obs

    steer = out_obs["obs_steering_angle"].reshape(len(out_obs["obs_steering_angle"]), -1)
    steer_abs = np.abs(steer[:, 0])
    mirror_mask = steer_abs > threshold

    n = len(steer_abs)
    m = int(mirror_mask.sum())
    if m == 0:
        print(f"[MIRROR] No samples above threshold={threshold} -> skipping.")
        return out_obs

    out = {k: v.copy() for k, v in out_obs.items()}

    # Concatenate mirrored subset
    for k, v in out_obs.items():
        v_m = v[mirror_mask].copy()

        # Flip lane-relative signals (only if present)
        if k in ("obs_steering_angle", "obs_lane_angle", "obs_ego_in_lane_position_x"):
            v_m = -v_m
        else:
            if v.ndim == 4:
                print(k, v.shape)

        out[k] = np.concatenate([v, v_m], axis=0)

    print(f"[MIRROR] Continuous: added {m} mirrored samples (from {n}) with threshold={threshold}.")
    return out


def apply_mirror_augmentation_discrete(out_obs, out_actions):
    """
    Duplicates samples by mirroring left/right turn in discrete actions.
    Expects out_actions[:,1] is the TURN index and TURN_MAP in config defines indices.
    """
    # We need to know indices for left/right. Use TURN_MAP if available.
    # Fallback: assume 0=left, 1=right, 2=no_turn, 3=... (won't be touched)
    left_idx = TURN_MAP.get("left", 0) if isinstance(TURN_MAP, dict) else 0
    right_idx = TURN_MAP.get("right", 1) if isinstance(TURN_MAP, dict) else 1

    turn = out_actions[:, 1]
    mirror_mask = (turn == left_idx) | (turn == right_idx)

    m = int(mirror_mask.sum())
    if m == 0:
        print("[MIRROR] Discrete: no left/right samples -> skipping.")
        return out_obs, out_actions

    obs_m = {k: v[mirror_mask].copy() for k, v in out_obs.items()}
    act_m = out_actions[mirror_mask].copy()

    # swap left <-> right
    act_m[act_m[:, 1] == left_idx, 1] = right_idx
    act_m[act_m[:, 1] == right_idx, 1] = left_idx

    # Mirror lane-relative obs if present
    for k in obs_m:
        if k in ("obs_lane_angle", "obs_ego_in_lane_position_x"):
            obs_m[k] = -obs_m[k]

    out_obs2 = {k: np.concatenate([out_obs[k], obs_m[k]], axis=0) for k in out_obs}
    out_actions2 = np.concatenate([out_actions, act_m], axis=0)

    print(f"[MIRROR] Discrete: added {m} mirrored samples.")
    return out_obs2, out_actions2




def gather_demo_files(dirs):
    """Collects all .npz demo files from given directories."""
    all_files = []
    for d in dirs:
        path = Path(d)
        if not path.exists():
            print(f"[WARN] Directory does not exist: {path}")
            continue
        found = sorted(path.glob("*.npz"))
        print(f"Found {len(found)} demos in {path.resolve()}")
        all_files.extend(found)

    assert len(all_files) > 0, "No demo files found in any provided directory."
    return sorted(all_files)






def should_keep(action, rng):
    """Determines if an action should be kept based on JOINT_KEEP_PROBS."""
    speed, turn = map(int, action)
    keep_prob = JOINT_KEEP_PROBS.get((speed, turn), 1.0)
    return rng.random() < keep_prob


def discrete_build_keep_mask(d, stats, rng):
    """
    Discrete-specific keep mask:
    - validate observations
    - idle trimming
    - termination trimming
    - sampling based on discrete actions
    """
    actions = d["actions"]
    terminated = d["terminated"].astype(bool)
    truncated = d["truncated"].astype(bool)
    T = actions.shape[0]
    
    stats["total_frames"] += T
    mask = validate_observations(d, stats)

    if FILTER_IDLE_FRAMES:
        idle_trim_mask = get_idle_trim_mask(d)
        trimmed_count = int(idle_trim_mask.sum())
        if trimmed_count > 0:
            stats["idle_frames_trimmed"] += trimmed_count
            mask[idle_trim_mask] = False

    term_idx = int(np.argmax(terminated)) if terminated.any() else None

    if DROP_TERMINATED:
        stats["terminated_dropped"] += np.sum(terminated)
        stats["truncated_dropped"] += np.sum(truncated)
        mask &= ~(terminated | truncated)

    if term_idx is not None and DROP_LAST_N_BEFORE_TERMINATION > 0:
        start = max(0, term_idx - DROP_LAST_N_BEFORE_TERMINATION)
        stats["pre_termination_dropped"] += np.sum(mask[start:term_idx + 1])
        mask[start:term_idx + 1] = False

    keep = np.zeros(T, dtype=bool)
    for t in range(T):
        if not mask[t]:
            continue
        if should_keep(actions[t], rng):
            keep[t] = True
        else:
            stats["sampling_dropped"] += 1

    stats["kept"] += int(keep.sum())
    return keep


def discrete_pass_1_compute_masks(files, stats, rng):
    """PASS 1 for discrete: compute keep masks using discrete rules."""
    keep_masks = []
    total_kept = 0
    obs_keys = None
    obs_shapes = {}

    for f in files:
        d = np.load(f, allow_pickle=True)
        
        if obs_keys is None:
            obs_keys = [k for k in d.files if k.startswith("obs_")]
            for k in obs_keys:
                arr = d[k]
                obs_shapes[k] = (1,) if arr.ndim == 1 else arr.shape[1:]

        keep = discrete_build_keep_mask(d, stats, rng)
        keep_masks.append(keep)
        total_kept += int(keep.sum())

    assert total_kept > 0, "No samples kept. Loosen sampling/drop rules for discrete."
    return keep_masks, total_kept, obs_keys, obs_shapes


def discrete_pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes):
    """PASS 2 for discrete with temporal stack (t, t-1, t-2)."""

    grid_keys = ["obs_presence", "obs_speed_x", "obs_speed_y"]
    scalar_keys = [k for k in obs_keys if k not in grid_keys]

    H, W = obs_shapes["obs_presence"]

    out_obs = {}

    # temporal grids
    for k in grid_keys:
        out_obs[k] = np.empty((total_kept, 3, H, W), dtype=np.float32)

    # scalars
    for k in scalar_keys:
        out_obs[k] = np.empty((total_kept, *obs_shapes[k]), dtype=np.float32)

    out_actions = np.empty((total_kept, 2), dtype=np.int64)

    idx = 0

    for f, keep in zip(files, keep_masks):

        d = np.load(f, allow_pickle=True)

        presence = d["obs_presence"]
        speed_x = d["obs_speed_x"]
        speed_y = d["obs_speed_y"]
        actions = d["actions"]

        valid_idx = np.where(keep)[0]

        for i in valid_idx:

            if i < 2:
                continue

            out_obs["obs_presence"][idx, 0] = presence[i]
            out_obs["obs_presence"][idx, 1] = presence[i-1]
            out_obs["obs_presence"][idx, 2] = presence[i-2]

            out_obs["obs_speed_x"][idx, 0] = speed_x[i]
            out_obs["obs_speed_x"][idx, 1] = speed_x[i-1]
            out_obs["obs_speed_x"][idx, 2] = speed_x[i-2]

            out_obs["obs_speed_y"][idx, 0] = speed_y[i]
            out_obs["obs_speed_y"][idx, 1] = speed_y[i-1]
            out_obs["obs_speed_y"][idx, 2] = speed_y[i-2]

            for k in scalar_keys:
                arr = d[k][i]
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                if arr.ndim == 0:
                    arr = np.array([arr], dtype=np.float32)
                out_obs[k][idx] = arr

            out_actions[idx] = actions[i].astype(np.int64)

            idx += 1

    out_obs = {k: v[:idx] for k, v in out_obs.items()}
    out_actions = out_actions[:idx]

    return out_obs, out_actions





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



def simplify_actions(out_obs, out_actions, remove_reverse=False, remove_no_turn=False):
    """
    Simplifies the action space by optionally removing Reverse and/or No Turn.
    Also reindexes remaining classes so they stay contiguous.
    """
    speed = out_actions[:, 0]
    turn = out_actions[:, 1]

    keep_mask = np.ones(len(out_actions), dtype=bool)

    if remove_reverse:
        keep_mask &= speed != 3  # Reverse index

    if remove_no_turn:
        keep_mask &= turn != 2  # No Turn index

    new_actions = out_actions[keep_mask].copy()
    new_obs = {k: v[keep_mask] for k, v in out_obs.items()}

    unique_speed = sorted(np.unique(new_actions[:, 0]))
    speed_map = {old: new for new, old in enumerate(unique_speed)}
    for old, new in speed_map.items():
        new_actions[new_actions[:, 0] == old, 0] = new

    unique_turn = sorted(np.unique(new_actions[:, 1]))
    turn_map = {old: new for new, old in enumerate(unique_turn)}
    for old, new in turn_map.items():
        new_actions[new_actions[:, 1] == old, 1] = new

    print("\nAction simplification applied:")
    print("Speed mapping:", speed_map)
    print("Turn mapping :", turn_map)

    return new_obs, new_actions




def discrete_pipeline(files, visualize=False):
    """
    Full discrete pipeline:
    - two-pass build
    - presence remap
    - stats + action distribution
    - optional viz
    - optional action simplification
    - save discrete dataset
    """
    print("[DISCRETE] Starting discrete dataset pipeline...")
    rng = np.random.default_rng(RNG_SEED)
    stats = init_stats()

    # PASS 1
    print("[DISCRETE] PASS 1: Computing filtering masks...")
    keep_masks, total_kept, obs_keys, obs_shapes = discrete_pass_1_compute_masks(files, stats, rng)
    print(f"[DISCRETE] Total entries kept for dataset: {total_kept}")

    # PASS 2
    print("[DISCRETE] PASS 2: Compiling final arrays...")
    out_obs, out_actions = discrete_pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes)

    print_minmax_summary(out_obs)
    out_obs = remap_presence_grid(out_obs)

    if MIRROR_DATASET:
        out_obs, out_actions = apply_mirror_augmentation_discrete(out_obs, out_actions)


    print_trim_statistics(stats)
    print_distribution_summary(out_actions, files)

    if visualize:
        print("\n[DISCRETE] Rendering visualization plots...")
        visualize_discrete(out_actions, out_obs)

    if SIMPLIFY_ACTIONS:
        out_obs, out_actions = simplify_actions(
            out_obs,
            out_actions,
            remove_reverse=REMOVE_REVERSE,
            remove_no_turn=REMOVE_NO_TURN
        )

    


    rename_keys = [
        "obs_throttle",
        "obs_brake",
        "obs_steering_angle",
        "obs_reverse"
    ]
    for k in rename_keys:
        if k in out_obs:
            out_obs[f"target_{k[4:]}"] = out_obs.pop(k)

    save_dict = {**out_obs, "actions": out_actions}
    np.savez_compressed(OUT_PATH, **save_dict)
    save_dataset_meta(stats, obs_keys, obs_shapes, total_kept, files, mode=config.ACTION_MODE)
    print(f"[DISCRETE] ✅ Successfully saved to: {OUT_PATH.resolve()}")


def continuous_build_keep_mask(d, stats):
    """
    Continuous-specific keep mask:
    - validate observations
    - idle trimming
    - termination trimming
    - NO sampling by actions (we keep all valid frames)
    """
    terminated = d["terminated"].astype(bool)
    truncated = d["truncated"].astype(bool)
    T = terminated.shape[0]

    stats["total_frames"] += T
    mask = validate_observations(d, stats)

    if FILTER_IDLE_FRAMES:
        idle_trim_mask = get_idle_trim_mask(d)
        trimmed_count = int(idle_trim_mask.sum())
        if trimmed_count > 0:
            stats["idle_frames_trimmed"] += trimmed_count
            mask[idle_trim_mask] = False

    term_idx = int(np.argmax(terminated)) if terminated.any() else None

    if DROP_TERMINATED:
        stats["terminated_dropped"] += np.sum(terminated)
        stats["truncated_dropped"] += np.sum(truncated)
        mask &= ~(terminated | truncated)

    if term_idx is not None and DROP_LAST_N_BEFORE_TERMINATION > 0:
        start = max(0, term_idx - DROP_LAST_N_BEFORE_TERMINATION)
        stats["pre_termination_dropped"] += np.sum(mask[start:term_idx + 1])
        mask[start:term_idx + 1] = False

    stats["kept"] += int(mask.sum())
    return mask


def continuous_pass_1_compute_masks(files, stats):
    """PASS 1 for continuous: compute keep masks (no action sampling)."""
    keep_masks = []
    total_kept = 0
    obs_keys = None
    obs_shapes = {}

    for f in files:
        d = np.load(f, allow_pickle=True)

        if obs_keys is None:
            obs_keys = [k for k in d.files if k.startswith("obs_")]
            for k in obs_keys:
                arr = d[k]
                obs_shapes[k] = (1,) if arr.ndim == 1 else arr.shape[1:]

        keep = continuous_build_keep_mask(d, stats)
        keep_masks.append(keep)
        total_kept += int(keep.sum())

    assert total_kept > 0, "No samples kept. Loosen filtering rules for continuous."
    return keep_masks, total_kept, obs_keys, obs_shapes


def continuous_pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes):

    grid_keys = ["obs_presence", "obs_speed_x", "obs_speed_y"]
    scalar_keys = [k for k in obs_keys if k not in grid_keys]

    H, W = obs_shapes["obs_presence"]

    out_obs = {}

    for k in grid_keys:
        out_obs[k] = np.empty((total_kept, 3, H, W), dtype=np.float32)

    for k in scalar_keys:
        out_obs[k] = np.empty((total_kept, *obs_shapes[k]), dtype=np.float32)

    out_actions = np.empty((total_kept, 2), dtype=np.int64)

    idx = 0

    for f, keep in zip(files, keep_masks):

        d = np.load(f, allow_pickle=True)

        presence = d["obs_presence"]
        speed_x = d["obs_speed_x"]
        speed_y = d["obs_speed_y"]
        actions = d["actions"]

        valid_idx = np.where(keep)[0]

        for i in valid_idx:

            if i < 2:
                continue

            out_obs["obs_presence"][idx, 0] = presence[i]
            out_obs["obs_presence"][idx, 1] = presence[i-1]
            out_obs["obs_presence"][idx, 2] = presence[i-2]

            out_obs["obs_speed_x"][idx, 0] = speed_x[i]
            out_obs["obs_speed_x"][idx, 1] = speed_x[i-1]
            out_obs["obs_speed_x"][idx, 2] = speed_x[i-2]

            out_obs["obs_speed_y"][idx, 0] = speed_y[i]
            out_obs["obs_speed_y"][idx, 1] = speed_y[i-1]
            out_obs["obs_speed_y"][idx, 2] = speed_y[i-2]

            for k in scalar_keys:
                arr = d[k][i]
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                if arr.ndim == 0:
                    arr = np.array([arr], dtype=np.float32)
                out_obs[k][idx] = arr

            out_actions[idx] = actions[i].astype(np.int64)

            idx += 1

    out_obs = {k: v[:idx] for k, v in out_obs.items()}
    out_actions = out_actions[:idx]

    return out_obs, out_actions




def visualize_continuous(out_obs):
    """Continuous visualization entry point."""

    features = {k: v.reshape(-1) for k, v in out_obs.items()}

    plot_continuous_deltas(features)
    plot_continuous_2d_relationships(features)
    plot_feature_distributions(features)


def continuous_pipeline(files, visualize=False):
    """
    Full continuous pipeline:
    - two-pass build (no discrete sampling)
    - presence remap
    - continuous stats
    - optional viz
    - save continuous dataset (with target_* keys)
    """
    print("[CONTINUOUS] Starting continuous dataset pipeline...")
    stats = init_stats()

    # PASS 1
    print("[CONTINUOUS] PASS 1: Computing filtering masks...")
    keep_masks, total_kept, obs_keys, obs_shapes = continuous_pass_1_compute_masks(files, stats)
    print(f"[CONTINUOUS] Total entries kept for dataset: {total_kept}")

    # PASS 2
    print("[CONTINUOUS] PASS 2: Compiling final arrays...")
    out_obs, out_actions = continuous_pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes)

    # Remap presence grid (still useful)
    out_obs = remap_presence_grid(out_obs)
    
    if MIRROR_DATASET:
        out_obs = apply_mirror_augmentation_continuous(out_obs, threshold=MIRROR_STEERING_THRESHOLD)


    print_trim_statistics(stats)
    print_obs_continuous_stats(out_obs)
    print_throttle_steer_bins(out_obs)

    if visualize:
        print("\n[CONTINUOUS] Rendering visualization plots...")
        visualize_continuous(out_obs)

    # Here we turn obs_* into target_*
    rename_keys = [
        "obs_throttle",
        "obs_brake",
        "obs_steering_angle",
        "obs_reverse"
    ]
    for k in rename_keys:
        if k in out_obs:
            out_obs[f"target_{k[4:]}"] = out_obs.pop(k)

    save_dict = {**out_obs, "actions": out_actions}  # actions just placeholder
    np.savez_compressed(OUT_PATH, **save_dict)
    save_dataset_meta(stats, obs_keys, obs_shapes, total_kept, files, mode="continuous")
    print(f"[CONTINUOUS] ✅ Successfully saved to: {OUT_PATH.resolve()}")








def main(mode=None, visualize=False):
    """
    Main entry point.
    
    Args:
        action_mode (str): "discrete" or "continuous".
                           If None, falls back to config.ACTION_MODE.
        visualize (bool): If True, enable visualizations.
    """

    if mode not in ("discrete", "continuous"):
        raise ValueError(f"Unknown action_mode: {mode} (expected 'discrete' or 'continuous')")

    print(f"Starting dataset generation pipeline in [{mode}] mode...")

    files = gather_demo_files(DEMO_DIRS)

    if mode == "discrete":
        discrete_pipeline(files, visualize=visualize)
    else:
        continuous_pipeline(files, visualize=visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--action-mode",
        type=str,
        default=config.ACTION_MODE,
        choices=["discrete", "continuous"],
        help="Which pipeline to run. Defaults to config.ACTION_MODE."
    )

    args = parser.parse_args()

    if args.out_path:
        OUT_PATH = Path(args.out_path)

    visualize_flag = args.visualize if args.visualize else config.BUILD_VISUALIZE

    main(mode=args.action_mode, visualize=visualize_flag)
