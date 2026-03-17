import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
DEMO_DIR = DATA_DIR / "demos"

OUT_PATH = DATA_DIR / "processed" / "dataset_bc.npz"
    
    


RNG_SEED = 42

# Termination Filtering
DROP_TERMINATED = True
DROP_LAST_N_BEFORE_TERMINATION = 10

# Idle / Silence Filtering
FILTER_IDLE_FRAMES = True
IDLE_FILTER_MODE = "all"  # Options: "start" (only trim beginning) OR "all" (remove every idle frame)
IDLE_SPEED_THRESHOLD = 0.3
IDLE_THROTTLE_THRESHOLD = 0.05
IDLE_BRAKE_THRESHOLD = 0.05


# JOINT_KEEP_PROBS = {
#     (4,3): 0.2,  
#     (4,0): 0.5,
#     (4,1): 0.7,   
# }

JOINT_KEEP_PROBS = {
    (4,3): 1,  
    (4,0): 1,
    (4,1): 1,   
}


# Observation Bounds
OBS_BOUNDS = {
    "obs_speed_x": dict(low=-np.inf, high=np.inf),
    "obs_speed_y": dict(low=-np.inf, high=np.inf),
    "obs_presence": dict(low=0, high=9),
    "obs_lane_angle": dict(low=-np.pi, high=np.pi),
    "obs_max_speed": dict(low=0.0, high=200.0),
    "obs_traffic_signs": dict(low=0.0, high=1.0),
    "obs_ego_speed_x": dict(low=-np.inf, high=np.inf),
    "obs_ego_speed_y": dict(low=-np.inf, high=np.inf),
    "obs_ego_in_lane_position_x": dict(low=-100.0, high=100.0),
    "obs_throttle": dict(low=0.0, high=1.0),
    "obs_brake": dict(low=0.0, high=1.0),
    "obs_steering_angle": dict(low=-1.0, high=1.0),
    "obs_reverse": dict(low=0.0, high=1.0),
}

# Action Maps
SPEED_MAP = {0: "Accelerate", 1: "Brake", 2: "Stop", 3: "Reverse", 4: "Constant"}
TURN_MAP = {0: "Right", 1: "Left", 2: "No Turn", 3: "Straight"}


# Optional simplification
SIMPLIFY_ACTIONS = True
REMOVE_REVERSE = True
REMOVE_NO_TURN = True


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


def should_keep(action, rng):
    """Determines if an action should be kept based on JOINT_KEEP_PROBS."""
    speed, turn = map(int, action)
    keep_prob = JOINT_KEEP_PROBS.get((speed, turn), 1.0)
    return rng.random() < keep_prob


def get_idle_trim_mask(d):
    """
    Returns a boolean mask of frames to trim based on idle conditions:
    speed < 0.3 AND throttle < 0.05 AND brake < 0.05.
    """
    speed = np.abs(d["obs_ego_speed_x"].reshape(-1))
    throttle = d["obs_throttle"].reshape(-1)
    brake = d["obs_brake"].reshape(-1)
    
    # Identify frames that meet the exact idle conditions
    is_idle = (speed < IDLE_SPEED_THRESHOLD) & \
              (throttle < IDLE_THROTTLE_THRESHOLD) & \
              (brake < IDLE_BRAKE_THRESHOLD)
              
    T = len(speed)
    trim_mask = np.zeros(T, dtype=bool)
    
    if IDLE_FILTER_MODE == "all":
        # Remove every frame that matches the condition
        trim_mask = is_idle
    elif IDLE_FILTER_MODE == "start":
        # Remove only continuous idle frames at the very beginning
        non_idle_idx = np.where(~is_idle)[0]
        if len(non_idle_idx) > 0:
            first_active = non_idle_idx[0]
            trim_mask[:first_active] = True
        else:
            trim_mask[:] = True  # The whole sequence was idle
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
    """
    Remaps categorical IDs in obs_presence to contiguous integers.

    mapping:
        0 -> empty
        1 -> car
        2 -> wall
        9 -> ego

    to

        0 -> empty
        1 -> car
        2 -> wall
        3 -> ego
    """

    if mapping is None:
        mapping = {
            0: 0,
            1: 1,
            2: 2,
            9: 3
        }

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




def build_keep_mask(d, stats, rng):
    """Generates a boolean mask indicating which frames to keep for a single demo."""
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

    stats["kept"] += np.sum(keep)
    return keep


def simplify_actions(out_obs, out_actions, remove_reverse=False, remove_no_turn=False):
    """
    Simplifies the action space by optionally removing Reverse and/or No Turn.
    Also reindexes remaining classes so they stay contiguous.

    Args:
        out_obs (dict): observation arrays
        out_actions (np.ndarray): (N,2) action array
        remove_reverse (bool)
        remove_no_turn (bool)

    Returns:
        new_obs, new_actions
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


def pass_1_compute_masks(files, stats, rng):
    """Iterates through files to determine keep masks and observation shapes."""
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

        keep = build_keep_mask(d, stats, rng)
        keep_masks.append(keep)
        total_kept += int(keep.sum())

    assert total_kept > 0, "No samples kept. Loosen sampling/drop rules."
    return keep_masks, total_kept, obs_keys, obs_shapes


def pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes):
    """Allocates arrays and fills them using the precomputed keep masks."""
    out_obs = {k: np.empty((total_kept, *obs_shapes[k]), dtype=np.float32) for k in obs_keys}
    out_actions = np.empty((total_kept, 2), dtype=np.int64)

    idx = 0
    for f, keep in zip(files, keep_masks):
        n = int(keep.sum())
        if n == 0:
            continue

        d = np.load(f, allow_pickle=True)
        out_actions[idx:idx+n] = d["actions"][keep].astype(np.int64)

        for k in obs_keys:
            arr = d[k][keep]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            out_obs[k][idx:idx+n] = arr

        idx += n
        
    assert idx == total_kept, f"Mismatch in expected records: {idx} vs {total_kept}"
    return out_obs, out_actions



def print_distribution_summary(out_actions, files):
    """Prints marginal and joint distribution statistics."""
    sp = out_actions[:, 0]
    tr = out_actions[:, 1]
    
    speed_counts = np.bincount(sp, minlength=5)
    turn_counts = np.bincount(tr, minlength=4)
    total = len(out_actions)
    
    print(f"\nFinal dataset size: {total}")
    print(f"Average samples per demo: {total / len(files):.2f}")

    print("\nSpeed distribution:")
    for i, c in enumerate(speed_counts):
        print(f"{SPEED_MAP[i]:10s} : {c:6d} ({100 * c / total:5.2f}%)")

    print("\nTurn distribution:")
    for i, c in enumerate(turn_counts):
        print(f"{TURN_MAP[i]:10s} : {c:6d} ({100 * c / total:5.2f}%)")

    print("\nJoint action distribution (speed, turn):")
    counter = Counter(map(tuple, out_actions))
    for (speed, turn), count in counter.most_common(20):
        print(f"{SPEED_MAP[speed]:10s} | {TURN_MAP[turn]:10s} : {count:6d} ({100 * count / total:5.2f}%)")



def print_minmax_summary(out_obs):
    """
    Prints min/max/mean/std for every observation in the dataset.
    """
    print("\n" + "="*50)
    print("DATASET FEATURE SUMMARY")
    print("="*50)

    for k, arr in out_obs.items():
        arr_flat = arr.reshape(-1)

        print(f"\n{k}")
        print(f"  shape : {arr.shape}")
        print(f"  min   : {arr_flat.min():.6f}")
        print(f"  max   : {arr_flat.max():.6f}")
        print(f"  mean  : {arr_flat.mean():.6f}")
        print(f"  std   : {arr_flat.std():.6f}")


def print_trim_statistics(stats):
    """Prints trimming, filtering, and violation statistics."""
    total = stats["total_frames"]
    def pct(x): return 100.0 * x / total if total > 0 else 0.0

    print("\n" + "="*50)
    print("TRIMMING / FILTERING STATISTICS")
    print("="*50)
    print(f"Total frames seen           : {total}")
    print("\n--- Hard trimming ---")
    print(f"Idle frames trimmed ({IDLE_FILTER_MODE:5s}) : {stats['idle_frames_trimmed']:8d} ({pct(stats['idle_frames_trimmed']):5.2f}%)")
    print(f"Terminated dropped          : {stats['terminated_dropped']:8d} ({pct(stats['terminated_dropped']):5.2f}%)")
    print(f"Truncated dropped           : {stats['truncated_dropped']:8d} ({pct(stats['truncated_dropped']):5.2f}%)")
    print(f"Pre-termination dropped     : {stats['pre_termination_dropped']:8d} ({pct(stats['pre_termination_dropped']):5.2f}%)")
    print("\n--- Sampling ---")
    print(f"Dropped by sampling         : {stats['sampling_dropped']:8d} ({pct(stats['sampling_dropped']):5.2f}%)")
    print("\n--- Final ---")
    print(f"Kept frames                 : {stats['kept']:8d} ({pct(stats['kept']):5.2f}%)")

    print("\n" + "="*50)
    print("OBSERVATION BOUND VIOLATIONS")
    print("="*50)
    
    viol = stats["obs_violations"]
    if not viol:
        print("No observation bound violations detected.")
    else:
        for k, c in viol.most_common():
            print(f"{k:30s} : {c:8d} ({pct(c):5.2f}%)")
        print(f"\nTotal frames dropped due to obs violations : {stats['obs_violation_frames']} ({pct(stats['obs_violation_frames']):.2f}%)")
    print("="*50)


def plot_feature_distributions(features, max_samples=100000):
    """
    Plots histograms for all collected scalar/array observation features.
    Automatically flattens arrays and subsamples to avoid memory issues.
    """
    print("\nRendering feature distributions...")
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5, rows * 3))

    for i, (k, arr) in enumerate(features.items()):
        # Flatten array to 1D for marginal histogram distribution
        arr_flat = arr.reshape(-1)
        
        # Subsample if the array is massive to prevent Matplotlib from freezing
        if len(arr_flat) > max_samples:
            arr_flat = np.random.choice(arr_flat, max_samples, replace=False)

        plt.subplot(rows, cols, i + 1)
        plt.hist(arr_flat, bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.title(k)
        
    plt.tight_layout()
    plt.show()


def plot_brake_throttle_joint(data, bins=50):
    """
    Plots joint distribution of throttle and brake values.

    Args:
        data (dict): dataset dict loaded from npz
        bins (int): histogram bins
    """

    throttle = data["obs_throttle"].reshape(-1)
    brake = data["obs_brake"].reshape(-1)

    plt.figure(figsize=(6,5))

    h = plt.hist2d(
        throttle,
        brake,
        bins=bins,
        range=[[0,1],[0,1]],
        cmap="viridis"
    )

    plt.colorbar(h[3], label="Frequency")
    plt.xlabel("Throttle")
    plt.ylabel("Brake")
    plt.title("Throttle vs Brake Joint Distribution")

    plt.tight_layout()
    plt.show()


def plot_speed_distribution(sp):
    speed_counts = np.bincount(sp, minlength=5)

    plt.figure(figsize=(6, 4))
    plt.bar([SPEED_MAP[i] for i in range(5)],
            speed_counts / max(1, speed_counts.sum()))
    plt.title("Speed Action Distribution")
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_turn_distribution(tr):
    turn_counts = np.bincount(tr, minlength=4)

    plt.figure(figsize=(6, 4))
    plt.bar([TURN_MAP[i] for i in range(4)],
            turn_counts / max(1, turn_counts.sum()))
    plt.title("Turn Action Distribution")
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_joint_action_distribution(sp, tr):
    joint_matrix = np.zeros((5, 4), dtype=np.int64)
    np.add.at(joint_matrix, (sp, tr), 1)
    joint_probs = joint_matrix / max(1, joint_matrix.sum())

    plt.figure(figsize=(7, 5))
    im = plt.imshow(joint_probs, cmap="viridis")
    plt.colorbar(im, label="Probability")

    plt.xticks(range(4), [TURN_MAP[i] for i in range(4)])
    plt.yticks(range(5), [SPEED_MAP[i] for i in range(5)])
    plt.xlabel("Turn")
    plt.ylabel("Speed")
    plt.title("Joint Action Distribution")

    for i in range(5):
        for j in range(4):
            if joint_probs[i, j] > 0:
                plt.text(
                    j, i,
                    f"{joint_probs[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9
                )

    plt.tight_layout()
    plt.show()


def plot_brake_throttle_hex(data):
    throttle = data["obs_throttle"].reshape(-1)
    brake = data["obs_brake"].reshape(-1)

    plt.figure(figsize=(6,5))

    plt.hexbin(
        throttle,
        brake,
        gridsize=40,
        cmap="inferno",
        mincnt=1
    )

    plt.colorbar(label="Count")
    plt.xlabel("Throttle")
    plt.ylabel("Brake")
    plt.title("Throttle vs Brake Joint Distribution")

    plt.tight_layout()
    plt.show()


def visualize_data(out_actions, out_obs):
    """Main visualization entry point."""
    sp = out_actions[:, 0]
    tr = out_actions[:, 1]

    plot_speed_distribution(sp)
    plot_turn_distribution(tr)
    plot_joint_action_distribution(sp, tr)
    plot_feature_distributions(out_obs)
    # plot_brake_throttle_joint(out_obs)
    # plot_brake_throttle_hex(out_obs)

def main(visualize=False):
    """
    Main function to execute the dataset building pipeline.
    
    Args:
        visualize (bool): If True, plots the dataset distribution distributions.
    """
    print(f"Starting dataset generation pipeline...")
    files = sorted(DEMO_DIR.glob("*.npz"))

    assert files, f"No demos found in {DEMO_DIR.resolve()}"

    rng = np.random.default_rng(RNG_SEED)
    stats = init_stats()

    # PASS 1: Calculate masks
    print("PASS 1: Computing filtering masks...")
    keep_masks, total_kept, obs_keys, obs_shapes = pass_1_compute_masks(files, stats, rng)
    print(f"Total entries kept for dataset: {total_kept}")

    # PASS 2: Compile
    print("PASS 2: Compiling final arrays...")
    out_obs, out_actions = pass_2_build_dataset(files, keep_masks, total_kept, obs_keys, obs_shapes)


    print_minmax_summary(out_obs)
    
    
    
    # Remap categorical grid IDs
    out_obs = remap_presence_grid(out_obs)




    # Output Reporting
    print_trim_statistics(stats)
    print_distribution_summary(out_actions, files)

    # Optional Visualization
    if visualize:
        print("\nRendering visualization plots...")
        visualize_data(out_actions, out_obs)
        
        
    if SIMPLIFY_ACTIONS:
        out_obs, out_actions = simplify_actions(
            out_obs,
            out_actions,
            remove_reverse=REMOVE_REVERSE,
            remove_no_turn=REMOVE_NO_TURN
        )
        
        
        
    # Save to disk
    
    
    # rename selected observation keys to target_*
    rename_keys = [
        "obs_throttle",
        "obs_brake",
        "obs_steering_angle",
        "obs_reverse"
    ]

    for k in rename_keys:
        out_obs[f"target_{k[4:]}"] = out_obs.pop(k)  # remove obs_ and add target_

    save_dict = {**out_obs, "actions": out_actions}

    print(save_dict.keys())

    np.savez_compressed(OUT_PATH, **save_dict)
    print(f"✅ Successfully saved to: {OUT_PATH.resolve()}")
    
    



if __name__ == "__main__":
    main(visualize=True)
