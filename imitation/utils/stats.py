import numpy as np
from .. import config
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[0]

DEMO_DIRS = config.DEMO_LIST
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
 


def print_dataset_structure(d):
    print("\n" + "="*50)
    print("DATASET STRUCTURE (from sample file)")
    print("="*50)
    for k in d.files:
        arr = d[k]
        print(f"{k:25s} | shape={str(arr.shape):15s} | dtype={arr.dtype}")


def print_discrete_stats(stats):
    total_T = stats["total_T"]
    sp_counts = stats["speed_counts"]
    tr_counts = stats["turn_counts"]
    
    print("\n" + "="*50)
    print("DISCRETE DATASET STATISTICS")
    print("="*50)
    print(f"Total timesteps:   {total_T}")
    print(f"Terminated steps:  {stats['terminated_ct']}")
    print(f"Total Episodes:    {len(stats['episode_lengths'])}")

    print("\nSpeed Distribution:")
    for i, count in enumerate(sp_counts):
        pct = 100 * count / max(1, sp_counts.sum())
        print(f"  {config.SPEED_MAP.get(i, str(i)):10s} : {count:8d} ({pct:5.2f}%)")

    print("\nTurn Distribution:")
    for i, count in enumerate(tr_counts):
        pct = 100 * count / max(1, tr_counts.sum())
        print(f"  {config.TURN_MAP.get(i, str(i)):10s} : {count:8d} ({pct:5.2f}%)")

    print("\nTop 20 Joint Actions (Speed | Turn):")
    for (speed, turn), count in stats["joint_counts"].most_common(20):
        print(f"  {config.SPEED_MAP.get(speed, str(speed)):10s} | {config.TURN_MAP.get(turn, str(turn)):10s} : {count:8d}")





def print_throttle_steer_bins(features, bin_size=0.1):
    """
    Text summary of throttle and steering distributions using fixed bins.
    """
    if "obs_throttle" not in features or "obs_steering_angle" not in features:
        print("Throttle or Steering data not found in features.")
        return

    throttle = features["obs_throttle"].ravel()
    steer = features["obs_steering_angle"].ravel()

    # Define bins
    t_bins = np.arange(0, 1 + bin_size, bin_size)
    s_bins = np.arange(-1, 1 + bin_size, bin_size)

    # Digitize
    t_idx = np.digitize(throttle, t_bins) - 1
    s_idx = np.digitize(steer, s_bins) - 1

    # Clamp to valid range
    t_idx = np.clip(t_idx, 0, len(t_bins)-2)
    s_idx = np.clip(s_idx, 0, len(s_bins)-2)

    # Marginal counts
    t_counts = np.bincount(t_idx, minlength=len(t_bins)-1)
    s_counts = np.bincount(s_idx, minlength=len(s_bins)-1)

    # Joint counts
    joint = np.zeros((len(s_bins)-1, len(t_bins)-1), dtype=int)
    for s, t in zip(s_idx, t_idx):
        joint[s, t] += 1

    print("\n" + "="*50)
    print("THROTTLE BIN DISTRIBUTION (0.1)")
    print("="*50)
    total = t_counts.sum()
    for i, c in enumerate(t_counts):
        low = t_bins[i]
        high = t_bins[i+1]
        pct = 100 * c / max(1, total)
        print(f"{low:4.1f} - {high:4.1f} : {c:8d} ({pct:5.2f}%)")

    print("\n" + "="*50)
    print("STEERING BIN DISTRIBUTION (0.1)")
    print("="*50)
    total = s_counts.sum()
    for i, c in enumerate(s_counts):
        low = s_bins[i]
        high = s_bins[i+1]
        pct = 100 * c / max(1, total)
        print(f"{low:5.2f} - {high:5.2f} : {c:8d} ({pct:5.2f}%)")

    print("\n" + "="*50)
    print("JOINT STEER × THROTTLE BINS")
    print("="*50)

    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] == 0:
                continue
            s_low, s_high = s_bins[i], s_bins[i+1]
            t_low, t_high = t_bins[j], t_bins[j+1]
            print(
                f"Steer[{s_low:5.2f},{s_high:5.2f}] "
                f"Throttle[{t_low:4.1f},{t_high:4.1f}] : {joint[i,j]}"
            )





def print_obs_continuous_stats(out_obs):
    """
    Basic continuous stats focused on throttle / brake / steering.
    """
    print("\n" + "="*50)
    print("CONTINUOUS TARGET STATS")
    print("="*50)

    for key in ["obs_throttle", "obs_brake", "obs_steering_angle"]:
        if key not in out_obs:
            continue
        arr = out_obs[key].reshape(-1)
        print(f"\n{key}")
        print(f"  min   : {arr.min():.6f}")
        print(f"  max   : {arr.max():.6f}")
        print(f"  mean  : {arr.mean():.6f}")
        print(f"  std   : {arr.std():.6f}")

def print_continuous_stats(stats):
    print("\n" + "="*50)
    print("CONTINUOUS DATASET STATISTICS")
    print("="*50)
    print(f"Total timesteps:   {stats['total_T']}")
    print(f"Terminated steps:  {stats['terminated_ct']}")
    print(f"Total Episodes:    {len(stats['episode_lengths'])}")
    if stats["episode_lengths"]:
        print(f"Avg Ep Length:     {np.mean(stats['episode_lengths']):.2f} steps")


# def print_throttle_steer_bins(out_obs, num_bins=10):
#     """Example binning stats for throttle and steering."""
#     if "obs_throttle" not in out_obs or "obs_steering_angle" not in out_obs:
#         return

#     throttle = out_obs["obs_throttle"].reshape(-1)
#     steer = out_obs["obs_steering_angle"].reshape(-1)

#     print("\nThrottle histogram (counts per bin):")
#     hist_t, edges_t = np.histogram(throttle, bins=num_bins, range=(0, 1))
#     print("  edges:", edges_t)
#     print("  counts:", hist_t)

#     print("\nSteering histogram (counts per bin):")
#     hist_s, edges_s = np.histogram(steer, bins=num_bins)
#     print("  edges:", edges_s)
#     print("  counts:", hist_s)



def print_distribution_summary(out_actions, files):
    """Prints marginal and joint distribution statistics for discrete actions."""
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
    print("\n--- Sampling (discrete only) ---")
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


