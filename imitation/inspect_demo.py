import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
import argparse
from . import config
ROOT = Path(__file__).resolve().parents[0]


DEMO_DIRS = config.DEMO_LIST

OBS_BOUNDS =config.OBS_BOUNDS


def process_demos(files, max_feature_samples=config.MAX_INSPECT_FEATURE_SAMPLES, is_continuous=False):
    """
    Makes a single pass over all demo files to collect data based on pipeline type.
    """
    total_T = 0
    terminated_ct = 0
    episode_lengths = []
    
    # Discrete specific
    speed_counts = np.zeros(5, dtype=np.int64)
    turn_counts = np.zeros(4, dtype=np.int64)
    joint_counts = Counter()
    
    # Store flattened arrays for histograms and 2D plots
    feature_collection = defaultdict(list)
    ignore_keys = {"actions", "terminated", "truncated", "dones", "t"}

    for f in files:
        d = np.load(f, allow_pickle=True)
        actions = d["actions"]
        presence = d["obs_presence"]
        terminated = d["terminated"]
        
        T = actions.shape[0]
        total_T += T
        episode_lengths.append(T)

        # Shape Assertions
        assert actions.shape[1] == 2, f"Expected actions shape (T, 2), got {actions.shape}"
        assert presence.shape[0] == T, f"Presence sequence length mismatch in {f.name}"
        assert 0 <= presence.min() and presence.max() <= 9, f"Presence out of bounds in {f.name}"

        lane_angle = None
        if "obs_lane_angle" in d.files:
            lane_angle = d["obs_lane_angle"].astype(np.float64).reshape(-1)
            lane_angle = np.unwrap(lane_angle)

        valid_mask = np.ones(T, dtype=bool)

        for k, bounds in OBS_BOUNDS.items():
            if k not in d.files:
                continue

            arr = d[k]

            if k == "obs_lane_angle" and lane_angle is not None:
                arr = lane_angle

            arr_flat = arr.reshape(T, -1)

            low = bounds["low"]
            high = bounds["high"]

            invalid = (arr_flat < low) | (arr_flat > high)
            invalid = np.any(invalid, axis=1)

            valid_mask &= ~invalid

        if not is_continuous:
            sp, tr = actions[:, 0], actions[:, 1]

            for t in range(T):
                if not valid_mask[t]:
                    continue
                if not (0 <= sp[t] <= 4 and 0 <= tr[t] <= 3):
                    continue

                speed_counts[sp[t]] += 1
                turn_counts[tr[t]] += 1
                joint_counts.update([(sp[t], tr[t])])

        else:
            if "obs_steering_angle" in d.files and T > 1:
                steer = d["obs_steering_angle"].flatten()
                delta = np.diff(steer)
                mask = valid_mask[1:]
                feature_collection["steer_delta"].append(delta[mask])

            if "obs_throttle" in d.files and T > 1:
                throttle = d["obs_throttle"].flatten()
                delta = np.diff(throttle)
                mask = valid_mask[1:]
                feature_collection["throttle_delta"].append(delta[mask])

        terminated_ct += terminated.astype(np.int32).sum()

        for k in d.files:
            if k not in ignore_keys:
                arr = d[k]

                if k == "obs_lane_angle" and lane_angle is not None:
                    arr = lane_angle

                if arr.ndim > 1:
                    arr = arr.reshape(T, -1)
                    arr = arr[valid_mask]
                    arr_flat = arr.reshape(-1)
                else:
                    arr = arr[valid_mask]
                    arr_flat = arr.flatten()

                feature_collection[k].append(arr_flat)

    # Post-process: Concatenate and Subsample features to save memory
    final_features = {}
    for k, v_list in feature_collection.items():
        if len(v_list) == 0:
            continue
        concat_arr = np.concatenate(v_list)
        if len(concat_arr) > max_feature_samples:
            idx = np.random.choice(len(concat_arr), max_feature_samples, replace=False)
            concat_arr = concat_arr[idx]
        final_features[k] = concat_arr

    stats = {
        "total_T": total_T,
        "terminated_ct": terminated_ct,
        "episode_lengths": episode_lengths,
        "speed_counts": speed_counts,
        "turn_counts": turn_counts,
        "joint_counts": joint_counts
    }
    
    return stats, final_features


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


def print_continuous_stats(stats):
    print("\n" + "="*50)
    print("CONTINUOUS DATASET STATISTICS")
    print("="*50)
    print(f"Total timesteps:   {stats['total_T']}")
    print(f"Terminated steps:  {stats['terminated_ct']}")
    print(f"Total Episodes:    {len(stats['episode_lengths'])}")
    if stats["episode_lengths"]:
        print(f"Avg Ep Length:     {np.mean(stats['episode_lengths']):.2f} steps")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_discrete_actions(stats):
    """1x2 Subplot for Speed and Turn discrete actions."""
    sp_counts = stats["speed_counts"]
    tr_counts = stats["turn_counts"]
    
    sp_probs = sp_counts / max(1, sp_counts.sum())
    tr_probs = tr_counts / max(1, tr_counts.sum())
    
    sp_labels = [config.SPEED_MAP.get(i, str(i)) for i in range(len(sp_counts))]
    tr_labels = [config.TURN_MAP.get(i, str(i)) for i in range(len(tr_counts))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(sp_labels, sp_probs, color='skyblue', edgecolor='black')
    axes[0].set_title("Speed Action Distribution")
    axes[0].set_ylabel("Probability")
    axes[0].tick_params(axis='x', rotation=30)
    
    axes[1].bar(tr_labels, tr_probs, color='lightgreen', edgecolor='black')
    axes[1].set_title("Turn Action Distribution")
    axes[1].set_ylabel("Probability")
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.show()


def plot_joint_heatmap(joint_counts):
    """Plots a 2D Heatmap of the joint (Speed x Turn) distribution."""
    joint_matrix = np.zeros((5, 4), dtype=np.int64)
    for (speed, turn), count in joint_counts.items():
        if 0 <= speed < 5 and 0 <= turn < 4:
            joint_matrix[speed, turn] = count

    joint_probs = joint_matrix / max(1, joint_matrix.sum())

    plt.figure(figsize=(7, 5))
    im = plt.imshow(joint_probs, cmap="viridis")
    plt.colorbar(im, label="Probability")

    plt.xticks(range(4), [config.TURN_MAP.get(i, str(i)) for i in range(4)])
    plt.yticks(range(5), [config.SPEED_MAP.get(i, str(i)) for i in range(5)])
    plt.xlabel("Turn Action")
    plt.ylabel("Speed Action")
    plt.title("Joint Action Distribution (Speed × Turn)")

    for i in range(5):
        for j in range(4):
            if joint_probs[i, j] > 0:
                plt.text(j, i, f"{joint_probs[i,j]:.2f}", 
                         ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_episode_lengths(episode_lengths):
    """1x1 plot for Episode Lengths."""
    plt.figure(figsize=(6, 4))
    plt.hist(episode_lengths, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Episode Lengths Distribution")
    plt.xlabel("Steps per Episode ($T$)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_continuous_deltas(features):
    """1x2 Subplot for Steer Delta and Throttle Delta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if "steer_delta" in features:
        axes[0].hist(features["steer_delta"], bins=50, color='teal', edgecolor='black', alpha=0.7)
        axes[0].set_title("Steering Angle Delta ($\Delta$ Steer)")
    else:
        axes[0].set_title("Steering Delta Not Found")
        
    if "throttle_delta" in features:
        axes[1].hist(features["throttle_delta"], bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_title("Throttle Delta ($\Delta$ Throttle)")
    else:
        axes[1].set_title("Throttle Delta Not Found")

    plt.tight_layout()
    plt.show()


def plot_continuous_2d_relationships(features):
    """Plots 2D relationships (hexbin/scatter) for continuous features."""
    pairs_to_plot = [
        ("obs_ego_speed_x", "obs_throttle", "Ego Speed X vs Throttle"),
        ("obs_throttle", "obs_steering_angle", "Throttle vs Steering Angle"),
        ("obs_lane_angle", "obs_steering_angle", "Lane Angle vs Steering Angle"),
        ("obs_ego_in_lane_position_x", "obs_steering_angle", "Lane Offset vs Steering Angle"),
        ("obs_lane_angle", "obs_ego_in_lane_position_x", "Lane Angle vs Lane Offset")
    ]
    
    valid_pairs = [(x, y, title) for (x, y, title) in pairs_to_plot if x in features and y in features]
    if not valid_pairs:
        return

    n = len(valid_pairs)
    cols = 3
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()
    
    for i, (x_key, y_key, title) in enumerate(valid_pairs):
        ax = axes[i]
        x_data = features[x_key]
        y_data = features[y_key]
        
        hb = ax.hexbin(x_data, y_data, gridsize=40, cmap='inferno', mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Count')
        
        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(features):
    """Plots histograms for all collected scalar/array observation features."""
    # Exclude deltas as they are plotted separately
    keys = [k for k in features.keys() if "delta" not in k]
    n = len(keys)
    if n == 0: return
    
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5, rows * 3))

    for i, k in enumerate(keys):
        plt.subplot(rows, cols, i + 1)
        plt.hist(features[k], bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.title(k)
        
    plt.tight_layout()
    plt.show()


def print_throttle_steer_bins(features, bin_size=0.1):
    """
    Text summary of throttle and steering distributions using fixed bins.
    """
    if "obs_throttle" not in features or "obs_steering_angle" not in features:
        print("Throttle or Steering data not found in features.")
        return

    throttle = features["obs_throttle"]
    steer = features["obs_steering_angle"]

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


def gather_demo_files(dirs):
    """
    Accepts a list of directories and gathers all .npz demo files from each.
    Returns a sorted list of file paths.
    """
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




# =============================================================================
# PIPELINES
# =============================================================================

def discrete_pipeline(stats, features, visualize=True):
    print_discrete_stats(stats)
    
    if visualize:
        print("\nRendering Discrete Visualizations...")
        plot_discrete_actions(stats)
        plot_joint_heatmap(stats["joint_counts"])
        plot_episode_lengths(stats["episode_lengths"])
        plot_feature_distributions(features)


def continuous_pipeline(stats, features, visualize=True):
    print_continuous_stats(stats)
    
    print_throttle_steer_bins(features)
    
    
    if visualize:
        print("\nRendering Continuous Visualizations...")
        plot_episode_lengths(stats["episode_lengths"])
        plot_continuous_deltas(features)
        plot_continuous_2d_relationships(features)
        plot_feature_distributions(features)

def main(demo_dirs=None, is_continuous=False, visualize=config.INSPECT_VISUALIZE, max_feature_samples=config.MAX_INSPECT_FEATURE_SAMPLES):




    # Gather all demo files from all directories
    files = gather_demo_files(demo_dirs)

    print(f"Found {len(files)} demos in ")
    
    assert len(files) > 0, f"No demos found in {demo_dirs}"

    # Print structural info using the first file
    first_demo = np.load(files[0], allow_pickle=True)
    print_dataset_structure(first_demo)

    # Process all demos
    print(f"\nProcessing datasets (Continuous Mode: {is_continuous})...")
    stats, features = process_demos(files, max_feature_samples=max_feature_samples, is_continuous=is_continuous)

    # Route to appropriate pipeline
    if is_continuous:
        continuous_pipeline(stats, features, visualize=visualize)
    else:
        discrete_pipeline(stats, features, visualize=visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["discrete", "continuous"],
        default=config.ACTION_MODE,
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-feature-samples", type=int, default=config.MAX_INSPECT_FEATURE_SAMPLES)

    args = parser.parse_args()

    visualize_flag = args.visualize if args.visualize else config.INSPECT_VISUALIZE
    is_continuous = True if args.mode=="continuous" else False
    main(
        demo_dirs=DEMO_DIRS,
        is_continuous=is_continuous,
        visualize=visualize_flag,
        max_feature_samples=args.max_feature_samples
    )
