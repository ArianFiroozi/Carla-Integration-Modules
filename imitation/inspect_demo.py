import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import argparse
from config import bc_config
from .utils.viz import *
from.utils.stats import * 


ROOT = Path(__file__).resolve().parents[0]


DEMO_DIRS = bc_config.DEMO_LIST

OBS_BOUNDS =bc_config.OBS_BOUNDS


def process_demos(files, max_feature_samples=bc_config.MAX_INSPECT_FEATURE_SAMPLES, is_continuous=False):
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

def main(demo_dirs=None, is_continuous=False, visualize=bc_config.INSPECT_VISUALIZE, max_feature_samples=bc_config.MAX_INSPECT_FEATURE_SAMPLES):




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
        default=bc_config.ACTION_MODE,
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-feature-samples", type=int, default=bc_config.MAX_INSPECT_FEATURE_SAMPLES)

    args = parser.parse_args()

    visualize_flag = args.visualize if args.visualize else bc_config.INSPECT_VISUALIZE
    is_continuous = True if args.mode=="continuous" else False
    main(
        demo_dirs=DEMO_DIRS,
        is_continuous=is_continuous,
        visualize=visualize_flag,
        max_feature_samples=args.max_feature_samples
    )
