import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict



ROOT = Path(__file__).resolve().parents[0]
DEMO_DIR = ROOT / "data" / "demos"


SPEED_MAP = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Reverse",
    4: "Constant"
}

TURN_MAP = {
    0: "Right",
    1: "Left",
    2: "No Turn",
    3: "Straight"
}



def process_demos(files, max_feature_samples=200000):
    """
    Makes a single pass over all demo files to collect:
    - Action distributions (marginal & joint)
    - Termination counts
    - Validates bounds/shapes
    - Subsampled observation features for plotting
    """
    total_T = 0
    terminated_ct = 0
    speed_counts = np.zeros(5, dtype=np.int64)
    turn_counts = np.zeros(4, dtype=np.int64)
    joint_counts = Counter()
    
    # Store flattened arrays for histograms
    feature_collection = defaultdict(list)
    ignore_keys = {"actions", "terminated", "truncated", "dones", "t"}

    for f in files:
        d = np.load(f, allow_pickle=True)
        actions = d["actions"]
        presence = d["obs_presence"]
        terminated = d["terminated"]
        
        T = actions.shape[0]
        total_T += T

        # Shape Assertions
        assert actions.shape[1] == 2, f"Expected actions shape (T, 2), got {actions.shape}"
        assert presence.shape[0] == T, f"Presence sequence length mismatch in {f.name}"

        # Range Assertions
        sp, tr = actions[:, 0], actions[:, 1]
        assert 0 <= sp.min() and sp.max() <= 4, f"Speed out of bounds in {f.name}: min={sp.min()}, max={sp.max()}"
        assert 0 <= tr.min() and tr.max() <= 3, f"Turn out of bounds in {f.name}: min={tr.min()}, max={tr.max()}"
        assert 0 <= presence.min() and presence.max() <= 9, f"Presence out of bounds in {f.name}"

        # Fast Vectorized Counting
        speed_counts += np.bincount(sp, minlength=5)
        turn_counts += np.bincount(tr, minlength=4)
        joint_counts.update(map(tuple, actions))
        terminated_ct += terminated.astype(np.int32).sum()

        # Collect features for histograms
        for k in d.files:
            if k not in ignore_keys:
                arr = d[k]
                arr_flat = arr.reshape(-1) if arr.ndim > 2 else arr.flatten()
                feature_collection[k].append(arr_flat)

    # Post-process: Concatenate and Subsample features to save memory
    final_features = {}
    for k, v_list in feature_collection.items():
        concat_arr = np.concatenate(v_list)
        if len(concat_arr) > max_feature_samples:
            idx = np.random.choice(len(concat_arr), max_feature_samples, replace=False)
            concat_arr = concat_arr[idx]
        final_features[k] = concat_arr

    stats = {
        "total_T": total_T,
        "terminated_ct": terminated_ct,
        "speed_counts": speed_counts,
        "turn_counts": turn_counts,
        "joint_counts": joint_counts
    }
    
    return stats, final_features



def print_dataset_structure(d):
    """Prints the structure (shapes and dtypes) of the arrays in the dataset."""
    print("\n" + "="*50)
    print("DATASET STRUCTURE (from sample file)")
    print("="*50)
    for k in d.files:
        arr = d[k]
        print(f"{k:25s} | shape={str(arr.shape):15s} | dtype={arr.dtype}")


def print_statistics_report(stats):
    """Prints a clean summary report of the parsed dataset statistics."""
    total_T = stats["total_T"]
    sp_counts = stats["speed_counts"]
    tr_counts = stats["turn_counts"]
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total timesteps:   {total_T}")
    print(f"Terminated steps:  {stats['terminated_ct']}")

    print("\nSpeed Distribution:")
    for i, count in enumerate(sp_counts):
        pct = 100 * count / max(1, sp_counts.sum())
        print(f"  {SPEED_MAP[i]:10s} : {count:8d} ({pct:5.2f}%)")

    print("\nTurn Distribution:")
    for i, count in enumerate(tr_counts):
        pct = 100 * count / max(1, tr_counts.sum())
        print(f"  {TURN_MAP[i]:10s} : {count:8d} ({pct:5.2f}%)")

    print("\nTop 20 Joint Actions (Speed | Turn):")
    for (speed, turn), count in stats["joint_counts"].most_common(20):
        print(f"  {SPEED_MAP[speed]:10s} | {TURN_MAP[turn]:10s} : {count:8d}")


def plot_bar_distribution(labels, counts, title):
    """Helper function to plot a standard bar distribution chart."""
    probs = counts / max(1, counts.sum())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_joint_heatmap(joint_counts):
    """Plots a 2D Heatmap of the joint (Speed x Turn) distribution."""
    joint_matrix = np.zeros((5, 4), dtype=np.int64)

    for (speed, turn), count in joint_counts.items():
        joint_matrix[speed, turn] = count

    joint_probs = joint_matrix / max(1, joint_matrix.sum())

    plt.figure(figsize=(7, 5))
    im = plt.imshow(joint_probs, cmap="viridis")
    plt.colorbar(im, label="Probability")

    plt.xticks(range(4), [TURN_MAP[i] for i in range(4)])
    plt.yticks(range(5), [SPEED_MAP[i] for i in range(5)])
    plt.xlabel("Turn Action")
    plt.ylabel("Speed Action")
    plt.title("Joint Action Distribution (Speed × Turn)")

    # Annotate cells
    for i in range(5):
        for j in range(4):
            if joint_probs[i, j] > 0:
                plt.text(j, i, f"{joint_probs[i,j]:.2f}", 
                         ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(features):
    """Plots histograms for all collected scalar/array observation features."""
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5, rows * 3))

    for i, (k, arr) in enumerate(features.items()):
        plt.subplot(rows, cols, i + 1)
        plt.hist(arr, bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.title(k)
        
    plt.tight_layout()
    plt.show()



def main(visualize=False):
    """
    Main execution function.
    
    Args:
        visualize (bool): If True, plots histograms, heatmaps, and bar charts.
    """
    files = sorted(DEMO_DIR.glob("*.npz"))
    print(f"Found {len(files)} demos in {DEMO_DIR.resolve()}")
    assert len(files) > 0, f"No demos found in {DEMO_DIR}"

    # Print structural info using the first file
    first_demo = np.load(files[0], allow_pickle=True)
    print_dataset_structure(first_demo)

    # Process all demos in a single optimized pass
    print("\nProcessing datasets (Extracting stats & features)...")
    stats, features = process_demos(files)

    print_statistics_report(stats)
    

    if visualize:
        print("\nRendering visualizations...")
        
        speed_labels = [SPEED_MAP[i] for i in range(5)]
        plot_bar_distribution(speed_labels, stats["speed_counts"], "Speed Action Distribution")

        turn_labels = [TURN_MAP[i] for i in range(4)]
        plot_bar_distribution(turn_labels, stats["turn_counts"], "Turn Action Distribution")

        plot_joint_heatmap(stats["joint_counts"])
        plot_feature_distributions(features)


if __name__ == "__main__":
    main(visualize=False)
