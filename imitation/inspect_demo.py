# this script checks the distribution of collected data for imitation learning

from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


speed_map = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Reverse",
    4: "Constant"
}

turn_map = {
    0: "Right",
    1: "Left",
    2: "No Turn",
    3: "Straight"
}



def print_dataset_structure(sample_file):
    """Show all stored arrays/features in the dataset."""
    d = np.load(sample_file, allow_pickle=True)

    print("\nDataset structure:")
    for k in d.files:
        arr = d[k]
        print(f"{k:15s} shape={arr.shape} dtype={arr.dtype}")


def update_action_counts(actions, speed_counts, turn_counts):
    sp = actions[:, 0]
    tr = actions[:, 1]

    for i in range(5):
        speed_counts[i] += (sp == i).sum()

    for i in range(4):
        turn_counts[i] += (tr == i).sum()

    return sp, tr


def compute_joint_actions(files):
    joint = Counter()

    for f in files:
        d = np.load(f, allow_pickle=True)
        actions = d["actions"]

        for a in actions:
            joint[tuple(map(int, a))] += 1

    return joint


def plot_distribution(labels, probs, title):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs)
    plt.title(title)
    plt.ylabel("Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_joint_heatmap(joint):
    joint_matrix = np.zeros((5, 4), dtype=np.int64)

    for (speed, turn), v in joint.items():
        joint_matrix[speed, turn] = v

    joint_probs = joint_matrix / max(1, joint_matrix.sum())

    plt.figure(figsize=(7, 5))
    im = plt.imshow(joint_probs, cmap="viridis")

    plt.colorbar(im, label="Probability")

    plt.xticks(range(4), [turn_map[i] for i in range(4)])
    plt.yticks(range(5), [speed_map[i] for i in range(5)])

    plt.xlabel("Turn Action")
    plt.ylabel("Speed Action")
    plt.title("Joint Action Distribution (Speed × Turn)")

    # annotate
    for i in range(5):
        for j in range(4):
            if joint_probs[i, j] > 0:
                plt.text(j, i, f"{joint_probs[i,j]:.2f}",
                         ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(files, max_samples=200000):
    """
    Plot histograms for all scalar observation features.
    """

    collected = {}

    for f in files:
        d = np.load(f, allow_pickle=True)

        for k in d.files:

            if k in ["actions", "terminated", "truncated", "dones", "t"]:
                continue

            arr = d[k]

            # flatten but avoid giant tensors
            if arr.ndim > 2:
                arr = arr.reshape(-1)
            else:
                arr = arr.flatten()

            if k not in collected:
                collected[k] = []

            collected[k].append(arr)

    # concatenate
    for k in collected:
        collected[k] = np.concatenate(collected[k])

        # random subsample if too big
        if len(collected[k]) > max_samples:
            idx = np.random.choice(len(collected[k]), max_samples, replace=False)
            collected[k] = collected[k][idx]

    n = len(collected)

    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols*5, rows*3))

    for i, (k, v) in enumerate(collected.items()):
        plt.subplot(rows, cols, i+1)

        plt.hist(v, bins=50)
        plt.title(k)

    plt.tight_layout()
    plt.show()




demo_dir = Path("demos")
files = sorted(demo_dir.glob("*.npz"))

print("num demos:", len(files))
assert len(files) > 0, "No demos found in ./demos"

# show structure using first file
print_dataset_structure(files[0])




total_T = 0
turn_counts = np.zeros(4, dtype=np.int64)
speed_counts = np.zeros(5, dtype=np.int64)
terminated_ct = 0

for f in files:
    d = np.load(f, allow_pickle=True)

    actions = d["actions"]          # (T,2)
    presence = d["obs_presence"]    # (T,25,11)
    terminated = d["terminated"]    # (T,)

    T = actions.shape[0]
    total_T += T

    assert actions.shape[1] == 2
    assert presence.shape[0] == T

    sp, tr = actions[:,0], actions[:,1]

    assert sp.min() >= 0 and sp.max() <= 4, (f, sp.min(), sp.max())
    assert tr.min() >= 0 and tr.max() <= 3, (f, tr.min(), tr.max())

    # presence range
    assert presence.min() >= 0 and presence.max() <= 9, (f, presence.min(), presence.max())

    update_action_counts(actions, speed_counts, turn_counts)

    terminated_ct += terminated.astype(np.int32).sum()


print("\nTotal timesteps:", total_T)

print("\nSpeed counts:")
print(speed_counts, "->", speed_counts / max(1, speed_counts.sum()))

print("\nTurn counts:")
print(turn_counts, "->", turn_counts / max(1, turn_counts.sum()))

print("\nTerminated steps:", terminated_ct)



joint = compute_joint_actions(files)

print("\nTop 20 joint actions in dataset:")

for (speed, turn), v in joint.most_common(20):
    print(f"{speed_map[speed]} | {turn_map[turn]} : {v}")




speed_labels = [speed_map[i] for i in range(5)]
speed_probs = speed_counts / max(1, speed_counts.sum())

plot_distribution(speed_labels, speed_probs, "Speed Action Distribution")

turn_labels = [turn_map[i] for i in range(4)]
turn_probs = turn_counts / max(1, turn_counts.sum())

plot_distribution(turn_labels, turn_probs, "Turn Action Distribution")

plot_joint_heatmap(joint)



plot_feature_distributions(files)
