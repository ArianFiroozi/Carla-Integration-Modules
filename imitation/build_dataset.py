from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# paths
DEMO_DIR = Path("demos")
OUT_PATH = Path("dataset_bc.npz")

# sampling
KEEP_GO_STRAIGHT = 1.0
KEEP_CONSTANT = 0.25

# termination filtering
DROP_TERMINATED = True
DROP_LAST_N_BEFORE_TERMINATION = 10

# silence trimming
TRIM_START_SILENCE = True
SPEED_KEY = "obs_ego_speed_x"
SILENCE_SPEED_THRESHOLD = 0.5
MAX_IDLE_STEPS = 10000


RNG_SEED = 42


trim_stats = {
    "total_frames": 0,

    "start_silence_trimmed": 0,

    "terminated_dropped": 0,
    "truncated_dropped": 0,

    "pre_termination_dropped": 0,

    "sampling_dropped": 0,
    "kept": 0,
}



# action maps
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


files = sorted(DEMO_DIR.glob("*.npz"))
assert files, "No demos found in ./demos"

rng = np.random.default_rng(RNG_SEED)



def should_keep(action):

    speed, turn = map(int, action)

    rare_turn = turn in (0, 1, 2)
    rare_speed = speed in (0, 1, 2, 3)

    if rare_turn or rare_speed:
        return True

    keep_prob = 1.0

    if turn == 3:
        keep_prob *= KEEP_GO_STRAIGHT

    if speed == 4:
        keep_prob *= KEEP_CONSTANT

    return rng.random() < keep_prob



def compute_start_trim(d):
    """
    Remove initial frames where vehicle speed is below threshold.
    """

    speed = d[SPEED_KEY]

    # ensure 1D
    speed = np.abs(speed.reshape(-1))

    T = len(speed)

    for i in range(min(T, MAX_IDLE_STEPS)):

        if speed[i] > SILENCE_SPEED_THRESHOLD:
            return max(0, i - 2) 

    return 0


def build_keep_mask(d, stats):

    actions = d["actions"]
    terminated = d["terminated"].astype(bool)
    truncated = d["truncated"].astype(bool)

    T = actions.shape[0]
    stats["total_frames"] += T

    mask = np.ones(T, dtype=bool)

  
    if TRIM_START_SILENCE:
        trim_idx = compute_start_trim(d)
        if trim_idx > 0:
            stats["start_silence_trimmed"] += trim_idx
            mask[:trim_idx] = False


    term_idx = None
    if terminated.any():
        term_idx = int(np.argmax(terminated))

    if DROP_TERMINATED:
        term_drop = np.sum(terminated)
        trunc_drop = np.sum(truncated)

        stats["terminated_dropped"] += term_drop
        stats["truncated_dropped"] += trunc_drop

        mask &= ~(terminated | truncated)

    if term_idx is not None and DROP_LAST_N_BEFORE_TERMINATION > 0:
        start = max(0, term_idx - DROP_LAST_N_BEFORE_TERMINATION)
        dropped = np.sum(mask[start:term_idx + 1])

        stats["pre_termination_dropped"] += dropped
        mask[start:term_idx + 1] = False

    keep = np.zeros(T, dtype=bool)

    for t in range(T):
        if not mask[t]:
            continue

        if should_keep(actions[t]):
            keep[t] = True
        else:
            stats["sampling_dropped"] += 1

    stats["kept"] += np.sum(keep)

    return keep


def print_joint_distribution(actions, top_k=20):

    print("\nJoint action distribution (speed, turn):")

    total = len(actions)
    counter = Counter(map(tuple, actions))

    for (speed, turn), count in counter.most_common(top_k):

        pct = 100.0 * count / total

        print(
            f"{speed_map[speed]:10s} | {turn_map[turn]:10s} : "
            f"{count:6d} ({pct:5.2f}%)"
        )

    print("total samples:", total)


# ------------------------------------------------------------
# PASS 1 — compute keep masks
# ------------------------------------------------------------

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

    keep = build_keep_mask(d, trim_stats)


    keep_masks.append(keep)
    total_kept += int(keep.sum())


print("PASS1 total_kept:", total_kept)

assert total_kept > 0, "No samples kept — loosen sampling/drop rules."

print("\nObservation structure:")
for k in obs_keys:
    print(f"{k:30s} -> {obs_shapes[k]}")


# ------------------------------------------------------------
# Allocate output arrays
# ------------------------------------------------------------

out_obs = {
    k: np.empty((total_kept, *obs_shapes[k]), dtype=np.float32)
    for k in obs_keys
}

out_actions = np.empty((total_kept, 2), dtype=np.int64)


# ------------------------------------------------------------
# PASS 2 — fill dataset
# ------------------------------------------------------------

idx = 0

for f, keep in zip(files, keep_masks):

    d = np.load(f, allow_pickle=True)

    n = int(keep.sum())

    if n == 0:
        continue

    out_actions[idx:idx+n] = d["actions"][keep].astype(np.int64)

    for k in obs_keys:

        arr = d[k][keep]

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if arr.ndim == 1:
            arr = arr[:, None]

        out_obs[k][idx:idx+n] = arr

    idx += n


print("PASS2 filled:", idx)

assert idx == total_kept, (idx, total_kept)

print_joint_distribution(out_actions)


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

save_dict = {**out_obs, "actions": out_actions}

np.savez_compressed(OUT_PATH, **save_dict)

print("saved:", OUT_PATH.resolve())


# ------------------------------------------------------------
# Final statistics
# ------------------------------------------------------------

sp = out_actions[:, 0]
tr = out_actions[:, 1]

speed_counts = np.bincount(sp, minlength=5)
turn_counts = np.bincount(tr, minlength=4)

print("\nFinal dataset size:", len(out_actions))
print("Average samples per demo:", len(out_actions) / len(files))


print("\nSpeed distribution:")
for i, c in enumerate(speed_counts):

    pct = 100 * c / len(sp)

    print(f"{speed_map[i]:10s} : {c:6d} ({pct:5.2f}%)")


print("\nTurn distribution:")
for i, c in enumerate(turn_counts):

    pct = 100 * c / len(tr)

    print(f"{turn_map[i]:10s} : {c:6d} ({pct:5.2f}%)")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

# Speed distribution
plt.figure(figsize=(6,4))

speed_labels = [speed_map[i] for i in range(5)]
speed_probs = speed_counts / speed_counts.sum()

plt.bar(speed_labels, speed_probs)

plt.title("Speed Action Distribution")
plt.ylabel("Probability")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# Turn distribution
plt.figure(figsize=(6,4))

turn_labels = [turn_map[i] for i in range(4)]
turn_probs = turn_counts / turn_counts.sum()

plt.bar(turn_labels, turn_probs)

plt.title("Turn Action Distribution")
plt.ylabel("Probability")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# Joint distribution heatmap

joint_matrix = np.zeros((5,4), dtype=np.int64)

np.add.at(joint_matrix, (sp, tr), 1)

joint_probs = joint_matrix / joint_matrix.sum()

plt.figure(figsize=(7,5))

im = plt.imshow(joint_probs, cmap="viridis")

plt.colorbar(im, label="Probability")

plt.xticks(range(4), [turn_map[i] for i in range(4)])
plt.yticks(range(5), [speed_map[i] for i in range(5)])

plt.xlabel("Turn")
plt.ylabel("Speed")
plt.title("Joint Action Distribution")

for i in range(5):
    for j in range(4):
        if joint_probs[i, j] > 0:
            plt.text(
                j,
                i,
                f"{joint_probs[i,j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9
            )

plt.tight_layout()
plt.show()


total = trim_stats["total_frames"]



def pct(x):
    return 100.0 * x / total if total > 0 else 0.0

print("\n" + "="*50)
print("TRIMMING / FILTERING STATISTICS")
print("="*50)

print(f"Total frames seen           : {total}")

print("\n--- Hard trimming ---")
print(f"Start silence trimmed       : {trim_stats['start_silence_trimmed']:8d} ({pct(trim_stats['start_silence_trimmed']):5.2f}%)")
print(f"Terminated dropped          : {trim_stats['terminated_dropped']:8d} ({pct(trim_stats['terminated_dropped']):5.2f}%)")
print(f"Truncated dropped           : {trim_stats['truncated_dropped']:8d} ({pct(trim_stats['truncated_dropped']):5.2f}%)")
print(f"Pre-termination dropped     : {trim_stats['pre_termination_dropped']:8d} ({pct(trim_stats['pre_termination_dropped']):5.2f}%)")

print("\n--- Sampling ---")
print(f"Dropped by sampling         : {trim_stats['sampling_dropped']:8d} ({pct(trim_stats['sampling_dropped']):5.2f}%)")

print("\n--- Final ---")
print(f"Kept frames                 : {trim_stats['kept']:8d} ({pct(trim_stats['kept']):5.2f}%)")
print("="*50)
