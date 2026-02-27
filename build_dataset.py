from pathlib import Path
import numpy as np

DEMO_DIR = Path("demos")
OUT_PATH = Path("dataset_bc.npz")

KEEP_GO_STRAIGHT = 0.25
KEEP_CONSTANT = 0.25

DROP_TERMINATED = True
DROP_LAST_N_BEFORE_TERMINATION = 10

files = sorted(DEMO_DIR.glob("*.npz"))
assert files, "No demos found in ./demos"

rng = np.random.default_rng(0)  # deterministic

def should_keep(action):
    speed, turn = int(action[0]), int(action[1])

    rare_turn = turn in [0, 1, 2]
    rare_speed = speed in [0, 1, 2, 3]
    if rare_turn or rare_speed:
        return True

    keep_prob = 1.0
    if turn == 3:
        keep_prob *= KEEP_GO_STRAIGHT
    if speed == 4:
        keep_prob *= KEEP_CONSTANT

    return rng.random() < keep_prob


def build_keep_mask(d):
    actions = d["actions"]
    terminated = d["terminated"].astype(bool)
    truncated = d["truncated"].astype(bool)
    T = actions.shape[0]

    term_idx = None
    if terminated.any():
        term_idx = int(np.argmax(terminated))

    mask = np.ones(T, dtype=bool)

    if DROP_TERMINATED:
        mask &= ~(terminated | truncated)

    if term_idx is not None and DROP_LAST_N_BEFORE_TERMINATION > 0:
        start = max(0, term_idx - DROP_LAST_N_BEFORE_TERMINATION)
        mask[start:term_idx + 1] = False

    keep = np.zeros(T, dtype=bool)
    for t in range(T):
        if not mask[t]:
            continue
        if should_keep(actions[t]):
            keep[t] = True

    return keep


# ---------- PASS 1: build + store keep masks ----------
keep_masks = []
total_kept = 0
obs_keys = None
obs_shapes = {}

for f in files:
    d = np.load(f, allow_pickle=True)

    if obs_keys is None:
        obs_keys = [k for k in d.files if k.startswith("obs_")]
        for k in obs_keys:
            obs_shapes[k] = d[k].shape[1:]  # drop time dim

    keep = build_keep_mask(d)
    keep_masks.append(keep)
    total_kept += int(keep.sum())

print("PASS1 total_kept:", total_kept)
assert total_kept > 0, "No samples kept — loosen sampling/drop rules."


# ---------- allocate ----------
out_obs = {k: np.empty((total_kept, *obs_shapes[k]), dtype=np.float32) for k in obs_keys}
out_actions = np.empty((total_kept, 2), dtype=np.int64)


# ---------- PASS 2: fill using saved masks ----------
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
        out_obs[k][idx:idx+n] = arr

    idx += n

print("PASS2 filled:", idx)
assert idx == total_kept, (idx, total_kept)


# ---------- save ----------
save_dict = {**out_obs, "actions": out_actions}
np.savez_compressed(OUT_PATH, **save_dict)
print("saved:", OUT_PATH.resolve())

# dist check
sp = out_actions[:, 0]
tr = out_actions[:, 1]
print("speed dist:", np.bincount(sp, minlength=5), np.bincount(sp, minlength=5) / len(sp))
print("turn  dist:", np.bincount(tr, minlength=4), np.bincount(tr, minlength=4) / len(tr))