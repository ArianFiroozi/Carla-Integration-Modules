# this script is for checking the distribution of collected data for imitation learning 

from pathlib import Path
import numpy as np

demo_dir = Path("demos")
files = sorted(demo_dir.glob("*.npz"))
print("num demos:", len(files))
assert len(files) > 0, "No demos found in ./demos"

total_T = 0
turn_counts = np.zeros(4, dtype=np.int64)
speed_counts = np.zeros(5, dtype=np.int64)
terminated_ct = 0

for f in files[:20]:  # inspect first 20
    d = np.load(f, allow_pickle=True)

    actions = d["actions"]          # (T,2)
    presence = d["obs_presence"]    # (T,25,11)
    terminated = d["terminated"]    # (T,)

    T = actions.shape[0]
    total_T += T

    assert actions.shape[1] == 2
    assert presence.shape[0] == T

    # action ranges
    sp = actions[:,0]
    tr = actions[:,1]
    assert sp.min() >= 0 and sp.max() <= 4, (f, sp.min(), sp.max())
    assert tr.min() >= 0 and tr.max() <= 3, (f, tr.min(), tr.max())

    # presence range
    assert presence.min() >= 0 and presence.max() <= 9, (f, presence.min(), presence.max())

    for i in range(5):
        speed_counts[i] += (sp == i).sum()
    for i in range(4):
        turn_counts[i] += (tr == i).sum()

    terminated_ct += terminated.astype(np.int32).sum()

print("total timesteps:", total_T)
print("speed_counts:", speed_counts, "->", speed_counts / max(1, speed_counts.sum()))
print("turn_counts :", turn_counts, "->", turn_counts / max(1, turn_counts.sum()))
print("terminated steps:", terminated_ct)