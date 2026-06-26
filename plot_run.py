"""
Standalone run plotter — read a SAC experiment's TensorBoard event file and save
a PNG of the key training curves. This bypasses the TensorBoard web UI entirely
(useful when the TB server is broken by a protobuf/version mismatch).

Usage (from repo root):
    python plot_run.py                                  # plots the latest run in experiments/rl/sac
    python plot_run.py experiments/rl/sac/2026-06-22_14-42-10
    python plot_run.py <run_dir> --out myplots.png

Output: <run_dir>/plots.png  (or the --out path)
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed (works over SSH / headless)
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (tensorboard tag, human title) for each panel we want to draw
PANELS = [
    ("train/critic_loss", "Critic loss"),
    ("grad_norms/critic", "Critic grad-norm (pre-clip)"),
    ("train/actor_loss", "Actor loss"),
    ("train/alpha", "Alpha (entropy temp)"),
    ("critic/q1_mean", "Q1 mean"),
    ("policy/log_std_mean", "Policy log_std (exploration)"),
    ("train/episode_return", "Episode return"),
    ("train/episode_length", "Episode length"),
]


def find_latest_run(base="experiments/rl/sac"):
    base = Path(base)
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "tb").exists()]
    if not runs:
        sys.exit(f"No runs with a tb/ folder found under {base}")
    return max(runs, key=lambda d: d.stat().st_mtime)


def smooth(y, k=15):
    if len(y) < k or k < 2:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None, help="experiment dir (defaults to latest)")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    tb_dir = run_dir / "tb" if (run_dir / "tb").exists() else run_dir
    print(f"Reading events from: {tb_dir}")

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 1_000_000})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    print(f"Found {len(available)} scalar tags.")

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(f"SAC run: {run_dir.name}", fontsize=15)

    for ax, (tag, title) in zip(axes.flat, PANELS):
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        if tag not in available:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            continue
        pts = ea.Scalars(tag)
        steps = np.array([p.step for p in pts])
        vals = np.array([p.value for p in pts])
        ax.plot(steps, vals, alpha=0.35, lw=0.8)
        if len(vals) > 30:
            ax.plot(steps, smooth(vals), color="C3", lw=1.6)
        ax.set_xlabel("step")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out) if args.out else (run_dir / "plots.png")
    fig.savefig(out, dpi=110)
    print(f"\nSaved: {out.resolve()}")
    print("Open that PNG to see the curves (no TensorBoard server needed).")


if __name__ == "__main__":
    main()
