import os
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from imitation.bc_policy import BCCNN

# -------------------------
# Dataset
# -------------------------
class BCDataset(Dataset):
    """
    Expects dataset_bc.npz created by build_dataset.py with keys like:
      - obs_presence: (N, 25, 11)  float32 (values like 0/1/2/9 etc.)
      - obs_lane_angle: (N, 1) or (N,)
      - obs_ego_in_lane_position_x: (N, 1) or (N,)
      - obs_ego_speed_x: (N, 1) or (N,)
      - obs_ego_speed_y: (N, 1) or (N,)
      - obs_traffic_signs: (N, K)  (optional)
      - actions: (N, 2) int64  [speed_action, turn_action]
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        # Required
        self.presence = data["obs_presence"].astype(np.float32)  # (N,25,11)
        self.actions = data["actions"].astype(np.int64)          # (N,2)

        # Optional scalars
        def get_key(key, default=None):
            return data[key] if key in data.files else default

        lane_angle = get_key("obs_lane_angle")
        ego_lat = get_key("obs_ego_in_lane_position_x")
        ego_vx = get_key("obs_ego_speed_x")
        ego_vy = get_key("obs_ego_speed_y")
        traffic = get_key("obs_traffic_signs")

        # Make sure shapes are (N, d)
        scalars = []
        for arr in [lane_angle, ego_lat, ego_vx, ego_vy]:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            scalars.append(arr.astype(np.float32))

        if traffic is not None:
            traffic = np.asarray(traffic)
            if traffic.ndim == 1:
                traffic = traffic[:, None]
            scalars.append(traffic.astype(np.float32))

        if len(scalars) == 0:
            # still keep a dummy scalar so model code stays simple
            self.scalars = np.zeros((self.presence.shape[0], 1), dtype=np.float32)
        else:
            self.scalars = np.concatenate(scalars, axis=1).astype(np.float32)

        # Normalize presence to something sane:
        # presence can contain categorical IDs (0,1,2,9...). We'll just scale to [0,1] by dividing by max.
        # This is a quick baseline. Later we can one-hot encode types.
        maxv = float(np.max(self.presence)) if np.max(self.presence) > 0 else 1.0
        self.presence = self.presence / maxv

    def __len__(self):
        return self.presence.shape[0]

    def __getitem__(self, idx):
        grid = self.presence[idx]  # (25,11)
        sc = self.scalars[idx]     # (d,)
        act = self.actions[idx]    # (2,)
        return torch.from_numpy(grid), torch.from_numpy(sc), torch.from_numpy(act)




# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_speed = 0
    correct_turn = 0
    loss_sum = 0.0

    ce = nn.CrossEntropyLoss()

    for grid, scalars, act in loader:
        grid = grid.to(device)
        scalars = scalars.to(device)
        act = act.to(device)

        speed_y = act[:, 0]
        turn_y = act[:, 1]

        speed_logits, turn_logits = model(grid, scalars)

        loss = ce(speed_logits, speed_y) + ce(turn_logits, turn_y)

        loss_sum += float(loss.item()) * grid.size(0)
        total += grid.size(0)

        correct_speed += int((speed_logits.argmax(dim=1) == speed_y).sum().item())
        correct_turn += int((turn_logits.argmax(dim=1) == turn_y).sum().item())

    return {
        "loss": loss_sum / max(1, total),
        "acc_speed": correct_speed / max(1, total),
        "acc_turn": correct_turn / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_bc.npz")
    ap.add_argument("--out", type=str, default="bc_cnn.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ds = BCDataset(args.data)
    n = len(ds)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    # infer scalar dim
    _, scalars0, _ = ds[0]
    scalar_dim = int(scalars0.numel())

    model = BCCNN(scalar_dim=scalar_dim).to(device)

    # --- class imbalance handling (optional but helpful)
    # We'll compute weights for speed and turn CE from the dataset.
    actions = ds.actions
    speed_counts = np.bincount(actions[:, 0], minlength=5).astype(np.float32)
    turn_counts = np.bincount(actions[:, 1], minlength=4).astype(np.float32)

    # inverse frequency weights (smoothed)
    speed_w = 1.0 / np.maximum(speed_counts, 1.0)
    turn_w = 1.0 / np.maximum(turn_counts, 1.0)
    speed_w = speed_w / speed_w.mean()
    turn_w = turn_w / turn_w.mean()

    ce_speed = nn.CrossEntropyLoss(weight=torch.tensor(speed_w, dtype=torch.float32, device=device))
    ce_turn  = nn.CrossEntropyLoss(weight=torch.tensor(turn_w, dtype=torch.float32, device=device))

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")

    print(f"Dataset: N={n} train={n_train} val={n_val}")
    print(f"Device: {device}")
    print("Speed counts:", speed_counts, "weights:", np.round(speed_w, 3))
    print("Turn  counts:", turn_counts, "weights:", np.round(turn_w, 3))

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        seen = 0

        for grid, scalars, act in train_loader:
            grid = grid.to(device)
            scalars = scalars.to(device)
            act = act.to(device)

            speed_y = act[:, 0]
            turn_y = act[:, 1]

            speed_logits, turn_logits = model(grid, scalars)
            loss = ce_speed(speed_logits, speed_y) + ce_turn(turn_logits, turn_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * grid.size(0)
            seen += grid.size(0)

        train_loss = loss_sum / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)

        dt = time.time() - t0
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc_speed={val_metrics['acc_speed']:.3f} | "
            f"val_acc_turn={val_metrics['acc_turn']:.3f} | "
            f"time={dt:.1f}s"
        )

        # save best
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "scalar_dim": scalar_dim,
                "meta": {
                    "n_speed": 5,
                    "n_turn": 4,
                    "presence_shape": (25, 11),
                },
            }
            torch.save(ckpt, args.out)
            print(f"  [saved best] {args.out} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()