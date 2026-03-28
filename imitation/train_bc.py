import argparse
import time
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import f1_score

from .datasets.bc_dataset import BCDataset, BCDatasetContinuous
from .models.imitation_policy import ImitationPolicy

from . import config


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_dataset(dataset, val_split, seed):

    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_ds, val_ds

@torch.no_grad()
def evaluate_discrete(model, loader, device):

    model.eval()

    ce = nn.CrossEntropyLoss()

    total = 0
    loss_sum = 0.0

    speed_preds = []
    speed_targets = []

    turn_preds = []
    turn_targets = []

    for grid, scalars, act in loader:

        grid = grid.to(device)
        scalars = scalars.to(device)
        act = act.to(device)

        speed_y = act[:, 0]
        turn_y = act[:, 1]

        speed_logits, turn_logits = model(grid, scalars)

        loss = ce(speed_logits, speed_y) + ce(turn_logits, turn_y)

        loss_sum += loss.item() * grid.size(0)
        total += grid.size(0)

        speed_preds.append(speed_logits.argmax(1).cpu())
        turn_preds.append(turn_logits.argmax(1).cpu())

        speed_targets.append(speed_y.cpu())
        turn_targets.append(turn_y.cpu())

    speed_preds = torch.cat(speed_preds).numpy()
    turn_preds = torch.cat(turn_preds).numpy()

    speed_targets = torch.cat(speed_targets).numpy()
    turn_targets = torch.cat(turn_targets).numpy()

    return {
        "loss": loss_sum / total,
        "f1_speed": f1_score(speed_targets, speed_preds, average="macro"),
        "f1_turn": f1_score(turn_targets, turn_preds, average="macro"),
    }

@torch.no_grad()
def evaluate_continuous(model, loader, device, is_gaussian=False):
    model.eval()

    mae_sum = 0.0
    mse_sum = 0.0
    nll_sum = 0.0
    n = 0

    for grid, scalars, target in loader:
        grid = grid.to(device)
        scalars = scalars.to(device)
        target = target.to(device)

        pred = model(grid, scalars)

        # Handle Gaussian vs Standard Continuous
        if is_gaussian:
            mean, std = pred
            dist = torch.distributions.Normal(mean, std)
            nll_sum += -dist.log_prob(target).sum().item()
            point_pred = mean  # Use the mean for MSE/MAE evaluation
        else:
            point_pred = pred

        mse_sum += nn.functional.mse_loss(point_pred, target, reduction="sum").item()
        mae_sum += torch.abs(point_pred - target).sum().item()
        n += np.prod(target.shape)

    metrics = {
        "mse": mse_sum / n,
        "mae": mae_sum / n,
    }
    
    # We use NLL as the primary validation loss for early-stopping if Gaussian
    if is_gaussian:
        metrics["nll"] = nll_sum / n
        metrics["loss"] = metrics["nll"]
    else:
        metrics["loss"] = metrics["mse"]
        
    return metrics



def train_epoch_discrete(model, loader, opt, ce_speed, ce_turn, device):

    model.train()

    total_loss = 0
    seen = 0

    for grid, scalars, act in loader:

        grid = grid.to(device)
        scalars = scalars.to(device)
        act = act.to(device)

        speed_y = act[:, 0]
        turn_y = act[:, 1]

        speed_logits, turn_logits = model(grid, scalars)

        loss = ce_speed(speed_logits, speed_y) + ce_turn(turn_logits, turn_y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        total_loss += loss.item() * grid.size(0)
        seen += grid.size(0)

    return total_loss / seen


def compute_class_weights(actions):
    n_classes = int(actions.max()) + 1
    counts = np.bincount(actions, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def debug_sampler_distribution(steer, weights, n_samples=200000):

    weights = weights / weights.sum()

    # sample indices according to weights
    idx = np.random.choice(len(weights), size=n_samples, p=weights)

    raw_bins = np.floor((steer + 1.0) / 0.1).astype(int)
    sampled_bins = raw_bins[idx]

    raw_counts = np.bincount(raw_bins, minlength=21)
    sampled_counts = np.bincount(sampled_bins, minlength=21)

    print("\nSteering distribution (raw vs sampled):")

    for i in range(21):
        lo = -1.0 + i * 0.1
        hi = lo + 0.1

        raw_pct = raw_counts[i] / raw_counts.sum() * 100
        samp_pct = sampled_counts[i] / sampled_counts.sum() * 100

        print(f"{lo:+.1f} to {hi:+.1f} | raw {raw_pct:5.2f}% -> sampled {samp_pct:5.2f}%")


def train_epoch_continuous(model, loader, opt, criterion, device):

    model.train()

    total_loss = 0
    seen = 0

    for grid, scalars, target in loader:

        grid = grid.to(device)
        scalars = scalars.to(device)
        target = target.to(device)

        pred = model(grid, scalars)

        loss = criterion(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        total_loss += loss.item() * grid.size(0)
        seen += grid.size(0)

    return total_loss / seen


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=config.DATASET_PATH)

    parser.add_argument(
        "--mode",
        choices=["discrete", "continuous"],
        default=config.ACTION_MODE,
    )

    parser.add_argument("--epochs", type=int, default=config.BC_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.BC_LR)
    parser.add_argument("--is_gaussian", action="store_true", help="Use Gaussian head for continuous mode" , default=config.IS_GAUSSIAN)
    
    parser.add_argument("--val_split", type=float, default=config.BC_VAL_SPLIT)

    parser.add_argument("--patience", type=int, default=config.BC_PATIENCE)

    parser.add_argument("--device", default=config.DEVICE)


    args = parser.parse_args()

    device = get_device(args.device)


    # Initialize dataset based on mode
    if args.mode == "discrete":
        ds = BCDataset(args.data)
        n_speed = int(ds.actions[:, 0].max()) + 1
        n_turn  = int(ds.actions[:, 1].max()) + 1
    else:
        # Incorporate friend's idea: one_hot_presence=True
        ds = BCDatasetContinuous(args.data)

    
    # Calculate weights based on mode
    if args.mode == "discrete":
        actions = ds.actions
        speed_weights = compute_class_weights(actions[:, 0]).to(device)
        turn_weights = compute_class_weights(actions[:, 1]).to(device)
    else:
        
        data_raw = np.load(args.data)
        
        throttle = data_raw["target_throttle"]
        brake = data_raw["target_brake"]
        steer = data_raw["target_steering_angle"]

        continuous_weights = np.ones_like(throttle, dtype=np.float32)
        # emphasize movement
        continuous_weights[throttle > 0.2] *= 5.0

        # emphasize turning
        continuous_weights[np.abs(steer) > 0.2] *= 3.0

        # emphasize braking
        continuous_weights[brake > 0.05] *= 3.0
        continuous_weights = continuous_weights.squeeze()
        # throttle = data_raw["target_throttle"].astype(np.float32)
        # steer = data_raw["target_steering_angle"].astype(np.float32)

        # throttle = throttle.squeeze()
        # steer = steer.squeeze()


        # # Bin steering
        # bin_size = 0.1
        # steer_bins = np.floor((steer + 1.0) / bin_size).astype(int)
        # n_bins = int(2.0 / bin_size) + 1

        # steer_bins = np.clip(steer_bins, 0, n_bins - 1)

        
        
        # # Compute bin frequencies
        # counts = np.bincount(steer_bins, minlength=n_bins).astype(np.float32)

        # freq = counts / counts.sum()

        # # avoid division by zero
        # freq = np.maximum(freq, 1e-6)

        # # 3. Steering weights
        # steer_weights = 1.0 / freq
        # steer_weights = steer_weights / steer_weights.mean()

        # sample_steer_weight = steer_weights[steer_bins]

        # # Throttle factor
        # sample_throttle_factor = 1.0 + 0.15 * throttle


        # # continuous_weights = sample_steer_weight * sample_throttle_factor
        
        # continuous_weights = sample_steer_weight
        # continuous_weights = continuous_weights.astype(np.float32)
        
        # debug_sampler_distribution(steer, continuous_weights)


    # Split dataset
    train_ds, val_ds = split_dataset(ds, args.val_split, seed=config.BC_SPLIT_SEED)

    # Initialize DataLoaders
    if args.mode == "continuous":
        # Get the indices of the data that ended up in the training split
        train_indices = train_ds.indices
        train_weights_subset = continuous_weights[train_indices]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights_subset),
            num_samples=len(train_weights_subset),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            # sampler=sampler,  # shuffle must be False when sampler is provided
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False
    )

    grid0, scalars0, _ = ds[0]
    grid_channels = grid0.shape[0]
    scalar_dim = scalars0.numel()

    # Model Initialization
    if args.mode == "discrete":
        model = ImitationPolicy(
            mode="discrete",
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
            n_speed=n_speed,
            n_turn=n_turn,
        ).to(device)
    else:
        model = ImitationPolicy(
            mode="continuous",
            is_gaussian=args.is_gaussian, # Pass the flag here
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
        ).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # Set up Continuous Loss Functions
    if args.mode == "continuous":
        if args.is_gaussian:
            # Negative Log-Likelihood for Gaussian
            def criterion(pred, target):
                mean, std = pred

                std = torch.clamp(std, 1e-3, 2.5)

                dist = torch.distributions.Normal(mean, std)
                return -dist.log_prob(target).mean()

        else:
            criterion = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0

    print("Dataset size:", len(ds))
    print("Train:", len(train_ds), "Val:", len(val_ds))
    print("Device:", device)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if args.mode == "discrete":
            
            ce_speed = nn.CrossEntropyLoss(weight=speed_weights)
            ce_turn = nn.CrossEntropyLoss(weight=turn_weights)

            train_loss = train_epoch_discrete(
                model,
                train_loader,
                opt,
                ce_speed,
                ce_turn,
                device,
            )

            val_metrics = evaluate_discrete(
                model,
                val_loader,
                device,
            )

            val_loss = val_metrics["loss"]

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"speed_f1={val_metrics['f1_speed']:.3f} | "
                f"turn_f1={val_metrics['f1_turn']:.3f} | "
                f"time={time.time()-t0:.1f}s"
            )
        else:
            # Because criterion is a custom function that unpacks (mean, std), 
            # train_epoch_continuous works out of the box without needing changes.
            train_loss = train_epoch_continuous(
                model, train_loader, opt, criterion, device
            )

            val_metrics = evaluate_continuous(
                model, val_loader, device, is_gaussian=args.is_gaussian
            )

            val_loss = val_metrics["loss"] # Pulls nll or mse depending on mode

            if args.is_gaussian:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_nll={train_loss:.5f} | "
                    f"val_nll={val_metrics['nll']:.5f} | "
                    f"val_mse={val_metrics['mse']:.5f} | "
                    f"val_mae={val_metrics['mae']:.4f} | "
                    f"time={time.time()-t0:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_mse={train_loss:.5f} | "
                    f"val_mse={val_metrics['mse']:.5f} | "
                    f"val_mae={val_metrics['mae']:.4f} | "
                    f"time={time.time()-t0:.1f}s"
                )

        # Early Stopping / Checkpointing
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0

            save_path = config.CONTINUOUS_MODEL_PATH if args.mode == "continuous" else config.DISCRETE_MODEL_PATH

            ckpt = {
                "model_state": model.state_dict(),
                "mode": args.mode,
                "is_gaussian": args.mode == "continuous" and args.is_gaussian, # Save Gaussian state!
                "scalar_dim": scalar_dim,
                "grid_channels": grid_channels,
            }

            if args.mode == "discrete":
                ckpt["n_speed"] = n_speed
                ckpt["n_turn"] = n_turn

            torch.save(ckpt, save_path)
            print("[saved best model]")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("\nEarly stopping triggered.")
                break



if __name__ == "__main__":
    main()

