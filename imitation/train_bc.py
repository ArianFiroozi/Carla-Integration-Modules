import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import time
import json
import os
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
from utils.experiment_logger import ExperimentLogger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def validate_dataset_config(dataset_meta, args):
    if dataset_meta is None:
        print("No dataset metadata found, skipping validation.")
        return

    dataset_mode = dataset_meta.get("pipeline_config", {}).get("mode")

    if dataset_mode is not None and dataset_mode != args.mode:
        raise ValueError(
            f"Dataset mode ({dataset_mode}) does not match training mode ({args.mode})"
        )

    print("Dataset configuration validated successfully.")


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
    std_sum = 0.0
    n = 0

    for grid, scalars, target in loader:
        grid = grid.to(device)
        scalars = scalars.to(device)
        target = target.to(device)

        pred = model(grid, scalars)

        # Handle Gaussian vs Standard Continuous
        if is_gaussian:
            mean, std = pred
            std = torch.clamp(std, min=config.MIN_STD, max=config.MAX_STD)
            dist = torch.distributions.Normal(mean, std)
            nll_sum += -dist.log_prob(target).sum().item()
            point_pred = mean  # Use the mean for MSE/MAE evaluation
        else:
            point_pred = pred

        mse_sum += nn.functional.mse_loss(point_pred, target, reduction="sum").item()
        mae_sum += torch.abs(point_pred - target).sum().item()
        std_sum += std.sum().item()
        n += np.prod(target.shape)

    metrics = {
        "mse": mse_sum / n,
        "mae": mae_sum / n,
    }
    
    # We use NLL as the primary validation loss for early-stopping if Gaussian
    if is_gaussian:
        metrics["nll"] = nll_sum / n
        metrics["loss"] = metrics["nll"]
        metrics["avg_std"] = std_sum / n 

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


def train_epoch_continuous(model, loader, opt, device):
    model.train()
    total_loss = 0
    seen = 0

    for grid, scalars, target in loader:
        grid = grid.to(device)
        scalars = scalars.to(device)
        target = target.to(device)

        pred = model(grid, scalars)


        if config.IS_GAUSSIAN:
            # GAUSSIAN MODE (NLL)
            mean, std = pred
            

            std = torch.clamp(std, min=config.MIN_STD, max=config.MAX_STD) 
            
            dist = torch.distributions.Normal(mean, std)
            per_element_loss = -dist.log_prob(target)
        else:
            # STANDARD MODE (MSE)
            per_element_loss = (pred - target) ** 2


        if config.USE_WEIGHTED_LOSS:
            throttle_loss = per_element_loss[:, 0] * config.THROTTLE_LOSS_WEIGHT
            brake_loss = per_element_loss[:, 1] * config.BRAKE_LOSS_WEIGHT
            
            steer_diff = torch.abs(target[:, 2])
            steer_weights = torch.where(
                steer_diff > config.WEIGHTED_LOSS_THRESHOLD,
                torch.tensor(config.STEER_LOSS_WEIGHT, device=device, dtype=torch.float32),
                torch.tensor(1.0, device=device, dtype=torch.float32)
            )
            steer_loss = per_element_loss[:, 2] * steer_weights
            
            # Combine weighted losses
            loss = torch.mean(throttle_loss + brake_loss + steer_loss)
        else:
            # If not weighted, just take the mean across the batch and controls
            loss = torch.mean(per_element_loss)


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


    dataset_meta = None
    meta_path = Path(args.data).with_suffix(".meta.json")

    if meta_path.exists():
        with open(meta_path, "r") as f:
            dataset_meta = json.load(f)
        print(f"[INFO] Loaded dataset meta from {meta_path}")
    else:
        print(f"[WARN] Dataset meta not found: {meta_path}")
    validate_dataset_config(dataset_meta, args)


    experiment_name = f"bc_{args.mode}"
    logger = ExperimentLogger(experiment_name)

    
    config_dict = {
    "mode": args.mode,
    "epochs": args.epochs,
    "batch_size": args.batch,
    "lr": args.lr,
    "val_split": args.val_split,
    "patience": args.patience,
    "device": args.device,
    "is_gaussian": args.is_gaussian,
    "use_continuous_undersampling": config.USE_CONTINUOUS_UNDERSAMPLING,
    "undersampling_threshold_continuous": config.UNDERSAMPLING_THRESHOLD,
    "undersampling_probability_continuous": config.UNDERSAMPLING_PROBABILITY,
    "use_weighted_loss": config.USE_WEIGHTED_LOSS,
    "steer_loss_weight_continuous": config.STEER_LOSS_WEIGHT,
    "throttle_loss_weight_continuous": config.THROTTLE_LOSS_WEIGHT,
    "brake_loss_weight_continuous": config.BRAKE_LOSS_WEIGHT,
    "weighted_loss_threshold_continuous": config.WEIGHTED_LOSS_THRESHOLD,
    "min_std": config.MIN_STD,
    "max_std": config.MAX_STD,
    "weight_sampling": config.WEIGHTED_SAMPLING
    
    }
    # attach dataset metadata
    if dataset_meta is not None:
        config_dict["dataset_meta"] = dataset_meta

    logger.save_config(config_dict)

    tb_writer = SummaryWriter(log_dir=f"{logger.logs_dir}/tb")
    

    # log dataset info to tensorboard
    # Log dataset distributions

    if dataset_meta is not None:

    # --- Core dataset info ---
        tb_writer.add_scalar("dataset/total_samples", dataset_meta.get("total_samples", 0), 0)
        tb_writer.add_text("dataset/created_at", dataset_meta.get("created_at", "unknown"), 0)

        # --- Stats section ---
        stats = dataset_meta.get("stats", {})
        if "total_frames" in stats:
            tb_writer.add_scalar("dataset/total_frames", stats["total_frames"], 0)
        if "idle_frames_trimmed" in stats:
            tb_writer.add_scalar("dataset/idle_trimmed", stats["idle_frames_trimmed"], 0)

        # --- Mirror augmentation info ---
        pipe = dataset_meta.get("pipeline_config", {})
        tb_writer.add_text("dataset/pipeline_config", json.dumps(pipe, indent=2), 0)

        mirror_enabled = pipe.get("mirror_enabled", False)
        mirror_thresh = pipe.get("mirror_steering_threshold", None)

        tb_writer.add_scalar("dataset/mirror_enabled", float(bool(mirror_enabled)), 0)
        if mirror_thresh is not None:
            tb_writer.add_scalar("dataset/mirror_steering_threshold", mirror_thresh, 0)

        print(f"[TB] ✅ Mirror info logged → enabled={mirror_enabled}, threshold={mirror_thresh}")
    else:
        print("[TB] ⚠️ No dataset metadata available to log.")





    device = get_device(args.device)


    # Initialize dataset based on mode
    if args.mode == "discrete":
        ds = BCDataset(args.data)
        n_speed = int(ds.actions[:, 0].max()) + 1
        n_turn  = int(ds.actions[:, 1].max()) + 1
    else:
        ds = BCDatasetContinuous(args.data)

    print(type(ds))
    
    
    if args.mode == "discrete" and config.USE_WEIGHTED_LOSS:
        actions = ds.actions
        speed_weights = compute_class_weights(actions[:, 0]).to(device)
        turn_weights = compute_class_weights(actions[:, 1]).to(device)
        
        


    if args.mode == "continuous" and config.WEIGHTED_SAMPLING in ["inverse", "handmade"]:
        # Only load the raw dataset into memory if we actually need it for weights
        data_raw = np.load(args.data)
        
        throttle = data_raw["target_throttle"].astype(np.float32).squeeze()
        steer = data_raw["target_steering_angle"].astype(np.float32).squeeze()
        
        if config.WEIGHTED_SAMPLING == "handmade":
            brake = data_raw["target_brake"].astype(np.float32).squeeze()
            
            continuous_weights = np.ones_like(throttle, dtype=np.float32)
            
            # emphasize movement
            continuous_weights[throttle > 0.2] *= 5.0
            # emphasize turning
            continuous_weights[np.abs(steer) > 0.2] *= 3.0
            # emphasize braking
            continuous_weights[brake > 0.05] *= 3.0
            
        elif config.WEIGHTED_SAMPLING == "inverse":
            # Bin steering
            bin_size = 0.1
            steer_bins = np.floor((steer + 1.0) / bin_size).astype(int)
            n_bins = int(2.0 / bin_size) + 1
            steer_bins = np.clip(steer_bins, 0, n_bins - 1)
            
            # Compute bin frequencies
            counts = np.bincount(steer_bins, minlength=n_bins).astype(np.float32)
            freq = counts / counts.sum()
            # avoid division by zero
            freq = np.maximum(freq, 1e-6)

            # Steering weights
            steer_weights = 1.0 / freq
            steer_weights = steer_weights / steer_weights.mean()
            sample_steer_weight = steer_weights[steer_bins]

            # sample_throttle_factor = 1.0 + 0.15 * throttle
            # continuous_weights = sample_steer_weight * sample_throttle_factor
            
            continuous_weights = sample_steer_weight.astype(np.float32)
            
            debug_sampler_distribution(steer, continuous_weights)



    train_ds, val_ds = split_dataset(ds, args.val_split, seed=config.BC_SPLIT_SEED)


    
    # If we are in continuous mode AND we created a sampler
    if args.mode == "continuous" and config.WEIGHTED_SAMPLING in ["inverse", "handmade"]:
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
            sampler=sampler, 
        )
        
    else:
        # This handles:
        # 1. args.mode == "discrete"
        # 2. args.mode == "continuous" but config.WEIGHTED_SAMPLING == "none"
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
            is_gaussian=args.is_gaussian,
            grid_channels=grid_channels,
            scalar_dim=scalar_dim,
        ).to(device)
    print("Grid channels:", grid_channels)
    print("Grid shape:", grid0.shape)

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # Set up Continuous Loss Functions
    if args.mode == "continuous":
        if args.is_gaussian:
            # Negative Log-Likelihood for Gaussian
            def criterion(pred, target):
                mean, std = pred

                std = torch.clamp(std, config.MIN_STD, config.MAX_STD)

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
            
            if config.USE_WEIGHTED_LOSS:
                ce_speed = nn.CrossEntropyLoss(weight=speed_weights)
                ce_turn = nn.CrossEntropyLoss(weight=turn_weights)
            else:
                ce_speed = nn.CrossEntropyLoss()
                ce_turn = nn.CrossEntropyLoss()
                
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

            # JSON logger
            logger.log_training(epoch, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1_speed": val_metrics["f1_speed"],
                "f1_turn": val_metrics["f1_turn"]
            })

            # TensorBoard logs
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("metrics/f1_speed", val_metrics["f1_speed"], epoch)
            tb_writer.add_scalar("metrics/f1_turn", val_metrics["f1_turn"], epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"speed_f1={val_metrics['f1_speed']:.3f} | "
                f"turn_f1={val_metrics['f1_turn']:.3f} | "
                f"time={time.time()-t0:.1f}s"
            )

        else:
            train_loss = train_epoch_continuous(
                model, train_loader, opt, device
            )

            val_metrics = evaluate_continuous(
                model, val_loader, device, is_gaussian=args.is_gaussian
            )

            val_loss = val_metrics["loss"]

            # JSON logger
            logger.log_training(epoch, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mse": val_metrics["mse"],
                "mae": val_metrics["mae"]
            })

            # TensorBoard logs
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("metrics/mse", val_metrics["mse"], epoch)
            tb_writer.add_scalar("metrics/mae", val_metrics["mae"], epoch)

            if args.is_gaussian:
                tb_writer.add_scalar("metrics/nll", val_metrics["nll"], epoch)
                tb_writer.add_scalar("metrics/avg_std", val_metrics["avg_std"], epoch)

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



        # # save every epoch
        # checkpoint_path = os.path.join(
        #     logger.model_dir,
        #     f"checkpoint_epoch_{epoch}.pt"
        # )

        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": opt.state_dict(),
        #     "val_loss": val_loss,
        #     "mode": args.mode,
        #     "is_gaussian": args.mode == "continuous" and args.is_gaussian,
        #     "scalar_dim": scalar_dim,
        #     "grid_channels": grid_channels
        # }, checkpoint_path)

        # -------- Best model + Early stopping --------
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0

            best_model_path = os.path.join(
                logger.model_dir,
                "best_model.pt"
            )
            
            ckpt= {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss,
                "mode": args.mode,
                "is_gaussian": args.mode == "continuous" and args.is_gaussian,
                "scalar_dim": scalar_dim,
                "grid_channels": grid_channels
                }
            
            if args.mode == "discrete":
                ckpt["n_speed"] = n_speed
                ckpt["n_turn"] = n_turn    
                
                        
            torch.save(ckpt, best_model_path)

            print("[saved best model]")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("\nEarly stopping triggered.")
                break


    tb_writer.close()




if __name__ == "__main__":
    main()




