import sys
import time
import json
import argparse
from pathlib import Path
import keyboard
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import carla
from CarlaEnv.env import CarlaEnv
from agents.bc.imitation_policy import ImitationPolicy
from utils.obs_wrapper import CarlaObsWrapper
from config import bc_config
from imitation.datasets.bc_dataset import BCDatasetContinuous
from utils.seed_utils import seed_everything
import concurrent.futures

seed_everything(bc_config.GLOBAL_SEED)

# =========================================================
# DAGGER HYPERPARAMETERS
# =========================================================
ENSEMBLE_SIZE = 5
WARMUP_EPOCHS = 10
UNCERTAINTY_THRESHOLD = 0.003 # ROUND 2: raised 0.001->0.003 so we only intervene on REAL confusion
                              # (round 1 at 0.001 fired ~every 175 steps, mostly on fine driving -> noise)
INTERVENTION_STEPS = 200      # Exact number of steps human takes over
COOLDOWN_STEPS = 100          # Steps to ignore variance after intervention
SMOOTHING_FRAMES = 15         # ROUND 2: 10->15 for a gentler human->AI handoff (less jerk at release)
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-4

# Directories
DAGGER_DATA_DIR = Path(bc_config.DATA_DIR) / "interventions_raw"
DAGGER_DATA_DIR.mkdir(parents=True, exist_ok=True)


class EnsembleDAgger:
    def __init__(self, env, wrapper, device, grid_channels, scalar_dim):
        self.env = env
        self.wrapper = wrapper
        self.device = device
        
        self.models = []
        self.optimizers = []
        
        self.grid_channels = grid_channels
        self.scalar_dim = scalar_dim
        
        print(f"[INIT] Initializing Ensemble of {ENSEMBLE_SIZE} Actor Models...")
        for i in range(ENSEMBLE_SIZE):
            torch.manual_seed(bc_config.GLOBAL_SEED + i * 100)
            model = ImitationPolicy(
                mode="continuous",
                is_gaussian=False,
                grid_channels=self.grid_channels,
                scalar_dim=self.scalar_dim,
                cnn_channels=bc_config.CNN_CHANNELS,
                kernel_sizes=bc_config.KERNEL_SIZES,
                head_n_mlp_layers=bc_config.HEAD_N_MLP_LAYERS,
                head_mlp_hidden_size=bc_config.HEAD_MLP_HIDDEN_SIZE,
                scalar_n_mlp_layers=bc_config.SCALAR_N_MLP_LAYERS,
                scalar_mlp_hidden_size=bc_config.SCALAR_MLP_HIDDEN_SIZE,
                latent_dim=bc_config.LATENT_DIM,
                decoupled=bc_config.IS_DECOUPLED
            ).to(self.device)
            
            opt = optim.AdamW(model.parameters(), lr=bc_config.BC_LR)
            self.models.append(model)
            self.optimizers.append(opt)
            
        self.criterion = nn.MSELoss()

    # ==============================================================================
    # PHASE 1: WARM-UP (TRUE BAGGING IMPLEMENTED)
    # ==============================================================================
    def warmup(self, dataset):
        print(f"\n--- [PHASE 1] Warming up ensemble concurrently with TRUE BAGGING ({WARMUP_EPOCHS} Epochs) ---")
        
        # Create 5 distinct DataLoaders, each seeing a random 80% of the dataset
        loaders = []
        N = len(dataset)
        subset_size = int(0.8 * N)
        
        for i in range(ENSEMBLE_SIZE):
            # Unique generator for each subset to guarantee different overlapping splits
            g = torch.Generator().manual_seed(bc_config.GLOBAL_SEED + i * 100)
            indices = torch.randperm(N, generator=g)[:subset_size].tolist()
            sampler = torch.utils.data.SubsetRandomSampler(indices, generator=g)
            
            loader = DataLoader(
                dataset, 
                batch_size=bc_config.BC_BATCH_SIZE, 
                sampler=sampler,
                drop_last=True # Ensure identical batch counts across zipped loaders
            )
            loaders.append(loader)
            
        num_batches = len(loaders[0])
        
        for epoch in range(WARMUP_EPOCHS):
            t0 = time.time()
            model_loss_sums = [0.0] * ENSEMBLE_SIZE
            
            # ZIP loaders: Each model gets its own distinct batch of data simultaneously
            for batches in zip(*loaders):
                
                # Helper function for the thread pool
                def train_single_model(i):
                    grid, scalars, targets = batches[i]
                    grid = grid.to(self.device)
                    scalars = scalars.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.models[i].train()
                    pred = self.models[i](grid, scalars)
                    loss = self.criterion(pred, targets)
                    
                    self.optimizers[i].zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizers[i].step()
                    return loss.item()

                # MULTITHREADING: Train all models concurrently on their unique batches
                with concurrent.futures.ThreadPoolExecutor(max_workers=ENSEMBLE_SIZE) as executor:
                    futures = [executor.submit(train_single_model, i) for i in range(ENSEMBLE_SIZE)]
                    results = [f.result() for f in futures]
                
                for i in range(ENSEMBLE_SIZE):
                    model_loss_sums[i] += results[i]
            
            # Format the output for the terminal
            epoch_losses = [s / num_batches if num_batches > 0 else 0 for s in model_loss_sums]
            time_taken = time.time() - t0
            mean_ensemble_loss = sum(epoch_losses) / len(epoch_losses)
            losses_str = " | ".join([f"M{i}: {l:.4f}" for i, l in enumerate(epoch_losses)])
            
            print(f"Warmup Epoch [{epoch+1:02d}/{WARMUP_EPOCHS:02d}] "
                  f"| Time: {time_taken:.1f}s "
                  f"| Mean Loss: {mean_ensemble_loss:.4f} "
                  f"| Models: [{losses_str}]")


    # ==============================================================================
<<<<<<< HEAD
    # PERSISTENCE: save / load the warmed ensemble, and export a single SAC-ready BC
    # ==============================================================================
    def save_ensemble(self, out_dir, norm_stats=None):
        """Persist all ENSEMBLE_SIZE warmed models + meta, so a collection session never
        has to re-warm from scratch (resume with --load-ensemble)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), out_dir / f"ensemble_model_{i}.pt")
        meta = {
            "ensemble_size": ENSEMBLE_SIZE,
            "grid_channels": int(self.grid_channels),
            "scalar_dim": int(self.scalar_dim),
            "is_gaussian": False,
            "normalization_stats": norm_stats or {},
        }
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[SAVE] Ensemble ({ENSEMBLE_SIZE} models) + meta saved -> {out_dir}")

    def load_ensemble(self, in_dir):
        """Reload a previously-saved ensemble (skips warmup)."""
        in_dir = Path(in_dir)
        for i, m in enumerate(self.models):
            sd = torch.load(in_dir / f"ensemble_model_{i}.pt", map_location=self.device)
            m.load_state_dict(sd)
        print(f"[LOAD] Ensemble ({ENSEMBLE_SIZE} models) reloaded <- {in_dir}")

    def export_for_sac(self, out_dir, norm_stats=None, model_index=0):
        """
        Convert ONE ensemble member into the EXACT checkpoint layout SAC expects:
            <out_dir>/models/best_model.pt  -> {"model_state_dict": <ImitationPolicy state_dict>}
            <out_dir>/config.json           -> {"dataset_meta": {"normalization_stats": ...}}
        SAC's load_norm_stats() reads <ckpt>.parents[1]/config.json and load_actor_from_bc()
        reads "model_state_dict", so <out_dir>/models/best_model.pt is a drop-in BC_CHECKPOINT_PATH.

        NOTE: an ensemble member is warmed on an 80% BAGGED subset, so it is NOT the strongest BC.
        For the real handoff, retrain a single model with train_bc on the AGGREGATED dataset (see
        the loop docs). This export exists for a quick SAC smoke-test of the format/handoff.
        """
        out_dir = Path(out_dir)
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.models[model_index].state_dict()}
        torch.save(ckpt, out_dir / "models" / "best_model.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump({"dataset_meta": {"normalization_stats": norm_stats or {}}}, f, indent=2)
        print(f"[EXPORT] SAC-compatible BC checkpoint -> {out_dir / 'models' / 'best_model.pt'}")
        print(f"         (set sac_config.BC_CHECKPOINT_PATH to that path to warm-start SAC)")

    # ==============================================================================
=======
>>>>>>> 1e7f54cccfec0f7d0a8e9470bb2c8210dc36574e
    # PHASE 4 & 5: IN-MEMORY FINE-TUNING & PERSISTENT STORAGE
    # ==============================================================================
    def finetune_and_store(self, buffer):
        if len(buffer) == 0: return
        
        self._dump_to_disk(buffer)
        
        grids, scalars, targets = [], [], []
        
        for step in buffer:
            o = step["obs"]
            target_act = np.array([o["throttle"][0], o["brake"][0], o["steering_angle"][0]], dtype=np.float32)
            
            g, s = self.wrapper.preprocess(o)
            grids.append(g)
            scalars.append(s)
            targets.append(target_act)
            
            self.wrapper.to_tensor(g, s) 
            
        grids_t = torch.tensor(np.stack(grids)).to(self.device)
        scalars_t = torch.tensor(np.stack(scalars)).to(self.device)
        targets_t = torch.tensor(np.stack(targets)).to(self.device)
        
        dataset_size = len(targets_t)
        
        # I COMMENTED THIS PART BECUASE FINETUNING ON THAT LITTLE 200 STEPS CAUSED CATASTROPHIC FORGGETING
        # print(f"\n--- [PHASE 4] In-Memory Concurrent Fine-Tuning on {dataset_size} Corrective Steps ---")
        # for epoch in range(FINETUNE_EPOCHS):
            
        #     def finetune_single_model(i):
        #         self.models[i].train()
        #         # Apply slight shuffle/dropout by randomly masking 10% of samples per model
        #         mask = torch.rand(dataset_size) > 0.1 
                
        #         if mask.sum() == 0: return
                
        #         g_batch = grids_t[mask]
        #         s_batch = scalars_t[mask]
        #         t_batch = targets_t[mask]
                
        #         pred = self.models[i](g_batch, s_batch)
        #         loss = self.criterion(pred, t_batch)
                
        #         self.optimizers[i].zero_grad(set_to_none=True)
        #         loss.backward()
        #         self.optimizers[i].step()

        #     with concurrent.futures.ThreadPoolExecutor(max_workers=ENSEMBLE_SIZE) as executor:
        #         futures = [executor.submit(finetune_single_model, i) for i in range(ENSEMBLE_SIZE)]
        #         concurrent.futures.wait(futures)

        # print("--- Fine-Tuning Complete. Returning control to AI. ---\n")

    # ==============================================================================
    # PHASE 2: INFERENCE & VARIANCE TRACKING
    # ==============================================================================
    def predict_ensemble(self, grid_t, scalars_t):
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(grid_t, scalars_t)
                preds.append(pred.cpu().numpy()[0])
        
        preds = np.array(preds) 
        
        mean_action = np.mean(preds, axis=0)
        
        steer_var = np.var(preds[:, 2])
        throttle_var = np.var(preds[:, 0])
        brake_var = np.var(preds[:, 1])
        
        return mean_action, [steer_var,throttle_var,brake_var]

    # ==============================================================================
    # PHASE 3: THE FREEZE & HUMAN INTERVENTION
    # ==============================================================================
    def get_manual_action(self):
        if keyboard.is_pressed('up'): speed_act = 0
        elif keyboard.is_pressed('down'): speed_act = 1
        elif keyboard.is_pressed('space'): speed_act = 2
        elif keyboard.is_pressed('r'): speed_act = 3
        else: speed_act = 4
            
        if keyboard.is_pressed('left'): turn_act = 1
        elif keyboard.is_pressed('right'): turn_act = 0
        elif keyboard.is_pressed('f'): turn_act = 3
        elif keyboard.is_pressed('t'): turn_act = 2
        else: turn_act = 3
        return np.array([speed_act, turn_act], dtype=np.int64)

    def human_intervention(self, current_obs):
        print(f"\n[!] HIGH UNCERTAINTY DETECTED. Simulation Frozen.")
        print("[!] Press [ENTER] to take manual control for 200 steps.")
        keyboard.wait('enter')
        print(">>> HUMAN IN CONTROL <<<")
        
        intervention_buffer = []
        obs = current_obs
        
        for t in range(INTERVENTION_STEPS):
            discrete_action = self.get_manual_action()
            
            next_obs, reward, term, trunc, info = self.env.step(discrete_action,  "discrete")
            self._update_spectator()
            
            intervention_buffer.append({
                "obs": obs,
                "action": discrete_action, 
                "reward": reward,
                "terminated": term,
                "truncated": trunc,
                "done": term or trunc,
                "t": t
            })
            
            obs = next_obs
            if term or trunc:
                print("[!] Episode terminated during intervention.")
                break
                
            time.sleep(bc_config.MANUAL_SLEEP_SECONDS)
            
        print(">>> HUMAN CONTROL COMPLETE. Freezing to Fine-Tune... <<<")
        return intervention_buffer, obs


    def _dump_to_disk(self, buffer):
        data = {}
        obs_keys = buffer[0]["obs"].keys()
        
        for k in obs_keys:
            data[f"obs_{k}"] = np.stack([s["obs"][k] for s in buffer], axis=0)

        data["actions"] = np.stack([s["action"] for s in buffer], axis=0).astype(np.int64)
        data["rewards"] = np.array([s["reward"] for s in buffer], dtype=np.float32)
        data["terminated"] = np.array([s["terminated"] for s in buffer], dtype=np.bool_)
        data["truncated"] = np.array([s["truncated"] for s in buffer], dtype=np.bool_)
        data["dones"] = np.array([s["done"] for s in buffer], dtype=np.bool_)
        data["t"] = np.array([s["t"] for s in buffer], dtype=np.int32)
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = DAGGER_DATA_DIR / f"intervention_{ts}.npz"
        np.savez_compressed(out_path, **data)
        print(f"[PHASE 5] Saved raw intervention data to -> {out_path}")

    # ==============================================================================
    # MAIN EXECUTION LOOP
    # ==============================================================================
    def run_live_dagger(self, max_steps=20000):
        obs, _ = self.env.reset()
        self.wrapper.reset()
        self._update_spectator()
        
        cooldown_counter = 0
        smoothing_counter = 0
        last_human_action = np.zeros(3)
        
        for t in range(max_steps):
            grid, scalars = self.wrapper.preprocess(obs)
            grid_t, scalars_t = self.wrapper.to_tensor(grid, scalars)
            
            ai_action, variances = self.predict_ensemble(grid_t, scalars_t)
            variance = max(variances)
            
            if t % 20 == 0:
                print(f"[DEBUG t={t:04d}] AI Thr: {ai_action[0]:.3f} | Brk: {ai_action[1]:.3f} | Str: {ai_action[2]:.3f} "
                      f"|| Thr Var: {variances[1]:.5f} | Brk Var: {variances[2]:.5f} | Str Var: {variances[0]:.5f} (Thresh: {UNCERTAINTY_THRESHOLD}) | Cooldown: {cooldown_counter}")
            
            if smoothing_counter > 0:
                alpha = 1.0 - (smoothing_counter / SMOOTHING_FRAMES)
                executed_action = (alpha * ai_action) + ((1.0 - alpha) * last_human_action)
                smoothing_counter -= 1
            else:
                executed_action = ai_action
                
            if cooldown_counter > 0:
                cooldown_counter -= 1
            elif variance > UNCERTAINTY_THRESHOLD:
                print(f"\n>>> VARIANCE SPIKE DETECTED: {variance:.6f} > {UNCERTAINTY_THRESHOLD} <<<")
                
                buffer, obs = self.human_intervention(obs)
                self.finetune_and_store(buffer)
                
                self.wrapper.reset() 
                cooldown_counter = COOLDOWN_STEPS
                smoothing_counter = SMOOTHING_FRAMES
                
                last_human_action = np.array([
                    obs["throttle"][0], 
                    obs["brake"][0], 
                    obs["steering_angle"][0]
                ])
                continue 
                
            obs, reward, term, trunc, info = self.env.step(executed_action.tolist())
            self._update_spectator() 
            
            if term or trunc:
                print(f"[t={t}] Episode ended (Terminated: {term}, Truncated: {trunc}). Resetting environment...")
                obs, _ = self.env.reset()
                self.wrapper.reset()
                self._update_spectator()
                cooldown_counter = 0

    def _update_spectator(self):
        if self.env.ego_vehicle is None:
            return
            
        world = self.env.world
        ego_vehicle = self.env.ego_vehicle
        spectator = world.get_spectator()
        tr = ego_vehicle.get_transform()
        forward = tr.get_forward_vector()

        cam_loc = tr.location - forward * 8.0 + carla.Location(z=3.0)
        cam_rot = carla.Rotation(pitch=-12.0, yaw=tr.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=bc_config.CARLA_MAP_PATH)
    parser.add_argument("--device", default=bc_config.DEVICE)
<<<<<<< HEAD
    parser.add_argument("--load-ensemble", type=str, default=None,
                        help="Dir of a previously-saved ensemble to resume (skips the warmup phase).")
=======
>>>>>>> 1e7f54cccfec0f7d0a8e9470bb2c8210dc36574e
    args = parser.parse_args()

    env = CarlaEnv(
        map_path=args.map,
        walkers_count=bc_config.CARLA_WALKERS,
        vehicles_count=bc_config.CARLA_VEHICLES,
        max_steps=bc_config.CARLA_MAX_STEPS,
        init_speed=bc_config.CARLA_INIT_SPEED,
<<<<<<< HEAD
        action_mode="continuous",
        random_ego_spawn=bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=bc_config.RANDOM_VEHICLE_START_POS,
        no_rendering=False,   # DAgger REQUIRES rendering: the human watches the spectator cam to drive.
                              # (Our SAC default is no_rendering=True for headless speed.) Launch CARLA
                              # WINDOWED for this run -- NOT with -RenderOffScreen.
=======
        action_mode="continuous", 
        random_ego_spawn=bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=bc_config.RANDOM_VEHICLE_START_POS
>>>>>>> 1e7f54cccfec0f7d0a8e9470bb2c8210dc36574e
    )

    print("[INIT] Loading metadata to extract dimensions and norm_stats...")
    meta_path = Path(bc_config.CONTINUOUS_DATASET_PATH).with_suffix(".meta.json")
    
    with open(meta_path, "r") as f:
        dataset_meta = json.load(f)
        
    norm_stats = dataset_meta.get("normalization_stats", {})
    if not norm_stats:
        print("[WARN] No norm_stats found in meta.json! Models may act blindly.")
        
    wrapper = CarlaObsWrapper(norm_stats=norm_stats, device=args.device, action_mode="continuous")
    
    print("[INIT] Loading offline dataset to extract dimensions...")
    dataset = BCDatasetContinuous(bc_config.CONTINUOUS_DATASET_PATH, one_hot_presence=bc_config.USE_ONE_HOT_GRID)
    grid0, scalars0, _ = dataset[0]
    grid_channels = grid0.shape[0]
    scalar_dim = scalars0.numel()
    
    dagger_system = EnsembleDAgger(env, wrapper, args.device, grid_channels, scalar_dim)
<<<<<<< HEAD

    # Persist the warmed ensemble (so re-runs don't re-warm) and drop a SAC-testable BC export.
    dagger_exp_dir = Path(bc_config.REPO_ROOT) / "experiments" / "dagger" / time.strftime("%Y%m%d_%H%M%S")
    if args.load_ensemble:
        dagger_system.load_ensemble(args.load_ensemble)
    else:
        dagger_system.warmup(dataset)
        dagger_system.save_ensemble(dagger_exp_dir, norm_stats=norm_stats)
        dagger_system.export_for_sac(dagger_exp_dir, norm_stats=norm_stats)

=======
    dagger_system.warmup(dataset)
    
>>>>>>> 1e7f54cccfec0f7d0a8e9470bb2c8210dc36574e
    try:
        dagger_system.run_live_dagger()
    finally:
        env.close()





#  TODO: use std instead of varaince for better understanding, add logging specially saving the 5 models(maybe add a flag to know it should lload the 5 models or train right now and then load and save) ,  add replay buffer , tune the var , add draw line function from the manual controller file so we know where we are suppsoed to go






 