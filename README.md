# Carla-Integration-Modules

Autonomous driving research framework built on **CARLA 0.9.15** and **PyTorch**, implementing an end-to-end **Behavior Cloning (BC) -> Interactive Ensemble DAgger -> BC-Regularized Soft Actor-Critic (SAC)** pipeline, alongside experimental offline RL (`AWAC`, `IQL`) modules.

For the full multi-phase development history, architecture analysis, per-run benchmark tables, and debugging forensics, see [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

---

## Prerequisites & Setup

### Runtime & Dependencies

- **Simulator**: CARLA `0.9.15` server running before launching environment, DAgger, training, or evaluation scripts.
- **Default Scenario** ([config/general_config.py](config/general_config.py)): OpenDRIVE map `CarlaEnv/LoadOpenDrive2/map1.xodr`, `CARLA_VEHICLES=30`, `CARLA_WALKERS=0`, `CARLA_MAX_STEPS=2000`, `TARGET_SPEED_MS=6.0`.
- **Python Packages** ([requirements.txt](requirements.txt)):

```text
carla==0.9.15
gymnasium==1.0.0
keyboard==0.13.5
matplotlib==3.7.5
numpy==1.24.4
opencv_python==4.11.0.86
scikit_learn==1.3.2
torch==2.4.1+cu124
tqdm==4.67.3
```

Install Python dependencies into your virtual environment with `pip install -r requirements.txt`.

### Cross-Platform & Storage Notes (Windows / Linux)

- **Batch scripts**: [scripts/rebuild_buffer.bat](scripts/rebuild_buffer.bat) and [scripts/start_sac_forensics_run.bat](scripts/start_sac_forensics_run.bat) are convenience wrappers for Windows lab setups. On Linux, invoke the equivalent `python -m ...` module commands listed below directly.
- **Experiment output directory**: At `HEAD` (`6bcc795`), [config/sac_config.py](config/sac_config.py) sets `SAVE_DIR = REPO_ROOT / 'experiments' / 'rl' / 'sac'` (replacing earlier hardcoded `E:\SAC_Experiments` paths on Windows).
- **Gitignored large artifacts**: [.gitignore](.gitignore) excludes `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.tfevents.*`, `imitation/data/`, `*.npz`, `videos/`, `*.mp4`, `checkpoints/`, and `extra_stats/`. Tracked experiment folders therefore preserve `config.json`, evaluation JSON summaries, and selected plots (`!experiments/**/*.png`, `!experiments_archive/**/*.png`), while large weights, replay buffers, and event files remain local.

---

## Repository Architecture

| Path | Responsibility |
| --- | --- |
| [CarlaEnv/](CarlaEnv/) | `CarlaEnv` Gymnasium environment (`env.py`), OpenDRIVE loader (`LoadOpenDrive2/LoadOpenDriveMap.py`), vehicle PID/control (`VehicleControl/VehicleController.py`), actor spawners (`ObjectSpawn/`), and observation adaptors (`ObservationAdaptors/`). |
| [utils/](utils/) | `CarlaObsWrapper` & `ObsHistory` (`obs_wrapper.py`) producing an `8`-dim scalar state and `5`-channel grid (`WINDOW_SIZE=1`, `USE_ONE_HOT_GRID=True`, `USE_SPATIAL_FEATURES=True`), unified reward function `compile_reward` (`reward_compiler.py`), `SACReplayBuffer` (`replay_buffer.py`), seeding (`seed_utils.py`), and experiment logging (`experiment_utils.py`, `copy_tensorboard_logs.py`). |
| [networks/](networks/) | Shared `FeatureExtractor` (`CNN_CHANNELS=[16, 32, 64]`, `KERNEL_SIZES=[3, 3, 3]`, `LATENT_DIM=128`), continuous/discrete actor heads (`actor_heads.py`), and `QCriticHead` / `TwinQCriticHead` (`critic_heads.py`). |
| [agents/](agents/) | Policy and RL agents: `ImitationPolicy` (`agents/bc/imitation_policy.py`), `SACAgent` / `SACActor` / `SACCritic` (`agents/sac/sac_agent.py`), and `AWACAgent` (`agents/awac/awac_agent.py`). |
| [config/](config/) | Central configuration modules: `general_config.py`, `bc_config.py`, `sac_config.py`, `awac_config.py`, `offline_rl_config.py`. |
| [imitation/](imitation/) | Demonstration recording, dataset inspection/compilation, BC training/evaluation, and interactive Ensemble DAgger (`train_interactive_dagger.py`, [imitation/DAGGER_LOOP.md](imitation/DAGGER_LOOP.md)). |
| [rl/](rl/) | BC-regularized SAC preflight, training, evaluation (`rl/sac/`), and online reward inspection (`rl/reward_verifier.py`). |
| [offline_rl/](offline_rl/) | Experimental AWAC and IQL training/evaluation scripts and synthetic demo generator. |
| [scripts/](scripts/) | Plotting utility ([scripts/plot_run.py](scripts/plot_run.py)) and Windows workflow `.bat` launchers. |
| [experiments/](experiments/) & [experiments_archive/](experiments_archive/) | Active and archived run configs, evaluation summaries, TensorBoard plots, and experiment notes ([experiments/README.md](experiments/README.md), [experiments_archive/README.md](experiments_archive/README.md)). |
| [docs/](docs/) | Full historical and quantitative engineering report ([docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)). |
| [archive/](archive/) | Early prototype CARLA/SB3 wrappers, loopback adaptor, and legacy helper scripts (retained for historical reference only). |

---

## Primary Workflows (`BC -> Optional DAgger -> SAC`)

Run all Python commands from the repository root with CARLA `0.9.15` running when environment interaction is required.

### 1. Demonstration Collection & Inspection

Record manual or autopilot demonstrations (`map1_30car` is configured in [config/bc_config.py](config/bc_config.py)) and verify demonstration contents before compilation:

```bash
python -m imitation.record_demos --mode manual --action-mode continuous --episodes 10
python -m imitation.inspect_demo --mode manual --visualize
```

Key flags for [imitation/record_demos.py](imitation/record_demos.py): `--map`, `--walkers`, `--vehicles`, `--max-steps`, `--init-speed`, `--mode`, `--action-mode`, `--demo-dir`, `--record`, `--base-name`, `--episodes`.  
Key flags for [imitation/inspect_demo.py](imitation/inspect_demo.py): `--mode`, `--visualize`, `--max-feature-samples`.

### 2. Compile Datasets (BC & Dual-Source SAC Buffer)

Compile continuous BC (`dataset_bc_continuous.npz`) and RL expert replay (`dataset_rl_buffer.npz`) archives:

```bash
python -m imitation.build_dataset --action-mode continuous
```

On Windows, [scripts/rebuild_buffer.bat](scripts/rebuild_buffer.bat) wraps dataset rebuilding.

> [!WARNING]
> **Known bug (`--out-path`)**: Do not pass `--out-path` to `python -m imitation.build_dataset` ([imitation/build_dataset.py#L862](imitation/build_dataset.py#L862)); omit the flag so outputs write to the default paths configured in [config/bc_config.py](config/bc_config.py).  
> Other supported flags: `--visualize`, `--action-mode`, `--extra-dirs`, `--upweight`.

### 3. Behavior Cloning (BC)

Train and evaluate the deterministic continuous BC policy (`BC_EPOCHS=30`, `BC_BATCH_SIZE=512`, `BC_LR=0.0003`, `ACTION_MODE='continuous'`, `SCALING_METHOD='z_score'`):

```bash
python -m imitation.train_bc --mode continuous
python -m imitation.evaluate_imitation --episodes 20 --max-steps 2000
```

- Key `train_bc` flags: `--mode`, `--epochs`, `--batch`, `--lr`, `--is_gaussian`, `--val_split`, `--patience`, `--device`.
- Key `evaluate_imitation` flags: `--map`, `--episodes`, `--max-steps`, `--mode`, `--device`, `--exp_id`, `--model_path`, `--experiments_root`, `--watch`.
- [config/sac_config.py](config/sac_config.py) initializes SAC actor weights (`LOAD_BC_WEIGHTS=True`) from `experiments/bc/2026_05_03_21_45_02_bc_continuous_Mani/models/best_model.pt` by default (ensure local `.pt` weights exist or point `--model_path` / config accordingly).

> [!WARNING]
> **Known bug (`--mode discrete`)**: Discrete BC training in [imitation/train_bc.py#L686-L693](imitation/train_bc.py#L686-L693) is currently broken. Use `--mode continuous`.

### 4. Interactive Ensemble DAgger (Optional Corrective Data)

Collect targeted human-in-the-loop recovery/corrective trajectories using a 5-model ensemble (`EnsembleDAgger`, configured via `ENSEMBLE_PATH` in [config/bc_config.py](config/bc_config.py)):

```bash
python -m imitation.train_interactive_dagger
```

Flags: `--map`, `--device`, `--load-ensemble`. After collecting DAgger corrections, re-run Step 2 (`python -m imitation.build_dataset`) with `--extra-dirs` / `--upweight` as needed so the compiled expert replay buffer includes the corrective episodes. See [imitation/DAGGER_LOOP.md](imitation/DAGGER_LOOP.md) for the interactive loop protocol.

### 5. BC-Regularized SAC Training & Evaluation

Verify environment, config synchronization, and local buffer/checkpoint prerequisites before starting a multi-hour SAC run:

```bash
python -m rl.sac.preflight
```

(`preflight.py` flags: `--wait-carla`, `--skip-buffer-check`, `--ignore-sync`, `--allow-cpu`.)

Launch SAC training (`CRITIC_WARMUP_STEPS=100000`, `MAX_TRAIN_STEPS=1000000`, `BC_PENALTY_INIT=30 -> BC_PENALTY_FLOOR=5.0`, `USE_Q_NORM=True`, `GAMMA=0.995`):

```bash
python -m rl.sac.train_sac --seed 42
```

On Windows, [scripts/start_sac_forensics_run.bat](scripts/start_sac_forensics_run.bat) runs preflight checks and starts SAC training with a seed argument (`scripts\start_sac_forensics_run.bat [SEED]`).  
`train_sac` flags: `--map`, `--vehicles`, `--max-steps`, `--device`, `--seed`, `--resume`, `--resume-dir`, `--branch-from`.

Evaluate a trained SAC policy (single-model benchmark or paired statistical comparison against another checkpoint):

```bash
python -m rl.sac.evaluate_sac --episodes 30 --max-steps 2000 --exp_id <RUN_DIR_NAME>
python -m rl.sac.evaluate_sac --episodes 30 --model_path <PATH_A> --compare_with <PATH_B>
```

`evaluate_sac` flags: `--map`, `--episodes`, `--max-steps`, `--device`, `--exp_id`, `--model_path`, `--experiments_root`, `--seed`, `--record`, `--no-record`, `--no-spectator`, `--watch`, `--compare_with`.

### 6. Reward Verification & Run Plotting

- Inspect live step-by-step reward term magnitudes in CARLA:

```bash
python -m rl.reward_verifier --total-steps 2000
```

(`--map`, `--total-steps`, `--max-steps`, `--device`, `--exp_id`, `--model_path`, `--experiments_root`.)

- Generate summary plots from TensorBoard event logs for a run directory:

```bash
python scripts/plot_run.py <run_dir>
```

(`plot_run.py [run_dir] [--out PATH]`.)

---

## Module Status Matrix

| Module / Component | Status | Notes |
| --- | --- | --- |
| Continuous BC ([imitation/train_bc.py](imitation/train_bc.py), [imitation/evaluate_imitation.py](imitation/evaluate_imitation.py)) | **Active** | Primary imitation baseline (`continuous` mode). |
| Discrete BC ([imitation/train_bc.py#L686-L693](imitation/train_bc.py#L686-L693)) | **Broken** | `--mode discrete` fails due to a training-loop bug; do not use. |
| Dataset Builder ([imitation/build_dataset.py](imitation/build_dataset.py)) | **Active (Caveat)** | `--out-path` has a bug at `L862`; omit `--out-path` to use `bc_config.py` defaults. |
| Interactive Ensemble DAgger ([imitation/train_interactive_dagger.py](imitation/train_interactive_dagger.py)) | **Active** | 5-model ensemble human-in-the-loop intervention workflow. |
| BC-Regularized SAC ([rl/sac/](rl/sac/)) | **Active** | Primary online RL pipeline with critic warmup, Q-norm BC anchor, and `preflight.py`. |
| Reward Verifier ([rl/reward_verifier.py](rl/reward_verifier.py)) & Plotting ([scripts/plot_run.py](scripts/plot_run.py)) | **Active** | Diagnostic utilities for reward balancing and TensorBoard visualization. |
| Traffic Sign Adaptor ([CarlaEnv/ObservationAdaptors/TrafficSigns.py](CarlaEnv/ObservationAdaptors/TrafficSigns.py)) | **Stub** | `get_nearby_signs` currently returns zeros. |
| Offline RL — AWAC ([offline_rl/awac/train_awac.py](offline_rl/awac/train_awac.py)) | **Experimental / Superseded** | Four June 2026 runs degraded during fine-tuning; superseded by SAC. |
| Offline RL — IQL ([offline_rl/iql/](offline_rl/iql/)) | **Experimental / Incomplete** | [config/offline_rl_config.py](config/offline_rl_config.py) notes `offline_noisy_data` is absent (`offline_rl/data` empty after lab sync); no trained models saved. |
| Early Prototypes ([archive/](archive/)) | **Archived** | Early SB3 PPO/SAC wrappers, loopback adapter, and legacy watchdog scripts. |
