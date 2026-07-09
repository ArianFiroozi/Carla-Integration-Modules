# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Autonomous-driving research on CARLA (0.9.15): a behavior-cloning (BC) → DAgger → SAC / offline-RL (IQL, AWAC) pipeline. The BC policy is the foundation; SAC and the offline-RL algorithms are initialized from a BC checkpoint and fine-tune it. There are no unit tests or linters — verification is done by running training/evaluation scripts against a live CARLA server.

## Prerequisites

- A CARLA server must be running on port 2000 before any env-touching script:
  `CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000`
  Training runs headless (`no_rendering=True` inside the env); DAgger/manual recording needs a **windowed** CARLA (no `-RenderOffScreen`) and an **Administrator terminal** (the `keyboard` library requires it).
- After force-killing a training run, restart `CarlaUE4.exe` — the sync-mode server wedges otherwise.
- If ports are unavailable (wifi adapter issue), run `CarlaEnv/loopback/run_loopback.py`.
- Custom maps are `.xodr` (OpenDRIVE), created in SUMO netedit and converted with `netconvert --sumo-net-file map.net.xml --opendrive-output map.xodr`. Default map is `map1.xodr` via `CARLA_MAP_PATH` in `config/general_config.py`.

## Commands

All scripts run as modules from the repo root. Hyperparameter defaults come from `config/*.py`; CLI flags override them.

```bash
# Imitation pipeline (in order)
python -m imitation.record_demos                  # record manual or autopilot demos (--mode manual|autopilot)
python -m imitation.build_dataset                 # compile demos -> dataset_bc_continuous.npz + .meta.json (norm stats)
python -m imitation.train_bc                      # train BC -> experiments/bc/<ts>/models/best_model.pt
python -m imitation.evaluate_imitation --model_path experiments/bc/<ts>/models/best_model.pt

# DAgger loop (full procedure in imitation/DAGGER_LOOP.md)
python -m imitation.train_interactive_dagger      # collect human corrections at uncertainty spikes
python -m imitation.build_dataset --extra-dirs imitation/data/interventions_raw --upweight 3
# then re-run train_bc

# Online RL
python -m rl.sac.train_sac                        # --resume / --resume-dir <dir> / --branch-from <state.pkl>
python -m rl.sac.evaluate_sac --exp_id <ts>       # --watch renders CARLA, --record saves video

# Offline RL
python -m offline_rl.iql.train_iql
python -m offline_rl.awac.train_awac
python -m offline_rl.iql.evaluate_iql --model_path <best_model.pt>

# Plot an experiment's training curves (defaults to latest run)
python plot_run.py [experiments/rl/sac/<ts>]
```

TensorBoard logs go to `experiments/.../<ts>/tb/`.

## Architecture

**Config layering** — `config/general_config.py` holds everything shared (seeds, device, network architecture sizes, obs/action space definitions, CARLA env defaults, and the "RL REWARD EXPERIMENTATION HUB" with all reward weights). `bc_config.py`, `sac_config.py`, `offline_rl_config.py`, and `awac_config.py` star-import it and add algorithm-specific hyperparameters. Configs are plain module-level UPPERCASE constants; each training run snapshots its config module to `config.json` in the experiment dir.

**Environment** — `CarlaEnv/env.py` (`CarlaEnv`) is a Gymnasium env wrapping a CARLA client (localhost:2000, synchronous mode, dt=0.05). Observations are a Dict of 25×11 bird's-eye grids (`presence`, `speed_x`, `speed_y`) plus scalars (lane angle, ego speed, current controls, ...), built from actor states and map queries — no cameras, only collision and lane-invasion sensors, which is why headless rendering works. Actions are either discrete (`MultiDiscrete [5,4]` speed/turn classes) or continuous (`[throttle, brake, steer]`); everything current uses continuous. Submodules: `LoadOpenDrive2` (loads `.xodr` maps), `ObjectSpawn` (ego/NPC vehicles, pedestrians), `ObservationAdaptors` (grid + lane-angle construction), `VehicleControl`. The env writes `extra_stats/runs/training.pid` and `heartbeat.txt` for external watchdogs.

**Observation wrapper** — `utils/obs_wrapper.py` (`CarlaObsWrapper`) converts the Dict obs into normalized grid+scalar tensors. Normalization stats (z-score by default) are computed by `build_dataset` and stored in the dataset `.meta.json`, then carried in the BC experiment's `config.json` as `dataset_meta.normalization_stats`. **Every downstream consumer (SAC, IQL, AWAC, evaluation) loads norm stats from the BC checkpoint's experiment folder** — a policy is only valid with the stats it was trained under.

**Shared networks** — `networks/` provides `FeatureExtractor` (CNN over grid channels + scalar MLP, fused to a latent vector), `actor_heads` (Gaussian/deterministic), and `critic_heads`. BC, SAC, IQL, and AWAC agents in `agents/` and `offline_rl/` all build on these, which is what makes BC→RL weight transfer work.

**Reward** — `utils/reward_compiler.py::compile_reward` is the single reward definition, usable with numpy or torch. `mode="info"` computes reward from live env info (online SAC); `mode="obs"` approximates it from recorded observations, used to (re)label demo datasets with rewards for offline RL and replay-buffer preloading. Reward weights live in `general_config.py`, so editing them there changes both online reward and offline relabeling.

**BC → SAC handoff** — `sac_config.py`: `LOAD_BC_WEIGHTS` initializes the actor from `BC_CHECKPOINT_PATH`; `PRELOAD_EXPERT_DATA` seeds the replay buffer from the compiled demo dataset; a decaying BC penalty (`BC_PENALTY_INIT/STEPS`) keeps the actor near the BC policy early on; `CRITIC_WARMUP_STEPS` trains the critic before the actor moves. After retraining BC, update `BC_CHECKPOINT_PATH` in **both** `sac_config.py` and `offline_rl_config.py`.

**Experiments layout** — each run creates `experiments/{bc,rl/sac,dagger,...}/<timestamp>/` containing `models/` (checkpoints, `best_model.pt`), `tb/`, and `config.json`. SAC checkpointing saves full state (`checkpoint_state_*.pkl` incl. replay buffer) so runs can `--resume` or `--branch-from` a prior state. `archive/` and `experiments_archive/` are dead code / old runs — don't build on them.

## Gotchas

- `bc_config.IS_GAUSSIAN = False` — the Gaussian BC head has a known bug (see comment in `config/bc_config.py`); don't enable it without fixing training + `agents/bc/imitation_policy.py`.
- The DAgger auto-export (`experiments/dagger/<ts>/models/best_model.pt`) is a plumbing smoke-test, not an improved policy — the real artifact comes from re-running `build_dataset` + `train_bc` (see `imitation/DAGGER_LOOP.md`).
- Dataset filtering/balancing (idle-frame filtering, undersampling, mirroring, crash-tail dropping) is all configured in `bc_config.py` and applied in `build_dataset` — dataset composition is a hyperparameter here.
