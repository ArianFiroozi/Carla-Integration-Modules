# Working Experiments Directory (`experiments/`)

This directory holds the **active/current experiments** across all training pipelines. Every experiment directory is suffixed with `_Mani` or `_Mehdi` (and tagged with `"creator"` inside `config.json`) via [`utils/experiment_utils.py`](../utils/experiment_utils.py).

Historical and archived runs live in [`../experiments_archive/`](../experiments_archive/).

## Directory Structure
- `bc/` — Behavior Cloning active models
- `dagger/` — Interactive / Ensemble DAgger active sessions
- `offline_rl/awac/` — Advantage Weighted Actor-Critic (AWAC) runs
- `offline_rl/iql/` — Implicit Q-Learning (IQL) runs
- `rl/sac/` — Soft Actor-Critic (SAC) online RL runs

## Active Experiment Runs

| Category | Experiment Run | Creator | Notes |
| :--- | :--- | :--- | :--- |
| `bc` | [`2026_05_03_21_45_02_bc_continuous_Mani`](bc/2026_05_03_21_45_02_bc_continuous_Mani/) | **Mani** | Active run |
| `bc` | [`2026_06_28_03_20_04_bc_continuous_Mehdi`](bc/2026_06_28_03_20_04_bc_continuous_Mehdi/) | **Mehdi** | Active run |
| `dagger` | [`20260628_024006_Mehdi`](dagger/20260628_024006_Mehdi/) | **Mehdi** | Active run |
| `dagger` | [`20260701_094519_Mani`](dagger/20260701_094519_Mani/) | **Mani** | Active run |
| `rl/sac` | [`2026-06-29_13-58-24_Mani`](rl/sac/2026-06-29_13-58-24_Mani/) | **Mani** | Active run |
| `rl/sac` | [`2026-07-05_02-54-05_Mehdi`](rl/sac/2026-07-05_02-54-05_Mehdi/) | **Mehdi** | Evaluated by Mani |
| `rl/sac` | [`2026-07-07_02-54-05_Mehdi`](rl/sac/2026-07-07_02-54-05_Mehdi/) | **Mehdi** | Evaluated by Mani |
| `rl/sac` | [`2026-07-15_02-54-05_Mehdi`](rl/sac/2026-07-15_02-54-05_Mehdi/) | **Mehdi** | Evaluated by Mani |
| `rl/sac` | [`2026-07-17_12-07-03_Mani`](rl/sac/2026-07-17_12-07-03_Mani/) | **Mani** | Active run |
| `rl/sac` | [`2026-11-05_08-39-08_Mani`](rl/sac/2026-11-05_08-39-08_Mani/) | **Mani** | Active run |

## current state:

the car can drive quite a while before crashing into other cars can sometimes even finish the 2000 step loop. it can overtake but fails in many conditions. for example, it can not turn at a corner and overtake shortly afterwards.