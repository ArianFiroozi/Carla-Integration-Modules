# Interactive DAgger loop

Goal: strengthen the BC foundation so it can **sustain** driving on `map1` (the bottleneck the SAC
investigation isolated — the agent *can* drive but crashes mid-route and retreats). DAgger fixes the
covariate-shift weakness of plain BC by collecting expert corrections **on the states the policy
actually fails at**, then retraining.

The loop reuses the existing `build_dataset` → `train_bc` pipeline, so the final policy is a single,
SAC-ready BC checkpoint (with `config.json` normalization stats). The 5-model ensemble exists only to
estimate uncertainty (when to ask the human to take over).

---

## Prerequisites (every collection session)
- **Launch CARLA WINDOWED** — the human must SEE the spectator cam. Do **not** use `-RenderOffScreen`:
  ```
  CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000
  ```
  (The DAgger script forces `no_rendering=False`; SAC's default headless mode is the opposite.)
- **Run the terminal as Administrator** on Windows — the `keyboard` library needs it to capture keys.
- Map is `map1.xodr` by default (via `CARLA_MAP_PATH`); `CARLA_VEHICLES=0` (no traffic to complicate
  the human's corrections).

---

## The loop

### 1. Collect corrections
```bash
python -m imitation.train_interactive_dagger
```
- Warms the 5-model ensemble on the current BC dataset, then **saves it** to
  `experiments/dagger/<timestamp>/` and drops a SAC-testable BC export there.
- Drives `map1`. On an uncertainty (inter-model variance) spike it **freezes**:
  press **ENTER**, then drive ~200 steps with the keyboard. Controls:
  - throttle/brake: `up` / `down` / `space` (stop) / `r` (reverse)
  - steer: `left` / `right` / `f` (straight) / `t`
- Each corrective segment is saved to `imitation/data/interventions_raw/intervention_<ts>.npz`.
- Resume a later session without re-warming: `--load-ensemble experiments/dagger/<timestamp>`

### 2. Aggregate the interventions into the dataset
```bash
python -m imitation.build_dataset --extra-dirs imitation/data/interventions_raw --upweight 3
```
- Rebuilds `dataset_bc_continuous.npz` (+ `.meta.json` with fresh norm stats) from the original demos
  **plus** the interventions. `build_dataset` derives continuous targets from `obs_throttle/brake/
  steering_angle`, so the intervention `.npz` files drop in with no conversion.
- `--upweight N` replicates each intervention episode N times so the model pays extra attention to the
  corrected failure states (start with 2–4; the interventions are a small fraction of the data).

### 3. Retrain the single, SAC-ready BC  ← the real handoff artifact
```bash
python -m imitation.train_bc
```
- Trains ONE model on the aggregated dataset → `experiments/bc/<timestamp>/models/best_model.pt`
  plus `config.json` carrying `dataset_meta.normalization_stats`. This is the gold-standard handoff.

### 4. Hand off to SAC
- Point `sac_config.BC_CHECKPOINT_PATH` (and `offline_rl_config.BC_CHECKPOINT_PATH`) at the new
  `experiments/bc/<timestamp>/models/best_model.pt`, then run SAC as usual.

### 5. Measure & iterate
```bash
python -m imitation.evaluate_imitation --model_path experiments/bc/<timestamp>/models/best_model.pt
```
- The average episode length is the BC **survival baseline**. Each loop should raise it.
- Re-run step 1 (it now warms on the aggregated data → smarter uncertainty), collect corrections on
  the **new** failure modes, and repeat 2–4 until BC sustains a full route.

---

## Two ways to get the single SAC BC
- **🥇 `train_bc` on the aggregated dataset (step 3) — use this for the real handoff.** Full training
  on all data; strongest model; exact SAC checkpoint+config format.
- **🥈 `export_for_sac`** (auto-run after the ensemble warmup in step 1) — exports ensemble member #0
  in the same SAC layout. It's an 80%-bagged-subset model, so weaker than `train_bc`, and after the
  first round it's ≈ the original BC. Use it only to smoke-test the SAC handoff *plumbing*.

## Gotchas
- After a force-kill, **restart `CarlaUE4.exe`** before the next run (sync-mode server wedges otherwise).
- The auto-exported `experiments/dagger/<ts>/models/best_model.pt` is a format smoke-test, **not** the
  improved policy — the improvement comes from steps 2+3.
- If you retrain BC, update **both** `sac_config.py` and `offline_rl_config.py` `BC_CHECKPOINT_PATH`
  (they share normalization stats with the BC checkpoint).
