from config.general_config import *
from pathlib import Path
# =========================================================
# SAC ALGORITHM HYPERPARAMETERS
# ========================================================= 

# Discount factor for future reward estimation (determining the agent's horizon)
GAMMA = 0.995    # ULTIMATE UPGRADE (was 0.99): effective horizon 100 -> 200 steps (5s -> 10s of
                 # sim time), enough to value a full overtake maneuver end-to-end. NOTE: value
                 # scale roughly DOUBLES (V ~ r/(1-gamma)) — feasible Q band is now ~[-400,+300],
                 # so judge critic health against those bounds, not the old +-200.
# Coefficient for soft updates of target Q-networks parameters (exponential moving average factor)
TAU = 0.005         

# =========================================================
# ENTROPY / TEMPERATURE
# =========================================================

# True to automatically tune the policy entropy temperature alpha, False to keep it static
AUTO_ENTROPY = True   # RE-ENABLED (round 3) with three mathematical rails that make the old
                      # 0.1->1.8 exponential ratchet impossible:
                      #   1. REACHABLE target: -5.1 sits INSIDE the achievable entropy band
                      #      (sigma in [0.082, 0.37] => entropy roughly [-7.5, -2]), so the
                      #      alpha gradient CHANGES SIGN across the band -> true regulation.
                      #      The old target (-0.9) was above the whole band -> constant-sign
                      #      gradient -> Adam integrated it into exponential growth.
                      #   2. Hard ceiling ALPHA_MAX (clamped in sac_agent after each update).
                      #   3. Tight LOG_STD band bounds |log pi| itself, capping the entropy
                      #      term's possible contamination of the TD target.
                      # Purpose: regulate the late-run sigma drift (the ~250k churn) instead
                      # of hardcoding it away — keeps SAC's max-entropy learning alive.
# Initial or static entropy temperature value alpha (controls exploration vs exploitation trade-off)
INIT_ALPHA = 0.01         # start low; the controller raises it only if entropy sags below target
# Scaling factor applied to target entropy target (typically matches negative action space dimension)
TARGET_ENTROPY_SCALE = 1.7  # target = -1.7*3 = -5.1 (about sigma~0.12 driving policy)
# Hard ceiling on alpha when AUTO_ENTROPY is on (clamped after each alpha update). Prevents the
# unbounded exponential ratchet even if the target entropy is misconfigured again.
ALPHA_MAX = 0.1   # TIGHTENED from 0.3: the 500k run's healthy operating range was alpha ~0.02-0.06,
                  # and in the last ~5% alpha drifted up with a small Q1 dip / log_std rise. 0.1
                  # gives headroom above the healthy band but caps the late drift on the 300k
                  # extension. Applied live each update, so it's safe on resume.
# =========================================================
# OPTIMIZATION  
# =========================================================
# True to use Huber loss instead of MSE for Q-value updates (protects gradients from huge gradients when crashing)
USE_HUBER_LOSS = True
# Learning rate for the actor network parameter optimizer
ACTOR_LR = 3e-5           # CONSERVATIVE FT: was 1e-4; nudge the already-good (1629) policy slowly
# Learning rate for the critic network parameter optimizer
CRITIC_LR = 1e-4          # STAGE 0: Mani's restricted value (reverted)
# Learning rate for the entropy temperature alpha parameter optimizer
ALPHA_LR = 5e-5           # halved for round-3 auto-entropy: alpha should adapt over ~10k-step
                          # scales, not chase batch noise (was 1e-4)

# L2 regularization weight decay penalty factor applied during optimizer steps
WEIGHT_DECAY = 1e-6          


# Initial weight/coefficient for the Behavior Cloning regularization penalty loss
BC_PENALTY_INIT = 30      # DIVERGENCE FIX: was 200. With USE_Q_NORM=True the RL term is O(1),
                          # so beta=200 made the RL gradient <1% of the total (400k steps of
                          # near-pure BC). 30 = strong anchor early, real learning signal.
# Number of environment steps over which the BC penalty weight decays or changes
BC_PENALTY_STEPS = 500_000   # was 2M: decay over the actual run length, not a horizon we never reach
# Floor the BC penalty never decays below (permanent tether). Protects against the actor
# free-maximizing an imperfect critic late in training — incl. rediscovering the stall exploit.
BC_PENALTY_FLOOR = 5.0
# =========================================================
# REPLAY BUFFER
# =========================================================

# Maximum capacity limit of the transitions replay buffer memory
REPLAY_BUFFER_SIZE = 300_000   # was 500k: halves the rolling-checkpoint footprint (~2x7 GB);
                               # still holds the expert preload + a 300k-step run's history
# Number of transitions sampled per gradient update step
BATCH_SIZE = 512

# Frequency of saving the replay buffer to disk (measured in checkpoint intervals)
SAVE_BUFFER_EVERY = 1        # ROLLING CHECKPOINTS: buffer in EVERY checkpoint. The old value (4)
                             # meant only 100k-mark checkpoints had a buffer — resuming from any
                             # other checkpoint reset the buffer and caused catastrophic
                             # forgetting (the 2026-07-04 incident).
# Number of recent checkpoint directories or files to preserve on disk
KEEP_CHECKPOINTS = 2

# =========================================================
# TRAINING SCHEDULE
# =========================================================

# Maximum total environment simulation steps allowed for the training run
MAX_TRAIN_STEPS = 800_000     # EXTENDED from 500k: the 2026-07-10 run (14/30 survivals, best yet)
                              # was STILL rising in return/length at 500k — resume to 800k to keep
                              # exploiting the un-plateaued curve. Resume is reward-compatible:
                              # nothing in the reward function changed, so the saved buffer is valid.

# Environment steps to collect before starting optimization updates on the actor network
CRITIC_WARMUP_STEPS = 100_000   # Mani's tested value for the 30-vehicle setup: longer critic
                                # warmup works better in traffic (his ablations). Note this is
                                # 1/3 of MAX_TRAIN_STEPS — extend a healthy run with --resume.
                                # (prev: 50_000 divergence-fix run; 20_000 empty-map conservative)
# (orig note) STAGE 0 TESTABILITY: with 100_000 the actor never engages before
                                # CARLA crashes (~17k), so the ablation can't reach the actor phase.
                                # 5_000 lets the actor turn on early so we can actually observe it.
# Minimum transitions that must accumulate in replay buffer before training updates begin
UPDATE_AFTER = 1_000          
# Number of gradient update steps performed per environment interaction step
GRADIENT_UPDATES = 1

# =========================================================
# UPDATE FREQUENCIES
# =========================================================

# Step interval between successive critic updates
CRITIC_UPDATE_EVERY = 1     
# Step interval between successive actor updates
ACTOR_UPDATE_EVERY = 10      
# Step interval between successive entropy temperature parameter updates
ALPHA_UPDATE_EVERY = 10       

# =========================================================
# TARGET NETWORK UPDATE
# =========================================================

# Critic update step interval between soft target network updates
TARGET_UPDATE_INTERVAL = 2    # Update target every 2 CRITIC updates

# =========================================================
# POLICY DISTRIBUTION
# =========================================================

# Lower bound limit constraint for the log standard deviation output of the policy
LOG_STD_MIN = -2.5   # DIVERGENCE FIX: was -5 (sigma floor 0.0067 => log-pi up to ~+14, the fuel
                     # of the alpha/critic spiral). sigma >= 0.082 bounds log-pi so the entropy
                     # term in the TD target stays negligible at alpha=0.01.
# Upper bound limit constraint for the log standard deviation output of the policy
LOG_STD_MAX = -1.0   # ROUND 3 (was 0): sigma <= 0.37. In the 300k run sigma drifted upward
                     # after ~200k, making COLLECTED rollouts noisier — the late "dip" in
                     # training returns tracks that drift (deterministic eval kept improving).
                     # This cap + auto-entropy regulation bound the collection noise. Also
                     # keeps the entropy band around the -5.1 target (see AUTO_ENTROPY note).

# =========================================================
# EXPLORATION
# =========================================================

# True to select purely random actions during initial exploration warmup phase
USE_RANDOM_POLICY_WARMUP = False
# Total environment steps for the initial exploration warmup phase
WARMUP_STEPS = 10_000 
# True to bypass actor warmup phase if critic training has already been initiated
FORCE_SKIP_WARMUP = False  
# =========================================================
# EVALUATION
# =========================================================

# Environment steps interval between evaluation phases
EVAL_INTERVAL = 25_000         # was 10k: fewer, better evals (see EVAL_EPISODES)
# Number of independent episodes to run during each evaluation phase
EVAL_EPISODES = 10             # was 5: n=5 from a bimodal return distribution (std ~850) has
                               # SE ~380 — best_model.pt selection was a winner's-curse lottery
                               # (the "best" model lost to the final checkpoint by ~360 return).
                               # n=10 halves the noise; final selection still needs n>=30 offline.

# =========================================================
# CHECKPOINTING
# =========================================================

# Environment steps interval between consecutive model saving operations
CHECKPOINT_INTERVAL = 5_000   # ROLLING CHECKPOINTS: full state (incl. buffer) every 5k steps,
                              # prune to KEEP_CHECKPOINTS. Max data loss on any crash: 5k steps.
# Absolute directory path where checkpoints, logs, and metadata are stored
SAVE_DIR = Path(r"E:\SAC_Experiments")   # LAB SERVER drive (563 GB free). NOTE: does not exist on
                                         # the local dev machine — train_sac fails fast with a
                                         # clear message if launched where this drive is absent.
                                         # (was: REPO_ROOT / "experiments" / "rl" / "sac")

# Training step interval for logging metrics to tensorboard/console
LOG_EVERY = 1000

# =========================================================
# BC INITIALIZATION
# =========================================================

# True to initialize policy network parameters using a pretrained behavior cloning model
LOAD_BC_WEIGHTS = True
# Path to the pretrained Behavior Cloning model checkpoint file
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"  # base BC (eval 1629 vs DAgger-r1 966 -> base is stronger)
# True to resume training from an existing SAC checkpoint found in SAVE_DIR
RESUME_CHECKPOINT = False    
# True to record video capture files of evaluation rollout episodes
RECORD_SAC_EVAL_VID = True
# True to pre-fill the replay buffer with expert demonstrations before online training starts
PRELOAD_EXPERT_DATA = True  
# Path to the consolidated demonstration file (.npz) containing expert trajectories
COMPILED_DATASET_PATH = REPO_ROOT / "imitation" / "data" / "processed" / "dataset_rl_buffer.npz"
# Path to a specific .pkl state checkpoint file to branch a new experiment from (None for starting fresh)
BRANCH_FROM = None 
# True to normalize actor loss against magnitude of Q-values to improve optimization stability
USE_Q_NORM = True   # DIVERGENCE FIX (was False in Mani's 400k run): TD3+BC-style normalization
                    # makes the actor's RL term O(1) regardless of Q scale, so critic drift can
                    # never blow up the actor gradient, and beta keeps a fixed meaning.
