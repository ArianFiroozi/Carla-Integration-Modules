from config.general_config import *

# =========================================================
# SAC ALGORITHM HYPERPARAMETERS
# ========================================================= 

GAMMA = 0.99               # discount factor 
TAU = 0.005                # target network soft update 

# =========================================================
# ENTROPY / TEMPERATURE
# =========================================================

AUTO_ENTROPY = True   
INIT_ALPHA = 0.005           # STAGE 0: Mani's restricted value (reverted)
TARGET_ENTROPY_SCALE = 1.0   # STAGE 1a: standard target_entropy = -ACTION_DIM = -3 (was 0.2 -> -0.6)

# =========================================================
# OPTIMIZATION  
# =========================================================
USE_HUBER_LOSS = True
ACTOR_LR = 1e-4           # STAGE 1a: was 1e-6 (froze the policy & log_std head). Coupled with LOG_STD_MAX
CRITIC_LR = 1e-4          # STAGE 0: Mani's restricted value (reverted)
ALPHA_LR = 1e-4           # STAGE 0: Mani's restricted value (reverted)

WEIGHT_DECAY = 0           


BC_PENALTY_INIT = 50      # STAGE 0: Mani's value (reverted) — now anchors to the REAL BC policy (bug #5 fixed)
BC_PENALTY_STEPS = 300_000  # STAGE 0: Mani's restricted value (reverted)
# =========================================================
# REPLAY BUFFER
# =========================================================

REPLAY_BUFFER_SIZE = 100_000   # was 500_000; 500k pre-allocates ~5.5GB RAM and is never filled
                               # in Stage 0. 100k (~1.1GB) relieves memory pressure that competes
                               # with CARLA. Safe to raise again once stable.
BATCH_SIZE = 256

SAVE_BUFFER_EVERY = 5         # buffer pickles are large; keep them infrequent (model saves are cheap)
KEEP_CHECKPOINTS = 3

# =========================================================
# TRAINING SCHEDULE
# =========================================================

MAX_TRAIN_STEPS = 500_000     

CRITIC_WARMUP_STEPS = 5_000     # STAGE 0 TESTABILITY: with 100_000 the actor never engages before
                                # CARLA crashes (~17k), so the ablation can't reach the actor phase.
                                # 5_000 lets the actor turn on early so we can actually observe it.
UPDATE_AFTER = 1_000          
GRADIENT_UPDATES = 1

# =========================================================
# UPDATE FREQUENCIES
# =========================================================

CRITIC_UPDATE_EVERY = 1     
ACTOR_UPDATE_EVERY = 10      
ALPHA_UPDATE_EVERY = 10       

# =========================================================
# TARGET NETWORK UPDATE
# =========================================================

TARGET_UPDATE_INTERVAL = 2    # Update target every 2 CRITIC updates

# =========================================================
# POLICY DISTRIBUTION
# =========================================================

LOG_STD_MIN = -5
LOG_STD_MAX = 2      # STAGE 1a: raise the ceiling so the policy can actually explore (was -3 = std 0.05)

# =========================================================
# EXPLORATION
# =========================================================

USE_RANDOM_POLICY_WARMUP = False
WARMUP_STEPS = 5_000 
FORCE_SKIP_WARMUP = False  # Set to True to bypass the actor for when critic is already warmupped
# =========================================================
# EVALUATION
# =========================================================

EVAL_INTERVAL = 10_000         
EVAL_EPISODES = 5

# =========================================================
# CHECKPOINTING
# =========================================================

CHECKPOINT_INTERVAL = 1_000   # CARLA freezes every few episodes; save model+state often (cheap/fast).
                              # On a force-kill you lose <1000 steps. (buffer saved every SAVE_BUFFER_EVERY)
SAVE_DIR = REPO_ROOT / "experiments" / "rl" / "sac"

LOG_EVERY = 1000

# =========================================================
# BC INITIALIZATION
# =========================================================

LOAD_BC_WEIGHTS = True
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
RESUME_CHECKPOINT = False    
RECORD_SAC_EVAL_VID = True
PRELOAD_EXPERT_DATA = False   # was True; disable 50/50 expert mixing until vanilla SAC is proven stable
COMPILED_DATASET_PATH = REPO_ROOT / "imitation" / "data" / "processed" / "dataset_rl_buffer.npz"

USE_Q_NORM = True    # was False; normalizes actor RL loss to O(1) so it can't explode with Q