from config.general_config import *

# =========================================================
# SAC ALGORITHM HYPERPARAMETERS
# =========================================================

GAMMA = 0.99               # discount factor
TAU = 0.005                # target network soft update

# =========================================================
# ENTROPY / TEMPERATURE
# =========================================================

AUTO_ENTROPY = True        # learn alpha automatically
INIT_ALPHA = 0.2           # initial entropy temperature
TARGET_ENTROPY_SCALE = 1.0 # multiplier for -|A|

# =========================================================
# OPTIMIZATION  
# =========================================================

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

WEIGHT_DECAY = 0.0

# =========================================================
# REPLAY BUFFER
# =========================================================

REPLAY_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256

# =========================================================
# TRAINING SCHEDULE
# =========================================================

MAX_TRAIN_STEPS = 1_000_000

WARMUP_STEPS = 10_000          # random policy steps before learning
UPDATE_AFTER = 1_000           # when gradient updates start
UPDATE_EVERY = 1               # env steps per update
GRADIENT_UPDATES = 1           # updates per training step

# =========================================================
# TARGET NETWORK UPDATE
# =========================================================

TARGET_UPDATE_INTERVAL = 1



# =========================================================
# POLICY DISTRIBUTION
# =========================================================

LOG_STD_MIN = -5
LOG_STD_MAX = 2

# =========================================================
# EXPLORATION
# =========================================================

USE_RANDOM_POLICY_WARMUP = False 

# =========================================================
# EVALUATION
# =========================================================

EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5

# =========================================================
# CHECKPOINTING
# =========================================================

CHECKPOINT_INTERVAL = 50_000
SAVE_DIR = REPO_ROOT / "experiments" / "rl" / "sac"

# =========================================================
# BC INITIALIZATION (OPTIONAL BUT HIGHLY RECOMMENDED)
# =========================================================

LOAD_BC_WEIGHTS = True
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" /"models" /  "best_model.pt"