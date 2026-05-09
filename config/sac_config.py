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
INIT_ALPHA = 0.02           
TARGET_ENTROPY_SCALE = 0.5 #  0.5 to reduce exploration with BC init

# =========================================================
# OPTIMIZATION  
# =========================================================

ACTOR_LR = 1e-4            
CRITIC_LR = 1e-4        
ALPHA_LR = 1e-4           

WEIGHT_DECAY = 1e-5        # ADD small regularization

# =========================================================
# REPLAY BUFFER
# =========================================================

REPLAY_BUFFER_SIZE = 500_000  #  your grid isn't huge, 500k is plenty
BATCH_SIZE = 128             

# How often to save the full replay buffer (in number of checkpoints)
SAVE_BUFFER_EVERY = 10  
# Only keep this many recent checkpoints on disk
KEEP_CHECKPOINTS = 3

# =========================================================
# TRAINING SCHEDULE
# =========================================================

MAX_TRAIN_STEPS = 500_000     #  with BC init, 500k might be enough


CRITIC_WARMUP_STEPS = 500_000 
UPDATE_AFTER = 1_000          # START earlier since BC gives good data
UPDATE_EVERY = 2              # Update every 2 steps to be more stable
GRADIENT_UPDATES = 1       

# =========================================================
# TARGET NETWORK UPDATE
# =========================================================

TARGET_UPDATE_INTERVAL = 2    # Update target less frequently

# =========================================================
# POLICY DISTRIBUTION
# =========================================================

LOG_STD_MIN = -5 
LOG_STD_MAX = -0.5     

# =========================================================
# EXPLORATION
# =========================================================

USE_RANDOM_POLICY_WARMUP = False  # keep False with BC init
WARMUP_STEPS = 5_000 
# =========================================================
# EVALUATION
# =========================================================

EVAL_INTERVAL = 10_000         
EVAL_EPISODES = 5             # keep

# =========================================================
# CHECKPOINTING
# =========================================================

CHECKPOINT_INTERVAL = 25_000  # Save more frequently
SAVE_DIR = REPO_ROOT / "experiments" / "rl" / "sac"

LOG_EVERY = 1000
# =========================================================
# BC INITIALIZATION
# =========================================================

LOAD_BC_WEIGHTS = True
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
RESUME_CHECKPOINT = False     # Set to True only when you want to resume