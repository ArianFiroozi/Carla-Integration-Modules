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
INIT_ALPHA = 0.02           
TARGET_ENTROPY_SCALE = 0.2  

# =========================================================
# OPTIMIZATION  
# =========================================================

ACTOR_LR = 1e-6           
CRITIC_LR = 1e-4          
ALPHA_LR = 1e-4           

WEIGHT_DECAY = 0           

# =========================================================
# REPLAY BUFFER
# =========================================================

REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 256             

SAVE_BUFFER_EVERY = 10  
KEEP_CHECKPOINTS = 3

# =========================================================
# TRAINING SCHEDULE
# =========================================================

MAX_TRAIN_STEPS = 500_000     

CRITIC_WARMUP_STEPS = 100_000 
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
LOG_STD_MAX = -3     

# =========================================================
# EXPLORATION
# =========================================================

USE_RANDOM_POLICY_WARMUP = False
WARMUP_STEPS = 5_000 

# =========================================================
# EVALUATION
# =========================================================

EVAL_INTERVAL = 10_000         
EVAL_EPISODES = 5

# =========================================================
# CHECKPOINTING
# =========================================================

CHECKPOINT_INTERVAL = 25_000
SAVE_DIR = REPO_ROOT / "experiments" / "rl" / "sac"

LOG_EVERY = 1000

# =========================================================
# BC INITIALIZATION
# =========================================================

LOAD_BC_WEIGHTS = True
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
RESUME_CHECKPOINT = False
RECORD_SAC_EVAL_VID = True