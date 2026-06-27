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
INIT_ALPHA = 0.005          
TARGET_ENTROPY_SCALE = 1.0  
# =========================================================
# OPTIMIZATION  
# =========================================================
USE_HUBER_LOSS = True
ACTOR_LR = 1e-4           
CRITIC_LR = 1e-4          
ALPHA_LR = 1e-4          

WEIGHT_DECAY = 0           


BC_PENALTY_INIT = 50
BC_PENALTY_STEPS = 300_000
# =========================================================
# REPLAY BUFFER
# =========================================================

REPLAY_BUFFER_SIZE = 500_000 
BATCH_SIZE = 256

SAVE_BUFFER_EVERY = 5         # buffer pickles are large; keep them infrequent (model saves are cheap)
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
LOG_STD_MAX = 2      # STAGE 1a: raise the ceiling so the policy can actually explore (was -3 = std 0.05)

# =========================================================
# EXPLORATION
# =========================================================

USE_RANDOM_POLICY_WARMUP = False
WARMUP_STEPS = 10_000 
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
PRELOAD_EXPERT_DATA = False  
COMPILED_DATASET_PATH = REPO_ROOT / "imitation" / "data" / "processed" / "dataset_rl_buffer.npz"
BRANCH_FROM = None # this is a path for which checkpoint to resume
USE_Q_NORM = True    # was False; normalizes actor RL loss to O(1) so it can't explode with Q
