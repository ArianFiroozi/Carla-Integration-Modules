from config.general_config import *

# =========================================================
# AWAC ALGORITHM HYPERPARAMETERS
# =========================================================
GAMMA = 0.99               # Discount factor 
TAU = 0.005                # Target network soft update 
AWAC_LAMBDA = 0.01          # The Lagrange multiplier (Controls how strictly to trust the expert)
AWAC_NUM_SAMPLES = 4       # How many actions to sample to estimate V(s)

# =========================================================
# OPTIMIZATION SAFETY RAILS
# =========================================================
USE_HUBER_LOSS = True      # True = Huber loss (Safe against -200 crashes), False = Standard MSE
ACTOR_LR = 3e-5            # AWAC uses standard LR, doesn't need microscopic Actor LR like SAC          
CRITIC_LR = 3e-4          
WEIGHT_DECAY = 1e-5        

# =========================================================
# REPLAY BUFFER & TRAINING SCHEDULE
# =========================================================
REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 256             

OFFLINE_PRETRAIN_STEPS = 100_000
MAX_TRAIN_STEPS = 500_000     
CRITIC_WARMUP_STEPS = 20_000  # AWAC requires less warmup than SAC

CRITIC_UPDATE_EVERY = 1     
ACTOR_UPDATE_EVERY = 1      
TARGET_UPDATE_INTERVAL = 2  

# =========================================================
# POLICY DISTRIBUTION
# =========================================================
LOG_STD_MIN = -5.0 
LOG_STD_MAX = -2.5     

# =========================================================
# BC INITIALIZATION
# =========================================================
LOAD_BC_WEIGHTS = True
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
PRELOAD_EXPERT_DATA = True
COMPILED_DATASET_PATH = REPO_ROOT / "imitation" / "data" / "processed" / "dataset_rl_buffer.npz"
RESUME_CHECKPOINT = False  
SAVE_DIR = REPO_ROOT / "experiments" / "offline_rl" / "awac"
CHECKPOINT_INTERVAL = 25_000

# How often to run backprop. 
# Set this to 1000 so the agent gathers a batch of online data before mixing it into the buffer.
UPDATE_AFTER = 1000        
GRADIENT_UPDATES = 1