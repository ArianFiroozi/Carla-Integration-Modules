from pathlib import Path
import numpy as np
import torch
from .general_config import *


# ---------- Data Collection ----------
EPSILON = 0.1                 # probability of random action
NUM_EPISODES = 200            # how many episodes to collect
MAX_STEPS = 2000              # max steps per episode (same as env)

# ---------- Paths ----------
BC_CHECKPOINTS_ROOT = REPO_ROOT / "experiments" / "bc" 
SAVE_DIR = REPO_ROOT / "offline_rl" / "data"
COLLECT_EPISODES = 500
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt" # choose the bc path manually

# =========================================================
# AWAC & RL HYPERPARAMETERS
# =========================================================
AWAC_LAMBDA = 2.0             
EXP_ADV_MAX = 20       
OFFLINE_TRAIN_STEPS = 70000
CRITIC_WARMUP_STEPS = 20000

ACTOR_LR = 3e-5            
CRITIC_LR = 1e-4        
WEIGHT_DECAY = 1e-5        

REPLAY_BUFFER_SIZE = 500_000  
BATCH_SIZE = 128             

GAMMA = 0.99               
TAU = 0.005                
TARGET_UPDATE_INTERVAL = 2

# =========================================================
# IQL HYPERPARAMETERS
# =========================================================
IQL_TAU = 0.7    
IQL_BETA = 3.0     