from pathlib import Path
import numpy as np
import torch
from .general_config import *


# ---------- Data Collection ----------
# Probability of taking a random action for epsilon-greedy exploration during offline data collection
EPSILON = 0.1                 # probability of random action
# Number of episodes to run during offline data collection
NUM_EPISODES = 200            # how many episodes to collect
# Maximum steps allowed per episode before terminating the environment
MAX_STEPS = 2000              # max steps per episode (same as env)

# ---------- Paths ----------
# Root directory where Behavior Cloning experiments and checkpoints are stored
BC_CHECKPOINTS_ROOT = REPO_ROOT / "experiments" / "bc" 
<<<<<<< HEAD
# NOTE: the original offline_noisy_data is gone (offline_rl/data is empty after the lab sync).
# Point IQL/AWAC at the BC demos, which have the same npz format incl. a 'rewards' key (178 eps).
# (To do offline RL *properly* later, re-record diverse/noisy rollouts via offline_rl/record_demos.py.)
SAVE_DIR = REPO_ROOT / "imitation" / "data" / "demos" / "map1_30car"
COLLECT_EPISODES = 500
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt" # base BC (stronger than DAgger round 1)
=======
# Output directory path where collected offline dataset files are saved
SAVE_DIR = REPO_ROOT / "offline_rl" / "data"
# Number of episodes of data to collect for offline RL training sets
COLLECT_EPISODES = 500
# Specific Behavior Cloning model checkpoint file path to use for offline data collection/warmup
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt" # choose the bc path manually
>>>>>>> 1e7f54cccfec0f7d0a8e9470bb2c8210dc36574e

# =========================================================
# AWAC & RL HYPERPARAMETERS
# =========================================================
# Lagrangian multiplier weight scaling factor for policy loss constraints in AWAC
AWAC_LAMBDA = 2.0             
# Maximum clipping value for exponential advantage weights to prevent numerical instability
EXP_ADV_MAX = 20       
# Total training updates/steps for offline reinforcement learning training
OFFLINE_TRAIN_STEPS = 70000
# Number of initial critic optimization updates prior to starting actor network updates
CRITIC_WARMUP_STEPS = 20000

# Learning rate for the actor network optimizer
ACTOR_LR = 3e-5            
# Learning rate for the critic/Q-value network optimizer
CRITIC_LR = 1e-4        
# L2 regularization weight decay factor applied to weights in Adam optimizers
WEIGHT_DECAY = 1e-5        

# Maximum capacity limit of transitions stored in the replay buffer memory
REPLAY_BUFFER_SIZE = 500_000  
# Minibatch sample size used during gradient update steps
BATCH_SIZE = 128             

# Discount factor for future reward estimations
GAMMA = 0.99               
# Soft update moving average coefficient for target networks parameter updates
TAU = 0.005                
# Critic update steps interval between soft target network updates
TARGET_UPDATE_INTERVAL = 2

# =========================================================
# IQL HYPERPARAMETERS
# =========================================================
# Expectile target parameter used in Implicit Q-Learning (IQL) value network updates
IQL_TAU = 0.7    
IQL_BETA = 3.0     