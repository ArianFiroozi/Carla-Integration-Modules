from config.general_config import *

# =========================================================
# AWAC ALGORITHM HYPERPARAMETERS
# =========================================================
# Discount factor for future reward estimation (determining the agent's horizon)
GAMMA = 0.99            
# Coefficient for soft updates of target Q-networks parameters (exponential moving average factor)
TAU = 0.005                # Target network soft update 
# Weight scale parameter (Lagrangian multiplier) controlling deviation from dataset actions in AWAC
AWAC_LAMBDA = 0.01          
# Number of actions to sample from the policy to approximate state value V(s) in AWAC Q-updates
AWAC_NUM_SAMPLES = 4      

# =========================================================
# OPTIMIZATION SAFETY RAILS
# =========================================================
# True to use Huber loss instead of MSE for Q-network updates (adds stability against large spikes)
USE_HUBER_LOSS = True      # Safe against -200 crashes
# Learning rate for the actor network parameter optimizer
ACTOR_LR = 3e-5            # AWAC uses standard LR, doesn't need microscopic Actor LR like SAC          
# Learning rate for the critic network parameter optimizer
CRITIC_LR = 3e-4          
# L2 regularization weight decay penalty factor applied during optimizer steps
WEIGHT_DECAY = 1e-5        

# =========================================================
# REPLAY BUFFER & TRAINING SCHEDULE
# =========================================================
# Maximum capacity limit of the transitions replay buffer memory
REPLAY_BUFFER_SIZE = 500_000
# Number of transitions sampled per gradient update step
BATCH_SIZE = 256             

# Number of initial offline pretraining updates using only dataset demonstrations
OFFLINE_PRETRAIN_STEPS = 100_000
# Maximum total updates allowed for the training run
MAX_TRAIN_STEPS = 500_000     
# Updates to optimize Q-networks before starting optimization updates on the policy/actor
CRITIC_WARMUP_STEPS = 20_000  # AWAC requires less warmup than SAC

# Step interval between successive critic updates
CRITIC_UPDATE_EVERY = 1     
# Step interval between successive actor updates
ACTOR_UPDATE_EVERY = 1      
# Critic update steps interval between target network soft updates
TARGET_UPDATE_INTERVAL = 2  

# =========================================================
# POLICY DISTRIBUTION
# =========================================================
# Lower bound limit constraint for the log standard deviation output of the policy
LOG_STD_MIN = -5.0 
# Upper bound limit constraint for the log standard deviation output of the policy
LOG_STD_MAX = -2.5     

# =========================================================
# BC INITIALIZATION
# =========================================================
# True to initialize policy network parameters using a pretrained behavior cloning model
LOAD_BC_WEIGHTS = True
# Path to the pretrained Behavior Cloning model checkpoint file
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
# True to pre-fill the replay buffer with expert demonstrations before online training starts
PRELOAD_EXPERT_DATA = True
# Path to the consolidated demonstration file (.npz) containing expert trajectories
COMPILED_DATASET_PATH = REPO_ROOT / "imitation" / "data" / "processed" / "dataset_rl_buffer.npz"
# True to resume training from an existing AWAC checkpoint found in SAVE_DIR
RESUME_CHECKPOINT = False  
# Absolute directory path where checkpoints, logs, and metadata are stored
SAVE_DIR = REPO_ROOT / "experiments" / "offline_rl" / "awac"
# Environment steps interval between consecutive model saving operations
CHECKPOINT_INTERVAL = 25_000


# Minimum transitions that must accumulate in replay buffer before training updates begin
UPDATE_AFTER = 1000        
# Number of gradient updates performed per environment interaction step
GRADIENT_UPDATES = 1