from config.general_config import *

# =========================================================
# SAC ALGORITHM HYPERPARAMETERS
# ========================================================= 

# Discount factor for future reward estimation (determining the agent's horizon)
GAMMA = 0.99     
# Coefficient for soft updates of target Q-networks parameters (exponential moving average factor)
TAU = 0.005         

# =========================================================
# ENTROPY / TEMPERATURE
# =========================================================

# True to automatically tune the policy entropy temperature alpha, False to keep it static
AUTO_ENTROPY = True   
# Initial or static entropy temperature value alpha (controls exploration vs exploitation trade-off)
INIT_ALPHA = 0.1          
# Scaling factor applied to target entropy target (typically matches negative action space dimension)
TARGET_ENTROPY_SCALE = 0.3  
# =========================================================
# OPTIMIZATION  
# =========================================================
# True to use Huber loss instead of MSE for Q-value updates (protects gradients from huge gradients when crashing)
USE_HUBER_LOSS = True
# Learning rate for the actor network parameter optimizer
ACTOR_LR = 5e-7           
# Learning rate for the critic network parameter optimizer
CRITIC_LR = 3e-4         
# Learning rate for the entropy temperature alpha parameter optimizer
ALPHA_LR = 1e-4          

# L2 regularization weight decay penalty factor applied during optimizer steps
WEIGHT_DECAY = 1e-6          


# Initial weight/coefficient for the Behavior Cloning regularization penalty loss
BC_PENALTY_INIT = 200
# Number of environment steps over which the BC penalty weight decays or changes
BC_PENALTY_STEPS = 2_000_000
# =========================================================
# REPLAY BUFFER
# =========================================================

# Maximum capacity limit of the transitions replay buffer memory
REPLAY_BUFFER_SIZE = 500_000 
# Number of transitions sampled per gradient update step
BATCH_SIZE = 512

# Frequency of saving the replay buffer to disk (measured in checkpoint intervals)
SAVE_BUFFER_EVERY = 4        
# Number of recent checkpoint directories or files to preserve on disk
KEEP_CHECKPOINTS = 2

# =========================================================
# TRAINING SCHEDULE
# =========================================================

# Maximum total environment simulation steps allowed for the training run
MAX_TRAIN_STEPS = 500_000     

# Environment steps to collect before starting optimization updates on the actor network
CRITIC_WARMUP_STEPS = 100_000    
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
LOG_STD_MIN = -5
# Upper bound limit constraint for the log standard deviation output of the policy
LOG_STD_MAX = 2      

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
EVAL_INTERVAL = 10_000         
# Number of independent episodes to run during each evaluation phase
EVAL_EPISODES = 5

# =========================================================
# CHECKPOINTING
# =========================================================

# Environment steps interval between consecutive model saving operations
CHECKPOINT_INTERVAL = 25_000  
# Absolute directory path where checkpoints, logs, and metadata are stored
SAVE_DIR = REPO_ROOT / "experiments" / "rl" / "sac"

# Training step interval for logging metrics to tensorboard/console
LOG_EVERY = 1000

# =========================================================
# BC INITIALIZATION
# =========================================================

# True to initialize policy network parameters using a pretrained behavior cloning model
LOAD_BC_WEIGHTS = True
# Path to the pretrained Behavior Cloning model checkpoint file
BC_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "bc" / "2026_05_03_21_45_02_bc_continuous" / "models" / "best_model.pt"
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
USE_Q_NORM = False 
