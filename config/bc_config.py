from pathlib import Path
import numpy as np
import torch
from .general_config import *

# =========================================================
# DATA PATHS
# =========================================================
# Base directory for imitation learning datasets and demonstrations
DATA_DIR = REPO_ROOT / "imitation" / "data"
# Directory path where autopilot expert demonstrations are stored
EXPERT_DIR = DATA_DIR / "expert_demos"
# Directory path where manual/human driver demonstrations are stored
MANUAL_DIR = DATA_DIR / "demos"
# Directory path where processed/compiled datasets are saved
PROCESSED_DIR = DATA_DIR / "processed"

# Target directory path for recording autopilot demonstration runs
AUTOPILOT_RECORD_DIR = EXPERT_DIR / "map1_0car"
# Target directory path for recording manual user demonstration runs
MANUAL_RECORD_DIR =  MANUAL_DIR / "test"

# Output path for the compiled discrete behavior cloning dataset file
DISCRETE_DATASET_PATH = PROCESSED_DIR / "dataset_bc_discrete.npz"
# Output path for the compiled continuous behavior cloning dataset file
CONTINUOUS_DATASET_PATH = PROCESSED_DIR / "dataset_bc_continuous.npz"
# Target path for compiling demonstrations to pre-load as a replay buffer
RL_REPLAY_BUFFER_PATH =  PROCESSED_DIR / "dataset_rl_buffer.npz"

# Main experiment base directory for behavior cloning logs and saved models
BC_EXPERIMENT_FOLDER = REPO_ROOT / "experiments" / "bc"


ENSEMBLE_PATH = REPO_ROOT / "experiments" / "dagger" / "20260701_094519"




# input of build dataset and inspect demo
# List of demonstration directories included when building the dataset
DEMO_LIST= [
    # EXPERT_DIR / "town01_0car",
    # MANUAL_DIR / "lab-map_0car",
    # MANUAL_DIR / "test",
    # MANUAL_DIR / "map1_0car",
    DATA_DIR/ "interventions_raw",
    # MANUAL_DIR / "map1_30car",
]


# =========================================================
# ACTION SPACE
# =========================================================
# Policy action space type: "discrete" or "continuous" (throttle, brake, steer)
ACTION_MODE = "continuous" 
# True if policy models action distribution as a Gaussian, False for deterministic output
IS_GAUSSIAN = False  #TODO: this has a bug and i was too lazy to debug , dont activate it unless you fixed the bug the bug is in training script and imitation policy 
# Lower bound limit constraint for Gaussian policy standard deviation parameter
MIN_STD=0.05
# Upper bound limit constraint for Gaussian policy standard deviation parameter
MAX_STD=1
# True to apply steering command exponential moving average smoothing filter
SMOOTH_STEERING = False

# Mapping from discrete class ID integers to descriptive speed action names
SPEED_MAP = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Reverse",
    4: "Constant",
}

# Mapping from discrete class ID integers to descriptive turning action names
TURN_MAP = {
    0: "Right",
    1: "Left",
    2: "No Turn",
    3: "Straight",
}

# =========================================================
# ACTION SIMPLIFICATION
# =========================================================
# True to condense discrete action space classes for easier policy learning
SIMPLIFY_ACTIONS = True
# True to filter out dataset transitions containing reverse actions
REMOVE_REVERSE = True
# True to filter out transitions containing redundant turn categories
REMOVE_NO_TURN = True

# Simplified map grouping speed actions into consolidated classes
SIMPLIFY_SPEED_MAP = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Constant",
}

# Simplified map grouping turning actions into consolidated classes
SIMPLIFY_TURN_MAP = {
    0: "Right",
    1: "Left",
    2: "Straight",
}



# =========================================================
# DATASET SETTINGS
# =========================================================
# True to discard termination/collision states from training trajectories
DROP_TERMINATED = True
# Number of trailing frames to discard prior to crash termination (discards bad collision behavior)
DROP_LAST_N_BEFORE_TERMINATION = 100

# True to filter out idle frames where the vehicle is stationary
FILTER_IDLE_FRAMES = False
# Mode used to specify frame filtering: "all" or specific filter strategy
IDLE_FILTER_MODE = "all"

# Speed threshold below which the vehicle can be flagged as idle (m/s)
IDLE_SPEED_THRESHOLD = 0.3
# Throttle input threshold below which the vehicle is flagged as idle
IDLE_THROTTLE_THRESHOLD = 0.05
# Brake input threshold below which the vehicle is flagged as idle
IDLE_BRAKE_THRESHOLD = 0.05

# Probabilities for downsampling combinations of (speed_action, turn_action) to balance datasets
JOINT_KEEP_PROBS = {
    (4, 3): 0.3,
    (4, 0): 0.5,
    (4, 1): 0.8,
}

# JOINT_KEEP_PROBS = {
#     (4, 3): 1,
#     (4, 0): 1,
#     (4, 1): 1,
# }


# 'min_max': Scale features to [-1, 1] based on dataset-wide min/max.
# 'z_score': Standardize features using dataset-wide mean/std.
# 'fixed':   Divide speed grids by MAX_SPEED. Scalars are not normalized. (Less recommended)
# Feature normalization technique: 'z_score', 'min_max', or 'fixed' scale
SCALING_METHOD = "z_score"
# Speed scaling denominator (m/s) under the 'fixed' scaling method
MAX_SPEED = 30.0


# True to downsample straight-driving transitions in continuous action space dataset building
USE_CONTINUOUS_UNDERSAMPLING = False
# Steering magnitude threshold below which a transition is flagged as straight
UNDERSAMPLING_THRESHOLD = 0.05 
# Chance of dropping a flagged straight transition sample if continuous undersampling is enabled
UNDERSAMPLING_PROBABILITY = 0.6    


# True to apply different loss multipliers depending on the magnitude of prediction error
USE_WEIGHTED_LOSS = False
# Loss multiplier weight applied when steering prediction error is above threshold
STEER_LOSS_WEIGHT = 3.0         
# Loss multiplier weight applied when throttle prediction error is above threshold
THROTTLE_LOSS_WEIGHT = 1.0
# Loss multiplier weight applied when brake prediction error is above threshold
BRAKE_LOSS_WEIGHT = 1.0
# Prediction error threshold above which loss weighting multipliers are applied
WEIGHTED_LOSS_THRESHOLD = 0.1      

# Sampling strategy for minibatches: "inverse" frequencies, "none", or "handmade" rules
WEIGHTED_SAMPLING = "none" 




# ================================
# Dataset Augmentation
# ================================
# True to perform horizontal mirroring data augmentation on dataset transitions
MIRROR_DATASET = False
# Minimum steering command threshold required to mirror transition (avoid mirroring straight runs)
MIRROR_STEERING_THRESHOLD = 0.04



# =========================================================
# IMITATION LEARNING TRAINING
# =========================================================
# Maximum training epochs (full passes through dataset) allowed
BC_EPOCHS = 30
# Training batch size for SGD/Adam updates
BC_BATCH_SIZE = 512
# Learning rate for the behavior cloning optimizer
BC_LR = 3e-4
# Fraction of training data set aside for validation evaluation
BC_VAL_SPLIT = 0.1
# Early stopping patience (number of epochs to wait without validation loss improvement)
BC_PATIENCE = 10

# =========================================================
# MANUAL CONTROL / DEMO RECORDING
# =========================================================
# Sleep interval (seconds) in keyboard manual control update loops
MANUAL_SLEEP_SECONDS = 0.001
# Step interval between logging manual control states to terminal
MANUAL_PRINT_EVERY = 500
# True to print environment occupancy grid variables to terminal logs for debugging
MANUAL_DEBUG_GRIDS = False

# True to record dataset trajectories during manual drive sessions
MANUAL_RECORD = False
# Prefix basename for manual demonstration run directories
MANUAL_BASE_NAME = "map1"

# Driving mode selection for data recording runs: "manual" human or "autopilot" agent
RECORD_DRIVE_MODE = "manual" # "manual" or "autopilot"

# Number of episodes to collect and record when running in autopilot drive mode
DEFAULT_AUTOPILOT_EPISODES = 1000
# Prefix basename for autopilot demonstration directories
AUTOPILOT_DEMO_BASENAME = "autopilot_map1"


# =========================================================
# POLICY EVALUATION / ROLLOUT
# =========================================================
# Number of episodes to run during behavior cloning evaluation phases
EVAL_NUM_EPISODES = 20
# Maximum steps allowed per episode before rollout termination
EVAL_MAX_STEPS = 2000
# Step interval between console reporting prints during evaluation rollouts
EVAL_RENDER_LOG_EVERY = 200
# True to record video files of evaluation rollout runs
RECORD_BC_EVAL_VID = False
# True to save trained model parameters to checkpoints folder
RECORD_MODEL = True

# =========================================================
# DEBUG / VISUALIZATION
# =========================================================
# Step interval for printing intermediate debug information during policy forward passes
DEBUG_PRINT_STEPS = 0

# True to open diagnostic inspection visualization tools
INSPECT_VISUALIZE = True
# Upper limit of dataset samples processed by the visual features inspector
MAX_INSPECT_FEATURE_SAMPLES = 200000

# True to build visualization diagnostics during dataset compilation
BUILD_VISUALIZE = False
# Limit on dataset samples compiled into feature distribution histograms
FEATURE_HIST_MAX_SAMPLES = 100000

# =========================================================
# Action Collapse Thresholds
# =========================================================
# Threshold below which steering variance triggers an action collapse warning
MIN_STEER_VAR = 0.005
# Threshold below which throttle variance triggers an action collapse warning
MIN_THROTTLE_VAR = 0.001 
# Threshold below which brake variance triggers an action collapse warning
MIN_BRAKE_VAR = 0.001

