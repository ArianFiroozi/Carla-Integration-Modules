from pathlib import Path
import numpy as np
import torch

# =========================================================
# SYSTEM
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# RANDOM SEEDS 
# =========================================================
BUILD_RNG_SEED = 42
BC_SPLIT_SEED = 42
GLOBAL_SEED = 42

# =========================================================
# ROOT PATHS
# =========================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# DATA PATHS
# =========================================================
DATA_DIR = REPO_ROOT / "imitation" / "data"
EXPERT_DIR = DATA_DIR / "expert_demos"
MANUAL_DIR = DATA_DIR / "demos"
PROCESSED_DIR = DATA_DIR / "processed"

AUTOPILOT_RECORD_DIR = EXPERT_DIR / "map1_0car"
MANUAL_RECORD_DIR =  MANUAL_DIR / "map1_30car"

DISCRETE_DATASET_PATH = PROCESSED_DIR / "dataset_bc_discrete.npz"
CONTINUOUS_DATASET_PATH = PROCESSED_DIR / "dataset_bc_continuous.npz"

# input of build dataset and inspect demo
DEMO_LIST= [
    # EXPERT_DIR / "town01_0car",
    # MANUAL_DIR / "lab-map_0car",
    MANUAL_DIR / "test",
    # MANUAL_DIR / "map1_0car",
    MANUAL_DIR / "map1_30car"
]

# =========================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =========================================================
# CNN settings

# ---BASELINE---
CNN_CHANNELS = [16, 32, 64]
KERNEL_SIZES = [3, 3, 3]
# Fully Connected settings (Scalars)
SCALAR_N_MLP_LAYERS = 2
SCALAR_MLP_HIDDEN_SIZE = 32
# Fusion & Latent
LATENT_DIM = 128
# Actor Head settings (Gaussian)
HEAD_N_MLP_LAYERS = 2
HEAD_MLP_HIDDEN_SIZE = 64



# # ---BALANCED---
# CNN_CHANNELS = [32, 64, 128]
# KERNEL_SIZES = [3, 3, 3]
# # Fully Connected settings (Scalars)
# SCALAR_N_MLP_LAYERS = 2
# SCALAR_MLP_HIDDEN_SIZE = 64
# # Fusion & Latent
# LATENT_DIM = 256
# # Actor Head settings (Gaussian)
# HEAD_N_MLP_LAYERS = 2
# HEAD_MLP_HIDDEN_SIZE = 128




# # ---BIG---
# CNN_CHANNELS = [32, 64, 128]
# KERNEL_SIZES = [5, 3, 3]
# # Fully Connected settings (Scalars)
# SCALAR_N_MLP_LAYERS = 2
# SCALAR_MLP_HIDDEN_SIZE = 64
# # Fusion & Latent
# LATENT_DIM = 256
# # Actor Head settings (Gaussian)
# HEAD_N_MLP_LAYERS = 2
# HEAD_MLP_HIDDEN_SIZE = 256





# =========================================================
# ACTION SPACE
# =========================================================
ACTION_MODE = "continuous"   # "discrete" or "continuous"
IS_GAUSSIAN = False
MIN_STD=0.05
MAX_STD=1
SMOOTH_STEERING = False

SPEED_MAP = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Reverse",
    4: "Constant",
}

TURN_MAP = {
    0: "Right",
    1: "Left",
    2: "No Turn",
    3: "Straight",
}

# =========================================================
# ACTION SIMPLIFICATION
# =========================================================
SIMPLIFY_ACTIONS = True
REMOVE_REVERSE = True
REMOVE_NO_TURN = True

SIMPLIFY_SPEED_MAP = {
    0: "Accelerate",
    1: "Brake",
    2: "Stop",
    3: "Constant",
}

SIMPLIFY_TURN_MAP = {
    0: "Right",
    1: "Left",
    2: "Straight",
}

# =========================================================
# OBSERVATION SPACE LIMITS
# =========================================================
OBS_BOUNDS = {
    "obs_speed_x": dict(low=-np.inf, high=np.inf),
    "obs_speed_y": dict(low=-np.inf, high=np.inf),
    "obs_presence": dict(low=0, high=9),
    "obs_lane_angle": dict(low=-np.pi, high=np.pi),
    "obs_max_speed": dict(low=0.0, high=200.0),
    "obs_traffic_signs": dict(low=0.0, high=1.0),
    "obs_ego_speed_x": dict(low=-np.inf, high=np.inf),
    "obs_ego_speed_y": dict(low=-np.inf, high=np.inf),
    "obs_ego_in_lane_position_x": dict(low=-100.0, high=100.0),
    "obs_throttle": dict(low=0.0, high=1.0),
    "obs_brake": dict(low=0.0, high=1.0),
    "obs_steering_angle": dict(low=-1.0, high=1.0),
    "obs_reverse": dict(low=0.0, high=1.0),
}

# =========================================================
# DATASET SETTINGS
# =========================================================
DROP_TERMINATED = True
DROP_LAST_N_BEFORE_TERMINATION = 100

FILTER_IDLE_FRAMES = False
IDLE_FILTER_MODE = "all"

IDLE_SPEED_THRESHOLD = 0.3
IDLE_THROTTLE_THRESHOLD = 0.05
IDLE_BRAKE_THRESHOLD = 0.05

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
SCALING_METHOD = "z_score"
MAX_SPEED = 30.0


USE_CONTINUOUS_UNDERSAMPLING = False
UNDERSAMPLING_THRESHOLD = 0.05      # Absolute steering value below which is "straight"
UNDERSAMPLING_PROBABILITY = 0.6     # 60% chance to drop straight samples if flag is True


USE_WEIGHTED_LOSS = False
STEER_LOSS_WEIGHT = 3.0             # Multiplier for loss when steering error is large
THROTTLE_LOSS_WEIGHT = 1.0
BRAKE_LOSS_WEIGHT = 1.0
WEIGHTED_LOSS_THRESHOLD = 0.1       # Threshold above which the weight is applied

WEIGHTED_SAMPLING = "none" # "inverse" or "none" or "handmade"




# ================================
# Dataset Augmentation
# ================================
MIRROR_DATASET = False
MIRROR_STEERING_THRESHOLD = 0.04
WINDOW_SIZE = 1
USE_ONE_HOT_GRID = True



# =========================================================
# IMITATION LEARNING TRAINING
# =========================================================
BC_EPOCHS = 1000
BC_BATCH_SIZE = 512
BC_LR = 3e-4
BC_VAL_SPLIT = 0.1
BC_PATIENCE = 10

# =========================================================
# MANUAL CONTROL / DEMO RECORDING
# =========================================================
MANUAL_SLEEP_SECONDS = 0.001
MANUAL_PRINT_EVERY = 500
MANUAL_DEBUG_GRIDS = False

MANUAL_RECORD = True
MANUAL_BASE_NAME = "map1"

RECORD_DRIVE_MODE = "manual" # "manual" or "autopilot"

DEFAULT_AUTOPILOT_EPISODES = 1000
AUTOPILOT_DEMO_BASENAME = "autopilot_map1"

# =========================================================
# CARLA ENVIRONMENT DEFAULTS
# =========================================================
CARLA_MAP_PATH = r"C:\carla\Carla-Integration-Modules\CarlaEnv\LoadOpenDrive2\map1.xodr"
CARLA_WALKERS = 0
CARLA_VEHICLES = 50
CARLA_MAX_STEPS = 2000
CARLA_INIT_SPEED = 0

RANDOM_VEHICLE_START_POS = True
RANDOM_EGO_START_POS = True

# =========================================================
# POLICY EVALUATION / ROLLOUT
# =========================================================
EVAL_NUM_EPISODES = 20
EVAL_MAX_STEPS = 2000
EVAL_RENDER_LOG_EVERY = 200

# =========================================================
# DEBUG / VISUALIZATION
# =========================================================
DEBUG_PRINT_STEPS = 500

INSPECT_VISUALIZE = True
MAX_INSPECT_FEATURE_SAMPLES = 200000

BUILD_VISUALIZE = False
FEATURE_HIST_MAX_SAMPLES = 100000

# =========================================================
# Action Collapse Thresholds
# =========================================================
MIN_STEER_VAR = 0.005
MIN_THROTTLE_VAR = 0.001 
MIN_BRAKE_VAR = 0.001