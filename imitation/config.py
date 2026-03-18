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


# =========================================================
# ROOT PATHS
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# DATA PATHS
# =========================================================

DATA_DIR = REPO_ROOT / "imitation" / "data"

DEMO_DIR = DATA_DIR / "demos"
DEMO_DIR_ALT = DATA_DIR / "demos2"

PROCESSED_DIR = DATA_DIR / "processed"

DATASET_PATH = PROCESSED_DIR / "dataset_bc.npz"


# =========================================================
# CHECKPOINT PATHS
# =========================================================

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
IMITATION_CHECKPOINT_DIR = CHECKPOINT_DIR / "imitation"

DISCRETE_MODEL_PATH = IMITATION_CHECKPOINT_DIR / "bc_cnn_discrete.pt"
CONTINUOUS_MODEL_PATH = IMITATION_CHECKPOINT_DIR / "bc_cnn_continuous.pt"


# =========================================================
# ACTION SPACE
# =========================================================

ACTION_MODE = "continuous"   # "discrete" or "continuous"

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
# DATASET BUILD SETTINGS
# =========================================================

DROP_TERMINATED = True
DROP_LAST_N_BEFORE_TERMINATION = 10

FILTER_IDLE_FRAMES = True
IDLE_FILTER_MODE = "all"

IDLE_SPEED_THRESHOLD = 0.3
IDLE_THROTTLE_THRESHOLD = 0.05
IDLE_BRAKE_THRESHOLD = 0.05

JOINT_KEEP_PROBS = {
    (4, 3): 1.0,
    (4, 0): 1.0,
    (4, 1): 1.0,
}


# =========================================================
# IMITATION LEARNING TRAINING
# =========================================================

BC_EPOCHS = 100
BC_BATCH_SIZE = 512
BC_LR = 3e-4
BC_VAL_SPLIT = 0.1
BC_PATIENCE = 10


# =========================================================
# MANUAL CONTROL / DEMO RECORDING
# =========================================================

MANUAL_DEMO_DIR = DEMO_DIR_ALT

MANUAL_SLEEP_SECONDS = 0.001
MANUAL_PRINT_EVERY = 1000
MANUAL_DEBUG_GRIDS = False

MANUAL_RECORD = False
MANUAL_BASE_NAME = "manual_demo13"


# =========================================================
# CARLA ENVIRONMENT DEFAULTS
# =========================================================

CARLA_MAP_PATH = r"C:\carla\Carla-Integration-Modules\CarlaEnv\LoadOpenDrive2\lab-map.xodr"

CARLA_WALKERS = 0
CARLA_VEHICLES = 0
CARLA_MAX_STEPS = 2000
CARLA_INIT_SPEED = 0


# =========================================================
# POLICY EVALUATION / ROLLOUT
# =========================================================

EVAL_NUM_EPISODES = 30
EVAL_MAX_STEPS = 2000
EVAL_RENDER_LOG_EVERY = 200


# =========================================================
# DEBUG / VISUALIZATION
# =========================================================

DEBUG_PRINT_STEPS = 50

INSPECT_VISUALIZE = False
MAX_INSPECT_FEATURE_SAMPLES = 200000

BUILD_VISUALIZE = True
FEATURE_HIST_MAX_SAMPLES = 100000
