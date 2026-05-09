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
# MODEL ARCHITECTURE HYPERPARAMETERS
# =========================================================
IS_DECOUPLED = False
LOG_STD_INIT_BIAS = -3.0
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
# OBS & ACTION SPACE
# =========================================================


WINDOW_SIZE = 1
USE_ONE_HOT_GRID = True

USE_SPATIAL_FEATURES = True
SCALAR_DIM = 4 + 4 * int(USE_SPATIAL_FEATURES)
GRID_CHANNELS = (3+ 2*int(USE_ONE_HOT_GRID)) * WINDOW_SIZE


ACTION_DIM = 3   # throttle, brake, steering

ACTION_LOW = [0.0, 0.0, -1.0]
ACTION_HIGH = [1.0, 1.0, 1.0]

# =========================================================
# CARLA ENVIRONMENT DEFAULTS
# =========================================================
CARLA_MAP_PATH = r"C:\carla\Carla-Integration-Modules\CarlaEnv\LoadOpenDrive2\map1.xodr"
CARLA_WALKERS = 0
CARLA_VEHICLES = 30
CARLA_MAX_STEPS = 2000
CARLA_INIT_SPEED = 0

RANDOM_VEHICLE_START_POS = True
RANDOM_EGO_START_POS = True








