from pathlib import Path
import numpy as np
import torch

# =========================================================
# SYSTEM
# =========================================================
# PyTorch computing device ('cuda' for GPU acceleration or 'cpu' for host CPU execution)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# RANDOM SEEDS 
# =========================================================
# Seed used for procedural map creation and environment spawning randomization
BUILD_RNG_SEED = 42
# Seed used to split the imitation learning / behavior cloning dataset into train/validation sets
BC_SPLIT_SEED = 42
# Main global seed for algorithm exploration and network initialization randomness
GLOBAL_SEED = 42

# =========================================================
# ROOT PATHS
# =========================================================
REPO_ROOT = Path(__file__).resolve().parents[1]



# =========================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =========================================================
# True to use separate MLP heads for speed/steering control actions, False to use a single shared head
IS_DECOUPLED = False
# Initial bias value for standard deviation parameter in the Gaussian policy distribution
LOG_STD_INIT_BIAS = -2.0  
# ---BASELINE---
# Channel output sizes for the CNN layers of the visual feature extractor
CNN_CHANNELS = [16, 32, 64]
# Convolutional kernel (filter) sizes for each layer of the feature extractor CNN
KERNEL_SIZES = [3, 3, 3]
# Fully Connected settings (Scalars)
# Number of MLP layers used to process scalar non-visual observation variables
SCALAR_N_MLP_LAYERS = 2
# Hidden units dimension in the scalar observations embedding MLP network
SCALAR_MLP_HIDDEN_SIZE = 32
# Fusion & Latent
# Dimensionality of the joint latent vector representing fused spatial and scalar observations
LATENT_DIM = 128
# Actor Head settings (Gaussian)
# Number of hidden MLP layers in the actor policy network output head
HEAD_N_MLP_LAYERS = 2
# Hidden units dimension inside the actor policy network output head MLP
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
# Normalization thresholds and value limits for active observation variables
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


# Number of consecutive history frames stacked as input observations
WINDOW_SIZE = 1   # REVERTED from 3 (executive call 2026-07-09). A/B result for the record:
                  # with the train/eval scalar-permutation bug FIXED, the window-3 BC still
                  # evaluated at ~248 avg length vs the window-1 May-3 baseline's 667 in
                  # traffic — frame stacking hurt BC at this capacity (likely the 16-filter
                  # first conv choking on 3x input channels). Revisit only with a wider CNN
                  # and its own BC retrain cycle. Window-1 restores compatibility with the
                  # proven 2026_05_03 BC checkpoint (5 grid channels).
# Whether to represent sign/presence elements in the grid as one-hot vectors
USE_ONE_HOT_GRID = True

# True to enable spatial feature map extraction, False to bypass
USE_SPATIAL_FEATURES = True
# Dimension of the final non-spatial scalar features vector
SCALAR_DIM = 4 + 4 * int(USE_SPATIAL_FEATURES)
# Final combined channel count of input observation grids (channels * window_size)
GRID_CHANNELS = (3+ 2*int(USE_ONE_HOT_GRID)) * WINDOW_SIZE


# Number of dimensions of agent's control output space: [throttle, brake, steering]
ACTION_DIM = 3   # throttle, brake, steering

# Minimum bounds allowed for action outputs: [min_throttle, min_brake, min_steer]
ACTION_LOW = [0.0, 0.0, -1.0]
# Maximum bounds allowed for action outputs: [max_throttle, max_brake, max_steer]
ACTION_HIGH = [1.0, 1.0, 1.0]

# =========================================================
# CARLA ENVIRONMENT DEFAULTS
# =========================================================

CARLA_MAP_PATH = str(REPO_ROOT / "CarlaEnv" / "LoadOpenDrive2" / "map1.xodr")
# Number of pedestrian/walker entities to spawn in the environment simulation
CARLA_WALKERS = 0
# Number of autonomous vehicle traffic obstacles to spawn in the environment simulation
CARLA_VEHICLES = 30
# Maximum time steps limit allowed per training/evaluation episode before truncation
CARLA_MAX_STEPS = 2000
# Initial velocity/speed of the ego vehicle upon initial spawning (m/s)
CARLA_INIT_SPEED = 0

# True to spawn non-ego vehicle traffic at randomized positions on the map
RANDOM_VEHICLE_START_POS = True
# True to spawn the ego vehicle at a randomized position on the map
RANDOM_EGO_START_POS = True







# ==============================================================================
# RL REWARD EXPERIMENTATION HUB
# ==============================================================================

# 1. The Progress Engine
# Target forward speed in meters/second for progress reward scaling
TARGET_SPEED_MS = 6.0
# Multiplier weight for the forward progress velocity reward.
# ULTIMATE UPGRADE: the progress term is now PEAKED, not saturating:
#   reward = clip(1 - |v_fwd - TARGET| / TARGET, 0, 1) * WEIGHT_PROGRESS
# Max at exactly TARGET_SPEED_MS, zero at standstill AND at 2x target — overspeeding
# (Mani's "pressing the gas pedal highly") is no longer free. See utils/reward_compiler.py.
WEIGHT_PROGRESS = 1.0

# 2. The Alignment Engine
# Multiplier weight for staying close to the center line of the lane
WEIGHT_CENTERING = 0.3         
# Curve steepness coefficient determining how quickly lane deviation drops centering reward
LANE_ALPHA = 0.5             
# Multiplier weight for reward based on alignment with the road's heading direction
WEIGHT_HEADING = 0.2           

# 3. The Control Penalty (Shock Absorbers)
# Penalty factor applied to the step reward for large steer changes (prevents erratic turning)
PENALTY_STEER_DELTA = 0.4     
# Penalty factor applied to the step reward for large throttle inputs changes (prevents rapid surging)
PENALTY_THROTTLE_DELTA = 0.15  
# Penalty applied when the agent presses throttle and brake pedals at the same time
PENALTY_PEDAL_OVERLAP = -0.5   

# 4. Terminals and Violations
# Terminal penalty/negative reward applied when a vehicle collision occurs
PENALTY_TERMINAL_CRASH = -10.0
# Step penalty/tax applied when crossing over lane markings/boundaries
PENALTY_LANE_INVASION = 0.0   # Tax for crossing the solid white line
# Step penalty/tax applied when vehicle velocity goes in the reverse direction
PENALTY_ROLLING_BACKWARD = -0.5 
# Velocity limit (m/s) below which the ego vehicle is considered static/stalled
STALL_SPEED_THRESHOLD = 1.0    
# Step penalty applied when vehicle remains stalled below threshold (discourages standing still)
PENALTY_STALLING = -0.5        
# 5. Proximity / headway shaping (anti-tailgate, pro-overtake)

# Max per-step penalty when bumper-to-bumper with the vehicle ahead (linear ramp from 0)
PENALTY_PROXIMITY = -0.8
# Center-to-center distance (m) where the proximity penalty starts (12 m centers ≈ 7.5 m gap)
PROXIMITY_DIST_M = 12.0
# Half-width (m) of the forward corridor in which a vehicle counts as "lead" (ego lane only)
PROXIMITY_LAT_HALFWIDTH_M = 2.0

# Overall scaling factor multiplier applied to the final calculated step reward
REWARD_SCALE_FACTOR = 1.0
