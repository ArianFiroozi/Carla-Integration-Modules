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