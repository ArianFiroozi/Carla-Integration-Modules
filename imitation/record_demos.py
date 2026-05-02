import time
from pathlib import Path
import argparse
import keyboard
import numpy as np
import carla
from .controllers.autopilot_controller import AutopilotController
from .controllers.manual_controller import ManualController
from config import bc_config













if __name__ == "__main__":
    from CarlaEnv.env import CarlaEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=bc_config.CARLA_MAP_PATH)
    parser.add_argument("--walkers", type=int, default=bc_config.CARLA_WALKERS)
    parser.add_argument("--vehicles", type=int, default=bc_config.CARLA_VEHICLES)
    parser.add_argument("--max-steps", type=int, default=bc_config.CARLA_MAX_STEPS)
    parser.add_argument("--init-speed", type=float, default=bc_config.CARLA_INIT_SPEED)
    parser.add_argument("--mode", type=str, default=bc_config.RECORD_DRIVE_MODE)
    parser.add_argument("--action-mode", type=str, default=bc_config.ACTION_MODE)
    parser.add_argument("--demo-dir", type=str, default=bc_config.MANUAL_RECORD_DIR)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--base-name", type=str, default=bc_config.MANUAL_BASE_NAME)
    parser.add_argument("--episodes", type=int, default=bc_config.DEFAULT_AUTOPILOT_EPISODES)
    args = parser.parse_args()
    env = CarlaEnv(
        map_path=args.map,
        walkers_count=args.walkers,
        vehicles_count=args.vehicles,
        max_steps=args.max_steps,
        init_speed=args.init_speed,
        action_mode=args.action_mode,
        random_ego_spawn=bc_config.RANDOM_EGO_START_POS,
        random_vehicle_spawn=bc_config.RANDOM_VEHICLE_START_POS
    )

    if args.mode == "autopilot":
        print("[INFO] Initializing Autopilot Controller...")
        controller = AutopilotController(
            env,
            record_dir=str(bc_config.AUTOPILOT_RECORD_DIR), 
            base_name=bc_config.AUTOPILOT_DEMO_BASENAME, 
            max_steps=args.max_steps
        )
        controller.run(episodes=args.episodes) 
        exit(0)

    elif args.mode == "manual":
        print("[INFO] Initializing Manual Controller...")
        controller = ManualController(
            env,
            demo_dir=args.demo_dir
        )
        record_flag = args.record if args.record else bc_config.MANUAL_RECORD
        controller.run(
            record=record_flag,
            base_name=args.base_name
        )
    else:
        print(f"[ERROR] Unknown mode '{args.mode}'. Use 'autopilot' or 'manual'.")
        exit(1)
    