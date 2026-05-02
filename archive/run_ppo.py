import os
import sys
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from IPython.display import clear_output
from imitation.record_demos import *
from CarlaEnv.env import CarlaEnv


# Add the project root to sys.path to easily import the CarlaEnv module
sys.path.append(str(Path(__file__).resolve().parent.parent))


def create_checkpoints_folder(base_path='./checkpoints/checkpoint'):
    folder_index = 0
    while True:
        folder_name = f"{base_path}{folder_index}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        folder_index += 1

def get_latest_checkpoint(base_path='./checkpoints/checkpoint'):
    folder_index = 0
    folder_name=""
    while True:
        folder_name = f"{base_path}{folder_index}"
        if not os.path.exists(folder_name):
            break
        folder_index += 1
    
    file_num=1
    while True:
        file_name = f"{base_path}{folder_index-1}/ppo_carla_checkpoint_{file_num}000_steps.zip"
        if not os.path.exists(file_name):
            if file_num==1 and folder_index != 0:
                folder_index-=1
                continue
            else:
                break
        file_num += 1
    if file_num==1 and folder_index==0:
        return ""
    return f"{base_path}{folder_index-1}/ppo_carla_checkpoint_{file_num-1}000_steps.zip"

def run(map_path, walkers_count, vehicles_count, steps, device, init_speed, manual_mode=False):
    env = CarlaEnv(map_path, walkers_count, vehicles_count, max_steps=steps, init_speed=init_speed)
    
    if manual_mode:
        # Use the manual controller
        controller = ManualController(env)
        controller.run()
        
    else :
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        model_path = "ppo_carla_model"
        latest_checkpoint=get_latest_checkpoint()
        if latest_checkpoint != "":
            print(f"Loading existing model: {latest_checkpoint}")
            model = PPO.load(latest_checkpoint, verbose=2, env=env, n_epochs=10, batch_size=128, learning_rate=3e-4, clip_range=0.2)
        else:
            print("Creating new model...")
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=2,
                tensorboard_log="./ppo_carla/",
                n_epochs=10,
                batch_size=128,
                learning_rate=3e-4,
                clip_range=0.2,
            )    
        try:
            checkpoints_folder = create_checkpoints_folder()
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints_folder, name_prefix='ppo_carla_checkpoint')

            # model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log="./ppo_carla/")
            model.learn(total_timesteps=steps, callback=checkpoint_callback)
            print ("saving model")
            model.save("ppo_carla_model")
        except ...:
            print(f"Error during model training: ")
            
# map_path = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/LoadOpenDrive2/lab-map.xodr"
if __name__ == "__main__":
    map_path = r"C:\carla\Carla-Integration-Modules\LoadOpenDrive2\harder.xodr"
    run(map_path, 0, 0, 2000000, "cuda", 0)
