import keyboard
import time
import carla
from VehicleControl import *

class ManualController:
    def __init__(self, env):
        """Initialize the manual controller with the CarlaEnv environment."""
        self.env = env
        self.world = env.world
        self.ego_vehicle = env.ego_vehicle
        self.done = False

        # Set up the spectator camera to follow the vehicle
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

    def run(self):
        """Run the manual control loop."""
        print("Manual control mode activated.")
        print("Controls: W (accelerate), S (brake), A (steer left), D (steer right)")
        print("Exit: Press 'q' | Reset after episode: Press 'r'")

        # Reset the environment to start
        obs, _ = self.env.reset()
        self.done = False

        while True:
            # Exit condition
            if keyboard.is_pressed('q'):
                print("Exiting manual control mode.")
                break

            # Default actions: no acceleration, no steering
            speed_action = 0
            turn_action = 0

            # Map keyboard inputs to actions
            if keyboard.is_pressed('w'):
                speed_action = Command.SPEED_UP  # Accelerate
            elif keyboard.is_pressed('s'):
                speed_action = Command.SPEED_DOWN  # Brake
            elif keyboard.is_pressed('space'):
                speed_action = Command.STOP  # stop
            elif keyboard.is_pressed('r'):
                speed_action = Command.REVERSE  # Brake
            if keyboard.is_pressed('a'):
                turn_action = Command.TURN_LEFT   # Steer left
            elif keyboard.is_pressed('d'):
                turn_action = Command.TURN_RIGHT   # Steer right
            elif keyboard.is_pressed('f'):
                turn_action = Command.GO_STRAIGHT  # Brake
            else:
                turn_action = Command.DO_NOT_TURN
            
            # Combine actions into a list compatible with env.step()
            action = [speed_action, turn_action]

            if not self.done:
                # Step the environment with the chosen action
                obs, reward, self.done, truncated, info = self.env.step(action)
                print(f"Step: {self.env.current_step}, Reward: {reward}, Done: {self.done}")
                print(f'obs is : {obs}')
            
            if self.done:
                # Episode ended, wait for reset or quit
                print("Episode ended. Press 'r' to reset or 'q' to quit.")
                while True:
                    if keyboard.is_pressed('r'):
                        obs, _ = self.env.reset()
                        self.done = False
                        print("Environment reset.")
                        break
                    if keyboard.is_pressed('q'):
                        print("Exiting manual control mode.")
                        return
                    time.sleep(0.1)  # Pause to avoid busy-waiting
            else:
                time.sleep(0.1)  # Control loop rate to match simulation