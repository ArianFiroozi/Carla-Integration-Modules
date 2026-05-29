import carla
import json
import numpy as np
from pathlib import Path
from config import general_config as config
from utils.reward_compiler import compile_reward
# The Progress Engine
TARGET_SPEED_MS = config.TARGET_SPEED_MS       
WEIGHT_PROGRESS = config.WEIGHT_PROGRESS         

# The Alignment Engine
WEIGHT_CENTERING = config.WEIGHT_CENTERING
LANE_ALPHA = config.LANE_ALPHA             
WEIGHT_HEADING = config.WEIGHT_HEADING         

# The Control Penalty (Shock Absorbers)
PENALTY_STEER_DELTA = config.PENALTY_STEER_DELTA     
PENALTY_THROTTLE_DELTA = config.PENALTY_THROTTLE_DELTA  
PENALTY_PEDAL_OVERLAP = config.PENALTY_PEDAL_OVERLAP  

# Terminals and Violations (Safety Net)
PENALTY_TERMINAL_CRASH =config.PENALTY_TERMINAL_CRASH
PENALTY_LANE_INVASION = config.PENALTY_LANE_INVASION
PENALTY_ROLLING_BACKWARD = config.PENALTY_ROLLING_BACKWARD
STALL_SPEED_THRESHOLD = config.STALL_SPEED_THRESHOLD
PENALTY_STALLING = config.PENALTY_STALLING

# ==============================================================================
# ACTION SPACE CONSTANTS
# ==============================================================================
SPEED_UP = 0
SPEED_DOWN = 1
STOP = 2
REVERSE = 3
CONSTANT = 4

TURN_RIGHT = 0
TURN_LEFT = 1
DO_NOT_TURN = 2
GO_STRAIGHT = 3

THROTTLE_STEP = 0.05        
MAX_THROTTLE = 1.0         
BRAKE_TAP = 0.10            
BRAKE_FULL = 1.00           
REVERSE_THROTTLE = 0.30     

STEER_STEP = 0.05           
MAX_STEER = 1.0         


class VehicleController():
    def __init__(self, world, vehicle=None):
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.vehicle = vehicle
        if vehicle is not None:
            self.__init_control()
        else:
            self.__spawn_vehicle()
        self.__init_reward_sensors()

        # State trackers for smoothness penalties
        self.prev_steer = 0.0
        self.prev_throttle = 0.0

    def __init_control(self):
        self.control = carla.VehicleControl()
        self.control.throttle = 0.0
        self.control.steer = 0.0
        self.control.brake = 0.0

    def __spawn_vehicle(self):
        try:
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            spawn_point = self.world.get_map().get_spawn_points()[0]
            spawn_point.rotation.yaw += 180
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.__init_control()
        except:
            print("Unknown error occured during spawn")

    def __init_reward_sensors(self):
        blueprint_library = self.world.get_blueprint_library()

        self.collision_happened = False
        self.lane_invaded = False

        def collision_callback(event):
            self.collision_happened = True

        def lane_callback(event):
            self.lane_invaded = True

        collision_sensor = blueprint_library.find('sensor.other.collision')
        lane_sensor = blueprint_library.find('sensor.other.lane_invasion')

        self.sensor_c = self.world.spawn_actor(collision_sensor, carla.Transform(), attach_to=self.vehicle)
        self.sensor_l = self.world.spawn_actor(lane_sensor, carla.Transform(), attach_to=self.vehicle)

        self.sensor_c.listen(collision_callback)
        self.sensor_l.listen(lane_callback)


    def get_reward(self, observation=None):
        info = {}

        # --- TELEMETRY GATHERING ---
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        car_forward = transform.get_forward_vector()
        
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        wp_forward = waypoint.transform.get_forward_vector()
        lane_center = waypoint.transform.location
        vehicle_loc = self.vehicle.get_location()

        # --- POPULATE RAW FACTS  ---
        # Terminals & Violations
        info['is_terminal_crash'] = int(self.collision_happened or vehicle_loc.z <= -5)
        info['is_lane_invaded'] = int(self.lane_invaded)
        info['is_pedal_overlap'] = int(self.control.throttle > 0.1 and self.control.brake > 0.1)
        
        # Vectors (Saved as raw lists/arrays)
        info['velocity_x'] = float(velocity.x)
        info['velocity_y'] = float(velocity.y)
        info['velocity_z'] = float(velocity.z)
        
        info['car_forward_x'] = float(car_forward.x)
        info['car_forward_y'] = float(car_forward.y)
        info['car_forward_z'] = float(car_forward.z)
        
        info['road_forward_x'] = float(wp_forward.x)
        info['road_forward_y'] = float(wp_forward.y)
        
        # Positions
        info['vehicle_loc_x'] = float(vehicle_loc.x)
        info['vehicle_loc_y'] = float(vehicle_loc.y)
        info['lane_center_x'] = float(lane_center.x)
        info['lane_center_y'] = float(lane_center.y)
        
        # Controls
        info['steer_change'] = float(abs(self.control.steer - self.prev_steer))
        info['throttle_change'] = float(abs(self.control.throttle - self.prev_throttle))

        # --- RESET SENSORS & STATE ---
        if info['is_terminal_crash'] == 1:
            self.collision_happened = False
            self.lane_invaded = False
        if info['is_lane_invaded'] == 1:
            self.lane_invaded = False
            
        self.prev_steer = self.control.steer
        self.prev_throttle = self.control.throttle

        # --- CALL THE COMPILER ---
        reward , _ = compile_reward(info, config, is_tensor=False)
        
        return reward, info
    # ==============================================================================
    # ACTION EXECUTION METHODS
    # ==============================================================================

    def speed_action_convertor(self, speed_action):
        if speed_action == SPEED_UP:
            return 0
        elif speed_action == SPEED_DOWN:
            return 1
        elif speed_action == STOP:
            return 4
        elif speed_action == REVERSE:
            return 7
        elif speed_action == CONSTANT:
            return 8
        else:
            print(f"speed_action unknown: {speed_action}")
            return -1
        
    def turn_action_convertor(self, turn_action):
        if turn_action == TURN_RIGHT:
            return 2
        elif turn_action == TURN_LEFT:
            return 3
        elif turn_action == DO_NOT_TURN:
            return 5
        elif turn_action == GO_STRAIGHT:
            return 6
        else:
            print(f"turn_action unknown: {turn_action}")
            return -1

    def exec_command(self, command):
        if command == 0:  # SPEED_UP
            self.control.throttle = min(self.control.throttle + THROTTLE_STEP, MAX_THROTTLE)
            self.control.brake = 0.0
            self.control.reverse = False
        elif command == 1:  # SPEED_DOWN
            self.control.throttle = max(self.control.throttle - THROTTLE_STEP, 0.0)
            self.control.brake = BRAKE_TAP
            self.control.reverse = False
        elif command == 4:  # STOP
            self.control.reverse = False
            self.control.throttle = 0.0
            self.control.brake = BRAKE_FULL
        elif command == 7:  # REVERSE
            self.control.reverse = True
            self.control.throttle = REVERSE_THROTTLE
            self.control.brake = 0.0
        elif command == 8:  # CONSTANT_SPEED
            self.control.brake = 0.0
        elif command == 2:  # TURN_RIGHT
            self.control.steer = min(self.control.steer + STEER_STEP, MAX_STEER)
        elif command == 3:  # TURN_LEFT
            self.control.steer = max(self.control.steer - STEER_STEP, -MAX_STEER)
        elif command == 5:  # DO_NOT_TURN
            pass
        elif command == 6:  # GO_STRAIGHT
            self.control.steer = 0.0
        else:
            print(f"Unknown command : {command}")

        self.vehicle.apply_control(self.control)

    def exec_continuous_command(self, throttle, brake, steer):
        """Cleanly execute a direct continuous action array from CarlaEnv"""
        self.control.throttle = float(np.clip(throttle, 0.0, 1.0))
        self.control.brake = float(np.clip(brake, 0.0, 1.0))
        self.control.steer = float(np.clip(steer, -1.0, 1.0))
        self.control.reverse = False  
        
        self.vehicle.apply_control(self.control)
        
    def exec_delta_command(self, throttle_action, steer_action):
        throttle = self.control.throttle
        throttle_change = throttle_action * 0.092
        
        if (not self.control.reverse and throttle + throttle_change < 0):
            self.control.throttle = -(throttle + throttle_change)
            self.control.reverse = True 
        elif (not self.control.reverse):
            self.control.throttle = throttle + throttle_change
        elif (self.control.reverse and throttle_change >= 0):
            if (throttle_change >= throttle):
                self.control.throttle = throttle_change - throttle
                self.control.reverse = False
            else: 
                self.control.throttle = throttle - throttle_change
        elif (self.control.reverse and throttle_change < 0):
            throttle = throttle - throttle_change
            
        self.control.throttle = min(self.control.throttle, 0.92)
        
        new_steer = self.control.steer + steer_action * 0.04
        self.control.steer = max(min(new_steer, 0.4), -0.4)
        
        self.vehicle.apply_control(self.control)