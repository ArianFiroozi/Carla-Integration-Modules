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
        # Per-episode cache of other vehicles for the lead-gap scan (the controller is
        # recreated on every env.reset, so the actor list is stable for its lifetime)
        self._nearby_vehicles = None
        # Raw (pre-exclusivity) policy pedals; env.step sets these each tick so the reward can
        # punish a throttle+brake hedge before the env zeroes the throttle.
        self.raw_throttle = 0.0
        self.raw_brake = 0.0

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
        self.last_collision = None   # forensics dict describing the most recent impact

        def collision_callback(event):
            self.collision_happened = True
            # COLLISION FORENSICS (headless debugging): classify who caused the impact so the
            # episode JSONs distinguish at-fault crashes (policy's problem) from NPC-caused
            # ones (environment's problem). Runs on the sensor thread — never raise from here.
            try:
                other = event.other_actor
                other_type = other.type_id if other is not None else "unknown"
                ego_tr = self.vehicle.get_transform()
                fwd = ego_tr.get_forward_vector()
                eloc = ego_tr.location
                oloc = other.get_location() if other is not None else eloc
                dx, dy = oloc.x - eloc.x, oloc.y - eloc.y
                lon = dx * fwd.x + dy * fwd.y   # impact source position along ego heading
                vel = self.vehicle.get_velocity()
                ego_speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

                if not other_type.startswith("vehicle."):
                    verdict = "at_fault_static"   # wall / pole / off-road object: lane-keeping failure
                elif lon > 1.0:
                    verdict = "at_fault_front"    # we drove into a vehicle ahead: following/merging failure
                elif lon < -1.0:
                    verdict = "npc_rear_end"      # a vehicle drove into our rear: likely blameless
                else:
                    verdict = "side_contact"      # lateral scrape: lane-change conflict, ambiguous

                self.last_collision = {
                    "verdict": verdict,
                    "other_type": other_type,
                    "impact_lon_m": round(float(lon), 2),
                    "ego_speed_ms": round(float(ego_speed), 2),
                    "loc_x": round(float(eloc.x), 1),
                    "loc_y": round(float(eloc.y), 1),
                }
            except Exception:
                self.last_collision = {"verdict": "unknown", "other_type": "unknown"}

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
        # Judge pedal overlap on the RAW policy action (before the env zeroed the throttle), so
        # the agent is punished for the throttle+brake hedge instead of escaping it for free.
        info['is_pedal_overlap'] = int(self.raw_throttle > 0.1 and self.raw_brake > 0.1)
        
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

        # Proximity shaping input: center-to-center distance to the nearest vehicle in the
        # forward ego-lane corridor (see PROXIMITY_* in general_config). 999 = nothing ahead.
        info['lead_gap_m'] = self._lead_vehicle_gap()

        # Attach collision forensics on the terminal step (see collision_callback)
        if info['is_terminal_crash'] == 1:
            info['collision_forensics'] = self.last_collision or {"verdict": "unknown",
                                                                  "other_type": "off_map_fall"}

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
    def _lead_vehicle_gap(self, max_dist=30.0):
        """
        Center-to-center longitudinal distance (m) to the nearest other vehicle inside the
        forward corridor (|lateral| < PROXIMITY_LAT_HALFWIDTH_M). Returns 999.0 when the
        corridor is clear (beyond max_dist the proximity penalty is zero anyway).
        """
        try:
            if self._nearby_vehicles is None:
                self._nearby_vehicles = [a for a in self.world.get_actors().filter("vehicle.*")
                                         if a.id != self.vehicle.id]
            if not self._nearby_vehicles:
                return 999.0
            tr = self.vehicle.get_transform()
            loc = tr.location
            fwd = tr.get_forward_vector()
            best = None
            for v in self._nearby_vehicles:
                vloc = v.get_location()
                dx, dy = vloc.x - loc.x, vloc.y - loc.y
                lon = dx * fwd.x + dy * fwd.y            # distance ahead along heading
                lat = -dx * fwd.y + dy * fwd.x           # lateral offset from heading axis
                if 0.0 < lon < max_dist and abs(lat) < config.PROXIMITY_LAT_HALFWIDTH_M:
                    if best is None or lon < best:
                        best = lon
            return float(best) if best is not None else 999.0
        except Exception:
            return 999.0   # actor churn mid-query: no penalty rather than a crashed step

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