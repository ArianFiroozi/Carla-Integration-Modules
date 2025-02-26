import carla
import enum
import json

class Command(enum.Enum):
    SPEED_UP = 0
    SPEED_DOWN = 1
    TURN_RIGHT = 2
    REVERSE = 7
    CONSTANT_SPEED = 8

    TURN_LEFT = 3
    STOP = 4
    DO_NOT_TURN = 5
    GO_STRAIGHT=6

    
SPEED_UP = 0
SPEED_DOWN = 1
STOP = 2
REVERSE = 3
CONSTANT = 4

TURN_RIGHT = 0
TURN_LEFT = 1
DO_NOT_TURN = 2
GO_STRAIGHT = 3

class VehicleController():
    def __init__(self, world, vehicle=None):
        self.world=world
        self.blueprint_library = world.get_blueprint_library()
        self.vehicle = vehicle
        if vehicle is not None:
            self.__init_control()
        else:
            self.__spawn_vehicle()
        self.__init_reward_sensors()
        self.get_config()

    def __init_control(self):
        self.control = carla.VehicleControl()
        self.control.throttle = 0.0
        self.control.steer = 0.0
        self.control.brake = 0.0

    def __spawn_vehicle(self):
        try:
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            spawn_point = self.world.get_map().get_spawn_points()[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.__init_control()
            # print("Vehicle spawned!")
        except:
            print("Unknown error occured")

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

    def get_config(self):
        with open("VehicleControl/reward_config.json", 'r') as file:
            config = json.load(file)

        self.speed_reward = config["speed_reward"]
        self.collision_penalty = config["collision_penalty"]
        self.lane_penalty = config["lane_penalty"]

    def get_reward(self, observation=None):
        reward = 0.0

        velocity = self.vehicle.get_velocity()
        # velocity *= 1 if self.control.reverse==False else -0.5
        speed = 3.6 * ((velocity.x**2 + velocity.y**2)**0.5) ##km/h
        reward += speed * self.speed_reward * (0.1 if self.control.reverse==False else -0.05)
        MIN_TRESH = 2
        # if (sp0eed < MIN_TRESH and speed > -MIN_TRESH):
        #     reward -=1

        # if (speed > 0 and observation["presence"][1][3]):
        #     reward -= 2
        # elif(speed < 0 and observation["presence"][1][1]):
        #     reward -= 2

        if self.collision_happened or self.vehicle.get_location().z <= -5:
            reward += self.collision_penalty
            print("Kalaps")
        # if self.lane_invaded: # DO NOT CLEAR THIS
        #     reward += self.lane_penalty

        
        self.collision_happened = False
        self.lane_invaded = False

        # print(reward)
        return reward
        # return 0
    
    def speed_action_convertor(self, speed_action):
        # speed_action//=3
        if speed_action == SPEED_UP:
            return Command.SPEED_UP.value
        elif speed_action == SPEED_DOWN:
            return Command.SPEED_DOWN.value
        elif speed_action==STOP:
            return Command.STOP.value
        elif speed_action==REVERSE:
            return Command.REVERSE.value
        elif speed_action==CONSTANT:
            return Command.CONSTANT_SPEED.value
        else:
            print(f"speed_action : {speed_action}")
            return -1
        
    def turn_action_convertor(self, turn_action):
        # turn_action//=3
        if turn_action == TURN_RIGHT:
            return Command.TURN_RIGHT.value
        elif turn_action == TURN_LEFT:
            return Command.TURN_LEFT.value
        elif turn_action == DO_NOT_TURN:
            return Command.DO_NOT_TURN.value
        elif turn_action == GO_STRAIGHT:
            return Command.GO_STRAIGHT.value
        else:
            print(f"turn_action : {turn_action}")
            return -1


    def exec_command(self, command):
        # print(f'taking action : {command}')
        if command == 0:#Command.SPEED_UP.value[0]:
            self.control.throttle = min(self.control.throttle + 0.3, 1.0)
            self.control.brake = 0.0
            self.control.reverse=False
        elif command == 1:#Command.SPEED_DOWN.value[0]:
            self.control.throttle = max(self.control.throttle - 0.3, 0)
            self.control.brake = 0.2
            self.control.reverse=False
        elif command == 2:#Command.TURN_RIGHT.value[0]:
            self.control.steer = min(self.control.steer + 0.2, 1.0)
        elif command == 3:#Command.TURN_LEFT.value[0]:
            self.control.steer = max(self.control.steer - 0.2, -1.0)
        elif command == 4:#Command.STOP.value[0]:
            self.control.reverse=False
            self.control.throttle = 0.0
            self.control.brake = 1.0
        elif command == 5:#Command.DO_NOT_TURN.value[0]:
            pass
        elif command == 6:#Command.DO_NOT_TURN.value[0]:
            self.control.steer = 0
        elif command == 7:#Command.DO_NOT_TURN.value[0]:
            self.control.reverse=True
            self.control.throttle=0.5
            self.control.brake = 0.0
        elif command == 8:
            pass
            
        else:
            print(f"Unknown command : {command}")
        self.vehicle.apply_control(self.control)
        # print(f"Command: {command} | Throttle: {self.control.throttle:.2f}, Steer: {self.control.steer:.2f}")
