import carla
import enum
import json

class Command(enum.Enum):
    SPEED_UP=0,
    SPEED_DOWN=1,
    TURN_RIGHT=2,
    TURN_LEFT=3,
    STOP=4

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
            print("Vehicle spawned!")
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

    def get_reward(self):
        reward = 0.0

        velocity = self.vehicle.get_velocity()
        speed = 3.6 * ((velocity.x**2 + velocity.y**2)**0.5) ##km/h
        reward += speed * self.speed_reward

        if self.collision_happened:
            reward -= self.collision_penalty

        if self.lane_invaded:
            reward -= self.lane_penalty

        
        self.collision_happened = False
        self.lane_invaded = False

        return reward

    def exec_command(self, command):
        print(command, Command.SPEED_UP.value)
        if command == 0:#Command.SPEED_UP.value[0]:
            self.control.throttle = min(self.control.throttle + 0.3, 1.0)
            self.control.brake = 0.0
        elif command == 1:#Command.SPEED_DOWN.value[0]:
            self.control.throttle = max(self.control.throttle - 0.3, 0.0)
            self.control.brake = 0.2
        elif command == 2:#Command.TURN_RIGHT.value[0]:
            self.control.steer = min(self.control.steer + 0.1, 1.0)
        elif command == 3:#Command.TURN_LEFT.value[0]:
            self.control.steer = max(self.control.steer - 0.1, -1.0)
        elif command == 4:#Command.STOP.value[0]:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        else:
            print("Unknown command!")
            
        self.vehicle.apply_control(self.control)
        print(f"Command: {command} | Throttle: {self.control.throttle:.2f}, Steer: {self.control.steer:.2f}")
