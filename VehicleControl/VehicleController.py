import carla
import enum
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

    def exec_command(self, command):
        if command == Command.SPEED_UP:
            self.control.throttle = min(self.control.throttle + 0.1, 1.0)
            self.control.brake = 0.0
        elif command == Command.SPEED_DOWN:
            self.control.throttle = max(self.control.throttle - 0.1, 0.0)
            self.control.brake = 0.2
        elif command == Command.TURN_RIGHT:
            self.control.steer = min(self.control.steer + 0.1, 1.0)
        elif command == Command.TURN_LEFT:
            self.control.steer = max(self.control.steer - 0.1, -1.0)
        elif command == Command.STOP:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        else:
            print("Unknown command!")
            
        self.vehicle.apply_control(self.control)
        print(f"Command: {command} | Throttle: {self.control.throttle:.2f}, Steer: {self.control.steer:.2f}")
