import carla
import random 

def spawn_ego_vehicle(world, init_speed=0):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]

    spawn_points = world.get_map().get_spawn_points()

    vehicle=None
    shuffeled_idx = [i for i in range(len(spawn_points))]
    random.shuffle(shuffeled_idx)
    for i in shuffeled_idx:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
        if vehicle!=None:
            break
        else:
            print("ego vehicle spawn failed! retrying...")
    if vehicle==None:
        raise Exception(0)

    
    control = carla.VehicleControl()
    control.throttle = init_speed
    control.steer = 0.0
    control.brake = 0.0

    vehicle.apply_control(control)
    return vehicle