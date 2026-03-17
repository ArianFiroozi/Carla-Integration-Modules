import carla
import random 

def spawn_ego_vehicle(world, init_speed=0):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]




    # spawn_points = world.get_map().get_spawn_points()
    # vehicle=None
    # shuffeled_idx = [i for i in range(len(spawn_points))]
    # random.shuffle(shuffeled_idx)
    # for i in shuffeled_idx:
    #     vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
    #     if vehicle!=None:
    #         break
    #     else:
    #         print("ego vehicle spawn failed! retrying...")
            
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]
    # shift vehicle slightly to the left (toward lane center)
    right_vec = spawn_point.get_right_vector()
    spawn_point.location.x -= right_vec.x * 2
    spawn_point.location.y -= right_vec.y * 2

    vehicle = None
    for _ in range(10):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            break
        world.tick()

    if vehicle is None:
        raise Exception("ego vehicle spawn failed")


    
    control = carla.VehicleControl()
    control.throttle = init_speed
    control.steer = 0.0
    control.brake = 0.0

    vehicle.apply_control(control)
    return vehicle





