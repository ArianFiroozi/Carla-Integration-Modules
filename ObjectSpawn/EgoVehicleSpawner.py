import carla

def spawn_ego_vehicle(world, init_speed=0):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    control = carla.VehicleControl()
    control.throttle = init_speed
    control.steer = 0.0
    control.brake = 0.0

    vehicle.apply_control(control)
    return vehicle