import carla
import random

def spawn_vehicles(client, num_vehicles=1):

    world = client.get_world()
    # traffic_manager = client.get_trafficmanager()
    # traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for i in range(num_vehicles):
        blueprint = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle:
            vehicle.apply_control(carla.VehicleControl(throttle=0.5))
            vehicles.append(vehicle)
            # traffic_manager.vehicle_percentage_speed_difference(vehicle, random.randint(-30, 30))
            print(f"Spawned vehicle {vehicle.id} at {spawn_point.location}")

    return vehicles
