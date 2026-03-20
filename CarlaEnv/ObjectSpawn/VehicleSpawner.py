import carla
import random

def spawn_vehicles(client, num_vehicles=0, random_spawn=True):

    world = client.get_world()
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()

    vehicles = []

    if random_spawn:
        random.shuffle(spawn_points)

    for i, spawn_point in enumerate(spawn_points):

        if len(vehicles) >= num_vehicles:
            break

        blueprint = random.choice(vehicle_blueprints)

        vehicle = world.try_spawn_actor(blueprint, spawn_point)

        if vehicle is None:
            continue

        vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.vehicle_percentage_speed_difference(
            vehicle,
            random.randint(-30, 30)
        )

        vehicles.append(vehicle)

    return vehicles
