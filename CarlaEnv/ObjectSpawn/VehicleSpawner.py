import carla
import random

def spawn_vehicles(client, num_vehicles=0, random_spawn=True):

    world = client.get_world()
    traffic_manager = client.get_trafficmanager()
    speed_profiles = [

    ## -- FOR FAST PACED DATASET
    # (25, 45), # very slow traffic (55–75% of speed limit)
    # (10, 25), # slow/near-limit traffic (75–90% of speed limit)
    # (-15, -5), # slightly fast (105–115% of speed limit)
    
    
    (30, 50),   # extremely slow traffic
    (40, 60),   # ultra slow 
    (50, 70),   # crawling traffic 

    

    ]
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



        speed_min, speed_max = random.choice(speed_profiles)
        speed_diff = random.randint(speed_min, speed_max)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
        
        
        # traffic_manager.random_left_lanechange_percentage(vehicle, random.randint(0, 30))
        # traffic_manager.random_right_lanechange_percentage(vehicle, random.randint(0, 30))
        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(2.0, 6.0))



        vehicles.append(vehicle)

    return vehicles
