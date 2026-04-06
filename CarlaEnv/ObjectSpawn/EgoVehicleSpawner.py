import carla
import random


def spawn_ego_vehicle(world,
                      init_speed=0.0,
                      random_spawn=True,
                      spawn_index=0,
                      max_retries=100):

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*model3*")[0]

    spawn_points = world.get_map().get_spawn_points()

    vehicle = None

    # RANDOM SPAWN
    if random_spawn:

        indices = list(range(len(spawn_points)))
        random.shuffle(indices)

        for i in indices[:max_retries]:

            spawn_point = spawn_points[i]

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at random spawn points")
        else:
            print("Spawned ego vehicle at random spawn point")

    # DETERMINISTIC SPAWN
    else:

        spawn_point = spawn_points[spawn_index]

        # small lateral offset (optional)
        right_vec = spawn_point.get_right_vector()
        spawn_point.location.x -= right_vec.x * 2
        spawn_point.location.y -= right_vec.y * 2

        for _ in range(max_retries):

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

            if vehicle is not None:
                break

            world.tick()

        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at deterministic spawn")
        else:
            print("Spawned ego vehicle at deterministic spawn")

    control = carla.VehicleControl(
        throttle=float(init_speed),
        steer=0.0,
        brake=0.0
    )

    vehicle.apply_control(control)

    return vehicle
