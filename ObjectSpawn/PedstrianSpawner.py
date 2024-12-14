import carla
import random

def spawn_pedestrians(world, num_pedestrians=1):
    blueprint_library = world.get_blueprint_library()

    walker_bp = blueprint_library.filter("walker.pedestrian.*")
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    spawn_points = []
    for _ in range(num_pedestrians):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        if spawn_point.location:
            spawn_points.append(spawn_point)

    walkers = []
    controllers = []
    for spawn_point in spawn_points:
        walker = world.try_spawn_actor(random.choice(walker_bp), spawn_point)
        if walker:
            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            controllers.append(controller)
            walkers.append(walker)

    for controller in controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.5)

    print(f"Spawned {len(walkers)} pedestrians.")
   
