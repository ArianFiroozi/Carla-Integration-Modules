import carla
import random
import time

def spawn_pedestrians(world, num_pedestrians=1, seed=None):
    rng = random.Random(seed)
    blueprint_library = world.get_blueprint_library()
    walkers = []
    num_spawned_peds = 0
    while num_spawned_peds < num_pedestrians:
        bp = rng.choice(blueprint_library.filter('walker'))
        spawn_point = rng.choice(world.get_map().get_spawn_points())
        pedestrain = world.try_spawn_actor(bp, spawn_point)
        if pedestrain:
            num_spawned_peds += 1
            walkers.append(pedestrain)
    return walkers
   
def step_peds(world, walkers, rng=None):
    """
    Step pedestrians with random walker control.
    
    Args:
        world: CARLA world
        walkers: list of walker actors
        rng: random.Random instance (optional). If None, uses global random.
    """
    if rng is None:
        rng = random  # fall back to global random module
    
    blueprint_library = world.get_blueprint_library()
    control = carla.WalkerControl()
    for i, pedestrain in enumerate(walkers):
        if (pedestrain.get_location().z < -10):
            new_guy = None
            while(new_guy is None):
                bp = rng.choice(blueprint_library.filter('walker'))
                spawn_point = rng.choice(world.get_map().get_spawn_points())
                new_guy = world.try_spawn_actor(bp, spawn_point)

            walkers[i].destroy()
            walkers[i] = new_guy
            
        try:
            control.speed = rng.uniform(0.5, 1.0)
            control.direction.y = rng.choice([1, -1])
            control.direction.x = rng.choice([1, -1])
            control.direction.z = 0
            pedestrain.apply_control(control)

        except Exception as e:
            pass