import carla
import random
import time

def spawn_pedestrians(world, num_pedestrians=1):
    blueprint_library = world.get_blueprint_library()
    walkers = []
    num_spawned_peds=0
    while num_spawned_peds<num_pedestrians:
        bp = random.choice(blueprint_library.filter('walker'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        pedestrain = world.try_spawn_actor(bp, spawn_point)
        if pedestrain:
            num_spawned_peds+=1
            walkers.append(pedestrain)
            #print(f"Spawned pedestrain {pedestrain.id} at {spawn_point.location}")
    return walkers
   
def step_peds(world, walkers):
    blueprint_library = world.get_blueprint_library()
    control = carla.WalkerControl()
    for i, pedestrain in enumerate(walkers):
        #print(f'ped {i} before if')//TODO destroy fallen peds
        if (pedestrain.get_location().z<-10):
            new_guy=None
            while(new_guy==None):
                bp = random.choice(blueprint_library.filter('walker'))
                spawn_point = random.choice(world.get_map().get_spawn_points())
                new_guy = world.try_spawn_actor(bp, spawn_point)

            walkers[i]=new_guy
            #print("we have a new folk")
        #print(f'ped {i} after if')
        try:
            control.speed = random.uniform(0.5, 1.0)
            control.direction.y = random.choice([1,-1])
            control.direction.x = random.choice([1,-1])
            control.direction.z = 0
            pedestrain.apply_control(control)

        except Exception as e:
            pass
            # print("nashod jadidesho sakhtam")