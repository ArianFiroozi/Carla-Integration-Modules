from ObservationAdaptors import *
from VehicleControl import *
from LoadOpenDrive2 import *
from ObjectSpawn import *
from LoadOpenDrive2 import *
import time

import carla, random

load_opendrive_map("./LoadOpenDrive2/simple_map.xodr")

client = carla.Client('localhost', 2000)
world = client.get_world()

ego_vehicle=spawn_ego_vehicle(world)
# spawn_pedestrians(world, 5)
spawn_vehicles(client, 5)

vehicleController=VehicleController(world, ego_vehicle)

while (True):
    world.tick()
    # spectator = world.get_spectator() 
    # transform = ego_vehicle.get_transform() 
    # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))) 
    time.sleep(0.5)

    vehicleController.exec_command(random.randint(0, 4))
    x_speed_matrix, y_speed_matrix, presence_matrix = get_speed_matrices(ego_vehicle, lanes=2, sections=6)
    # lane_angle = get_lane_angle(ego_vehicle)
    # traffic_signs = get_nearby_signs(ego_vehicle, radius=10)

    print("reward:", vehicleController.get_reward())
    # print("lane angle:", lane_angle)
    # for sign in traffic_signs:
    #     print(f"Traffic Sign: {sign.id}, Type: {sign.type}, Location: {sign.transform.location}")
    print("presence:\n", presence_matrix)
    print('-------------------------------------------------------------------------------')
