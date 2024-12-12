from ObservationAdaptors import *
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
ego_vehicle = world.get_actors().filter('vehicle.*')[0]

speed_matrix = get_speed_matrices(ego_vehicle, lanes=2, sections=6)

print(speed_matrix)
