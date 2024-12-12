from ObservationAdaptors import *
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
ego_vehicle = world.get_actors().filter('vehicle.*')[0]

x_speed_matrix, y_speed_matrix, presence_matrix = get_speed_matrices(ego_vehicle, lanes=2, sections=6)
lane_angle = get_lane_angle(ego_vehicle)
traffic_signs = get_nearby_signs(ego_vehicle, radius=10)

print(presence_matrix)
print(lane_angle)
for sign in traffic_signs:
    print(f"Traffic Sign: {sign.id}, Type: {sign.type}, Location: {sign.transform.location}")
