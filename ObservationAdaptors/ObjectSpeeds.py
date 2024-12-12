import carla
import numpy as np

def get_speed_matrices(ego_vehicle, lanes=2, sections=6, lane_width=4.0, section_length=1.0):
    """
    Generate a speed matrix for nearby objects.

    Args:
        ego_vehicle: The ego vehicle actor.
        lanes (int): Number of lanes.
        sections (int): Number of sections per lane.
        lane_width (float): Width of each lane (meters).
        section_length (float): Length of each section (meters).
    
    Returns:
        1. np.ndarray: A matrix where each cell contains the speed of objects in that lane and section on x axis.
        2. np.ndarray: A matrix where each cell contains the speed of objects in that lane and section on y axis.
        3. np.ndarray: A matrix where each cell contains presence of objects in that lane and section.
    """

    x_speed_matrix = np.zeros((lanes, sections))
    y_speed_matrix = np.zeros((lanes, sections))
    presence_matrix = np.zeros((lanes, sections))

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    world = ego_vehicle.get_world()
    actors = world.get_actors()
    dynamic_objects = actors.filter('vehicle.*') + actors.filter('walker.*')
    
    for obj in dynamic_objects:
        if obj.id == ego_vehicle.id:
            continue

        obj_transform = obj.get_transform()
        obj_velocity = obj.get_velocity()

        dx = obj_transform.location.x - ego_location.x
        dy = obj_transform.location.y - ego_location.y

        lane_index = int((dy + lane_width * lanes / 2) // lane_width) # fix if you want to drift inbetween the lanes
        if lane_index < 0 or lane_index >= lanes:
            continue

        section_index = int((dx + sections * section_length / 2) // section_length)
        if section_index < 0 or section_index >= sections:
            continue

        x_speed_matrix[lane_index, section_index] += obj_velocity.x
        y_speed_matrix[lane_index, section_index] += obj_velocity.y
        presence_matrix[lane_index, section_index] = 1

    return x_speed_matrix, y_speed_matrix, presence_matrix