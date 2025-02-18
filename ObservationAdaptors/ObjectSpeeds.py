import carla
import numpy as np
import torch

def get_speed_matrices(ego_vehicle, matrix_length=3, matrix_width=6, cell_width=4.0, cell_length=1.0):
    """
    Generate a speed matrix for nearby objects and mark cells that are off the road.

    Args:
        ego_vehicle: The ego vehicle actor.
        matrix_length (int): Number of matrix_length.
        matrix_width (int): Number of matrix_width per lane.
        cell_width (float): Width of each lane (meters).
        cell_length (float): Length of each section (meters).
    
    Returns:
        1. np.ndarray: A matrix where each cell contains the speed of objects in that lane and section on x axis.
        2. np.ndarray: A matrix where each cell contains the speed of objects in that lane and section on y axis.
        3. np.ndarray: A matrix where each cell contains presence of objects in that lane and section.
    """

    x_speed_matrix = torch.zeros((matrix_length, matrix_width))
    y_speed_matrix = torch.zeros((matrix_length, matrix_width))
    presence_matrix = torch.zeros((matrix_length, matrix_width))

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    world = ego_vehicle.get_world()
    actors = world.get_actors()
    dynamic_objects = actors.filter('vehicle.*') #+ actors.filter('walker.*')
    
    # Function to check if a location is on the road
    def is_on_road(location):
        waypoint = world.get_map().get_waypoint(location, project_to_road=False)
        return waypoint is not None and not waypoint.is_intersection

    for obj in dynamic_objects:
        if obj.id == ego_vehicle.id:
            continue

        obj_transform = obj.get_transform()
        obj_velocity = obj.get_velocity()

        dx = obj_transform.location.x - ego_location.x
        dy = obj_transform.location.y - ego_location.y

        x_idx = int((dx + cell_width * matrix_width / 2) // cell_width) # fix if you want to drift inbetween the matrix_width
        if x_idx < 0 or x_idx >= matrix_width:
            continue

        y_idx = int((dy + matrix_length * cell_length / 2) // cell_length)
        if y_idx < 0 or y_idx >= matrix_length:
            continue

        x_speed_matrix[y_idx, x_idx] += obj_velocity.x
        y_speed_matrix[y_idx, x_idx] += obj_velocity.y
        presence_matrix[y_idx, x_idx] = 1

    # Mark cells that are out of the road with 1
    for i in range(matrix_length):
        for j in range(matrix_width):
            cell_x = ego_location.x + (j - matrix_width // 2) * cell_width
            cell_y = ego_location.y + (i - matrix_length // 2) * cell_length
            cell_location = carla.Location(x=cell_x, y=cell_y)
            if not is_on_road(cell_location):
                presence_matrix[i, j] = 1

    return x_speed_matrix, y_speed_matrix, presence_matrix