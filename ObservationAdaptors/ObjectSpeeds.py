import carla
import numpy as np
import torch
import math

def get_speed_matrices(ego_vehicle, matrix_length=25, matrix_width=11, cell_width=2.0, cell_length=2.0):
    """
    Generate speed matrices aligned with the ego vehicle's heading.
    """
    x_speed_matrix = torch.zeros((matrix_length, matrix_width))
    y_speed_matrix = torch.zeros((matrix_length, matrix_width))
    presence_matrix = torch.zeros((matrix_length, matrix_width))

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_yaw = ego_transform.rotation.yaw
    theta = math.radians(ego_yaw)

    world = ego_vehicle.get_world()
    actors = world.get_actors()
    dynamic_objects = actors.filter('vehicle.*')

    for obj in dynamic_objects:
        if obj.id == ego_vehicle.id:
            continue

        obj_transform = obj.get_transform()
        obj_velocity = obj.get_velocity()

        # Calculate global displacement
        dx = obj_transform.location.x - ego_location.x
        dy = obj_transform.location.y - ego_location.y

        # Rotate to local coordinates
        dx_local = dx * math.cos(theta) + dy * math.sin(theta)
        dy_local = -dx * math.sin(theta) + dy * math.cos(theta)

        # Calculate x_idx (lateral)
        x_offset = (matrix_width // 2) * cell_width
        x_idx = int((dy_local + x_offset) // cell_width)
        if x_idx < 0 or x_idx >= matrix_width:
            continue

        # Calculate y_idx (longitudinal)
        y_offset = (matrix_length // 2) * cell_length
        y_idx = int((dx_local + y_offset) // cell_length)
        if y_idx < 0 or y_idx >= matrix_length:
            continue

        # Rotate velocity to local coordinates
        vx = obj_velocity.x
        vy = obj_velocity.y
        vx_local = vx * math.cos(theta) + vy * math.sin(theta)
        vy_local = -vx * math.sin(theta) + vy * math.cos(theta)

        x_speed_matrix[y_idx, x_idx] += vx_local
        y_speed_matrix[y_idx, x_idx] += vy_local
        presence_matrix[y_idx, x_idx] = 1

    # Mark off-road cells
    for i in range(matrix_length):
        for j in range(matrix_width):
            if presence_matrix[i, j] == 0:
                # Local cell position
                dx_cell_local = (i - matrix_length // 2) * cell_length
                dy_cell_local = (j - matrix_width // 2) * cell_width

                # Rotate to global coordinates
                dx_global = dx_cell_local * math.cos(theta) - dy_cell_local * math.sin(theta)
                dy_global = dx_cell_local * math.sin(theta) + dy_cell_local * math.cos(theta)

                cell_x = ego_location.x + dx_global
                cell_y = ego_location.y + dy_global
                cell_location = carla.Location(x=cell_x, y=cell_y)

                if not is_on_road(cell_location, world):
                    presence_matrix[i, j] = 2

    # Mark ego vehicle's position
    presence_matrix[matrix_length // 2, matrix_width // 2] = 9

    return x_speed_matrix, y_speed_matrix, presence_matrix

def is_on_road(location, world):
    waypoint = world.get_map().get_waypoint(location, project_to_road=False)
    return waypoint is not None