import numpy as np
import torch

def get_lane_angle(ego_vehicle, world_map):
    """
    Get lane angle relative to ego vehicle heading 

    Args:
        ego_vehicle: The ego vehicle actor.
    
    Returns:
        float: relative angle between road and vehicle heading in radians
    """

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_vehicle.get_location()
    ego_waypoint = world_map.get_waypoint(ego_location)

    lane_vector = ego_waypoint.transform.get_forward_vector()
    road_angle = np.arctan2(lane_vector.y, lane_vector.x)

    car_forward_vector = ego_transform.get_forward_vector()
    car_heading = np.arctan2(car_forward_vector.y, car_forward_vector.x)
    
    relative_angle = car_heading - road_angle
    return relative_angle
