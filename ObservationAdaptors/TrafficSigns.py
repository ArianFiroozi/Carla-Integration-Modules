def get_nearby_signs(ego_vehicle, world_map, radius=10):
    """
    Get traffic signs in a given radius of ego vehicle 

    Args:
        ego_vehicle: The ego vehicle actor.
        radius: Radius in which the signs will be returned.
    
    Returns:
        list[signs]: Traffic signs present in given radius
    """
        
    ego_location = ego_vehicle.get_location()
    ego_waypoint = world_map.get_waypoint(ego_location)
    landmarks = ego_waypoint.get_landmarks(radius, True)
    
    traffic_signs = [lm for lm in landmarks if lm.type == '1000001']

    return traffic_signs
