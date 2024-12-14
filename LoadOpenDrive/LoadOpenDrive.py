import carla

def load_opendrive_map(xodr_file_path):
    """
    Loads given opendrive file in carla
    
    Args:
        string(xodr_file_path): string that contains map path

    Returns:
        world: Carla world created using given map
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    with open(xodr_file_path, 'r') as file:
        xodr_data = file.read()

    world = client.generate_opendrive_world(
        xodr_data,
        carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=1.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True
        )
    )
    print("Map successfully loaded into CARLA.")
    return world