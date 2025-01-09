import carla

def load_opendrive_map(xodr_file_path, _client=None):
    """
    Loads given opendrive file in carla
    
    Args:
        string(xodr_file_path): string that contains map path

    Returns:
        world: Carla world created using given map
    """
    if (_client is None):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
    else :
        client = _client

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

def destroy_all_actors(client):
    world = client.get_world()
    actors = world.get_actors()
    for actor in actors:
        if actor.is_alive:
            actor.destroy()
# load_opendrive_map()
    