def read_obj(file_path):
    obj_data = {
        'vertices': [],
        'normals': [],
        'textures': [],
        'faces': []
    }

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            line_type, *values = line.split()

            if line_type == 'v':  # Vertex
                obj_data['vertices'].append(parse_vertex(values))
            elif line_type == 'vt':  # Texture coordinate
                obj_data['textures'].append(parse_texture(values))
            elif line_type == 'vn':  # Normal
                obj_data['normals'].append(parse_normal(values))
            elif line_type == 'f':  # Face
                obj_data['faces'].append(parse_face(values))

    return obj_data

def parse_vertex(values):
    """Parses a vertex line into a tuple (x, y, z)."""
    return tuple(map(float, values[:3]))

def parse_texture(values):
    """Parses a texture coordinate line into a tuple (u, v)."""
    return tuple(map(float, values[:2]))

def parse_normal(values):
    """Parses a normal vector line into a tuple (nx, ny, nz)."""
    return tuple(map(float, values[:3]))

def parse_face(values):
    """
    Parses a face line into a list of tuples (vertex_idx, texture_idx, normal_idx).
    Handles the optional absence of texture and normal indices.
    """
    face = []
    for value in values:
        indices = value.split('/')
        vertex_idx = int(indices[0]) - 1  # Convert to 0-based index
        texture_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
        normal_idx = int(indices[2]) - 1 if len(indices) > 2 else None
        face.append((vertex_idx, texture_idx, normal_idx))
    return face