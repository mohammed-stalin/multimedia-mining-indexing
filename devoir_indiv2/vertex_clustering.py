import numpy as np
from read_obj import read_obj


def create_grid(vertices, grid_size):
    """
    Creates a 3D grid for vertex clustering.

    :param vertices: List of (x, y, z) vertex coordinates
    :param grid_size: Size of each grid cell
    :return: Dictionary mapping grid cells to lists of vertex indices
    """
    grid = {}
    for vertex_index, vertex in enumerate(vertices):
        cell_coord = calculate_cell(vertex, grid_size)
        grid.setdefault(cell_coord, []).append(vertex_index)
    return grid


def calculate_cell(vertex, grid_size):
    """Calculates the grid cell coordinates for a given vertex."""
    return tuple(int(coord // grid_size) for coord in vertex)


def compute_representative_vertex(vertices, vertex_indices):
    """
    Computes the representative vertex for a cell by averaging all vertices in it.

    :param vertices: List of all vertices
    :param vertex_indices: List of vertex indices in the cell
    :return: (x, y, z) coordinates of representative vertex
    """
    cell_vertices = [vertices[idx] for idx in vertex_indices]
    return tuple(np.mean(cell_vertices, axis=0))


def vertex_clustering(obj_data, grid_size):
    """
    Performs vertex clustering simplification on the mesh.

    :param obj_data: Dictionary containing mesh data from read_obj
    :param grid_size: Size of grid cells for clustering
    :return: Dictionary with simplified mesh data
    """
    # Step 1: Create grid and map vertices to cells
    grid = create_grid(obj_data['vertices'], grid_size)

    # Step 2: Compute new vertices and map old indices to new ones
    new_vertices, old_to_new_index = build_new_vertices(grid, obj_data['vertices'])

    # Step 3: Create new faces using the new vertex indices
    new_faces = create_new_faces(obj_data['faces'], old_to_new_index)

    # Step 4: Construct the simplified mesh data
    return {
        'vertices': new_vertices,
        'textures': obj_data['textures'],
        'normals': obj_data['normals'],
        'faces': new_faces,
    }


def build_new_vertices(grid, vertices):
    """
    Builds a new vertex list and maps old indices to new indices.

    :param grid: Grid mapping cells to vertex indices
    :param vertices: List of original vertices
    :return: (List of new vertices, Mapping of old indices to new indices)
    """
    new_vertices = []
    old_to_new_index = {}

    for cell, vertex_indices in grid.items():
        new_vertex = compute_representative_vertex(vertices, vertex_indices)
        new_index = len(new_vertices)
        new_vertices.append(new_vertex)

        for old_index in vertex_indices:
            old_to_new_index[old_index] = new_index

    return new_vertices, old_to_new_index


def create_new_faces(faces, old_to_new_index):
    """
    Reconstructs faces using the new vertex indices.

    :param faces: Original list of faces
    :param old_to_new_index: Mapping of old vertex indices to new indices
    :return: List of new faces
    """
    new_faces = []

    for face in faces:
        new_face = [
            (old_to_new_index[v_idx], vt_idx, vn_idx)
            for v_idx, vt_idx, vn_idx in face
        ]

        # Only add face if all vertices are unique (to prevent degenerate faces)
        if len(set(v[0] for v in new_face)) >= 3:
            new_faces.append(new_face)

    return new_faces


def write_obj(data, output_path):
    """
    Writes mesh data to an OBJ file.

    :param data: Dictionary containing mesh data
    :param output_path: Path to output OBJ file
    """
    with open(output_path, 'w') as file:
        # Write vertices
        for vertex in data['vertices']:
            file.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')

        # Write texture coordinates
        for texture in data['textures']:
            file.write(f'vt {texture[0]:.6f} {texture[1]:.6f}\n')

        # Write normals
        for normal in data['normals']:
            file.write(f'vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n')

        # Write faces
        for face in data['faces']:
            file.write(format_face(face) + '\n')


def format_face(face):
    """
    Formats a face for writing to an OBJ file.

    :param face: List of (vertex_idx, texture_idx, normal_idx) tuples
    :return: Formatted string for the face
    """
    face_str = 'f'
    for v_idx, vt_idx, vn_idx in face:
        if vt_idx is not None and vn_idx is not None:
            face_str += f' {v_idx+1}/{vt_idx+1}/{vn_idx+1}'
        elif vt_idx is not None:
            face_str += f' {v_idx+1}/{vt_idx+1}'
        else:
            face_str += f' {v_idx+1}'
    return face_str


def main():
    """
    Main function to perform vertex clustering on a given OBJ file.
    """
    input_path = 'DVI2/obj3.obj'
    output_path = 'simplified.obj'
    grid_size = 0.1  # Adjust this value to control the level of simplification

    # Read input mesh
    obj_data = read_obj(input_path)

    # Perform vertex clustering
    simplified_data = vertex_clustering(obj_data, grid_size)

    # Write simplified mesh
    write_obj(simplified_data, output_path)

    # Print statistics
    print(f"Original vertices: {len(obj_data['vertices'])}")
    print(f"Simplified vertices: {len(simplified_data['vertices'])}")
    print(f"Original faces: {len(obj_data['faces'])}")
    print(f"Simplified faces: {len(simplified_data['faces'])}")


if __name__ == '__main__':
    main()