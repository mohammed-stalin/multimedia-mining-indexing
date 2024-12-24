import heapq
from read_obj import read_obj

def edge_collapse(obj_data, reduction_ratio):
    vertices = obj_data['vertices']
    faces = obj_data['faces']
    target_faces = int(len(faces) * reduction_ratio)

    # Initialize priority queue and edge storage
    edge_queue = []
    edge_map = build_edge_map(faces)

    # Add edges to the queue with their initial cost
    for edge, related_faces in edge_map.items():
        cost = calculate_edge_cost(edge, vertices, faces, related_faces)
        heapq.heappush(edge_queue, (cost, edge, related_faces))

    # Perform edge collapses until the target number of faces is reached
    while len(faces) > target_faces and edge_queue:
        cost, edge_to_collapse, affected_faces = heapq.heappop(edge_queue)
        collapse_edge(edge_to_collapse, vertices, faces, affected_faces)
        update_edge_queue(edge_queue, edge_to_collapse, vertices, faces)

    return obj_data

def write_obj(file_path, obj_data):
    with open(file_path, 'w') as file:
        for vertex in obj_data['vertices']:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for normal in obj_data['normals']:
            file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        for texture in obj_data['textures']:
            file.write(f"vt {texture[0]} {texture[1]}\n")
        for face in obj_data['faces']:
            face_str = ' '.join(
                ['/'.join([str(idx + 1) if idx is not None else '' for idx in vertex]) for vertex in face]
            )
            file.write(f"f {face_str}\n")

def build_edge_map(faces):
    """Build a mapping of edges to the faces they belong to."""
    edge_map = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            vertex1 = face[i][0]
            vertex2 = face[(i + 1) % 3][0]
            edge = tuple(sorted((vertex1, vertex2)))  # Ensure edge is always ordered (v1, v2)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(face_idx)
    return edge_map

def calculate_edge_cost(edge, vertices, faces, related_faces):
    """Calculate the cost of collapsing an edge."""
    vertex1, vertex2 = edge
    point1 = vertices[vertex1]
    point2 = vertices[vertex2]
    # Cost is the Euclidean distance between two vertices
    return sum((point1[i] - point2[i]) ** 2 for i in range(3)) ** 0.5

def collapse_edge(edge, vertices, faces, affected_faces):
    """Perform the edge collapse operation."""
    vertex1, vertex2 = edge
    # Compute the midpoint of the edge
    midpoint = tuple((vertices[vertex1][i] + vertices[vertex2][i]) / 2 for i in range(3))
    vertices[vertex1] = midpoint

    # Remove faces that include vertex2
    faces[:] = [face for face in faces if vertex2 not in [vertex[0] for vertex in face]]

    # Update remaining faces to replace vertex2 with vertex1
    for face in faces:
        for i in range(len(face)):
            if face[i][0] == vertex2:
                face[i] = (vertex1, face[i][1], face[i][2])

def update_edge_queue(edge_queue, collapsed_edge, vertices, faces):
    """Rebuild the priority queue after an edge collapse."""
    heapq.heapify(edge_queue)

if __name__ == "__main__":
    # Read input OBJ file
    obj_data = read_obj('DVI2/obj3.obj')
    reduction_ratio = 0.9  # Example: reduce to 90% of the original number of faces
    # Perform edge collapse
    simplified_obj = edge_collapse(obj_data, reduction_ratio)
    # Write the simplified OBJ file
    write_obj('result1.obj', simplified_obj)