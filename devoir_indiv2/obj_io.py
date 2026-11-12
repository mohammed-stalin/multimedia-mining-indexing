"""Minimal Wavefront OBJ reader / writer.

Mesh data is a plain dict:

    {
        'vertices': [(x, y, z), ...],
        'textures': [(u, v), ...],
        'normals':  [(nx, ny, nz), ...],
        'faces':    [[(v_idx, vt_idx, vn_idx), (..), (..)], ...]
    }

All indices are 0-based; vt_idx / vn_idx are None when absent.
Polygons with more than 3 vertices are triangulated (fan) on load so that
every face is a triangle, which the simplification algorithms rely on.
"""
import copy


def _parse_index(token, count):
    """Convert an OBJ index (1-based, may be negative/relative) to 0-based."""
    if not token:
        return None
    idx = int(token)
    return idx - 1 if idx > 0 else count + idx


def read_obj(file_path, triangulate=True):
    obj_data = {"vertices": [], "normals": [], "textures": [], "faces": []}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            line_type, *values = line.split()

            if line_type == "v":
                obj_data["vertices"].append(tuple(map(float, values[:3])))
            elif line_type == "vt":
                obj_data["textures"].append(tuple(map(float, values[:2])))
            elif line_type == "vn":
                obj_data["normals"].append(tuple(map(float, values[:3])))
            elif line_type == "f":
                face = _parse_face(values, obj_data)
                if triangulate and len(face) > 3:
                    for k in range(1, len(face) - 1):
                        obj_data["faces"].append([face[0], face[k], face[k + 1]])
                else:
                    obj_data["faces"].append(face)

    return obj_data


def _parse_face(values, obj_data):
    """Parse 'v', 'v/vt', 'v//vn' or 'v/vt/vn' tokens of a face line."""
    nv, nt, nn = len(obj_data["vertices"]), len(obj_data["textures"]), len(obj_data["normals"])
    face = []
    for value in values:
        parts = value.split("/") + ["", ""]
        face.append((
            _parse_index(parts[0], nv),
            _parse_index(parts[1], nt),
            _parse_index(parts[2], nn),
        ))
    return face


def _format_corner(v_idx, vt_idx, vn_idx):
    if vt_idx is None and vn_idx is None:
        return f"{v_idx + 1}"
    if vn_idx is None:
        return f"{v_idx + 1}/{vt_idx + 1}"
    vt = "" if vt_idx is None else str(vt_idx + 1)
    return f"{v_idx + 1}/{vt}/{vn_idx + 1}"


def write_obj(obj_data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("# Simplified mesh\n")
        file.write(f"# {len(obj_data['vertices'])} vertices, {len(obj_data['faces'])} faces\n")
        for x, y, z in obj_data["vertices"]:
            file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in obj_data["textures"]:
            file.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in obj_data["normals"]:
            file.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        for face in obj_data["faces"]:
            file.write("f " + " ".join(_format_corner(*corner) for corner in face) + "\n")


def copy_mesh(obj_data):
    """Deep copy, so simplification never modifies the original mesh."""
    return copy.deepcopy(obj_data)


def remove_unused_vertices(obj_data):
    """Drop vertices no face refers to and re-index the faces."""
    used = sorted({corner[0] for face in obj_data["faces"] for corner in face})
    remap = {old: new for new, old in enumerate(used)}
    return {
        "vertices": [obj_data["vertices"][i] for i in used],
        "textures": obj_data["textures"],
        "normals": obj_data["normals"],
        "faces": [[(remap[v], vt, vn) for v, vt, vn in face] for face in obj_data["faces"]],
    }


def mesh_stats(obj_data):
    return len(obj_data["vertices"]), len(obj_data["faces"])
