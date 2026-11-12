"""Mesh simplification by vertex clustering (Rossignac & Borrel).

1. Overlay a uniform 3D grid on the bounding box of the mesh.
2. All vertices falling in the same cell are merged into one representative
   vertex (the mean of the cell's vertices).
3. Faces are re-indexed; faces that become degenerate (two corners in the
   same cell) and duplicate faces are removed.

The grid can be given either as an absolute cell size (`grid_size`) or as a
number of cells along the longest side of the bounding box (`resolution`),
which is independent of the model's units and therefore easier to tune.

Usage:
    python vertex_clustering.py [input.obj] [-r RESOLUTION | -g GRID_SIZE] [-o output.obj]
"""
import argparse
from pathlib import Path

import numpy as np

from obj_io import mesh_stats, read_obj, write_obj

HERE = Path(__file__).resolve().parent


def cell_size_from_resolution(vertices, resolution):
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return float(extent.max()) / resolution


def vertex_clustering(obj_data, grid_size=None, resolution=None):
    """Return a NEW simplified mesh dict (the input is not modified)."""
    if (grid_size is None) == (resolution is None):
        raise ValueError("Give exactly one of grid_size or resolution")

    vertices = np.asarray(obj_data["vertices"], dtype=np.float64)
    if grid_size is None:
        grid_size = cell_size_from_resolution(vertices, resolution)

    # Step 1: integer cell coordinates of every vertex
    origin = vertices.min(axis=0)
    cells = np.floor((vertices - origin) / grid_size).astype(np.int64)

    # Step 2: one representative (mean) vertex per non-empty cell
    _, old_to_new, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    old_to_new = old_to_new.ravel()
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, old_to_new, vertices)
    new_vertices = sums / counts[:, None]

    # Step 3: re-index faces, drop degenerate and duplicate triangles
    new_faces, seen = [], set()
    for face in obj_data["faces"]:
        new_face = [(int(old_to_new[v]), vt, vn) for v, vt, vn in face]
        ids = [c[0] for c in new_face]
        if len(set(ids)) < 3:
            continue
        key = frozenset(ids)
        if key in seen:
            continue
        seen.add(key)
        new_faces.append(new_face)

    return {
        "vertices": [tuple(v) for v in new_vertices],
        "textures": obj_data["textures"],
        "normals": obj_data["normals"],
        "faces": new_faces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=HERE / "DVI2" / "obj3.obj")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--resolution", type=int, help="cells along the longest bbox side (default 30)")
    group.add_argument("-g", "--grid-size", type=float, help="absolute cell size in model units")
    parser.add_argument("-o", "--output", help="output OBJ (default: <input>_clustered.obj in results/)")
    args = parser.parse_args()

    if args.resolution is None and args.grid_size is None:
        args.resolution = 30

    obj_data = read_obj(args.input)
    simplified = vertex_clustering(obj_data, grid_size=args.grid_size, resolution=args.resolution)

    output = Path(args.output) if args.output else HERE / "results" / f"{Path(args.input).stem}_clustered.obj"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(simplified, output)

    (v0, f0), (v1, f1) = mesh_stats(obj_data), mesh_stats(simplified)
    print(f"Original   : {v0} vertices, {f0} faces")
    print(f"Simplified : {v1} vertices, {f1} faces ({f1 / f0:.1%} of the faces kept)")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
