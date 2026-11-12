"""Mesh simplification by iterative edge collapse.

Repeatedly pick the cheapest edge (a, b), merge b into a and move a to the
edge midpoint. The two triangles sharing the edge disappear, every other
triangle that used b now uses a. Stop when the requested fraction of faces
remains.

Two cost functions are available:
* "length"  - Euclidean length of the edge (shortest edges first).
* "quadric" - Quadric Error Metric (Garland & Heckbert 1997) evaluated at the
              midpoint: sum of squared distances to the planes of the
              original faces around a and b. Preserves shape much better.

Safety checks applied before each collapse:
* link condition - the two endpoints must share exactly as many neighbours as
  they have common faces, otherwise the collapse makes the mesh non-manifold;
* flip test      - a collapse that would turn a surrounding triangle upside
  down is rejected.

Implementation: a binary heap with lazy deletion (each entry stores the
"version" of both endpoints; stale entries are skipped when popped) and a
vertex -> faces adjacency table, so each collapse is O(local neighbourhood)
instead of O(number of faces).

Usage:
    python edge_collapse.py [input.obj] [-k KEEP_RATIO] [--cost quadric|length] [-o output.obj]
"""
import argparse
import heapq
import math
import time
from pathlib import Path

from obj_io import mesh_stats, read_obj, remove_unused_vertices, write_obj

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ geometry
def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _triangle_normal(p0, p1, p2):
    return _cross(_sub(p1, p0), _sub(p2, p0))


# ------------------------------------------------------------------ quadrics
# A symmetric 4x4 quadric is stored as its 10 upper-triangular coefficients:
# [a2, ab, ac, ad, b2, bc, bd, c2, cd, d2] for the plane ax + by + cz + d = 0.
def _plane_quadric(p0, p1, p2):
    n = _triangle_normal(p0, p1, p2)
    length = math.sqrt(_dot(n, n))
    if length == 0.0:
        return None
    a, b, c = n[0] / length, n[1] / length, n[2] / length
    d = -(a * p0[0] + b * p0[1] + c * p0[2])
    return [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d]


def _quadric_error(q, p):
    x, y, z = p
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x
            + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
            + q[7] * z * z + 2 * q[8] * z + q[9])


# ------------------------------------------------------------- simplifier
def edge_collapse(obj_data, reduction_ratio=0.5, cost="quadric", verbose=False):
    """Return a NEW mesh keeping about `reduction_ratio` of the faces.

    reduction_ratio = 0.1 keeps 10 % of the triangles.
    The input mesh is not modified.
    """
    if not 0 < reduction_ratio <= 1:
        raise ValueError("reduction_ratio must be in (0, 1]")
    if cost not in ("quadric", "length"):
        raise ValueError("cost must be 'quadric' or 'length'")

    positions = [tuple(v) for v in obj_data["vertices"]]
    # Keep only proper triangles (some input files contain degenerate faces
    # such as "f 730 726 730" which would corrupt the adjacency).
    faces = [list(face) for face in obj_data["faces"]
             if len(face) == 3 and len({c[0] for c in face}) == 3]
    n_vertices, n_faces = len(positions), len(faces)
    target = max(1, int(n_faces * reduction_ratio))

    face_alive = [True] * n_faces
    vertex_faces = [set() for _ in range(n_vertices)]
    for fi, face in enumerate(faces):
        for v, _, _ in face:
            vertex_faces[v].add(fi)
    removed = [False] * n_vertices
    version = [0] * n_vertices

    quadrics = None
    if cost == "quadric":
        quadrics = [[0.0] * 10 for _ in range(n_vertices)]
        for face in faces:
            ids = [c[0] for c in face]
            q = _plane_quadric(*(positions[i] for i in ids))
            if q is None:
                continue
            for i in ids:
                qi = quadrics[i]
                for k in range(10):
                    qi[k] += q[k]

    def midpoint(a, b):
        pa, pb = positions[a], positions[b]
        return ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5)

    def edge_cost(a, b):
        if quadrics is None:
            d = _sub(positions[a], positions[b])
            return math.sqrt(_dot(d, d))
        qa, qb = quadrics[a], quadrics[b]
        return _quadric_error([qa[k] + qb[k] for k in range(10)], midpoint(a, b))

    def neighbours(v):
        return {c[0] for fi in vertex_faces[v] for c in faces[fi]} - {v}

    def push(heap, a, b):
        if a > b:
            a, b = b, a
        heapq.heappush(heap, (edge_cost(a, b), a, b, version[a], version[b]))

    def flips(a, b, new_pos, shared):
        """True if moving a and b to new_pos turns any remaining face over."""
        for fi in (vertex_faces[a] | vertex_faces[b]) - shared:
            ids = [c[0] for c in faces[fi]]
            old = [positions[i] for i in ids]
            new = [new_pos if i in (a, b) else positions[i] for i in ids]
            n_old = _triangle_normal(*old)
            n_new = _triangle_normal(*new)
            if _dot(n_old, n_new) <= 0.0:
                return True
        return False

    # Initial queue: every unique edge
    heap, edges = [], set()
    for face in faces:
        ids = [c[0] for c in face]
        for k in range(3):
            a, b = ids[k], ids[(k + 1) % 3]
            edges.add((a, b) if a < b else (b, a))
    for a, b in edges:
        push(heap, a, b)
    del edges

    alive_faces = n_faces
    start = time.perf_counter()
    while alive_faces > target and heap:
        _, a, b, va, vb = heapq.heappop(heap)
        if removed[a] or removed[b] or version[a] != va or version[b] != vb:
            continue  # stale entry

        shared = vertex_faces[a] & vertex_faces[b]
        if not shared:
            continue
        if len(neighbours(a) & neighbours(b)) != len(shared):
            continue  # link condition violated -> non-manifold result
        new_pos = midpoint(a, b)
        if flips(a, b, new_pos, shared):
            continue

        # --- perform the collapse b -> a
        positions[a] = new_pos
        for fi in shared:
            face_alive[fi] = False
            alive_faces -= 1
            for v, _, _ in faces[fi]:
                vertex_faces[v].discard(fi)
        for fi in vertex_faces[b]:
            faces[fi] = [(a, vt, vn) if v == b else (v, vt, vn) for v, vt, vn in faces[fi]]
            vertex_faces[a].add(fi)
        vertex_faces[b] = set()
        removed[b] = True
        if quadrics is not None:
            qa, qb = quadrics[a], quadrics[b]
            for k in range(10):
                qa[k] += qb[k]

        version[a] += 1
        for n in neighbours(a):
            push(heap, a, n)

        if verbose and alive_faces % 10000 == 0:
            print(f"  {alive_faces} faces left ({time.perf_counter() - start:.1f}s)")

    result = {
        "vertices": positions,
        "textures": obj_data["textures"],
        "normals": obj_data["normals"],
        "faces": [faces[i] for i in range(n_faces) if face_alive[i]],
    }
    return remove_unused_vertices(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=HERE / "DVI2" / "obj3.obj")
    parser.add_argument("-k", "--keep", type=float, default=0.5,
                        help="fraction of faces to keep, e.g. 0.1 keeps 10%% (default 0.5)")
    parser.add_argument("--cost", choices=("quadric", "length"), default="quadric")
    parser.add_argument("-o", "--output", help="output OBJ (default: results/<input>_collapsed.obj)")
    args = parser.parse_args()

    obj_data = read_obj(args.input)
    t0 = time.perf_counter()
    simplified = edge_collapse(obj_data, args.keep, cost=args.cost, verbose=True)
    elapsed = time.perf_counter() - t0

    output = Path(args.output) if args.output else HERE / "results" / f"{Path(args.input).stem}_collapsed.obj"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(simplified, output)

    (v0, f0), (v1, f1) = mesh_stats(obj_data), mesh_stats(simplified)
    print(f"Original   : {v0} vertices, {f0} faces")
    print(f"Simplified : {v1} vertices, {f1} faces ({f1 / f0:.1%} of the faces kept) in {elapsed:.1f}s")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
