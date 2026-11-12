# Assignment 2 - 3D mesh simplification

Two classic simplification algorithms for triangle meshes (Wavefront OBJ) and an
OpenGL viewer comparing the original and the simplified mesh side by side.

## Files

| File | Role |
|---|---|
| `obj_io.py` | OBJ reader/writer (`v`, `vt`, `vn`, `f` in all index forms, negative indices, polygon triangulation) |
| `vertex_clustering.py` | simplification by vertex clustering |
| `edge_collapse.py` | simplification by iterative edge collapse (length or quadric cost) |
| `main.py` | runs a simplification and opens the comparison viewer |
| `DVI2/obj1-6.obj` | test models (3D Pottery Benchmark, ILSP / Athena) |

| Model | Vertices | Faces |
|---|---|---|
| obj1 | 1 369 | 2 734 |
| obj2 | 10 505 | 20 823 |
| obj3 | 2 165 | 4 438 (142 degenerate) |
| obj4 | 17 403 | 34 644 |
| obj5 | 1 932 | 3 824 |
| obj6 | 327 323 | 654 666 |

## Algorithms

### Vertex clustering

1. A uniform 3D grid is laid over the bounding box. The cell size is given
   either in model units (`--grid-size`) or as a number of cells along the
   longest side (`--resolution`, independent of the model's scale).
2. Vertices in the same cell are replaced by their mean.
3. Faces are re-indexed; degenerate and duplicate triangles are removed.

Very fast (fully vectorised with NumPy, ~1 s on obj6) but the result can be
non-manifold and quality depends on the grid alignment.

### Edge collapse

Repeatedly collapse the cheapest edge `(a, b)` into its midpoint: the two
triangles sharing the edge disappear and `b` is replaced by `a` everywhere else.

* **Cost**: `quadric` (default) - Garland & Heckbert's Quadric Error Metric, the
  sum of squared distances to the planes of the original surrounding faces,
  which keeps the shape much better; or `length` - shortest edges first.
* **Link condition**: the collapse is rejected if it would make the mesh
  non-manifold.
* **Flip test**: the collapse is rejected if a surrounding triangle would be
  turned over.
* **Data structures**: binary heap with lazy deletion (a version number per
  vertex invalidates stale entries) and a vertex → faces table, so each collapse
  only touches its neighbourhood.

Timings (keep 10 %, quadric): obj1-obj5 in 0.1-1.2 s, obj6 (650k faces) in ~66 s.
The output is always a valid manifold (every edge shared by at most two faces).

## Run

```bash
pip install numpy glfw PyOpenGL pyrr

# Viewer: original (blue, left) vs simplified (orange, right)
python main.py                                  # obj3, edge collapse, keep 10 %
python main.py DVI2/obj1.obj -m edge -k 0.2
python main.py DVI2/obj4.obj -m edge --cost length -k 0.3
python main.py DVI2/obj6.obj -m cluster -r 80
python main.py DVI2/obj2.obj --save             # also writes results/obj2_edge.obj

# Command line only, writes an OBJ in results/
python edge_collapse.py DVI2/obj2.obj -k 0.1
python vertex_clustering.py DVI2/obj2.obj -r 40
```

Viewer controls: mouse drag to rotate, scroll to zoom, **W** wireframe,
**R** reset view, **Esc** quit. The window title shows the face counts.

## Fixes made

* `main.py` called `edge_collapse(original_data, ratio=0.1)` but the parameter
  was named `reduction_ratio` → `TypeError`, the program never started.
* `edge_collapse` modified the mesh **in place**, so the "original" and
  "simplified" meshes were the same object. All algorithms now return a new mesh.
* `main.py` subtracted 1 from face indices that `read_obj` had already made
  0-based, so index −1 wrapped to 4 294 967 295 in `uint32`.
* The models have coordinates around ±150 while the camera was at z = 3, so
  nothing was visible. Both meshes are now centred and scaled to the unit
  sphere with the same transform.
* `collapse_edge` deleted **every** face touching `v2` instead of only the
  faces sharing the collapsed edge, tearing holes in the mesh; the heap was never
  updated after a collapse (stale costs, removed vertices reused) and each
  collapse rebuilt the face list (O(F²), obj6 could never finish).
* Degenerate input faces (`obj3`) are now ignored.
* `grid_size = 0.1` barely simplified models of size ~300; added `--resolution`.
* Viewer: lighting with smooth normals, wireframe mode, framebuffer-size
  viewports (HiDPI), OpenGL 3.3 core context (also works on macOS), CLI options.
* The duplicated `write_obj` functions and `read_obj.py` were merged into `obj_io.py`.
