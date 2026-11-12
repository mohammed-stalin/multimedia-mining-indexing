# Lab 7 - Interactive Bézier surface (OpenGL)

`script.py` renders a bi-quadratic Bézier surface defined by a 3×3 grid of
control points and lets you edit the points from the keyboard.

$$S(u,v)=\sum_{i=0}^{n}\sum_{j=0}^{m} B_i^n(u)\,B_j^m(v)\,P_{ij},\qquad B_i^n(t)=\binom{n}{i}t^i(1-t)^{n-i}$$

## Implementation

* The surface is evaluated on a 21×21 grid in a single NumPy `einsum`
  using Bernstein matrices, and only re-evaluated when a point moves.
* Smooth per-vertex normals from finite differences, rendered with fixed-function
  OpenGL lighting (two-sided).
* The control net and the control points (selected one in red) are drawn on top.

## Controls

| Key | Action |
|---|---|
| Tab | select next control point |
| X / Y / Z | move the selected point along that axis by ±0.1 |
| S | flip the direction (+/−) |
| W | wireframe on/off |
| N | control net on/off |
| R | reset |
| mouse drag / scroll | rotate / zoom |
| Esc | quit |

The window title shows the selected point, its coordinates and the direction.

## Run

```bash
pip install glfw PyOpenGL numpy
python script.py
```

## Improvements

* Surface evaluation vectorised (it was recomputed point by point with nested
  Python loops every frame).
* Normals were sent but lighting was never enabled, so the surface looked flat;
  lighting is now enabled and the normals are smooth.
* Added mouse rotation/zoom, wireframe, control net, reset, window resize
  handling and key repeat; removed a duplicated import.
