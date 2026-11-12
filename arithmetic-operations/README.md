# Arithmetic operations on images

Pixel-wise arithmetic with OpenCV. Both operands must have the same size,
so the second image is resized to the first one.

| Script | What it does |
|---|---|
| `addition.py` | Weighted sum `alpha*img1 + beta*img2 + gamma` with `cv2.addWeighted` |
| `addition2.py` | Interactive cross-fade `(1-a)*img1 + a*img2` controlled by a trackbar |
| `subss.py` | `cv2.subtract` (saturating, negatives clip to 0) vs `cv2.absdiff` |

## Run

```bash
python addition.py                               # mountain + horse
python addition.py img1.jpg img2.jpg --alpha 0.7 --beta 0.3
python addition2.py                              # move the slider, Esc/q to quit
python subss.py                                  # star - square
```

Default images come from [`../assets`](../assets).

## Notes

* `uint8` arithmetic in NumPy **wraps around** (`250 + 10 = 4`), while OpenCV
  functions **saturate** (`250 + 10 = 255`). Prefer `cv2.add` / `cv2.subtract` /
  `cv2.addWeighted`, or compute in float and convert back.
* `cv2.subtract(a, b)` only keeps where `a` is brighter than `b`; use
  `cv2.absdiff` to detect changes in both directions.

## Fixes made

* `addition2.py`: the blend formula used `+ b` instead of `* b`; the images
  were not resized to a common shape; `waitKey(0)` blocked the loop so the
  slider never refreshed.
* All scripts: paths are resolved relative to the script, so they work from any
  working directory; missing files raise a clear error.
