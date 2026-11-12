# Lab 3 - Texture segmentation with Gabor filters

`ex4.py` lets you select a region containing a texture, then highlights all
areas of the image with a similar texture.

## Method

1. **Gabor filter bank** - 4 orientations (0°, 45°, 90°, 135°) × 3 scales.
   Kernels are made zero-mean so they respond to texture and not to brightness.
2. **Energy** - each filter response is replaced by its magnitude `|I * g|`.
3. **Local descriptor** - for every pixel, mean and standard deviation of each
   energy map over a window the size of the ROI (computed with box filters:
   `std = sqrt(E[x²] − E[x]²)`), i.e. a 24-dimensional vector.
4. **Standardisation** - each feature is z-scored so no filter dominates.
5. **Similarity** - Euclidean distance to the mean descriptor of the ROI.
   A pixel matches when its distance is below
   `factor × (95th percentile of the distances inside the ROI)`.

The figure shows the original with the ROI, the ROI, the distance map and the
extracted texture.

## Run

```bash
python ex4.py                       # a.jpg, b.jpg, c.jpg one after another
python ex4.py c.jpg --factor 2      # more tolerant
python ex4.py a.jpg --threshold 3   # absolute threshold instead of adaptive
```

Draw a rectangle, then press **Enter** or **Space** to confirm (**c** to cancel).

| Image | Content |
|---|---|
| `a.jpg` | mosaic of 6 textures (fabric, carpet, plaster, bark, wood, sand) |
| `b.jpg` | synthetic pattern mosaic |
| `c.jpg` | aerial view: forest and road |

## Fixes and improvements

* Filtering used `cv2.CV_8UC3` as output depth on a grayscale image, which
  clipped every negative response; responses are now `float32`.
* The original sliding window re-applied all 12 filters to every window
  (thousands of convolutions, minutes per image). The bank is now applied once
  and local statistics come from box filters (under a second).
* Kernel normalisation divided by `kernel.sum()`, which can be ~0; L1
  normalisation is used instead.
* The fixed threshold (80) did not transfer between images; the threshold now
  adapts to the selected ROI.
* Cancelled ROI selections are handled; the distance map is displayed.
