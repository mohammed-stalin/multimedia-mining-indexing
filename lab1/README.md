# Lab 1 - Colour segmentation in HSV & chroma key

## Exercise 1 - `ex1.py`: interactive HSV thresholding

Six trackbars set the lower and upper bounds of H, S and V. The binary mask
produced by `cv2.inRange` is shown next to the masked image so you can tune the
bounds that isolate the green background. The final bounds are printed when
you quit, ready to paste into `ex2.py`.

```bash
python ex1.py            # Esc or q to quit
```

OpenCV ranges: **H ∈ [0, 179]**, S ∈ [0, 255], V ∈ [0, 255]. Pure green is H ≈ 60.

## Exercise 2 - `ex2.py`: background replacement (green screen)

1. Convert to HSV and threshold the green range → background mask.
2. Clean the mask (morphological opening + closing) and blur it slightly for soft edges.
3. Blend: `result = foreground * (1 - mask) + new_background * mask`.

```bash
python ex2.py
python ex2.py fondvert.png newback.jpeg --save result.png
```

| File | Role |
|---|---|
| `fondvert.png` | subject on a green background |
| `newback.jpeg` | replacement background |

## Fixes made

* `ex1.py`: each trackbar's *maximum* was equal to its initial value, so the
  bounds could never be raised; maxima are now 179/255.
* `ex2.py`: mask clean-up and soft blending remove the green fringe around the subject.
* Both scripts work from any working directory and accept custom images.
