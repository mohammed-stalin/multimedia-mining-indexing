# Lab 6 - Marker-based watershed segmentation

`code3.py` is an interactive tool: you draw one marker per object, and the
watershed algorithm ("ligne de partage des eaux") grows each marker until the
regions meet.

## How it works

1. **User markers** - each stroke is written into an `int32` marker image with
   the current label (labels 2, 3, … after shifting).
2. **Background marker** - the image is thresholded with Otsu (inverted
   automatically when objects are darker than the background) and dilated; the
   pixels still at 0 are certainly background and get label 1.
3. **`cv2.watershed`** floods from the markers; boundaries get label −1.
4. The result is shown as a label image and as an overlay on the original, with
   boundaries in red.

## Controls

| Key | Action |
|---|---|
| left drag | draw with the current marker |
| `n` | next marker (new label and colour) |
| `+` / `-` | brush thickness |
| `w` or Space | run the watershed now (you can keep drawing) |
| `c` | clear all markers |
| `s` | save the segmentation as `<image>_segmented.png` |
| `q` / Esc | quit (shows the final segmentation) |

## Run

```bash
python code3.py              # coins.png
python code3.py coins1.png
```

Tip: press `n` before each new coin so every coin gets its own label.

## Fixes made

* The image path was hard-coded to `C:/Users/lenovo/Desktop/...`.
* Boundaries were painted `[255, 0, 0]`, which is **blue** in OpenCV's BGR
  order, not red as intended.
* Only 10 colours were available (labels wrapped around); each marker now has its own colour.
* Fixed threshold 100 replaced by Otsu, with automatic polarity.
* Global state moved into a class; added live preview, overlay and save.
