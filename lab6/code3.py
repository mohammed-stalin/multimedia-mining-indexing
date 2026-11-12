"""Lab 6 - Interactive marker-based watershed segmentation.

Draw markers on the objects you want to separate (one marker label per
object), then run the watershed ("ligne de partage des eaux") algorithm.
The background marker is created automatically from a thresholded and
dilated version of the image.

Controls (in the drawing window)
    left mouse drag  draw with the current marker
    n                next marker (new label, new colour)
    + / -            increase / decrease brush thickness
    w or Space       run watershed and show the result (you can keep drawing)
    c                clear all markers
    s                save the last segmentation next to the image
    q or Esc         quit (runs the watershed one last time)

Usage:
    python code3.py [image]
"""
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
DRAW_WINDOW = "Dessinez vos marqueurs"
RESULT_WINDOW = "Segmentation"

rng = np.random.default_rng(0)


def random_color():
    return tuple(int(c) for c in rng.integers(50, 256, size=3))


def background_mask(image, dilate_iterations=12):
    """Sure-background mask: pixels far from any foreground object.

    Otsu threshold, inverted automatically when the background is bright,
    then dilated: what stays at 0 is safely background.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if np.count_nonzero(thresh) > thresh.size / 2:  # objects are dark on bright bg
        thresh = cv.bitwise_not(thresh)
    kernel = np.ones((3, 3), np.uint8)
    fg_dilated = cv.dilate(thresh, kernel, iterations=dilate_iterations)
    return fg_dilated == 0


class WatershedApp:
    def __init__(self, image):
        self.original = image
        self.sure_background = background_mask(image)
        self.result = None
        self.reset()

    def reset(self):
        self.canvas = self.original.copy()
        self.markers = np.zeros(self.original.shape[:2], dtype=np.int32)
        self.label = 1
        self.colors = {1: random_color()}
        self.thickness = 3
        self.drawing = False
        self.last = (-1, -1)

    # ------------------------------------------------------------------ mouse
    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last = (x, y)
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self._stroke(x, y)
        elif event == cv.EVENT_LBUTTONUP and self.drawing:
            self._stroke(x, y)
            self.drawing = False

    def _stroke(self, x, y):
        color = self.colors[self.label]
        cv.line(self.canvas, self.last, (x, y), color, self.thickness)
        cv.line(self.markers, self.last, (x, y), self.label, self.thickness)
        self.last = (x, y)

    # -------------------------------------------------------------- watershed
    def segment(self):
        if self.markers.max() == 0:
            print("Draw at least one marker first.")
            return None

        markers = self.markers.copy()
        markers[markers > 0] += 1                  # user labels start at 2
        markers[self.sure_background & (markers == 0)] = 1  # label 1 = background
        markers = cv.watershed(self.original, markers)

        result = np.zeros_like(self.original)
        for label in range(2, markers.max() + 1):
            result[markers == label] = self.colors.get(label - 1, (255, 255, 255))
        result[markers == -1] = (0, 0, 255)       # boundaries in red (BGR)

        overlay = cv.addWeighted(self.original, 0.5, result, 0.5, 0)
        overlay[markers == -1] = (0, 0, 255)
        self.result = np.hstack([result, overlay])
        cv.imshow(RESULT_WINDOW, self.result)
        return self.result

    # ------------------------------------------------------------------- loop
    def run(self, save_path):
        cv.namedWindow(DRAW_WINDOW)
        cv.setMouseCallback(DRAW_WINDOW, self.on_mouse)
        print(__doc__.split("Usage:")[0])

        while True:
            cv.imshow(DRAW_WINDOW, self.canvas)
            k = cv.waitKey(20) & 0xFF

            if k in (ord("q"), 27):
                break
            elif k == ord("c"):
                self.reset()
            elif k == ord("n"):
                self.label += 1
                self.colors[self.label] = random_color()
                print(f"Marker {self.label}")
            elif k in (ord("+"), ord("=")):
                self.thickness += 1
            elif k == ord("-") and self.thickness > 1:
                self.thickness -= 1
            elif k in (ord("w"), ord(" ")):
                self.segment()
            elif k == ord("s") and self.result is not None:
                cv.imwrite(str(save_path), self.result)
                print(f"Saved {save_path}")

        cv.destroyWindow(DRAW_WINDOW)
        if self.segment() is not None:
            cv.waitKey(0)
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=HERE / "coins.png")
    args = parser.parse_args()

    img = cv.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    save_path = Path(args.image).with_name(Path(args.image).stem + "_segmented.png")
    WatershedApp(img).run(save_path)


if __name__ == "__main__":
    main()
