"""Interactive cross-fade between two images using a trackbar.

    c = (1 - alpha) * a + alpha * b          with alpha in [0, 1]

Usage:
    python addition2.py [image1] [image2]
Move the "Alpha" slider to blend. Press Esc or q to quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
WINDOW = "addition"


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def blend(a, b, alpha_percent):
    alpha = alpha_percent / 100.0
    return np.uint8((1.0 - alpha) * a + alpha * b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image1", nargs="?", default=ASSETS / "mountain.jpg")
    parser.add_argument("image2", nargs="?", default=ASSETS / "rose.jpg")
    args = parser.parse_args()

    a = load_image(args.image1)
    b = load_image(args.image2)
    # Pixel-wise arithmetic requires identical shapes
    b = cv2.resize(b, (a.shape[1], a.shape[0]))

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("Alpha", WINDOW, 0, 100, lambda _: None)

    alpha = -1
    while True:
        pos = cv2.getTrackbarPos("Alpha", WINDOW)
        if pos != alpha:
            alpha = pos
            cv2.imshow(WINDOW, blend(a, b, alpha))

        key = cv2.waitKey(30) & 0xFF  # non-blocking so the slider stays responsive
        if key in (27, ord("q")):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
