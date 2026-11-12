"""Exercise 1 - Interactive HSV thresholding.

Tune the six trackbars (H/S/V lower & upper bounds) to isolate the green
background of `fondvert.png`. The binary mask and the masked image are
shown side by side. When you quit, the final bounds are printed so they can
be reused in ex2.py.

OpenCV HSV ranges: H in [0, 179], S in [0, 255], V in [0, 255].

Usage:
    python ex1.py [image]
Press Esc or q to quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WINDOW = "HSV threshold"

# (name, initial value, max value)
TRACKBARS = [
    ("H Lower", 35, 179),
    ("H Upper", 85, 179),
    ("S Lower", 40, 255),
    ("S Upper", 255, 255),
    ("V Lower", 40, 255),
    ("V Upper", 255, 255),
]


def read_bounds():
    values = [cv2.getTrackbarPos(name, WINDOW) for name, _, _ in TRACKBARS]
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = values
    return np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=HERE / "fondvert.png")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name, initial, maximum in TRACKBARS:
        cv2.createTrackbar(name, WINDOW, initial, maximum, lambda _: None)

    while True:
        lower, upper = read_bounds()
        mask = cv2.inRange(hsv_image, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=mask)

        preview = np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked])
        cv2.imshow(WINDOW, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    lower, upper = read_bounds()
    print(f"lower_hsv = np.array({lower.tolist()})")
    print(f"upper_hsv = np.array({upper.tolist()})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
