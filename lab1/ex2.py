"""Exercise 2 - Chroma key (green-screen background replacement).

1. Convert the image to HSV and build a mask of the green pixels.
2. Clean the mask with morphological opening/closing and a small blur
   so the edges of the subject are smoother.
3. Keep the subject from the original image and take the background
   pixels from the new background image.

Usage:
    python ex2.py [foreground] [background] [--save result.png]
Press any key to close the window.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# Bounds found with ex1.py
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def green_mask(image, lower=LOWER_GREEN, upper=UPPER_GREEN):
    """Return a float mask in [0, 1] where 1 means 'green background'."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove isolated specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    mask = cv2.GaussianBlur(mask, (5, 5), 0)                # soft edges
    return mask.astype(np.float32) / 255.0


def replace_background(foreground, background, mask):
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    alpha = mask[..., None]  # broadcast over the 3 colour channels
    result = foreground * (1.0 - alpha) + background * alpha
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("foreground", nargs="?", default=HERE / "fondvert.png")
    parser.add_argument("background", nargs="?", default=HERE / "newback.jpeg")
    parser.add_argument("--save", help="optional path to save the composited image")
    args = parser.parse_args()

    image = load_image(args.foreground)
    background = load_image(args.background)

    mask = green_mask(image)
    result = replace_background(image, background, mask)

    if args.save:
        cv2.imwrite(args.save, result)
        print(f"Saved {args.save}")

    cv2.imshow("Original", image)
    cv2.imshow("Mask (white = background)", (mask * 255).astype(np.uint8))
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
