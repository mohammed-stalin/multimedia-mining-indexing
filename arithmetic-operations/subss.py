"""Image subtraction with cv2.subtract (saturated: negative values clip to 0).

Also shows cv2.absdiff, which keeps the magnitude of the difference
in both directions and is what you usually want for change detection.

Usage:
    python subss.py [image1] [image2]
Press any key to close the windows.
"""
import argparse
from pathlib import Path

import cv2

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image1", nargs="?", default=ASSETS / "star.png")
    parser.add_argument("image2", nargs="?", default=ASSETS / "square.png")
    args = parser.parse_args()

    image1 = load_image(args.image1)
    image2 = load_image(args.image2)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    subtracted = cv2.subtract(image1, image2)  # image1 - image2, clipped at 0
    difference = cv2.absdiff(image1, image2)   # |image1 - image2|

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("subtract(image1, image2)", subtracted)
    cv2.imshow("absdiff(image1, image2)", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
