"""Weighted addition (blending) of two images with cv2.addWeighted.

    result = alpha * img1 + beta * img2 + gamma

Usage:
    python addition.py [image1] [image2] [--alpha 0.5] [--beta 0.5]
Press any key (or Esc) to close the window.
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
    parser.add_argument("image1", nargs="?", default=ASSETS / "mountain.jpg")
    parser.add_argument("image2", nargs="?", default=ASSETS / "horse.jpg")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of image1")
    parser.add_argument("--beta", type=float, default=0.5, help="weight of image2")
    parser.add_argument("--gamma", type=float, default=0.0, help="scalar added to the sum")
    args = parser.parse_args()

    image1 = load_image(args.image1)
    image2 = load_image(args.image2)

    # Both images must have the same size for addWeighted
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    weighted_sum = cv2.addWeighted(image1, args.alpha, image2, args.beta, args.gamma)

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("Weighted sum", weighted_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
