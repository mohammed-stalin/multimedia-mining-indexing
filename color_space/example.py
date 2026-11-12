"""Split an image into its colour channels in several colour spaces.

Shows the B, G, R channels, then the H, S, V channels and the
L*, a*, b* channels of the same image.

Usage:
    python example.py [image]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent.parent / "assets"

SPACES = [
    ("BGR", None, ("Blue", "Green", "Red")),
    ("HSV", cv2.COLOR_BGR2HSV, ("Hue", "Saturation", "Value")),
    ("Lab", cv2.COLOR_BGR2LAB, ("L*", "a*", "b*")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=ASSETS / "rose.jpg")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # matplotlib expects RGB

    fig, axes = plt.subplots(len(SPACES), 4, figsize=(14, 9))
    for row, (name, code, channel_names) in enumerate(SPACES):
        converted = image if code is None else cv2.cvtColor(image, code)
        axes[row, 0].imshow(original_rgb)
        axes[row, 0].set_title(f"Original ({name} row)")
        for col, (channel, ch_name) in enumerate(zip(cv2.split(converted), channel_names), start=1):
            axes[row, col].imshow(channel, cmap="gray")
            axes[row, col].set_title(f"{name}: {ch_name}")
        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
