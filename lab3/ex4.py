"""Exercise 4 - Texture segmentation with a bank of Gabor filters.

The user selects a region of interest (ROI) containing a texture. Every
pixel of the image is described by the local mean and standard deviation of
the responses of a Gabor filter bank (4 orientations x 3 scales), computed
over a window the same size as the ROI. Pixels whose descriptor is close to
the ROI descriptor are marked as belonging to the same texture.

Implementation notes
--------------------
* Kernels are made zero-mean so they respond to texture, not to brightness,
  and the energy |response| is used as the feature.
* The filter bank is applied ONCE to the whole image (float32). Local
  mean / std over a sliding window are obtained with box filters:
  std = sqrt(E[x^2] - E[x]^2). This replaces the original
  "re-filter every window" loop and is orders of magnitude faster.
* Each feature is standardised (z-score) so that no single filter dominates
  the Euclidean distance.
* The threshold adapts to the ROI: factor x the 95th percentile of the
  distances measured inside the ROI itself (override with --threshold).

Usage:
    python ex4.py [image ...] [--factor 1.5] [--threshold T]
For each image: draw a rectangle, press Enter/Space to confirm (c to cancel).
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

ORIENTATIONS = (0, 45, 90, 135)  # degrees
SCALES = (3, 5, 7)


def build_gabor_filters(ksize=31, sigma=5.0, lambd=10.0, gamma=0.5):
    """Bank of Gabor kernels, 4 orientations x 3 scales."""
    filters = []
    for theta in ORIENTATIONS:
        for scale in SCALES:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma * scale / 3,
                theta=np.deg2rad(theta),
                lambd=lambd * scale / 3,
                gamma=gamma,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernel -= kernel.mean()          # zero DC: respond to texture, not brightness
            kernel /= np.abs(kernel).sum()  # L1 normalisation
            filters.append(kernel)
    return filters


def apply_gabor_filters(img, filters):
    """Return the energy |I * g| of every filter response (float32)."""
    img = img.astype(np.float32)
    return [np.abs(cv2.filter2D(img, cv2.CV_32F, k)) for k in filters]


def local_features(responses, window):
    """Per-pixel (mean, std) of every response over a `window`=(w, h) box.

    Returns an array of shape (H, W, 2 * n_filters).
    """
    feats = []
    for r in responses:
        mean = cv2.boxFilter(r, cv2.CV_32F, window)
        sq_mean = cv2.boxFilter(r * r, cv2.CV_32F, window)
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
        feats.extend([mean, std])
    return np.stack(feats, axis=-1)


def texture_extraction(img, x, y, width, height, threshold=None, factor=1.5):
    """Return (binary mask, distance map) of pixels with the ROI texture.

    If `threshold` is None it is derived from the ROI itself:
    factor * (95th percentile of the distances measured inside the ROI),
    i.e. "as close to the reference as the ROI is to itself, with a margin".
    """
    filters = build_gabor_filters()
    responses = apply_gabor_filters(img, filters)
    pixel_stats = local_features(responses, (width, height))

    # Standardise every feature using statistics of the whole image
    flat = pixel_stats.reshape(-1, pixel_stats.shape[-1])
    z = (pixel_stats - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-6)

    roi_vector = z[y:y + height, x:x + width].reshape(-1, z.shape[-1]).mean(axis=0)
    distance = np.linalg.norm(z - roi_vector, axis=-1)

    if threshold is None:
        threshold = factor * np.percentile(distance[y:y + height, x:x + width], 95)
    mask = np.where(distance < threshold, 255, 0).astype(np.uint8)

    # Light clean-up of the binary result
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, distance


def select_region(img, title="Select ROI"):
    roi = cv2.selectROI(title, img, showCrosshair=True)
    cv2.destroyWindow(title)
    return roi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", default=[HERE / n for n in ("a.jpg", "b.jpg", "c.jpg")])
    parser.add_argument("--factor", type=float, default=1.5,
                        help="tolerance relative to the ROI's own spread (default 1.5)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="absolute distance threshold in standardised units (overrides --factor)")
    args = parser.parse_args()

    for image_path in args.images:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        x, y, w, h = select_region(img, f"Select ROI - {Path(image_path).name}")
        if w == 0 or h == 0:
            print(f"No region selected for {image_path}, skipping.")
            continue

        mask, distance = texture_extraction(img, x, y, w, h, threshold=args.threshold, factor=args.factor)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        axes[0].set_title("Original + ROI")
        axes[1].imshow(img[y:y + h, x:x + w], cmap="gray")
        axes[1].set_title("Selected region")
        axes[2].imshow(distance, cmap="viridis_r")
        axes[2].set_title("Distance to ROI texture")
        axes[3].imshow(mask, cmap="gray")
        axes[3].set_title("Extracted texture")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(Path(image_path).name)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
