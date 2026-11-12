"""Colour features and distances shared by the `devoir` notebooks.

Features
    * colour histogram   - one normalised 256-bin histogram per B, G, R channel
    * dominant colours   - KMeans clusters covering more than `threshold` of the
                           pixels, with their proportions

Distances
    * histogram distance        - sum over channels of the Bhattacharyya distance (0..3)
    * dominant colour distance  - weighted symmetric "chamfer" distance: every colour
                                  is matched to its nearest colour in the other image
    * global distance           - weighted sum of both, each normalised by its maximum
                                  over the database so they live on the same scale
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import distance as sp_distance
from sklearn.cluster import KMeans

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
CHANNELS = ("b", "g", "r")


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def load_bgr(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def category(image_name):
    """Ground-truth category of the dataset images, deduced from the file number."""
    n = int(Path(image_name).stem)
    if n < 2000:
        return "castles"
    if n < 4000:
        return "coast"
    if n < 8500:
        return "floral art"
    return "tulip fields"


# ------------------------------------------------------------------ features
def calculate_color_histogram(image_path, bins=256):
    img = load_bgr(image_path)
    histograms = {}
    for chan, name in zip(cv2.split(img), CHANNELS):
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms[name] = hist.astype(np.float32).tolist()
    return histograms


def find_dominant_colors(image_path, k=20, threshold=0.05, max_pixels=20000, seed=0):
    """Return {'colors': [[R, G, B], ...], 'weights': [p, ...]} sorted by weight."""
    img = cv2.cvtColor(load_bgr(image_path), cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > max_pixels:  # subsample for speed, the palette barely changes
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(len(pixels), max_pixels, replace=False)]

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(pixels)
    counts = np.bincount(kmeans.labels_, minlength=k)
    proportions = counts / counts.sum()

    keep = np.where(proportions > threshold)[0]
    if len(keep) == 0:  # very uniform image: keep at least the biggest cluster
        keep = [int(np.argmax(proportions))]
    order = sorted(keep, key=lambda i: -proportions[i])
    weights = proportions[order] / proportions[order].sum()
    return {
        "colors": kmeans.cluster_centers_[order].tolist(),
        "weights": weights.tolist(),
    }


def extract_features(folder, k=20, threshold=0.05):
    histograms, dominant = {}, {}
    for path in list_images(folder):
        name = os.path.basename(path)
        histograms[name] = calculate_color_histogram(path)
        dominant[name] = find_dominant_colors(path, k=k, threshold=threshold)
    return histograms, dominant


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- distances
def histogram_distance(hist1, hist2):
    return float(sum(
        cv2.compareHist(np.asarray(hist1[c], np.float32), np.asarray(hist2[c], np.float32),
                        cv2.HISTCMP_BHATTACHARYYA)
        for c in CHANNELS
    ))


def dominant_color_distance(dom1, dom2):
    c1, w1 = np.asarray(dom1["colors"]), np.asarray(dom1["weights"])
    c2, w2 = np.asarray(dom2["colors"]), np.asarray(dom2["weights"])
    d = sp_distance.cdist(c1, c2, "euclidean")
    return float(0.5 * (np.dot(w1, d.min(axis=1)) + np.dot(w2, d.min(axis=0))))


def rank(query_hist, query_dom, histograms, dominant, weight=0.5):
    """Distances from a query to every database image, for the three metrics.

    Returns {metric: [(image_name, distance), ...]} sorted ascending.
    """
    names = list(histograms)
    h = np.array([histogram_distance(query_hist, histograms[n]) for n in names])
    c = np.array([dominant_color_distance(query_dom, dominant[n]) for n in names])
    g = weight * h / (h.max() or 1) + (1 - weight) * c / (c.max() or 1)
    return {
        metric: sorted(zip(names, values.tolist()), key=lambda x: x[1])
        for metric, values in (("histogram", h), ("dominant colours", c), ("global", g))
    }
