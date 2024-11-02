
---

# Image Similarity Analysis with Color Histograms and Dominant Colors

## Overview

This project aims to perform image similarity analysis by calculating color-based characteristics (histograms and dominant colors) of images in two folders. The first folder (`DVI1`) contains 16 reference images categorized into four types (animals, flowers, ocean, sky), and the second folder (`DVI2`) contains one representative image per category. The goal is to identify and display the top similar images from the first folder based on three different metrics: color histograms, dominant colors, and a global metric that combines the two.

## Project Structure

- **DVI1**: Folder with 16 reference images for similarity comparison.
- **DVI2**: Folder with 4 test images (1 per category) used for similarity testing.

The process is split into two main parts:
1. **Feature Extraction**: Calculate and save color histograms and dominant colors for images in `DVI1`.
2. **Similarity Analysis**: Select one image from `DVI2`, compute its characteristics, and calculate its similarity to all images in `DVI1` using histogram, dominant color, and global metrics.

## Key Features

1. **Color Histogram Calculation**: Calculates color histograms for each image, with normalization, across RGB channels.
2. **Dominant Color Extraction**: Uses KMeans clustering to identify the dominant colors in each image, with a threshold to retain only prominent colors.
3. **Similarity Calculation**: Computes the similarity between images based on:
   - **Histogram Distance**: Bhattacharyya distance across RGB histograms.
   - **Dominant Color Distance**: Euclidean distance between dominant colors.
   - **Global Metric**: A weighted combination of histogram and dominant color distances.
4. **Visualization**: Displays the selected image alongside its most similar images for each metric (histogram, dominant color, and global metric).

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Libraries:
  - `opencv-python` for image processing
  - `numpy` for numerical computations
  - `scikit-learn` for KMeans clustering
  - `scipy` for distance calculations
  - `Pillow` for image handling
  - `matplotlib` for visualization

### Installation

Install required packages via pip:

```bash
pip install opencv-python-headless numpy scikit-learn scipy pillow matplotlib
```

## Usage

1. **Feature Extraction**: Calculate and save histograms and dominant colors for all images in `DVI1`.
   
2. **Similarity Analysis**: Select a random image from `DVI2` and calculate its distance from all images in `DVI1` based on the stored characteristics. The top similar images for each metric are displayed.

### Code Execution Steps

In a Jupyter notebook, separate the code into small, manageable cells as outlined below:

#### 1. Import Libraries

```python
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial import distance
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
```

#### 2. Define Helper Functions

**Color Histogram Calculation**

```python
def calculate_color_histogram(image_path):
    img = cv2.imread(image_path)
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    histograms = {}

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms[color] = hist

    return histograms
```

**Dominant Color Extraction**

```python
def find_dominant_colors(image_path, k=20, threshold=0.05):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}. Skipping...")
        return []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img)
    counts = Counter(kmeans.labels_)
    total_pixels = sum(counts.values())

    dominant_colors = [kmeans.cluster_centers_[idx].tolist() for idx, count in counts.items() if count / total_pixels > threshold]

    return dominant_colors
```

**Distance Calculations**

```python
def bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def dominant_color_distance(colors1, colors2):
    return distance.cdist(colors1, colors2, 'euclidean').mean()
```

#### 3. Feature Extraction and JSON Saving

```python
# Set directories
folder1_dir = './DVI1'
image_paths_folder1 = [os.path.join(folder1_dir, f) for f in os.listdir(folder1_dir) if f.endswith(('.jpg', '.png'))]

# Calculate and save histograms and dominant colors
image_histograms = {os.path.basename(path): calculate_color_histogram(path) for path in image_paths_folder1}
image_dominant_colors = {os.path.basename(path): find_dominant_colors(path) for path in image_paths_folder1}

# Save features to JSON
with open("histograms.json", "w") as f:
    json.dump(image_histograms, f)

with open("dominant_colors.json", "w") as f:
    json.dump(image_dominant_colors, f)
```

#### 4. Load Features and Perform Similarity Analysis

```python
# Load characteristics from JSON files
with open("histograms.json", "r") as f:
    image_histograms = json.load(f)

with open("dominant_colors.json", "r") as f:
    image_dominant_colors = json.load(f)

# Randomly select an image from DVI2 and calculate similarity
folder2_dir = './DVI2'
image_paths_folder2 = [os.path.join(folder2_dir, f) for f in os.listdir(folder2_dir) if f.endswith(('.jpg', '.png'))]
selected_image_path = random.choice(image_paths_folder2)
selected_histogram = calculate_color_histogram(selected_image_path)
selected_dominant_colors = find_dominant_colors(selected_image_path)

# Calculate distances
histogram_distances = {name: sum(bhattacharyya_distance(selected_histogram[c], hist[c]) for c in ('b', 'g', 'r')) for name, hist in image_histograms.items()}
dominant_color_distances = {name: dominant_color_distance(selected_dominant_colors, colors) for name, colors in image_dominant_colors.items()}
global_distances = {name: (histogram_distances[name] + dominant_color_distances[name]) for name in histogram_distances}

# Sort distances
top_n = 6
top_histogram = sorted(histogram_distances.items(), key=lambda x: x[1])[:top_n]
top_dominant_color = sorted(dominant_color_distances.items(), key=lambda x: x[1])[:top_n]
top_global = sorted(global_distances.items(), key=lambda x: x[1])[:top_n]
```

#### 5. Visualization

```python
def plot_similar_images(similar_pairs, selected_image_path, title):
    plt.figure(figsize=(15, 12))
    plt.suptitle(title, fontsize=20)
    
    plt.subplot(1 + len(similar_pairs), 2, 1)
    selected_img = Image.open(selected_image_path)
    plt.imshow(selected_img)
    plt.axis('off')
    plt.title("Selected Image")
    
    for i, (img_name, score) in enumerate(similar_pairs):
        img_path = os.path.join(folder1_dir, img_name)
        img = Image.open(img_path)
        
        plt.subplot(1 + len(similar_pairs), 2, 2 * (i + 1) + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Similar Image {i+1} - Score: {score:.4f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_similar_images(top_histogram, selected_image_path, "Top Similar Images by Histogram")
plot_similar_images(top_dominant_color, selected_image_path, "Top Similar Images by Dominant Colors")
plot_similar_images(top_global, selected_image_path, "Top Similar Images by Global Metric")
```

## License

This project is licensed under the MIT License.

---

