
---

# Image Similarity Analysis

This project is a Python-based application that analyzes and compares images based on color characteristics. It computes color histograms, identifies dominant colors, and measures the similarity between images using histogram and color-based metrics. The results include visualizations of the most similar images, making it useful for image processing and content-based image retrieval (CBIR) applications.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1: Load and Display Images](#step-1-load-and-display-images)
  - [Step 2: Calculate and Display Color Histograms](#step-2-calculate-and-display-color-histograms)
  - [Step 3: Calculate and Display Dominant Colors](#step-3-calculate-and-display-dominant-colors)
  - [Step 4: Calculate Similarity Distances](#step-4-calculate-similarity-distances)
  - [Step 5: Plot Results](#step-5-plot-results)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project reads a dataset of images, calculates various color-based features for each image, and compares them to determine similarities. The similarity metrics include:

- **Color Histograms**: Measures color distribution across channels.
- **Dominant Colors**: Identifies primary colors within each image.
- **Similarity Measures**: Uses Bhattacharyya distance for histograms and Euclidean distance for dominant colors to quantify image similarity.

This code can be useful in various applications, including content-based image retrieval systems and image clustering tasks.

---

## Requirements

- **Python 3.x**
- **Jupyter Notebook**
- **Libraries**:
  - `opencv-python`: For image processing and color histogram calculations.
  - `numpy`: For numerical operations.
  - `scikit-learn`: For KMeans clustering.
  - `scipy`: For calculating color-based similarity.
  - `matplotlib`: For visualizing histograms and image pairs.
  - `Pillow`: For handling and displaying images.

You can install these libraries with:

```bash
pip install opencv-python numpy scikit-learn scipy matplotlib pillow
```

## Setup

1. Clone or download the repository.
2. Place your image dataset in a directory and update the `image_dir` path in the code to point to your images folder (e.g., `'../devoir/DVI1'`).
3. Open the project in Jupyter Notebook or any compatible IDE.

---

## Usage

Run each cell step-by-step in Jupyter Notebook for modular analysis and visualization. Here is a breakdown of each part:

### Step 1: Load and Display Images

Load images from the specified directory, showing the paths of all loaded images.

### Step 2: Calculate and Display Color Histograms

Compute and display histograms for each color channel in each image. This gives an understanding of the color distribution across RGB channels.

### Step 3: Calculate and Display Dominant Colors

Identify and display dominant colors for each image using KMeans clustering. This can help in identifying the primary color themes present in the images.

### Step 4: Calculate Similarity Distances

Compute pairwise distances between images:
- **Histogram Distance**: Based on Bhattacharyya distance between histograms for each color channel.
- **Dominant Color Distance**: Based on Euclidean distance between the dominant color sets of each image.
- **Global Distance**: A normalized, combined metric of histogram and dominant color distances.

### Step 5: Plot Results

Display the top 5 most similar image pairs based on each similarity metric (histogram, dominant colors, global metric) for visual comparison.

---

## Results

- **Histogram-based similarity**: Measures similarity based on color distribution.
- **Dominant color-based similarity**: Measures similarity based on prominent colors in each image.
- **Global metric**: A combination of both histogram and dominant color distances.

The project outputs sorted lists of similar image pairs and displays them for visual analysis.

---

## Contributing

Contributions to improve the code or add new similarity measures are welcome. Feel free to fork this repository and submit a pull request with any enhancements or fixes.

---

