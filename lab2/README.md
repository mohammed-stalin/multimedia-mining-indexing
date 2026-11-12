# Lab 2 - Dominant colours with K-Means

Notebook: [`dominantecolor.ipynb`](dominantecolor.ipynb)

Each pixel is treated as a point in RGB space and clustered with K-Means. The
cluster centres are the representative colours and the share of pixels in each
cluster gives how dominant it is.

## Steps

1. Load `car.jpg` (BGR → RGB) and resize it, keeping the aspect ratio.
2. Reshape to an `(N, 3)` array of pixels and run `KMeans(n_clusters=K)`.
3. Compute the proportion of each colour from the label histogram.
4. Display the original image, the image quantised to `K` colours and a palette
   bar sorted by dominance (hex codes and percentages are printed).
5. Elbow curve (inertia vs `K`) to help choose `K`.

## Run

```bash
jupyter notebook dominantecolor.ipynb
```

Change `IMAGE_PATH` and `K` in the *Parameters* cell.

## Fixes made

* Cells could fail when run out of order (`NameError: image`); the notebook is
  now linear and runs top to bottom.
* `n_init` and `random_state` are set explicitly (no scikit-learn warning,
  reproducible palette).
* Colours are sorted by proportion; added quantised image and elbow method.
