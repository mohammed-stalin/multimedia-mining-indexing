# Assignment 1 - Colour-based image similarity and retrieval

Content-based image retrieval using two colour descriptors (colour histograms and
dominant colours) and a combined global metric. A full write-up is in
[`images_similarity.pdf`](images_similarity.pdf).

## Data

| Folder | Content |
|---|---|
| `DVI1/` | database: 16 images, 4 per category |
| `DVI2/` | queries: 1 image per category |

Categories (deduced from the file number):

| Files | Category |
|---|---|
| 14xx | castles |
| 34xx | coast / sea |
| 81xx-82xx | floral art |
| 89xx | tulip fields |

## Files

| File | Role |
|---|---|
| [`features.py`](features.py) | feature extraction and distances shared by both notebooks |
| [`pairwise_similarity.ipynb`](pairwise_similarity.ipynb) | compares all DVI1 images with each other: distance matrices and most similar pairs |
| [`image_retrieval.ipynb`](image_retrieval.ipynb) | indexes DVI1 to JSON, queries with a DVI2 image, evaluates all queries |
| `histograms.json`, `dominant_colors.json` | generated feature index of DVI1 |

## Method

**Colour histogram** - one normalised 256-bin histogram per B, G, R channel.
Distance = sum of the three Bhattacharyya distances (0 to 3).

**Dominant colours** - K-Means with k = 20 on the RGB pixels; clusters covering
more than 5 % of the pixels are kept together with their proportions.
Distance = weighted symmetric nearest-colour distance: each colour is matched to
the closest colour of the other image, weighted by its proportion.

**Global metric** - `w · d_hist / max(d_hist) + (1 − w) · d_colour / max(d_colour)`
with `w = 0.5`. Both distances are normalised first so they are on the same scale.

## Results (precision@4 over the 4 queries)

| Query | Category | Histogram | Dominant colours | Global |
|---|---|---|---|---|
| 1468.jpg | castles | 50 % | 75 % | 75 % |
| 3486.jpg | coast | 75 % | 100 % | 75 % |
| 8116.jpg | floral art | 100 % | 100 % | 100 % |
| 8994.jpg | tulip fields | 100 % | 100 % | 100 % |
| **mean** | | **81 %** | **94 %** | **88 %** |

## Run

```bash
pip install opencv-python numpy scikit-learn scipy pillow matplotlib
jupyter notebook image_retrieval.ipynb
```

Run the notebooks from this folder. In `image_retrieval.ipynb`, set
`QUERY = "8994.jpg"` to choose the query (default random) and `WEIGHT` to change
the balance of the global metric.

## Fixes made

* `updated.ipynb` displayed the "chosen image" using a leftover loop variable
  (`filename`, the last image of DVI1), so it showed the wrong image or failed.
* The global metric added a Bhattacharyya sum (0-3) to an RGB distance (0-441),
  so it was in practice only the colour distance; both are now normalised.
* The dominant-colour distance averaged *all* pairs of colours, so even an image
  compared with itself had a large distance; it now matches nearest colours and
  uses the cluster proportions.
* K-Means runs on a pixel subsample with explicit `n_init` (faster, no warnings).
* Duplicate code from the notebooks moved into `features.py`; notebooks renamed
  (`Untitled.ipynb` → `pairwise_similarity.ipynb`, `updated.ipynb` →
  `image_retrieval.ipynb`) and an evaluation section was added.
