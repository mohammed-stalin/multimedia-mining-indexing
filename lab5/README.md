# Lab 5 - Shape retrieval with Hu moments

Notebook: [`code.ipynb`](code.ipynb)

The 7 Hu moments are invariant to translation, scale and rotation, and are used
here as a compact shape descriptor for content-based shape retrieval.

## Data

Silhouettes from the MPEG-7 shape dataset (GIF, white object on black):

| Folder | Content |
|---|---|
| `Formes/formes1` | database: 5 classes (apple, bell, bird, camel, device0) × 4 images |
| `Formes/formes2` | queries: 1 new image per class |
| `hu_moments_folder1.json` | generated index (image → 7 log-scaled Hu moments) |

## Pipeline

1. Binarise the image and fill the **largest contour by area** (the silhouette).
2. Compute the Hu moments of the filled region, take `|h7|` so mirrored shapes
   match, and log-scale: `-sign(h) · log10|h|`.
3. Index `formes1` into `hu_moments_folder1.json`.
4. For a query from `formes2`, rank the database by L2 distance and show the top 4.
5. Evaluate every query with **precision@4** (correct matches shown in green).

Result on this dataset: **mean precision@4 = 50 %**. Bird and bell are retrieved
well; camel and device are confused because Hu moments only capture the global
mass distribution of the shape.

## Run

```bash
jupyter notebook code.ipynb
```

Set `QUERY = "camel-5.gif"` in the query cell to choose a specific image
(default: random).

## Fixes made

* The log transform was `log10(|h + 1e-10|)` (epsilon added *inside* the
  absolute value, which shifts small moments); now `log10(|h| + eps)`.
* The largest contour was chosen by number of points, which is unreliable with
  `CHAIN_APPROX_SIMPLE`; it is now chosen by area.
* Moments are computed on the filled silhouette and `h7` is made
  reflection-invariant: precision@4 went from 35 % to 50 %.
* Paths no longer depend on going up a level (`../lab5/...`).
* Added the evaluation over all queries.
