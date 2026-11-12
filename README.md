# Multimedia Mining & Indexing

Lab work for the *Multimedia Mining and Indexing* course: image processing
with OpenCV, content-based image retrieval (colour, texture, shape), image
segmentation, and 3D graphics / mesh simplification with OpenGL.

## Contents

| Folder | Topic | Main techniques |
|---|---|---|
| [arithmetic-operations](arithmetic-operations/) | Image arithmetic | weighted addition, interactive cross-fade, subtraction / absdiff |
| [color_space](color_space/) | Colour spaces | BGR, HSV and L\*a\*b\* channel decomposition |
| [lab1](lab1/) | Colour segmentation | interactive HSV thresholding, chroma key (green screen) |
| [lab2](lab2/) | Dominant colours | K-Means colour quantisation, palette, elbow method |
| [lab3](lab3/) | Texture | Gabor filter bank, texture similarity segmentation |
| [lab5](lab5/) | Shape retrieval | Hu moments, JSON index, precision@k evaluation |
| [lab6](lab6/) | Segmentation | interactive marker-based watershed |
| [lab7](lab7/) | 3D modelling | interactive Bézier surface (OpenGL / GLFW) |
| [devoir](devoir/) | Assignment 1 - colour CBIR | colour histograms, dominant colours, Bhattacharyya, retrieval evaluation |
| [devoir_indiv2](devoir_indiv2/) | Assignment 2 - mesh simplification | vertex clustering, edge collapse (QEM), OpenGL comparison viewer |

Shared test images live in [assets/](assets/).

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All scripts locate their data relative to their own file, so they can be run
from any directory, e.g. `python lab1/ex2.py`. Every script accepts `--help`.

Notebooks (`lab2`, `lab5`, `devoir`) should be opened with their own folder as
the working directory (the default with `jupyter notebook` / VS Code).

## Repository layout

```
.
├── assets/                  shared sample images
├── arithmetic-operations/
├── color_space/
├── lab1/ lab2/ lab3/ lab5/ lab6/ lab7/
├── devoir/                  colour-based image retrieval (assignment 1)
├── devoir_indiv2/           mesh simplification (assignment 2)
└── requirements.txt
```
