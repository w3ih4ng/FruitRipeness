# Fruit Ripeness — Mode A

A local desktop application comparing five image-processing methods and two
hybrid variants across multiple fruits. The final classifier is **one shared
MobileNetV2 CNN**: a frozen pretrained backbone with a trained ripeness head.
The earlier **Random Forest models** remain available as experimental benchmarks.
Predictions are image-level **unripe, ripe or overripe**, not fruit-species detection.

## Open the application

From this existing folder in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

1. Choose **CNN - MobileNetV2 (shared model)** in Classifier.
2. Choose the training date **28 Aug 2026, 08:58:10 UTC**.
3. Use **Images & folders** for image/folder predictions, **Saved validation** for
   development metrics, or **Saved final Test** for the completed Test report.

In **Saved final Test → Open Test report**, select:

```text
outputs/final_test/20260828T100939_923549Z
```

That report already contains all eight variants, comparison PNGs, CSVs and example
images under `previews/`. Do not retrain or tune settings using Test results.
The normal image picker still blocks original-Test content and identical copies;
use the saved final report for Test evidence. See [UI guide](docs/UI.md).

## Names used in the application

| Display name | Existing ID / folder | Meaning |
|---|---|---|
| CNN - MobileNetV2 (shared model) | `shared_cnn` | One classifier shared by every input variant |
| Random Forest (per-method models) | Individual method folders | Earlier RF model for each input variant |
| Raw image (baseline) | `baseline` | No segmentation; reference input |
| Method 1 - HSV colour threshold | `hsv` | Saturation/value foreground threshold |
| Method 2 - Otsu threshold | `otsu` | Automatic grayscale threshold |
| Method 3 - K-means clustering | `kmeans` | Colour-based grouping |
| Method 4 - GrabCut | `grabcut` | Rectangle-initialised foreground extraction |
| Method 5 - Watershed | `watershed` | Marker-controlled segmentation |
| Hybrid A - HSV + GrabCut union | `hybrid` | Union of two foreground masks |
| Hybrid B - HSV-seeded GrabCut | `hybrid_refined` | Colour-seeded, bounded GrabCut refinement |

Names are display-only. Folder names, model bytes, metadata, CSV IDs, splits and
processing rules are unchanged. Dates are shown in **UTC**, not local time. Runs
created in the same second also show their exact IDs so they remain distinct.
Hybrid A/B identify designs, not a ranking. Keep both in the report.

## What to keep

| Location | Purpose |
|---|---|
| `src/fruitripeness/` | Processing, training, evaluation and desktop UI code |
| `tests/` | Regression tests, retained to protect reproducibility |
| `docs/` | Method explanations, experiment protocols and dataset limitations |
| `data/raw/` | Original dataset, unchanged |
| `outputs/` | Complete trained runs, validation results and final Test reports |
| `requirements.txt`, `requirements-cnn.txt`, `pyproject.toml` | Pinned installation requirements |

The application automatically selects the newest compatible saved run. Models
live inside complete output runs, not a separate deployment folder.
Do not rename or split up saved run folders: models depend on accompanying metadata
and manifests. Generated data/models/results are ignored by Git; back them up
separately. The source repository alone cannot restore them.

Old downloaded `.patch` files are only installers. After an update is applied and
committed, you can move those downloads out of the project. This cleanup does not
delete dataset archives, caches, environments or generated runs on your PC.
The obsolete planning document and unused `models/README.md` placeholder are removed;
their contents remain recoverable from the patch or your Git history.

## Setup on another computer

Clone the existing private repository; do not create another project. Install
Python 3.12, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-cnn.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Copy your backed-up original dataset and complete output runs separately. Select
`.venv\Scripts\python.exe` as the VS Code interpreter. Keep pinned dependencies;
saved models check compatible versions. No retraining is needed to use them.

## Technical references for the report

- [Raw/RF baseline](docs/BASELINE.md)
- [HSV](docs/METHOD1_HSV.md), [Otsu](docs/METHOD2_OTSU.md),
  [K-means](docs/METHOD3_KMEANS.md), [GrabCut](docs/METHOD4_GRABCUT.md),
  [Watershed](docs/METHOD5_WATERSHED.md)
- [Hybrid A](docs/HYBRID.md), [Hybrid B](docs/HYBRID_REFINED.md)
- [Shared CNN protocol](docs/SHARED_CNN.md), [final Test protocol](docs/FINAL_TEST.md)
- [Dataset limitations](docs/DATASET_AUDIT.md), [output files](outputs/README.md)

The method/protocol documents retain historical reproduction commands; they are
not a request to rerun experiments after final Test evaluation. Keep an AI-use
record and ensure every member can explain their contribution.

Dataset: [Fruits Ripeness Classification Dataset](https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset).
Original images, labels and duplicates are retained. Cross-split duplicates limit
claims about independent generalisation; all model scores are uncalibrated.
