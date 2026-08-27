# Mode A experiment and UI plan

## Current status

Baseline and HSV Method 1 training are implemented, sharing RGB features (32 x 32)
and a Random Forest. The audit is optional, not a prerequisite. The remaining four
processing methods, the hybrid and the UI remain future milestones. Continue editing this repository;
no replacement starter folder is needed.

## Confirmed dataset policy

- Use the supplied dataset as it is. Retain every image, duplicate, original label
  and original Train/Test membership. No cleaning, deduplication or relabelling.
- The folder spelling Overipe maps to overripe in metadata only. Do not rename files.
- Use the extracted Train and Test folders under data/raw. A single wrapper folder,
  such as data/raw/archive/Train, is also supported automatically.
- Reserve a reproducible 20% of original Train for validation (seed 42; stratify by
  fruit and stage). This changes only metadata, not source files or supplied Test.
- Current counts: 3,547 fitting images, 887 validation images, 180 original Test images.
- Use validation while developing the methods/hybrid. Evaluate original Test only
  after all choices are frozen. Duplicates can inflate scores, and conflicting
  labels can distort scores. Acknowledge this limitation without blocking the work.
- All methods must use the same split ID and image IDs. The saved split manifest
  contains source paths, labels, content hashes and assigned experiment partitions.
- Do not use names/labels as model features. This baseline predicts three ripeness
  stages across five fruits; it does not independently recognise the fruit species.
- Treat existing labels as image-level categories. Multi-fruit/mixed-stage scenes
  remain in this experiment, but no per-object maturity or blemish ground truth
  can be claimed from those labels alone.

## Controlled comparison

Each member applies a different processing pipeline to the SAME image IDs. The
methods are parallel experiments, not a five-stage serial pipeline. Train a
separate instance of the same classifier for each method, keeping shared features,
training settings, split and random seed consistent.

Provisional candidates to test on sample images before assigning members:

| ID | Main technique | Qualification |
|---|---|---|
| hsv | HSV S/V foreground thresholding (implemented) | Accept all hues; S >= 0.20, V >= 0.10, documented mask cleanup; not a semantic fruit detector |
| otsu | Otsu threshold segmentation | Choose the channel and foreground rule on validation data |
| kmeans | K-means colour clustering | Select foreground clusters without access to true test labels |
| grabcut | GrabCut foreground extraction | Automatic initialisation; no manual test-image tuning |
| watershed | Marker-controlled watershed | Document automatic foreground/background markers |
| hybrid | Enhanced combined method | Specify from validation evidence, then freeze before testing |

HSV v1 is now implemented with fixed parameters. The other four techniques remain
candidates until implemented and reviewed. The uploaded images have varied
backgrounds and multiple objects; preliminary trials may justify replacing a method.
Use a raw/minimal-preprocessing baseline as a seventh EXPERIMENTAL reference; the
requested UI still has five methods plus one hybrid. The initial fixed classifier
is a 300-tree Random Forest with balanced class weights and seed 42, using flattened
32 x 32 RGB pixels for every branch. This is minimal baseline preparation, not an
enhancement technique. If the common feature extractor changes later, rerun the
baseline and every method with that same extractor; never compare incompatible
feature/model configurations as if preprocessing were the only difference.

Compare macro F1, balanced accuracy, accuracy, per-stage precision/recall, per-fruit
results and confusion matrices. Record preprocessing and prediction time separately
on the same machine. Run repeated seeds where practical. If evaluating masks, create
a small manually annotated evaluation set and report IoU/Dice separately.

For confidence displays, label Random Forest probability estimates as model scores;
do not imply they are calibrated probabilities. Inspect class calibration before
presenting stronger confidence claims.

## Planned local comparison UI

The intended implementation is a local Python dashboard, not a public hosted service.
Choose the UI framework at the implementation milestone; no framework installed yet.

### Single-image view

- Upload an image, select one of five methods or hybrid, then run inference.
- Show original image, processed image, fruit mask/overlay, predicted stage,
  class scores, preprocessing time and model prediction time.
- Offer Compare all six: the identical input appears in a consistent six-card grid.
- Do not show per-image accuracy when the true label is unknown. Never fabricate
  missing predictions; missing model files must produce a clear not-trained state.
- Use exactly the training-time preprocessing, dimensions and feature order.

### Folder and batch view

- Accept a single image, multiple selected images, or a folder of supported images.
- Offer an explicit include-subfolders option; keep relative paths to distinguish
  files with the same name. The source images must not be changed.
- Apply the selected method, selected subset of methods, or all six consistently.
- Show progress, successful/failed file counts and a per-image results table.
- Handle unsupported files and per-image failures with visible messages; continue
  other valid batch images without silently suppressing failures.
- Show predicted ripeness, class scores and processing times. Do not invent true
  labels for arbitrary uploads; accuracy/F1 require an explicit labelled dataset.
- Export CSV predictions and PNG comparison images. Keep the destination outside
  the source folder, and never overwrite inputs.

### Evaluation view

- Load measured results from the fixed test manifest. Show model version, dataset
  split ID, class support, macro F1, balanced accuracy and confusion matrices.
- Filter by fruit; distinguish aggregate metrics from single-image predictions.
- Export comparison figures as PNG and metrics/predictions as CSV for the report.
- The raw baseline can appear in the evaluation table without becoming another
  member's method. Mark hybrid results honestly even if it does not improve.

### Report and assignment safeguards

- Include original/processed/mask images for successes AND failures.
- No claims of chemical ripeness, sweetness or entire 3D surface coverage from RGB.
- The brief also calls for surface quality and blemish quantification. Class labels
  alone do not support area evaluation. Plan a manually labelled subset and a
  visible-area blemish metric, or obtain explicit tutor agreement to narrow scope.
- Image resizing is not physical calibration. Do not report cm/mm without a scale
  reference; document how calibration requirements apply with the tutor.
- Confirm five-person approval and disclose AI assistance in the assignment form.

## Incremental milestones

1. Private GitHub repository and Python environment: completed by the user.
2. Run the baseline on the original extracted dataset and inspect validation outputs.
3. HSV Method 1 implemented; run it locally and inspect comparison/previews.
4. Add the remaining member methods with the same split/features/model.
5. Develop the hybrid using validation evidence.
6. Add single-image, multiple-image and folder UI workflows with comparison/export.
7. Freeze choices, run final original-Test comparison and produce report figures.
