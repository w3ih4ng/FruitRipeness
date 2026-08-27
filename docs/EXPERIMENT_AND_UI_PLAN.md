# Mode A experiment and UI plan

## Current status

This starter implements the dataset audit only. Splitting, image-processing
methods, feature extraction, training, inference and the comparison UI are not
implemented yet. No placeholder predictions or invented performance metrics are included.
Continue modifying this repository; do not generate replacement starter folders.

## Dataset decisions before training

- Canonical stages: unripe, ripe, overripe. Map the original folder Overipe to
  overripe in metadata only. Retain source paths and labels for traceability.
- Keep fruit identity in the manifest. Do not infer maturity from the filename
  or supply it as a predictor; filenames are label metadata only.
- Manually review exact duplicates with contradictory labels. Quarantine unresolved
  conflicts, rather than retaining a convenient label.
- Perceptual hashes flag candidates, not proven duplicates. Review them before
  making grouping or exclusion decisions. Repeated views of one specimen/scene
  should stay together wherever that identity can be established.
- Do not trust the original Train/Test split without leakage checks. Create one
  versioned group-aware train/validation/test split after review. Aim for 70/15/15
  by fruit and stage, allowing deviations to preserve groups. Report actual counts.
- Keep all test data out of parameter selection and hybrid selection. A final
  split must not be claimed leak-free solely because a hash check passes.
- Audit all five fruits first; choose final coverage based on usable independent
  groups and label clarity. More images of the same scene are not more independent
  examples. Do not silently relabel mouldy fruit as merely overripe.
- Scene-level labels do not provide per-object boxes or masks. A photo can contain
  several fruits or mixed stages. First scope the application to one dominant fruit,
  after review; mixed-stage scenes require exclusion or separate annotation.

## Controlled comparison

Each member applies a different processing pipeline to the SAME image IDs. The
methods are parallel experiments, not a five-stage serial pipeline. Train a
separate instance of the same classifier for each method, keeping shared features,
training settings, split and random seed consistent.

Provisional candidates to test on sample images before assigning members:

| ID | Main technique | Qualification |
|---|---|---|
| hsv | HSV colour segmentation | No universal green = unripe rule; avoid retaining only a maturity-specific patch |
| otsu | Otsu threshold segmentation | Choose the channel and foreground rule on validation data |
| kmeans | K-means colour clustering | Select foreground clusters without access to true test labels |
| grabcut | GrabCut foreground extraction | Automatic initialisation; no manual test-image tuning |
| watershed | Marker-controlled watershed | Document automatic foreground/background markers |
| hybrid | Enhanced combined method | Specify from validation evidence, then freeze before testing |

These are candidates, not a confirmed final list. The uploaded images have varied
backgrounds and multiple objects; preliminary trials may justify replacing a method.
Use a raw/minimal-preprocessing baseline as a seventh EXPERIMENTAL reference; the
requested UI still has five methods plus one hybrid. A Random Forest with shared
colour and texture features is the provisional classifier. No deep detector should
replace the fundamental image-processing contribution.

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

1. Create private GitHub repository, install starter, reproduce audit.
2. Review duplicate/label problems, freeze cleaned manifest and shared split.
3. Establish a measured baseline.
4. Implement and test one member method at a time, reusing the same interfaces.
5. Develop hybrid using validation evidence.
6. Add six-method UI, test end-to-end inference and export real report figures.
