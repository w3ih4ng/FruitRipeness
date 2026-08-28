# One shared CNN for all processing methods

The training workflow below still reserves Test. Once settings are frozen, the
separate `fruitripeness.final_test` evaluator scores all eight variants without
retraining; see `docs/FINAL_TEST.md`. Saved final Test is a separate UI tab.

This optional update adds one MobileNetV2 model shared by raw images, HSV, Otsu,
K-means, GrabCut, Watershed, original hybrid and refined hybrid. Existing Random
Forest code, saved models, masks and results remain available. Do not compare new
CNN numbers against old RF numbers as if preprocessing were the only change.

## Install in the same project folder

Close the UI and save `shared_cnn_update.patch` beside `pyproject.toml`. This patch
expects the refined-hybrid update (project version 0.10.0) to be installed already.
Run each command separately and stop on errors. Successful git apply is usually silent.

```powershell
git apply --check .\shared_cnn_update.patch
git apply .\shared_cnn_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-cnn.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The optional file installs CPU builds of PyTorch 2.10.0 and torchvision 0.25.0
from the official PyTorch wheel index for Python 3.12 on Windows/Linux. No GPU,
CUDA setup or new project folder is needed. Existing dependencies stay pinned.
The first package installation downloads several hundred MB. Do not substitute
other package versions if pip reports an error; share the error first.

Random Forest and its UI still work without installing the CNN dependencies.
CNN operations then give an installation message. JSON-only CNN result viewing
does not load PyTorch. The CNN integration tests skip when its dependencies are
absent; install them before treating that part of the suite as verified.

## Train once, not once per method

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.cnn --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual baseline run path if different. First use downloads approximately
14 MB of ImageNet pretrained weights from PyTorch. A failed download stops training;
there is no silent random-weight fallback. Model loading in the UI never downloads.

The command verifies that the current dataset reproduces the baseline split ID,
then prints progress for each processing method and each training epoch. It runs
on CPU with two threads by default; keep the terminal open. Segmentation and CNN
feature extraction are the main work, not fitting the small classifier head.
Duration depends on the computer; no fixed completion time is promised.

For reduced peak memory, use `--batch-size 8` (default 32). `--threads 2` controls
CNN training CPU use. Defaults are `--epochs 30 --patience 5`. Changing these is
a new experimental configuration and should be recorded rather than silently
mixed with earlier runs. RF settings and image-processing parameters are unchanged.

Outputs go to a NEW `outputs/shared_cnn/<timestamp>/`:

- `model.pt`: ONE checkpoint, containing the frozen backbone and trained head.
- `metadata.json`: preprocessing, method list, versions, split, training settings,
  selected epoch and checkpoint SHA-256. Written as completed only after outputs finish.
- `metrics.json`: separate validation results for each of the eight input variants.
- `comparison_validation.csv`, per-method prediction and confusion-matrix CSVs.
- `split_manifest.csv`, `processing_diagnostics.csv`, `history.csv`.
- `features/`: frozen CNN feature arrays, approximately 173 MiB for this dataset.
  These are intermediates; the UI needs model/metadata/metrics/manifest, not features.

All outputs are already ignored by Git. No pretrained or fitted weights go in the
patch. Existing runs are never overwritten. An interrupted run remains marked
incomplete and is ignored by the UI; rerun creates a new run. This first version
does not resume interrupted feature extraction or fine-tune the convolutional layers.

## Exact experimental protocol

### Same source split, multiple views

Keep the existing 3,547 fitting images, 887 validation images and 180 original-Test
images. All labels, duplicates and original membership are retained. The baseline
split hash must match before training starts. Test bytes may be hashed for the
manifest, but Test images are never decoded, trained on, previewed or evaluated by
this training command. There is deliberately no `--evaluate-test` option yet.

For EACH fitting image, create eight pixel-derived views: raw RGB plus the seven
implemented processing outputs. All views of that source stay in its fitting
partition. Repeat this separately for validation; validation views never enter
classifier fitting. There are 28,376 fitting views, not 28,376 independent source
images. Existing dataset duplicates can still cross partitions; this protocol
does not claim to remove the dataset's pre-existing leakage limitations.

Every fitting source contributes exactly one view per method per epoch. There is
no method-specific sampling advantage or filename/fruit-label feature. Empty
masks remain black inputs with diagnostics; images are not skipped or replaced
with their raw versions. Unexpected processing errors fail visibly.

### Input and model

Apply the selected frozen processing method at its existing settings. Convert its
processed RGB to a 224 x 224 input by aspect-preserving bilinear resize and centred
black padding. Upscaling is allowed; there is no centre crop. Convert to float32
CHW, divide RGB by 255, then normalise with ImageNet mean (0.485, 0.456, 0.406) and
standard deviation (0.229, 0.224, 0.225). The same function serves training and UI.

The backbone is torchvision MobileNetV2 with `IMAGENET1K_V2` weights. Freeze all
backbone parameters AND keep it in evaluation mode, so batch-normalisation running
statistics do not change. Global average pooling gives 1,280 features. Replace
the ImageNet classifier with one trainable linear layer with three logits, ordered
`unripe`, `ripe`, `overripe`. Softmax gives the displayed uncalibrated scores.

This is transfer learning with a frozen CNN feature extractor and a trained
classification head, NOT end-to-end CNN fine-tuning and NOT a Random Forest on
CNN features. The deployed model includes both backbone and head.

Letterboxing intentionally differs from the pretrained weights' default resize/
centre-crop transform so whole-scene fruit is not cropped out. This is an explicit
project adaptation, not a claim of using the exact ImageNet benchmark preprocessing.
No random colour jitter, crop, flip or geometric augmentation is added in this version.

### Training and checkpoint selection

Cache each frozen feature vector once, then train the shared head with AdamW
(learning rate 0.001, weight decay 0.0001), head minibatches of 128 and weighted
cross-entropy. Class weights are inverse fitting-stage frequency only. Shuffle
all fitting views each epoch with seed 42. Use CPU deterministic algorithms;
exact numerical identity across different hardware/software is not guaranteed.

At each epoch, evaluate all eight validation variants using the SAME head. Select
the checkpoint with the highest arithmetic mean of the eight per-method macro F1
scores; keep the earlier epoch on ties. Stop after five consecutive non-improving
epochs or 30 total epochs by default. There is no separate best checkpoint per
method. Restore the selected shared head before saving any final comparison.

Validation participates in model selection: these remain development results.
The method scores are correlated because they use the same source images. Do not
pool the eight view sets and present that as eight times as many independent cases.

## UI: image, folder and saved comparison

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

1. Set Classifier to **CNN - MobileNetV2 (shared model)** (`shared_cnn`). The
   initial default remains **Random Forest (per-method models)** (`random_forest`).
2. Select ONE CNN training date. Every method uses this same CNN.
3. Use **Refresh models** if saved runs were added after opening the application.
4. Choose image(s) or a folder. Select a method, Compare all six, or Compare hybrid
   versions as before. The six-method view remains five individual methods plus
   the explicitly selected hybrid variant. `baseline` is also available as an
   individual raw-image prediction reference, not a seventh processing technique.
5. Saved validation loads all eight variant measurements from the selected shared
   run. Filter by fruit and export PNG/CSV as before.
6. Switch Classifier back to **Random Forest (per-method models)** to use the
   previous models/results. Full IDs remain unchanged in metadata and exports.

The UI automatically accepts compatible models from this team's configured local
outputs folder; do not place downloaded or unknown models there. Model loading is
shared once per batch, not repeated for each method. All CNN
prediction rows record the same checkpoint hash, while method IDs differ. The
loader uses CPU and `weights_only=True`, checks the checkpoint hash, embedded
metadata, input/processing versions and class order.

Classifier/run controls lock during work. Switching classifier retains prior
displayed/exportable results until explicitly rerun or reloaded: their recorded
backend does not change. Prediction cards identify their backend, evaluation PNGs
name it in the heading, and both CSV types include a `backend` column. Mixed RF/CNN
runs, or different shared CNN checkpoints in one method comparison, are rejected.

CNN UI `feature_ms` measures input resizing/normalisation; `prediction_ms` includes
the convolutional backbone and classification head. It is not just the cached
head-training speed. As before, timing excludes model loading, image disk read,
report rendering and UI refresh. Single-run timing is not a performance benchmark.

## Interpretation and verification

- Comparing methods under the shared CNN asks: which input processing works best
  for this SAME fixed classifier trained on a balanced mixture of methods?
- It does not establish the best achievable model specialised for each method.
  RF comparisons trained separate classifiers; that is a different protocol.
- CNN and RF also differ in input resolution, feature extraction and pretraining.
  Do not attribute any RF-to-CNN gain solely to one processing method.
- The CNN predicts image-level ripeness, not fruit species, instance boundaries,
  chemical ripeness or per-object labels in mixed scenes. It does not repair masks.
- Cleaner masks may remove useful dark/pale surface cues. Record successes and
  failures, macro F1, balanced accuracy, per-class recall and per-fruit results.
- Preserve the Random Forest benchmark. No improved full-dataset accuracy is
  claimed until the team runs the shared CNN and inspects its saved measurements.

Verification includes all 79 prior non-GUI tests, four dependency-free CNN checks,
and 11 optional real-MobileNet integration tests (94 non-GUI tests in total).
Integration tests use explicitly random weights ONLY to keep tests offline and
check training/inference mechanics; production always requests pretrained weights.
An additional local smoke check exercises the actual pretrained weights on a small
original-Train subset, not the project's full validation set. Six desktop tests
are supplied; graphical Tk verification must be done on Windows because the build
environment has no display. Run the full suite and inspect the new controls there.

## Primary references

- [PyTorch transfer learning: fixed feature extractor](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Torchvision 0.25 MobileNetV2 weights and input transforms](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.mobilenet_v2.html)
- [Official matching CPU package versions](https://pytorch.org/get-started/previous-versions/)
- [Data leakage and split-first evaluation guidance](https://scikit-learn.org/stable/common_pitfalls.html)

Record AI assistance and ensure the team can explain frozen-backbone transfer
learning, shared-versus-separate classifiers, validation selection and limitations.
