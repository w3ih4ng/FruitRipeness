# Baseline: original dataset, minimal processing

## Purpose

Measure a reproducible starting point before implementing five member methods and
the hybrid. Do not deliberately weaken the baseline or assume it will be inaccurate.
No enhancement, denoising, segmentation or augmentation is applied. Ordinary input
preparation is EXIF orientation, RGB conversion, bilinear resize to 32 x 32, flattening
to 3,072 features and scaling to [0,1]. This same extractor should be reused after
each method's processing so the comparison isolates that method's effect.

Model: RandomForestClassifier with 300 trees, max_features=sqrt,
class_weight=balanced and random_state=42. It predicts unripe, ripe or overripe;
it is not a fruit-type classifier or multi-object detector. Fruit names and labels
are metadata for splitting/reporting only and never model inputs.

## Original data policy

All source images, duplicates and labels are retained. Supplied Train/Test membership
does not change. A fixed 20% validation holdout is selected ONLY from original Train,
stratified by fruit and stage. No files are moved or renamed.

| Partition | Images | Purpose |
|---|---:|---|
| Fit (80% of original Train) | 3,547 | Fit the model |
| Validation (20% of original Train) | 887 | Develop and compare methods |
| Original Test | 180 | Final comparison after all choices are frozen |

All 4,614 source records appear in split_manifest.csv. Its split_id is derived from
the assigned records and SHA-256 input hashes. These hashes identify the dataset;
they are not used to remove duplicates. The split is reproducible for unchanged
data, package versions and seed. It is not claimed to be duplicate-independent.

No test-image pixel features are loaded by the default run. Test filenames, labels
and file hashes are recorded to preserve the manifest, but do not influence fitting
or parameter selection. Do not tune against test results when final evaluation is run.

## Run in the existing VS Code project

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m fruitripeness.baseline --data data/raw
```

If CPU usage is inconvenient, set --jobs 2. This changes worker count, not the
experimental method. Keep the same setting when comparing timing on your machine.
If your extraction has additional nested folders, --data must point at the folder
that directly contains Train and Test (one wrapper level is detected automatically).

Each run writes a fresh UTC timestamped folder under outputs/baseline/; existing
runs are not overwritten. The final terminal line gives its exact location.

| File | Contents |
|---|---|
| metrics.json | Validation accuracy, balanced accuracy, macro precision/recall/F1, per-class and per-fruit metrics, confusion matrix and timing |
| metadata.json | Configuration, counts, split ID, package versions, hardware platform description and known limitations |
| split_manifest.csv | Every source path, label, experiment partition and content hash |
| predictions_validation.csv | Actual validation predictions, truth labels and class scores |
| confusion_matrix_validation.csv | Counts; rows are true labels, columns predicted labels |
| model.joblib | Locally generated model plus feature/method/split metadata |

Metrics are fractions (0.90 = 90%). Probability outputs are uncalibrated model scores.
Prediction timing is a batch average, not interactive single-image latency. Image
loading/feature time is measured separately from model fitting and prediction.
Invalid images cause a clear failure rather than silent removal.

The later UI will use the saved model without retraining on every upload. The
predict_image helper in baseline.py uses the same feature preparation as training.
It does not require a specially named input file.

## Final Test evaluation (later, not the next command)

After settings and methods are frozen, append --evaluate-test to the baseline
command. This fits the same seeded model on the same fit partition, evaluates
validation AND original Test, and adds predictions_test.csv and
confusion_matrix_test.csv. It does not merge validation into training. Use the same
protocol for all methods. Do not change parameters after inspecting final scores.

## Interpretation and limits

Retained duplicates and conflicting labels can inflate or distort scores. Treat
results as performance under the supplied dataset protocol, not a guarantee of
generalisation to independently collected fruit images. Do not claim physical
ripeness, food safety, fruit-specific localisation or blemish-area accuracy from
this classifier. Multi-fruit scenes retain the supplied image-level labels.

Keep the baseline even if it outperforms an enhancement. A changed feature extractor,
model or split requires rerunning every method for a fair comparison. Use validation
results to develop the hybrid; the final test should be a one-time comparison.

## Technical references

- Random Forest: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Holdout splitting: https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.train_test_split.html
- Model persistence/security: https://scikit-learn.org/1.8/model_persistence.html

Only load your own model.joblib files. Loading pickle/joblib from an untrusted source
can execute code. Preserve the recorded library versions when reusing a saved model.
