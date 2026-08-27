# Method 1: HSV foreground segmentation

## What changes

The baseline uses minimally prepared image pixels. Method 1 first makes a foreground
mask using HSV-space thresholding, applies that mask to the original RGB image, and
then uses the SAME 32 x 32 RGB feature extractor and Random Forest configuration.
Each method has its own fitted model. This method does not change source images,
remove duplicates, relabel examples or alter Train/Test membership.

HSV describes hue (colour family), saturation (colour intensity) and value
(brightness). This implementation thresholds S and V, accepting every hue because
the dataset contains different fruit colours and ripeness stages. It does not use
green = unripe as a segmentation rule, nor receive the true label or fruit name.

## Fixed v1 algorithm

1. Apply EXIF orientation and convert the image to RGB, then HSV.
2. Normalise the Pillow HSV channels from 0..255 to 0..1.
3. Candidate foreground: saturation >= 0.20 AND value >= 0.10; accept all hues.
4. Apply one 3 x 3 binary opening and one 5 x 5 binary closing to the mask.
   Edge padding avoids introducing an artificial black rim on a fully coloured image.
5. Fill enclosed mask holes. Original RGB values inside them remain unchanged,
   allowing enclosed dark blemishes to remain in the extracted region.
6. Keep every connected component with area >= max(16 pixels, 0.5% of image area).
   Eight-neighbour connectivity is used. More than one component can remain.
7. Set background RGB pixels to black, retaining original foreground RGB values.
   There is no crop or colour replacement inside retained regions.
8. Resize the result to 32 x 32 using the same bilinear operation as the baseline,
   flatten and scale it using the shared feature extractor.

These parameters define the first implementation; they were not selected by searching
the original Test set. Morphology is supporting mask cleanup, not the team's final
hybrid. The later hybrid must specify what it combines and be evaluated separately.

Processing operates at source resolution (300 x 300 for this dataset). Kernel sizes
are fixed pixels; behaviour on uploads at substantially different resolutions is a
limitation to address consistently before the UI is finalised.

## Failure handling and interpretation

- An empty mask produces a black processed input and an explicit empty flag. The
  image is still included in training/evaluation; there is no silent deletion or
  substitution of the baseline image.
- Masks covering more than 98% of the image are marked near_full for inspection.
- Other masks are marked nonempty, not correct. Mask status/coverage is diagnostic,
  not segmentation accuracy. No reference masks are available in this dataset.
- Saturated leaves, hands, trays or backgrounds may remain. Pale fruit can be
  removed. The algorithm finds coloured regions, not semantic fruit boundaries.
- Filling holes can also include unwanted background enclosed by foreground.
- The retained dataset limitations still apply. Better classification does not by
  itself prove better masks, biological maturity estimates or robustness.
- Improvement over the baseline is not guaranteed. Keep measured failures for the
  Mode A comparison rather than hiding them or changing the baseline.

## Apply the update to the existing project

Save method1_hsv_update.patch beside pyproject.toml. Run the check first and stop if
it reports an error:

```powershell
git apply --check .\method1_hsv_update.patch
```

If no error is reported:

```powershell
git apply .\method1_hsv_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run commands one at a time. Expect 20 passing tests. Do not apply the same patch twice.

## Run Method 1

Use the exact baseline run folder you already generated:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method hsv --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

If using another baseline run, replace only the final path with that run's folder.
No baseline retraining is required if its metadata and environment remain compatible.
The command checks dataset/split identity before training, and checks the feature
extractor, classifier settings, seed and relevant package versions before comparing.
Use --jobs 2 if you prefer fewer CPU workers; timing is not included in the comparison
table because worker count and hardware can differ.

The method trains on 3,547 images, evaluates 887 validation images, and does not
evaluate the original 180 Test images. Default model parameters match the baseline.

## Output

A new timestamped folder is written under outputs/hsv/. Existing baseline and HSV
runs remain unchanged. The terminal prints its full path.

| File | Purpose |
|---|---|
| model.joblib | HSV-trained model and processing/version metadata |
| metadata.json | Fixed preprocessing specification, model settings and split ID |
| metrics.json | Actual validation metrics, per-fruit results and mask diagnostics |
| split_manifest.csv | Same source images and partitions as the baseline |
| predictions_validation.csv | Predictions, true labels and uncalibrated class scores |
| confusion_matrix_validation.csv | True labels in rows, predictions in columns |
| processing_diagnostics.csv | Coverage, status and processing time for every fit/validation image |
| comparison_validation.csv | Baseline and HSV metrics overall and per fruit |
| comparison_validation.json | Measured differences and comparison identifiers |
| previews/*.png | Original, mask and processed image with actual truth/prediction |
| previews/index.csv | Preview source path and selection reason |

Previews contain the first validation image for each fruit/stage pair plus up to
three additional misclassifications. They are deliberately not a best-case-only
gallery. Filenames in index.csv allow locating the exact source and prediction.

Processing functions are shared by training, inference and preview generation.
Saved models record their processing specification. A mismatched specification is
rejected rather than applying new preprocessing to an old model silently.
Only load joblib models produced by your own project; never load untrusted files.

## Technical references

- Pillow modes and HSV representation: https://pillow.readthedocs.io/en/stable/handbook/concepts.html
- Binary opening: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html
- Connected components: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

These document the implementation primitives; they are not evidence that this
particular threshold configuration is optimal for fruit images.
