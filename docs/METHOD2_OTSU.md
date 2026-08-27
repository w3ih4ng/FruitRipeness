# Method 2: Otsu automatic threshold segmentation

## Controlled experiment

Otsu v1 replaces the foreground-selection step. It retains the same source data,
3,547 fit images, 887 validation images, RGB feature extractor and 300-tree Random
Forest as the baseline and HSV method. Original Test is not evaluated during this
development command. No images, duplicates or labels are changed or removed.

Grayscale is used only to make the mask. The retained foreground keeps its original
RGB colours; the classifier still receives flattened 32 x 32 RGB pixels, not a
grayscale image or a binary mask. Each method is trained separately.

## Fixed algorithm

1. Apply EXIF orientation and convert to RGB, then Pillow's 8-bit L grayscale.
2. Calculate a 256-bin intensity histogram. For each threshold t from 0 to 254,
   split dark pixels (intensity <= t) from bright pixels (intensity > t).
3. Choose the threshold maximising between-class variance, equivalent to minimising
   weighted within-class variance. If several thresholds tie, use the first.
   Constant-intensity images have no meaningful two-class separation: flag an empty
   mask instead of arbitrarily selecting the entire image.
4. Otsu itself does not say which class is fruit. This implementation uses an
   explicit heuristic: treat the class occupying more of the outer image frame as
   background. The frame width is round(5% of the smaller dimension), at least one
   pixel (15 pixels on this dataset). Count each frame pixel once.
5. Keep the other class as foreground. On equal border counts, choose the class with
   fewer pixels in the whole image. If that also ties, choose the bright class.
6. Use the same cleanup as HSV: one 3 x 3 opening, one 5 x 5 closing, enclosed-hole
   filling, then retain every eight-connected component with area at least
   max(16 pixels, 0.5% of total image area). Apply the same edge padding.
7. Set excluded RGB pixels to black. Retain the full image dimensions; do not crop.
8. Pass the processed RGB image through the unchanged common feature extractor
   and train/evaluate the same Random Forest configuration.

Threshold values are calculated separately for each image from its own pixels.
That is part of the algorithm, not test-label tuning. No fruit names, filenames or
true labels influence threshold selection or foreground polarity. The initial
grayscale channel and border rule are fixed, not selected through a parameter search.

## Important distinctions and limits

- Automatic threshold does not mean automatic identification of fruit. A histogram
  split can separate shadows or leaves instead of the fruit/background boundary.
- The border heuristic assumes background dominates the frame. Large fruit touching
  the edge, multiple fruit and cluttered scenes can violate that assumption.
- Similar fruit/background brightness can make grayscale thresholding ineffective,
  even where their colours differ. Some fruit detail can also be removed.
- Empty masks produce black processed images but remain in training/evaluation.
  The diagnostics explicitly flag them; there is no silent fallback to baseline.
- Mask coverage/status is not segmentation accuracy. Reference masks are unavailable.
- A performance change alone does not prove a particular cause. Inspect predictions
  and masks, and describe possible explanations as hypotheses unless separately tested.
- All limitations of the original unchanged dataset remain applicable.

## Apply and run

Save method2_otsu_update.patch beside pyproject.toml. Check first:

```powershell
git apply --check .\method2_otsu_update.patch
```

If no error appears, run the following one at a time, stopping on an error:

```powershell
git apply .\method2_otsu_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expect 25 passing tests. Do not apply the patch twice. This version uses the same
dependencies as the HSV version; reinstalling updates the project's version metadata.

Run against the existing baseline:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method otsu --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual timestamped baseline folder if it differs. The command checks the
data/split identity before training and checks feature/model compatibility before
comparison. Existing baseline and HSV models do not need to be retrained.

## Output and report evidence

The terminal prints a new run folder under outputs/otsu/. It includes model.joblib,
metadata.json, metrics.json, split_manifest.csv, predictions_validation.csv,
confusion_matrix_validation.csv, comparison_validation.csv/.json and
processing_diagnostics.csv. The comparison contains baseline and Otsu results overall
and per fruit; the stored HSV run remains a separate result for later aggregation.

Otsu diagnostics add otsu_threshold (0..254, blank for a constant image) and
foreground_polarity (bright, dark or none). Preview PNGs show original, mask and
processed images, plus the actual prediction, threshold and selected polarity.

Preview selection is unchanged: first validation example per fruit/stage and up to
three additional failures. The index.csv records each source and selection reason.
Neither previews nor classification accuracy establish ground-truth mask quality.

The same process_image function is used during training, saved-model inference and
preview generation. Saved processing specifications must match the implementation.
Only load your own project-generated joblib files, never untrusted models.

## Implementation reference

OpenCV's tutorial explains the Otsu objective and its distinction from a manually
chosen global threshold: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

This project implements that histogram objective directly in NumPy and tests it
against a brute-force within-class-variance calculation. OpenCV is not an added
dependency. The border-polarity rule and shared mask cleanup are project design
choices, not claims made by the Otsu algorithm itself.
