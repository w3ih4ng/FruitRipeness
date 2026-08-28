# Method 5: marker-controlled watershed

## Controlled experiment

Keep every source image, duplicate and label unchanged. This branch uses the same
3,547 fit images, 887 validation images, 32 x 32 RGB feature extractor and 300-tree
Random Forest settings as the earlier methods. It trains its own classifier, not
a new classifier architecture. The original 180-image Test set remains reserved.
Existing baseline, HSV, Otsu, K-means and GrabCut models do not need retraining.

Watershed starts from labelled seed regions and expands regions according to image
boundaries. This version supplies markers automatically, without hand-drawn masks,
fruit names, filenames or ripeness labels. Its Otsu mask only INITIALISES the seeds;
the final mask comes from OpenCV watershed followed by the shared cleanup.

## Fixed algorithm

1. Apply EXIF orientation and convert to RGB. Keep the original RGB image for the
   final output. Make a working copy with longest side at most 300 pixels, keeping
   aspect ratio, rounding dimensions (minimum one) and using Pillow BILINEAR.
   Never enlarge small inputs. The dataset's 300 x 300 images remain that size.
2. Reuse the Otsu v1 coarse-mask helper: Pillow L grayscale, maximum between-class
   variance threshold, and the class occupying fewer pixels in the outer 5% frame
   as foreground. Equal frame counts choose the smaller whole-image class, with
   a final tie choosing bright. Constant grayscale has no coarse foreground.
   This calls the helper, not the entire Method 2 processing pipeline.
3. Apply one 3 x 3 opening, with edge padding, to the coarse mask. Clear a border
   of max(2 pixels, round(2% of the smaller working dimension)). Then retain every
   eight-connected coarse component with area at least max(16 pixels, 0.5% of
   working image area). No closing or hole filling is done at this marker stage.
4. Compute exact Euclidean distance to background inside the coarse foreground.
   In EACH retained component, select pixels whose distance is at least 50% of
   that component's own maximum. This gives interior cores. A per-component
   threshold avoids automatically suppressing small components just because a
   larger one has a greater maximum distance. It does not guarantee correct seeds.
5. Label the eight-connected core regions with foreground marker IDs 2, 3, ... .
   A coarse component can have more than one disconnected core. Marker count is
   therefore not a fruit count or a count of ripeness classes.
6. Dilate the coarse mask three times with a 3 x 3 square. Pixels outside this
   expanded region, plus the fixed border from step 3, become background marker 1.
   Leave the remaining transition area at marker 0 (unknown). Foreground cores
   are inside the coarse mask, so they do not overlap the background seeds.
7. Keep a copy of these INITIAL int32 marker labels for previews. Run cv2.watershed
   on the unblurred working BGR uint8 image. OpenCV modifies the supplied labels
   using colour differences; this implementation does not flood a grayscale mask.
8. Retain labels greater than 1 as candidate foreground. Exclude background 1,
   unresolved 0 and watershed boundaries -1. Union all foreground regions: this
   project predicts image-level ripeness, not separate object instances.
9. Resize only the binary mask to the original dimensions with NEAREST. Apply
   the common cleanup at original resolution: one 3 x 3 opening, one 5 x 5 closing,
   enclosed-hole filling and all eight-connected components of at least
   max(16 pixels, 0.5% of original image area), with the same edge padding.
10. Keep the ORIGINAL RGB values inside the final mask and set excluded pixels
    to black. Do not crop or substitute a different method for an empty result.
    Use the unchanged RGB feature extractor and Random Forest training path.

The background border has a two-pixel minimum because OpenCV replaces its outermost
row/column with watershed boundaries. A second row/column leaves background seeds
inside that boundary. All numerical settings are fixed before the full validation
run. No validation-label parameter search is used for this initial version.

## Empty cases, reproducibility and limitations

- A working dimension below five pixels yields too_small. No retained coarse
  region yields no_coarse_foreground. An all-background watershed output yields
  empty_result. These inputs remain in training/evaluation as black images.
- Unexpected OpenCV errors stop the experiment with a clear error; no images are
  silently discarded. The runtime version must match the recorded OpenCV version.
- Dependencies are unchanged from GrabCut: OpenCV 4.13.0 from the pinned
  opencv-python-headless 4.13.0.92 package, SciPy 1.17.0 and the existing numerical
  libraries. Watershed has no random initialisation in this implementation.
- Markers are estimates, not verified fruit/background annotations. Similar
  brightness, clutter, shadows and fruit at the image edge can make them wrong.
  Watershed cannot reliably recover a fruit that has no foreground marker.
- The coarse-mask choice constrains where the method can expand. Distance cores
  do not prove that the retained region is fruit, nor that all fruit is retained.
- Downsampling large inputs can lose fine detail. Fixed-pixel morphology behaves
  differently at different resolutions. Shared cleanup may fill or reconnect
  watershed boundaries, so the final mask is not an instance-segmentation result.
- No reference masks are supplied: neither mask coverage, seed count nor boundary
  fraction measures segmentation accuracy. Ripeness labels alone do not establish
  blemish-area accuracy. Preserve the unchanged dataset's existing limitations.

## Apply and run

Save method5_watershed_update.patch beside pyproject.toml in the existing project:

```powershell
git apply --check .\method5_watershed_update.patch
```

If no error appears, run these one at a time, stopping on any error:

```powershell
git apply .\method5_watershed_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expect 47 passing tests. Apply the patch only once. Reinstalling updates the project
version metadata; no new dependency is added by this patch. Then train:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method watershed --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual baseline timestamp if different. The command checks data/split
identity before fitting and common feature/model compatibility before comparison.
It prints a new run folder under outputs/watershed/. Earlier runs are not overwritten.

## Outputs and report evidence

The run contains model.joblib, metadata.json, metrics.json, split_manifest.csv,
validation predictions/confusion matrix, comparison_validation.csv/.json against
baseline overall and per fruit, processing_diagnostics.csv and previews/.

| Diagnostic | Meaning |
|---|---|
| otsu_threshold, foreground_polarity | Coarse-mask initialisation, not final watershed labels |
| watershed_width, watershed_height | Working image dimensions |
| watershed_markers | Number of foreground core markers, not fruit count |
| watershed_seed_fraction | Initial foreground-marker fraction of the working image |
| watershed_background_fraction | Initial background-marker fraction |
| watershed_boundary_fraction | Fraction labelled -1 before resizing/cleanup, including the outside frame |
| watershed_status | completed, empty_result, too_small or no_coarse_foreground |

foreground_fraction and mask_status describe the FINAL cleaned mask. Initial
markers are all zero when the method is skipped before marker construction.

The watershed previews are 1330 x 460 with four panels: original, actual initial
markers, final mask and processed RGB. Blue means foreground markers (all IDs use
the same blue), black background and grey unknown. All-grey markers in a skipped
case mean no markers were initialised. Earlier methods keep their three-panel
1000 x 460 previews. The optional ProcessingResult.markers field contains the
working-resolution initial labels; it is None for the earlier methods.

Selection remains the first validation example per fruit/stage plus up to three
additional errors, with source and selection reason in previews/index.csv. Include
both successes and failures in the report; these previews are not random samples.
Training, previews and inference share the same processing function. Only load
your own project-generated joblib models.

This completes implementation of the five individual methods. The hybrid and
single-image/multiple-image/folder comparison UI remain subsequent milestones.

## Implementation reference

[OpenCV watershed tutorial](https://docs.opencv.org/4.13.0/d3/db4/tutorial_py_watershed.html)
explains marker labels, distance-based initialisation and watershed boundaries.
The automatic polarity rule, per-component core thresholds, border policy, size
limit, cleanup and preview layout are this project's explicit design choices.
