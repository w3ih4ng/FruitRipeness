# Method 4: automatic GrabCut foreground extraction

## Controlled experiment

Keep all source images, duplicates and labels unchanged. This method uses the same
3,547 fit images, 887 validation images, 32 x 32 RGB feature extractor and 300-tree
Random Forest settings as the earlier runs. Train a separate classifier for the
GrabCut branch. The experiment command does not evaluate the original 180-image
Test set. Existing baseline, HSV, Otsu and K-means models remain usable.

GrabCut estimates foreground and background using colour models and graph cuts.
It needs an initial foreground/background separation. This implementation creates
that initialisation automatically for every image; it does not ask a user to draw
a rectangle or correct masks. This is a fixed image-processing heuristic, not a
trained semantic fruit detector.

## Fixed algorithm

1. Apply EXIF orientation and convert to RGB. Keep this original image for the
   final masked output.
2. Make a temporary working image with longest side at most 160 pixels. Preserve
   aspect ratio, round each dimension to the nearest integer (minimum one), and
   resize with Pillow BILINEAR. Never enlarge a small input. The dataset's 300 x 300
   images therefore use a 160 x 160 working image. This limits graph computation
   for later folder input; it is a documented part of this method, not a change
   to the shared classifier feature extractor or source files.
3. Set horizontal and vertical margins to round(5% of their respective working
   dimensions), with at least one pixel on each side. The interior rectangle is
   (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y). For this dataset,
   that is (8, 8, 144, 144). Outside is definite background; the rectangle is
   initially probable foreground. No pixels are manually marked definite foreground.
4. If the rectangle interior or its complement contains fewer than five pixels,
   return an empty mask with status too_small. OpenCV's colour models require
   enough initial samples. If the working image has a single RGB colour, return
   an empty mask with status constant_working_image. Retain both cases in the data.
5. Convert the working image to BGR uint8 for OpenCV. Allocate fresh foreground and
   background models for this image. Reset OpenCV's random seed to 42 and run
   cv2.grabCut with GC_INIT_WITH_RECT for five iterations, using one OpenCV thread.
6. Keep labels GC_FGD and GC_PR_FGD. Treat GC_BGD and GC_PR_BGD as background.
   Enlarge only this binary mask to the original image dimensions with NEAREST
   interpolation. Never use centroid colours or the resized image as the final RGB.
7. Apply the shared cleanup at original resolution: one 3 x 3 opening, one 5 x 5
   closing, enclosed-hole filling, and retain every eight-connected component with
   area at least max(16 pixels, 0.5% of image area), with the same edge padding.
8. Keep original RGB values inside the final mask and set the rest to black. Do not
   crop or fall back to the baseline when the mask is empty. Pass the result to the
   unchanged 32 x 32 RGB feature extractor and Random Forest training pipeline.

All choices above are fixed before the full validation run. No filename, fruit
name, ripeness label, other image or manually edited mask influences segmentation.
The image-specific colour models are rebuilt during inference, using the same
function as training and preview generation.

## Reproducibility and errors

The project pins opencv-python-headless==4.13.0.92 (cv2 version 4.13.0), which accepts
the existing NumPy 2.3.5 requirement. Headless means OpenCV's desktop window system
is omitted; image processing still works and can be used by a later browser UI.
Do not install another OpenCV package alongside it in this project's virtual env.

GrabCut checks the OpenCV runtime version against its saved processing specification.
The wrapper serialises its OpenCV calls, resets the random seed for each image and
restores the previous OpenCV thread count even after failure. It does not restore
OpenCV RNG state; subsequent project GrabCut calls always set their own seed.
Fixed seeds and packages support repeatability, but do not promise identical
floating-point results across every operating system or OpenCV build.

Expected small/constant/empty cases produce black images with diagnostics, not
image exclusions. Unexpected OpenCV errors stop training with an explicit error;
they are not silently converted into successful masks or dropped samples.

## Limitations to discuss

- Fruit touching the excluded frame can be removed because the frame is initially
  definite background. The automatic rectangle is a heuristic, not a fruit box.
- Similar fruit/background colours, busy backgrounds, overlapping objects and
  background shadows can lead to wrong or empty masks.
- Downsampling can lose thin edges and small blemishes. Nearest-neighbour mask
  enlargement can produce coarse boundaries. This is not a validated blemish-area
  measurement method.
- Shared morphology can modify the graph-cut result, including filling enclosed
  background gaps. The final pipeline is GrabCut plus documented cleanup.
- No reference masks are supplied, so mask coverage is not segmentation accuracy.
  Classification improvements or declines alone do not prove why a mask succeeded
  or failed. Inspect both successful and failed examples.
- The original unchanged dataset's limitations remain in force. Develop with
  validation only, and reserve original Test until all method choices are frozen.

## Apply and run

Save method4_grabcut_update.patch beside pyproject.toml in the existing project.
Check first:

```powershell
git apply --check .\method4_grabcut_update.patch
```

No output means the check passed. Run these one at a time, stopping on any error:

```powershell
git apply .\method4_grabcut_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expect 39 passing tests. Apply the patch only once. The requirements install adds
OpenCV; do not skip it. Then use the same saved baseline:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method grabcut --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual baseline timestamp if different. The command checks data/split
identity before fitting, then compares common feature/model settings and library
versions before writing the comparison. GrabCut prints progress every 250 images.
Allow several minutes for processing; its graph-cut step is slower than the earlier
thresholding methods, and elapsed time depends on your CPU and image content.

## Outputs and report evidence

The command prints a new timestamped folder under outputs/grabcut/. It contains
model.joblib, metadata.json, metrics.json, split_manifest.csv, validation predictions
and confusion matrix, comparison_validation.csv/.json against the baseline overall
and per fruit, processing_diagnostics.csv and previews/. Earlier runs are untouched.

| GrabCut diagnostic | Meaning |
|---|---|
| grabcut_width, grabcut_height | Dimensions used for graph-cut processing |
| grabcut_rect | Automatically generated x,y,width,height in working pixels |
| grabcut_iterations | Requested iterations, or zero when intentionally skipped |
| grabcut_status | completed, empty_result, too_small or constant_working_image |

foreground_fraction and mask_status describe the FINAL resized/cleaned mask.
grabcut_status describes the earlier GrabCut stage; for example, completed can
still result in a final empty mask if cleanup removes every small component.

Preview PNGs show original, final mask and processed image, actual label/prediction,
working dimensions and GrabCut status. Selection remains the first validation
example per fruit/stage plus up to three additional errors. previews/index.csv
records the sources and selection reasons; these are not random samples.

Only load your own project-generated joblib models. The folder-input comparison UI
remains a later milestone; this patch adds Method 4 to the experiment runner.

## Implementation reference

[OpenCV GrabCut tutorial](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html)
explains initialisation, mask labels and the graph-cut approach. The automatic
rectangle, fixed resolution, seed, special-case handling and morphology are this
project's design choices, not fruit-detection guarantees from GrabCut itself.
