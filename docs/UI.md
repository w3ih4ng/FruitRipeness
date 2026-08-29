# Desktop comparison UI

Launch from the existing project:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

This is a local Tkinter application, not a web server. If Tkinter is missing,
modify the same Python 3.12 installation to include Tcl/Tk. The UI never trains
a model or changes dataset images. Run tests after source updates;
desktop integration tests require a graphical Tk display.

## Automatic model selection

- **CNN - MobileNetV2 (shared model)**: one frozen checkpoint for raw input,
  all five individual methods and both hybrids. Select its UTC training date at
  the top. The team's completed run is
  `outputs/shared_cnn/20260828T085810_496281Z`.
- **Random Forest (per-method models)**: earlier benchmark models. The application
  automatically uses each method's newest compatible completed run.
- Dates and descriptive method names are display labels only. Exact run IDs
  remain in CSVs and metadata. Same-second runs remain separately selectable.
- **Refresh models** rescans the configured `outputs` folder. Selection is based
  on completion timestamp, never Test score. Loading still rejects incompatible
  package versions, processing settings, hashes, class order and split metadata.
- There is no Models & runs tab or repeated trust checkbox. The application is
  intended for this team's locally generated models under its configured outputs
  folder; do not copy unknown model files into that folder.
- Changing the selector does not relabel existing results. Rerun predictions or
  reload validation to show the new selection. Saved final Test always displays
  the checkpoint recorded in that report, independently of this selector.

## Images & folders

1. Choose images or a folder. Enable **Include subfolders** before choosing
   the folder when needed.
2. Select a descriptive method name and **Run selected method**.
3. **Compare all six** runs the five individual methods and the chosen hybrid.
   **Hybrid A - HSV + GrabCut union** is the original version;
   **Hybrid B - HSV-seeded GrabCut** is the refined version.
   **Compare hybrid versions** runs both. Raw input is a separate reference option.
4. Select a result row and choose **Preview selected image** to inspect it.
5. Export predictions as CSV, the displayed comparison as PNG, or a combined PDF
   report containing visual evidence and the recorded prediction table.

Cards show original/processed images, masks, ripeness predictions, uncalibrated
class scores, retained area and timing. Neither a predicted label nor mask coverage
is dataset accuracy. Input filenames do not supply labels or features. CSVs
keep stable internal method IDs; the results table uses readable names.

JPG/JPEG, PNG, WEBP and BMP are supported. Unsupported entries appear in the log;
corrupt images and method failures appear as error rows. Cancellation retains
completed rows. Images above 20 megapixels are rejected, not silently resized.
Preview reconstruction checks that source bytes still match the saved prediction.

**Original-Test images and identical copies remain blocked in this tab.** Use
the completed final Test report and its exported previews instead.

## Saved validation

Choose **Load selected runs** to display validation metrics and confusion matrices
from the selected classifier/runs. Filter by fruit and reload as needed.
Comparisons reject incompatible splits or classifiers. Validation figures are
development measurements, not final Test scores. Export the displayed scope as
CSV, PNG or PDF; raw baseline is a reference, not another member's method.

## Saved final Test

Choose **Open Test report** and select:

```text
outputs/final_test/20260828T100939_923549Z
```

Choose the report folder, not the dataset Test folder or the model folder. This
read-only view shows all eight variants using the one frozen CNN, checks the saved
report and metrics checksum, and offers per-fruit/overall CSV, PNG and PDF exports.
The full report already contains comparisons, confusion matrices, predictions
and example PNGs under `previews/`. It neither evaluates images nor retrains.
See [final Test protocol](FINAL_TEST.md) for formats and limitations.

## Exports and reproducibility

Choose new filenames outside source and selected run/report folders. Existing
files are not overwritten. Raw IDs and full paths in CSVs preserve provenance;
readable date/method labels appear in newly rendered figures. Existing saved
PNGs/CSVs are not rewritten by this naming update.

Processing, input preparation and prediction times exclude disk reads, loading,
rendering and GUI refresh. Hardware and warm-up affect them; one observation is
not a timing benchmark. Masks remain heuristics, not annotated fruit masks.
Keep successes, failures, validation and final Test evidence in the report.

## Surface analysis

The **Surface analysis** tab reuses images selected on **Images & folders**. It
applies median denoising, contrast stretching, the selected segmentation method,
heuristic blemish detection, contour-based object detection and pixel-based sizing.
It does not use dataset labels or a trained classifier.

For a technically valid blemish fraction, the application reports a transparent
project quality grade: **Good** for at most 5% detected blemish area,
**Acceptable** for above 5% through 15%, and **Poor** for above 15%. An empty or
too-small foreground is **Not graded**, never silently treated as clean. These
thresholds are an assignment rule rather than a biological or commercial fruit
inspection standard because the dataset has no verified blemish masks.

Surface results can be exported as CSV, the displayed evidence as PNG, or both
measurements and visual evidence as PDF. Diameter remains in pixels because the
dataset contains no consistent reference object of known physical size.

## Camera and video

The **Camera & video** tab uses the classifier selected in the top bar and one
selected processing method. It does not train, alter the source media or read
labels. Predictions apply to the complete frame; the displayed object boxes are
contours from the segmentation mask, not separate per-fruit classifications.

For a webcam, choose a non-negative camera index (normally `0`) and select
**Start live camera**. The preview updates with the frame-level ripeness scores
and detected-object count. **Capture snapshot for comparison** saves the original
unannotated frame, selects it on **Images & folders**, and stops the camera. After
the camera finishes closing, use **Compare all six** as for any other image.

For uploaded video, choose an MP4, AVI, MOV or MKV source, then select
**Process and export annotated video**. Choose a new MP4 or AVI filename outside
the source and saved-model folders. Every decoded frame is processed, so slower
methods such as K-means or GrabCut can take substantially longer than the video
duration. The application writes to a temporary file first and only finalises the
annotated video after successful completion. Cancelling discards that incomplete
video while retaining completed frame rows for optional CSV export.

The exported CSV records frame number, timestamp, method, prediction, three
uncalibrated scores, timing and detected-object count. Video containers/codecs
depend on the local OpenCV build; use MP4 first and AVI if a computer cannot create
MP4 output. Camera permission and a camera not already occupied by another program
are required for live mode.
