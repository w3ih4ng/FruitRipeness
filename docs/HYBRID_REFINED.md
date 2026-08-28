# Revised hybrid: HSV-seeded GrabCut

Method ID: `hybrid_refined`. Processing version:
`hybrid_hsv_seeded_grabcut_v1`. This is a separately trained hybrid experiment,
not a sixth member's technique and not an overwrite of the original `hybrid`.

## Why this experiment exists

The original hybrid uses the union of two masks. Anything retained by either
component remains in the union. Both components can retain the tray, and unrestricted
hole filling can fill a tray-shaped region around fruit. The new method replaces
that union rule with selective colour seeds, bounded growth and limited hole filling.

This is still an automatic foreground heuristic, not semantic fruit recognition.
It has no learned fruit detector and cannot reliably distinguish green leaves from
green fruit, brown wood from brown fruit, or every reflection from fruit skin.

## Apply in the existing folder

Close the UI. Save `hybrid_refined_update.patch` beside `pyproject.toml`. Run each
command separately; stop if a command fails. Successful git apply commands normally
produce no output. Do not force a patch over unrelated edits or reapply it twice.

```powershell
git apply --check .\hybrid_refined_update.patch
git apply .\hybrid_refined_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No dependency versions change. This updates project version 0.9.0 to 0.10.0.
Old processing specifications, masks and saved models remain compatible.

## Train only the new variant

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method hybrid_refined --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual baseline run path if different. The same split, RGB32 features and
300-tree Random Forest settings are used. All source images and labels are retained;
empty masks are flagged, not excluded or replaced by the raw image. The command does
not evaluate original Test. It writes a fresh `outputs/hybrid_refined/<timestamp>/`
with its own model, metrics, diagnostics and previews. The baseline comparison is
saved as usual; use Saved validation to compare both hybrids as well.

Preprocessing changes the classifier's input distribution, so a new model is
required. Never feed refined images to the original hybrid model or overwrite its
metadata to bypass compatibility checks. The other models do not need retraining.

## Fixed algorithm

All decisions use pixels only, never filenames, fruit labels, ripeness labels or
manual rectangles. Parameters are shared across images and fruits.

1. Apply EXIF orientation and convert to RGB using the shared image path.
2. Resize a working copy to maximum side 200, preserving aspect ratio with rounded
   dimensions and bilinear interpolation; never upscale.
3. Convert to Pillow HSV, normalise S/V by 255, and select S >= 0.30 and V >= 0.10.
   Accept all hues. Apply a 5 x 5 binary opening and keep 8-connected components
   with at least max(16, ceil(0.5% of working-image pixels)) pixels.
4. Fill only enclosed holes of at most max(9, floor(2% of working-image pixels)).
   Erode the cleaned colour mask BEFORE hole filling by 3 x 3 to obtain hard
   foreground cores. Hole-filled areas are not promoted to hard colour evidence.
5. Define support as pixels within Euclidean distance
   max(3, round(3% of the longer working dimension)) of the filled candidate.
   The one-pixel outer frame and all pixels outside support are hard background.
6. Initialise GrabCut labels: core = definite foreground (1), other candidate =
   probable foreground (3), remaining support = probable background (2), outside =
   definite background (0). Background takes precedence on the outer frame.
7. Run OpenCV 4.13.0 GrabCut with `GC_INIT_WITH_MASK`, five iterations, seed 42 and
   one OpenCV thread under the existing lock. Restore the prior thread count.
8. Retain GrabCut labels 1 or 3. Apply 3 x 3 closing, the same minimum-component
   filter and limited hole filling, then intersect with support. Do not OR with
   the original full HSV or rectangle-GrabCut mask.
9. Resize the binary mask to original dimensions using nearest-neighbour sampling.
   Keep original RGB pixels inside and black outside. No extra original-resolution
   cleanup is applied. The shared feature extractor then prepares RGB32 pixels.

Working images with a dimension below seven, constant images, or fewer than five
eroded colour-core pixels return an empty mask with an explicit status. Unexpected
OpenCV failures are reported as errors, not hidden behind an original-image fallback.
Training fails visibly on unexpected processing errors; the UI records an error row.

## Diagnostics and previews

`processing_diagnostics.csv` records working dimensions, candidate/core/support
fractions, iteration count, status, final retained area and review flag. Flags:

- `no_foreground`: empty output; inspect the source and limitation.
- `broad_colour_candidates`: candidate coverage exceeds 85%; background retention
  is possible, but fruit-dominated close-ups can also legitimately be broad.
- `inspect_mask`: a nonempty result still needs inspection; this is not a pass grade.

Fractions refer to the working image except final retained area, which refers to
the original-resolution mask. Neither coverage nor a flag is segmentation accuracy.
1330 x 460 training previews show original, actual initial GrabCut labels, final
binary mask and processed RGB. Initial-label colours: black hard background, grey
probable background, orange probable foreground, green hard foreground.

## UI workflow

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

1. Models & runs: refresh and confirm trusted team models. The new variant is
   visibly untrained until its completed run exists. Old runs remain selectable.
2. Choose images or a folder, as before. For one variant, select `hybrid_refined`
   and Run selected method.
3. To compare the two hybrids directly, click Compare hybrid versions.
4. For the assignment's six-method grid, set Hybrid used by Compare all six to
   `hybrid_refined`, then click Compare all six. Its default remains original
   `hybrid` for compatibility; the single-method selector does not change it.
5. Saved validation can show baseline, all five individuals and both hybrid
   versions. Load selected runs again after changing run selection, then export.

Prediction cards/CSV include the refined review flag. Inputs still have no assumed
ground truth, model scores remain uncalibrated, and original-Test content is blocked.

## Development observations, not held-out performance

Parameters were developed after the team's observation of tray retention. Visual
checks used three previously discussed Train examples plus the lexicographically
first Train image in each of the 15 fruit/stage groups (18 distinct images total).
This is a convenience development sample, not a random or independent evaluation.
No original-Test image was decoded. No new full-dataset validation result has been
measured in the build environment; that requires the team's training command.

Verified examples from the final implementation:

| Original Train example | Union retained area | Refined retained area | Visual observation |
|---|---:|---:|---|
| Ripe/apple_ripe_218.png | 70.86% | 27.72% | Most tray removed; four apples retained, some coloured reflections still remain |
| Ripe/apple_ripe_207.jpg | 37.42% | 36.10% | Main apple body retained; much of the thin stem removed |
| Ripe/apple_ripe_006.jpg | 93.26% | 66.21% | Some background removed, but leaves and coloured background remain |
| Overipe/banana_overripe_001.jpg | 100.00% | 98.64% | Brown wood remains; poor separation |
| Overipe/orange_overripe_001.jpg | 44.72% | 37.13% | Some mould/pale fruit pixels are incorrectly removed |

Smaller retained area does not itself prove better segmentation. Other observed
failures include dark damaged mango areas being removed and leafy scenes remaining
mostly foreground. Minimum-size filtering and downsampling can remove small fruit
or stems, and the hard frame can clip fruit touching the image edge. Do not use this
mask for reliable blemish-area measurement. Do not present only the successful tray
example as evidence that the method works on all fruits/backgrounds.

The automated synthetic tray test has an exact constructed foreground mask; it
checks expected behaviour but is not a real-dataset IoU/Dice evaluation. There are
no supplied reference fruit masks for the real examples.

## Verification and report decisions

- 79 non-GUI tests pass, including new synthetic tray, bounded-hole, multi-colour,
  tiny/constant/no-seed, determinism, failure reporting, metadata/model round-trip,
  diagnostics, preview and no-Test-decoding checks.
- Five desktop integration tests are provided but cannot run without a graphical
  Tk environment. Run the full suite on Windows and inspect the added controls.
- All seven original method specifications matched the saved pre-update source;
  all 126 old-method/image comparisons (seven methods x 18 Train images) had exactly
  matching masks, processed pixels, diagnostics and RGB32 resized pixels.
- Full training and validation of the new variant are intentionally pending. Do
  not call it more accurate based on the tray preview. Compare macro F1, balanced
  accuracy, per-stage recall and per-fruit results as well as overall accuracy.
- Keep both hybrid runs in the report as versions/experiments. If the refined model
  scores lower or removes important damage cues, report that trade-off rather than
  deleting the original or describing refinement as a guaranteed improvement.
- Preserve the dataset-as-is policy and its duplicate/label limitations. Freeze
  final choices before evaluating the reserved original Test.

Technical reference: [OpenCV GrabCut tutorial and mask initialisation](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html).
The colour-seed thresholds, support constraint and cleanup rules are this project's
heuristics, not guarantees supplied by GrabCut. Record AI assistance and ensure the
team can explain the implementation and limitations.
