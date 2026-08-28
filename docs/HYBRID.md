# Hybrid: HSV + GrabCut mask union

## Why these two methods?

Development validation results reproduced by the team on the same 887 images:

| Method | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Minimal-preprocessing baseline | 0.7948 | 0.7202 | 0.7400 |
| HSV | 0.7813 | 0.6882 | 0.6989 |
| Otsu | 0.7520 | 0.6732 | 0.6895 |
| K-means | 0.7576 | 0.6691 | 0.6822 |
| GrabCut | 0.7666 | 0.6887 | 0.7056 |
| Watershed | 0.7204 | 0.6402 | 0.6547 |

HSV has the highest accuracy among the processed methods; GrabCut has the highest
macro F1 among them. Neither beats the baseline. The hybrid uses this aggregate
validation evidence to select two complementary foreground heuristics, then fixes
a simple union rule before evaluating it. It does not need to combine all five
methods, and it does not guarantee an improvement.

Hypothesis: HSV may retain coloured areas that GrabCut removes; GrabCut may retain
low-saturation areas that HSV removes. Their union discards less of the original
image than either component alone. It may also retain unwanted coloured background.
Check the previews; these are hypotheses, not conclusions from measured masks.

## Fixed pipeline: hybrid_hsv_grabcut_union_v1

1. Read the image, apply EXIF orientation and convert to RGB, as in existing methods.
2. Run the complete existing HSV pipeline, including its mask cleanup.
3. Independently run the complete existing GrabCut pipeline on the same RGB image,
   including its size cap, automatic rectangle, seed, iterations and mask cleanup.
4. Combine the two final masks at original resolution with logical OR:
   `hybrid_mask = hsv_mask | grabcut_mask`.
5. Do not apply another morphology/cleanup pass to the union. The final hybrid mask
   is exactly the union of the component masks displayed in the preview.
6. Keep original RGB pixels inside the union and set pixels outside it to black.
7. Use the unchanged 32 x 32 RGB feature extractor and train a separate, identically
   configured 300-tree Random Forest on the same 3,547 fitting images.

No filenames, fruit names, ripeness labels or model predictions enter segmentation.
This is image-processing fusion, NOT voting between trained classifiers. Previous
HSV/GrabCut model files are not required; their processing code is reused directly.
The full nested component settings are stored with the hybrid model. Inference and
preview exports reject mismatched saved settings.

If one component mask is empty, the union equals the other. If both are empty, the
processed image is black and the sample remains in training/evaluation. There is
no fallback to baseline predictions and no silent image exclusion. Unexpected
processing errors stop training rather than silently changing the dataset.

## Run in the existing project

Apply `hybrid_update.patch` once, from the project root, after the Watershed update.
Do not re-create your environment or download another starter folder.

```powershell
git apply --check .\hybrid_update.patch
git apply .\hybrid_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method hybrid --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Expect 55 tests. Stop if a command reports an error. Successful `git apply` commands
normally print nothing. Dependencies are unchanged from the Watershed update.
This hybrid runs GrabCut for every image, so expect runtime comparable to GrabCut
plus HSV; progress prints every 250 images. Use `--jobs 2` if Random Forest fitting
uses too much CPU; it does not parallelise the image-processing loop.

Every original image, label, duplicate and Train/Test membership remains unchanged.
The command checks the baseline's dataset/split ID before training. It evaluates
only validation, never the original 180-image Test set. Earlier models and run
folders remain unchanged and do not need to be regenerated.

## Outputs and reporting

The command prints a new timestamped folder under `outputs/hybrid/` containing:

- Model, metadata, split manifest, validation metrics/predictions and confusion matrix.
- `comparison_validation.csv` and `.json`: measured comparison against the baseline,
  including per-fruit metrics in the CSV and signed overall metric changes in JSON.
- `processing_diagnostics.csv`: final coverage/status, each component's coverage/status,
  intersection/disagreement fractions, mask agreement and GrabCut diagnostics.
- `previews/`: original, HSV mask, GrabCut mask, union mask and processed RGB in five
  panels. Selection is unchanged: first validation example per fruit/stage and up to
  three additional misclassified examples, with their paths and selection reasons.

`hybrid_mask_jaccard` is intersection area divided by union area between the two
heuristic masks. Both empty is defined as 1.0 agreement. This is NOT ground-truth
segmentation IoU, classification accuracy or evidence that either mask is correct.
The disagreement fraction is their exclusive-OR area divided by image area.

Record measured results even if the hybrid underperforms. Validation has informed
this design choice, so its score is a development result, not an unbiased final
generalisation estimate. Do not repeatedly adjust this rule to chase validation
scores without documenting each change. Freeze choices before the final original-
Test evaluation, and acknowledge the existing dataset limitations in the report.

For preview discussion, include failures such as retained coloured backgrounds,
fruit touching the frame, missing pale regions and mixed-object scenes. Classification
scores alone cannot establish segmentation quality without annotated masks.

## Next milestone

Build the local comparison UI in this same repository: five individual methods and
this hybrid, single-image/multiple-image/folder input, processed images/masks,
predictions, model scores, timings and PNG/CSV exports. Keep baseline in the
evaluation reference table and keep validation distinct from final Test results.
