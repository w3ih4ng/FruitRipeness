# Final original-Test comparison: all methods, one frozen CNN

This completes the existing comparison workflow. It does not add a new processing
method or train a new model. The raw baseline, HSV, Otsu, K-means, GrabCut,
Watershed, original hybrid and refined hybrid are ALL evaluated on the same
original-Test images using the SAME saved shared-CNN weights.

## Apply once in the existing project

Close the UI. Save `final_test_update.patch` beside `pyproject.toml`. This patch
expects the shared-CNN update (version 0.11.0). Run each command separately and
stop if it fails. Successful git apply commands are normally silent.

```powershell
git apply --check .\final_test_update.patch
git apply .\final_test_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No new dependencies. Keep the already installed requirements-cnn.txt versions.
No model needs retraining. Source data, saved models and validation outputs remain
unchanged. The project version becomes 0.12.0.

## Run the frozen comparison

The team's completed shared CNN is `20260828T085810_496281Z`. Confirm that this is
your own trusted model and that the processing/model settings are frozen:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.final_test --data data/raw --run outputs/shared_cnn/20260828T085810_496281Z --confirm-frozen --trusted-model
```

Use the actual completed run path if your team deliberately selected another run
BEFORE looking at Test results. The two explicit flags are required; they are not
a way to bypass a download or an untrusted-model warning. Do not approve unknown
checkpoints. No internet download occurs during evaluation.

The evaluator checks every dataset entry against the original saved split manifest
and split ID, including source hashes. It checks the saved checkpoint hash, embedded
metadata, class order, package versions and processing specifications. It then loads
the checkpoint once and decodes only the 180 original-Test images for inference.
There are 1,440 predictions (180 images x eight variants), NOT 1,440 independent
Test images. Original labels and duplicates are retained; nothing is excluded.

No optimiser, fitting, epoch selection, augmentation or parameter tuning is run.
Unexpected image/method errors stop evaluation rather than silently dropping an
image. Source/run hashes are checked again before the report is completed. Partial
failed reports remain marked failed and cannot be loaded as completed Test results.

Freeze the settings before the first run. Do not retrain/tune in response to Test
results or repeatedly select whichever checkpoint scores best on Test. A repeated
technical verification run creates a separate timestamped report, never overwrites
the earlier one, and is not new independent evidence. This tool records confirmation;
it cannot enforce a team-wide prohibition on retraining elsewhere.

## Outputs for every method

The command prints the exact new `outputs/final_test/<timestamp>/` folder:

| File | Contents |
|---|---|
| `comparison_test.csv` | All eight methods, overall and per-fruit accuracy, balanced accuracy, precision, recall and macro F1 |
| `comparison_test_overall.png` | All eight overall summaries and confusion matrices |
| `comparison_test_<fruit>.png` | All eight method summaries for that fruit |
| `predictions_test_all.csv` | Every image/method prediction, supplied label, match flag, scores, timing and checkpoint hash |
| `predictions_test_<method>.csv` | Same prediction detail for one method |
| `confusion_matrix_test_<method>.csv` | Overall confusion matrix for one method; rows true, columns predicted |
| `metrics.json` | Full overall/per-fruit/per-class Test measurements |
| `metadata.json` | Frozen model provenance, settings, completion status and metrics checksum |
| `split_manifest.csv` | Copy of the unchanged original split manifest |
| `previews/` | Multi-method PNG examples and an index identifying their source paths |

Previews are chosen BEFORE prediction: the first lexicographic Test path in each
fruit/stage stratum. This normally gives 15 examples for five fruits x three stages.
Every preview includes all eight variants and is labelled FINAL TEST. Each card
shows the supplied label and whether the prediction matches it; it does not call
one prediction a dataset accuracy measurement. Preview selection never favours
successes or the refined hybrid. These are examples, not segmentation ground truth.

The report includes processing, input preparation and prediction times from the
same inference path as the UI. Disk reads, model loading and rendering are outside
those method timings. Do not treat one timing observation as a benchmark.

## View results in the UI

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.ui
```

Open **Saved final Test**, then **Open Test report**, and choose the new
`outputs/final_test/<timestamp>` folder (not data/raw/Test and not the model folder).
The tab reads saved measurements only, with no model loading or retraining.
Select a fruit scope, click Load scope, and export the displayed CSV/PNG if needed.
The automatically generated full reports are already available in the output folder.

This tab is independent of the top Classifier selector and clearly identifies its
source CNN run. It never mixes Random Forest results or validation scores into a
Test comparison. The existing Saved validation tab retains the earlier results.
Test reports are evaluation-only and cannot be selected as inference checkpoints.

The Images & folders tab remains a DEVELOPMENT preview tool. Its reserved-Test
warning still applies, including identical copies inside Train. This does not
exclude any processing method from the report. Use the generated final-Test
previews for Test-image examples rather than renaming images to evade the guard.

## Report interpretation and verification

- Present all methods, including weaker results. Keep validation and final Test
  in separately labelled tables/figures.
- The shared CNN's raw input is a reference, not another member's method. Both
  hybrids can be reported as versions of the team's combined technique.
- Dataset duplicates and label issues remain; original Test is not guaranteed
  independent of fitting images. Do not claim leakage-free evaluation or use
  these scores to guarantee performance on new photographs.
- Compare macro F1, balanced accuracy, class recall and per-fruit results, not
  accuracy alone. Small differences on 180 images should be described cautiously.
- Test score rankings may differ from validation. Report that outcome without
  changing the frozen pipeline to improve the Test numbers.

The suite has 106 non-GUI tests (20 require optional CNN dependencies) and seven
desktop tests. New checks cover explicit confirmation, full-manifest consistency,
all Test rows/all methods, a single checkpoint load, no fitting, unchanged sources,
separate Test labels, report integrity, exports, failure handling and the unchanged
development guard. Desktop tests require graphical Tk and must be run on Windows;
the build environment cannot verify the actual window layout. Local functional
checks use synthetic Test fixtures, not the team's reserved 180-image evaluation.
