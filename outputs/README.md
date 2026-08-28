# Generated output (not committed)

The baseline writes a new timestamped folder under outputs/baseline/ for each run.
It contains metrics.json, metadata.json, split_manifest.csv, model.joblib,
predictions_validation.csv and confusion_matrix_validation.csv. Test files appear
only when --evaluate-test is explicitly supplied for the final comparison.

The optional audit still writes outputs/audit/{summary.json,manifest.csv,
duplicate_groups.json,near_duplicate_candidates.csv}. Neither command changes
source images. Generated outputs are ignored by Git.

HSV Method 1 writes outputs/hsv/<run timestamp>/ using the same result layout. It
also saves processing_diagnostics.csv, comparison_validation.json/.csv, and a
previews/ folder with original/mask/processed PNGs and an index.csv. The baseline
run remains untouched. The method command does not evaluate original Test.

Otsu Method 2 uses the same layout under outputs/otsu/<run timestamp>/. Its
processing_diagnostics.csv also records the automatic threshold and chosen
foreground polarity for every fit/validation image. Preview PNGs display both.
The common experiment command selects its output folder from --method unless
--out is explicitly provided; existing runs are never overwritten.

K-means Method 3 writes the same layout under outputs/kmeans/<run timestamp>/.
Its diagnostics add the effective cluster count, sampled pixel count, background
cluster ID, background share of the frame and fit iterations. Preview PNGs show
the cluster count and selected background ID. These IDs are colour groups, not
ripeness classes. See docs/METHOD3_KMEANS.md for the fixed processing rules.

GrabCut Method 4 uses outputs/grabcut/<run timestamp>/ with the same layout.
Diagnostics record working dimensions, automatic rectangle, requested iteration
count and GrabCut status. The final mask fraction/status is recorded separately
after resizing and cleanup. See docs/METHOD4_GRABCUT.md for the exact rules.

Watershed Method 5 uses outputs/watershed/<run timestamp>/ with the same layout.
It adds initial-marker count, seed/background coverage, working dimensions and
watershed boundary/status diagnostics. Its 1330 x 460 previews contain four panels:
original, actual initial markers, final mask and processed RGB. Earlier methods'
1000 x 460 previews are unchanged. See docs/METHOD5_WATERSHED.md for the fixed rules.
