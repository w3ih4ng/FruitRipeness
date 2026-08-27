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
