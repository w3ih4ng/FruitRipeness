# Saved runs and reports

Generated output is ignored by Git. Keep complete run folders and back them up
separately; do not rename individual files or detach models from metadata.

| Folder | Contents |
|---|---|
| `shared_cnn/<timestamp>/` | One `model.pt`, metadata, split manifest, all eight validation variants, predictions, confusion matrices, training history, diagnostics and feature arrays |
| `final_test/<timestamp>/` | All eight Test variants, per-fruit/overall metrics and figures, predictions, confusion matrices, previews and frozen model provenance; no new model |
| `baseline/<timestamp>/` | Raw-input Random Forest, validation results and split manifest |
| `hsv/`, `otsu/`, `kmeans/`, `grabcut/`, `watershed/` | Per-method RF runs, diagnostics and processing previews |
| `hybrid/`, `hybrid_refined/` | Separate RF runs for Hybrid A and Hybrid B |
| `audit/` | Optional historical dataset audit results |
| `ui_reports/` (if created) | Manually exported UI figures and CSVs |

The team's completed CNN is `shared_cnn/20260828T085810_496281Z`.
The final Test report is `final_test/20260828T100939_923549Z`.
The reference split comes from `baseline/20260827T153126_909623Z`.

UI dates are readable aliases for these exact folder IDs, not renamed files.
No cleanup automatically deletes old runs, source data or cached features.
Earlier RF evidence remains available for the report.

Open the final report in **Saved final Test**, not in the image picker.
See [final report files](../docs/FINAL_TEST.md) and
[CNN run files](../docs/SHARED_CNN.md) for detailed formats.
