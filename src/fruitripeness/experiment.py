"""Run an implemented processing method, compare baseline, and export previews."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

from .audit import write_csv
from .baseline import assign_splits, collect_records, resolve_dataset_root, train_baseline
from .processing import process_image, processing_spec

METHOD_LABELS = {"hsv": "Method 1 - HSV", "otsu": "Method 2 - Otsu", "kmeans": "Method 3 - K-means",
                 "grabcut": "Method 4 - GrabCut", "watershed": "Method 5 - Watershed",
                 "hybrid": "Hybrid - HSV + GrabCut Union",
                 "hybrid_refined": "Hybrid refined - HSV-seeded GrabCut"}


def read_run(path: Path) -> tuple[dict, dict]:
    path = Path(path)
    try:
        metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((path/"metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read run {path}; select the timestamped run folder containing metadata.json and metrics.json") from exc
    if metadata.get("status") != "completed" or "validation" not in metrics:
        raise ValueError(f"Incomplete validation run: {path}")
    return metadata, metrics


def compare_runs(baseline_run: Path, method_run: Path) -> dict:
    """Compare actual stored validation metrics; incompatible experiments are rejected."""
    baseline, baseline_metrics = read_run(baseline_run)
    current, current_metrics = read_run(method_run)
    if baseline.get("method") != "baseline":
        raise ValueError("The reference must be a baseline run")
    method_id = current.get("method")
    if method_id not in METHOD_LABELS:
        raise ValueError("Unsupported processing method in run")
    for key in ["split_id", "feature_id", "seed", "counts", "model_type"]:
        if baseline.get(key) != current.get(key):
            raise ValueError(f"Cannot compare runs: {key} differs. Use the same dataset and experiment settings.")
    # Thread count affects timing, but does not change this seeded estimator's design.
    for key in set(baseline["model_parameters"]) | set(current["model_parameters"]):
        if key != "n_jobs" and baseline["model_parameters"].get(key) != current["model_parameters"].get(key):
            raise ValueError(f"Cannot compare runs: model parameter {key} differs")
    for key in ["numpy", "Pillow", "scikit-learn"]:
        if baseline["versions"].get(key) != current["versions"].get(key):
            raise ValueError(f"Cannot compare runs: {key} version differs; rerun baseline with the installed environment")
    fields = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    rows = []
    for method, result in [("baseline", baseline_metrics["validation"]), (method_id, current_metrics["validation"])]:
        for scope, values in [("overall", result), *sorted(result["per_fruit"].items())]:
            rows.append({"method": method, "scope": scope, "n_images": values["n_images"],
                         **{key: values[key] for key in fields}})
    write_csv(Path(method_run)/"comparison_validation.csv",rows,["method", "scope", "n_images", *fields])
    old, new = baseline_metrics["validation"], current_metrics["validation"]
    comparison = {
        "split": "validation", "split_id": current["split_id"],
        "baseline_run": Path(baseline_run).name, "method_run": Path(method_run).name,
        "baseline": {key: old[key] for key in fields},
        method_id: {key: new[key] for key in fields},
        f"{method_id}_minus_baseline": {key: new[key]-old[key] for key in fields},
        "delta_units": "fraction differences; multiply accuracy differences by 100 for percentage points",
        "timing_note": "Timing comparison omitted; batch size, worker count and hardware influence timings.",
    }
    (Path(method_run)/"comparison_validation.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    return comparison


def render_preview(image: Image.Image, title: str, footer: str, *, method: str = "hsv") -> Image.Image:
    result = process_image(image, method)
    canvas = Image.new("RGB", ({"watershed": 1330, "hybrid": 1660, "hybrid_refined": 1330}.get(method, 1000), 460), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text((20, 12), title, fill="black", font=font)
    panels = [("Original", result.original),
              (f"{method.upper()} foreground mask", Image.fromarray(result.mask.astype("uint8")*255)),
              ("Processed input", result.processed)]
    if method == "hybrid":
        for index, name in enumerate(("hsv", "grabcut"), 1):
            panels.insert(index, (f"{name.upper()} component mask",
                                 Image.fromarray(result.component_masks[name].astype("uint8")*255)))
    if method == "hybrid_refined":
        colours = np.zeros((*result.markers.shape, 3), dtype=np.uint8)
        colours[result.markers == 1] = (0, 150, 70)
        colours[result.markers == 2] = (160, 160, 160)
        colours[result.markers == 3] = (230, 150, 30)
        panels.insert(1, ("GrabCut initial labels", Image.fromarray(colours)))
    if method == "watershed":
        # These are the actual initial markers, not a reconstruction from the final mask.
        colours = np.full((*result.markers.shape, 3), 160, dtype=np.uint8)
        colours[result.markers == 1] = (0, 0, 0)
        colours[result.markers > 1] = (0, 110, 220)
        panels.insert(1, ("Initial markers", Image.fromarray(colours)))
    for i, (caption, panel) in enumerate(panels):
        x = 20+i*330
        draw.text((x, 48), caption, fill="black", font=font)
        resized = ImageOps.contain(panel.convert("RGB"), (300, 300))
        canvas.paste(resized, (x+(300-resized.width)//2, 80+(300-resized.height)//2))
        draw.rectangle((x-1, 79, x+300, 380), outline="#bbbbbb")
    status = f"Foreground: {result.details['foreground_fraction']:.1%} | Mask status: {result.details['mask_status']}"
    if method == "otsu":
        status += f" | Threshold: {result.details['otsu_threshold']} | Keep: {result.details['foreground_polarity']}"
    elif method == "kmeans":
        status += f" | Clusters: {result.details['kmeans_clusters']} | Background ID: {result.details['background_cluster']}"
    elif method == "grabcut":
        status += (f" | Working: {result.details['grabcut_width']}x{result.details['grabcut_height']}"
                   f" | GrabCut: {result.details['grabcut_status']}")
    elif method == "watershed":
        status += (f" | Seeds: {result.details['watershed_markers']} | {result.details['watershed_status']}"
                   " | Markers: blue=foreground, black=background, grey=unknown")
    elif method == "hybrid":
        status += (f" | HSV: {result.details['hybrid_hsv_fraction']:.1%}"
                   f" | GrabCut: {result.details['hybrid_grabcut_fraction']:.1%}"
                   f" | Mask agreement IoU: {result.details['hybrid_mask_jaccard']:.3f} (not accuracy)")
    elif method == "hybrid_refined":
        status += (f" | {result.details['refined_status']} | Seeds: green=FG, orange=probable FG, grey=probable BG, black=BG")
    draw.text((20, 393), status, fill="black", font=small)
    for i, line in enumerate(textwrap.wrap(footer, width={"watershed": 140, "hybrid": 180, "hybrid_refined": 140}.get(method, 105))[:2]):
        draw.text((20, 418+i*18), line, fill="black", font=small)
    return canvas


def export_previews(data_root: Path, method_run: Path) -> int:
    """First validation example per fruit/stage, plus up to 3 additional failures.

    Selection is fixed/documented and independent of the segmentation parameters.
    True labels only select/report examples; process_image receives only pixels.
    """
    metadata, _ = read_run(method_run)
    method = metadata.get("method")
    if method not in METHOD_LABELS:
        raise ValueError("Unsupported processing method for previews")
    if metadata.get("processing_spec") != processing_spec(method):
        raise ValueError("Preview implementation does not match the saved processing settings")
    with (method_run/"predictions_validation.csv").open(encoding="utf-8", newline="") as f:
        predictions = list(csv.DictReader(f))
    with (method_run/"split_manifest.csv").open(encoding="utf-8", newline="") as f:
        manifest = {row["path"]: row for row in csv.DictReader(f)}
    selected, groups, paths = [], set(), set()
    for row in predictions:
        group = (row["fruit"], row["true_stage"])
        if group not in groups:
            selected.append((row,"first_per_fruit_stage"))
            groups.add(group); paths.add(row["path"])
    added = 0
    for row in predictions:
        if row["correct"].lower() == "false" and row["path"] not in paths and added < 3:
            selected.append((row,"additional_failure")); paths.add(row["path"]); added += 1
    output = method_run/"previews"
    output.mkdir(exist_ok=False)
    index = []
    root = Path(data_root).resolve()
    for i, (row, reason) in enumerate(selected, 1):
        record = manifest[row["path"]]
        if record["split"] != "validation":
            raise ValueError("Only validation examples can be previewed in this development command")
        source = (root/row["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Invalid source path in saved run")
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Preview source changed since training: {row['path']}")
        name = f"{i:02d}_{row['fruit']}_{row['true_stage']}.png"
        title = f"{METHOD_LABELS[method]} | Truth: {row['true_stage']} | Predicted: {row['predicted_stage']}"
        with Image.open(source) as image:
            preview = render_preview(image, title, row["path"], method=method)
        preview.save(output/name)
        index.append({"file":name,"source":row["path"],"selection":reason,
                      "true_stage":row["true_stage"],"predicted_stage":row["predicted_stage"]})
    write_csv(output/"index.csv",index,["file","source","selection","true_stage","predicted_stage"])
    return len(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHOD_LABELS), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Default: outputs/<method>")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    try:
        reference, _ = read_run(args.baseline_run)
        if reference.get("method") != "baseline":
            raise ValueError("--baseline-run must point to your saved baseline run")
        root = resolve_dataset_root(args.data)
        records = assign_splits(collect_records(root))
        split_id = hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        if split_id != reference.get("split_id"):
            raise ValueError("Dataset/split differs from the selected baseline. No training started.")
        run, metrics = train_baseline(root,args.out or Path("outputs")/args.method,method=args.method,jobs=args.jobs)
        print(f"{args.method.upper()} model and metrics saved: {run.resolve()}", flush=True)
        comparison = compare_runs(args.baseline_run,run)
        n = export_previews(root,run)
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(1,f"Error: {exc}\n")
    result = metrics["validation"]
    delta = comparison[f"{args.method}_minus_baseline"]
    print(f"{args.method.upper()} validation: images={result['n_images']}, accuracy={result['accuracy']:.4f}, "
          f"balanced_accuracy={result['balanced_accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")
    print(f"Change vs baseline: accuracy={delta['accuracy']*100:+.2f} percentage points, macro_f1={delta['macro_f1']:+.4f}")
    print(f"Saved {n} preview PNGs: {(run/'previews').resolve()}")
    print("Original Test not evaluated. Source images and labels unchanged.")


if __name__ == "__main__":
    main()
