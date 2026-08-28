"""Train a fixed RGB-pixel/Random-Forest baseline without changing source data.

Default: reserve 20% of supplied Train for validation. Supplied Test is never
used for fitting/tuning and its pixels are only loaded with --evaluate-test.
Duplicates and all source labels are retained, as requested for this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import PIL
from PIL import Image, ImageOps
import sklearn
import scipy
import cv2
import threadpoolctl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .audit import EXTENSIONS, parse_labels, write_csv
from .processing import process_image, processing_spec

CLASSES = ["unripe", "ripe", "overripe"]
FEATURE_ID = "rgb_pixels_32x32_v1"
SEED = 42
VALIDATION_FRACTION = 0.2


def resolve_dataset_root(path: Path) -> Path:
    """Accept Train/Test directly or one unambiguous extraction wrapper folder."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"Dataset folder not found: {path}. Extract archive.zip into data/raw first.")

    def has_splits(p: Path) -> bool:
        names = {c.name.lower() for c in p.iterdir() if c.is_dir()}
        return {"train", "test"}.issubset(names)

    if has_splits(path):
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and has_splits(p)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Expected Train and Test inside {path}. Use --data with their parent folder.")


def collect_records(root: Path) -> list[dict]:
    """Record every supported source image. Hashes identify inputs, not exclusions."""
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Image links outside the dataset are not supported: {relative}")
        label = parse_labels(relative)
        if label["source_split"] not in {"train", "test"}:
            raise ValueError(f"Unexpected supplied split: {relative}")
        records.append({"path": relative, **label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for split in ["train", "test"]:
        found = {r["stage"] for r in records if r["source_split"] == split}
        if found != set(CLASSES):
            raise ValueError(f"{split} must contain images for all stages {CLASSES}; found {sorted(found)}")
    return records


def assign_splits(records: list[dict]) -> list[dict]:
    """Deterministic stratification by fruit+stage inside supplied Train only."""
    train_indices = [i for i,r in enumerate(records) if r["source_split"] == "train"]
    strata = [f"{records[i]['fruit']}/{records[i]['stage']}" for i in train_indices]
    try:
        _, validation = train_test_split(train_indices, test_size=VALIDATION_FRACTION,
                                         random_state=SEED, stratify=strata)
    except ValueError as exc:
        raise ValueError(f"Cannot reserve validation by fruit/stage without dropping images: {exc}") from exc
    validation = set(validation)
    return [{**r, "split": ("test" if r["source_split"] == "test" else
                            "validation" if i in validation else "train")}
            for i,r in enumerate(records)]


def feature_vector(image: Image.Image) -> np.ndarray:
    """Shared minimal preparation: EXIF orientation, RGB, 32x32, float [0,1]."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((32, 32), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).reshape(-1) / np.float32(255)


def image_features(path: Path, expected_sha256: str | None = None, *, method: str = "baseline",
                   diagnostics: list | None = None) -> np.ndarray:
    blob = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError(f"Image changed during this run: {path}")
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        if method == "baseline":
            # Preserve the exact original baseline feature path and old-model behaviour.
            return feature_vector(image)
        start = time.perf_counter()
        result = process_image(image, method)
        elapsed = time.perf_counter() - start
        if diagnostics is not None:
            diagnostics.append({**result.details, "processing_seconds": elapsed})
        return feature_vector(result.processed)


def feature_matrix(root: Path, rows: list[dict], *, method: str = "baseline",
                   diagnostics: list | None = None) -> np.ndarray:
    features = np.empty((len(rows), 32*32*3), dtype=np.float32)
    progress_every = 250 if method == "grabcut" else 1000
    for i,row in enumerate(rows):
        try:
            details = []
            features[i] = image_features(root/row["path"], row["sha256"], method=method, diagnostics=details)
            if diagnostics is not None and details:
                diagnostics.append({"path": row["path"], **details[0]})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Cannot process {row['path']}; no image was silently skipped: {exc}") from exc
        if (i+1) % progress_every == 0:
            print(f"  Loaded {i+1}/{len(rows)} images", flush=True)
    return features


def compute_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    return {
        "n_images": len(y_true),
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true,y_pred)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: report[c] for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_true,y_pred,labels=CLASSES).tolist(),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix_axes": "rows=true, columns=predicted",
    }


def evaluate(model, x: np.ndarray, rows: list[dict]) -> tuple[dict, list[dict]]:
    start = time.perf_counter()
    scores = model.predict_proba(x)
    elapsed = time.perf_counter()-start
    predicted = model.classes_[scores.argmax(axis=1)]
    truth = [r["stage"] for r in rows]
    result = compute_metrics(truth, predicted)
    result["prediction_seconds"] = elapsed
    result["prediction_ms_per_image_batch_average"] = elapsed*1000/len(rows)
    result["per_fruit"] = {}
    for fruit in sorted({r["fruit"] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r["fruit"] == fruit]
        result["per_fruit"][fruit] = compute_metrics([truth[i] for i in idx], predicted[idx])
    predictions = []
    for row,pred,score in zip(rows,predicted,scores):
        predictions.append({"path": row["path"], "fruit": row["fruit"], "true_stage": row["stage"],
                            "predicted_stage": str(pred), "correct": bool(pred == row["stage"]),
                            **{f"score_{c}": float(v) for c,v in zip(model.classes_,score)}})
    return result, predictions


def predict_image(bundle: dict, path: Path) -> dict:
    """Inference helper for the later UI; path names never supply model features."""
    method = bundle.get("method")
    if bundle.get("feature_id") != FEATURE_ID or method not in {"baseline", "hsv", "otsu", "kmeans", "grabcut"}:
        raise ValueError("Model feature/method version is incompatible")
    # Legacy baseline bundles did not store a processing_spec; their path is unchanged.
    if method != "baseline" or "processing_spec" in bundle:
        if bundle.get("processing_spec") != processing_spec(method):
            raise ValueError("Saved model processing settings differ from the installed implementation")
    scores = bundle["model"].predict_proba(image_features(path, method=method).reshape(1,-1))[0]
    classes = bundle["model"].classes_
    return {"predicted_stage": str(classes[scores.argmax()]),
            "scores": {str(c): float(v) for c,v in zip(classes,scores)}}


def train_baseline(data: Path, out: Path, *, evaluate_test: bool = False,
                   jobs: int = -1, trees: int = 300, method: str = "baseline") -> tuple[Path,dict]:
    """Shared experiment engine; default remains the backward-compatible baseline."""
    spec = processing_spec(method)
    if jobs == 0 or trees < 1:
        raise ValueError("jobs must not be zero and trees must be positive")
    root = resolve_dataset_root(data)
    if Path(out).resolve().is_relative_to(root):
        raise ValueError("Write results outside the source dataset folder")
    print(f"Dataset: {root}", flush=True)
    records = assign_splits(collect_records(root))
    counts = dict(Counter(r["split"] for r in records))
    print(f"Images: {counts}; all source images and labels retained", flush=True)
    manifest_json = json.dumps(records,sort_keys=True,separators=(",",":"))
    split_id = hashlib.sha256(manifest_json.encode()).hexdigest()
    sets = {s:[r for r in records if r["split"] == s] for s in ["train","validation","test"]}
    print(f"Preparing {method} training features (RGB pixels, 32 x 32)...", flush=True)
    start = time.perf_counter()
    train_diagnostics = []
    x_train = feature_matrix(root,sets["train"],method=method,diagnostics=train_diagnostics)
    train_feature_seconds = time.perf_counter()-start
    model = RandomForestClassifier(n_estimators=trees,max_features="sqrt",class_weight="balanced",
                                   random_state=SEED,n_jobs=jobs)
    print(f"Training Random Forest ({trees} trees)...", flush=True)
    start = time.perf_counter()
    model.fit(x_train,[r["stage"] for r in sets["train"]])
    fit_seconds = time.perf_counter()-start
    del x_train
    metrics, tables = {}, {}
    diagnostics = [{"split": "train", **r} for r in train_diagnostics]
    for split in ["validation"] + (["test"] if evaluate_test else []):
        print(f"Evaluating {split}...", flush=True)
        start = time.perf_counter()
        split_diagnostics = []
        x = feature_matrix(root,sets[split],method=method,diagnostics=split_diagnostics)
        diagnostics.extend({"split": split, **r} for r in split_diagnostics)
        feature_seconds = time.perf_counter()-start
        metrics[split],tables[split] = evaluate(model,x,sets[split])
        metrics[split]["image_loading_and_feature_seconds"] = feature_seconds
        metrics[split]["image_loading_and_feature_ms_per_image"] = feature_seconds*1000/len(sets[split])
        if split_diagnostics:
            metrics[split]["mask_diagnostics"] = {
                "mean_foreground_fraction": float(np.mean([r["foreground_fraction"] for r in split_diagnostics])),
                "status_counts": dict(Counter(r["mask_status"] for r in split_diagnostics)),
                "processing_seconds": sum(r["processing_seconds"] for r in split_diagnostics),
            }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = Path(out)/stamp
    run.mkdir(parents=True,exist_ok=False)
    metadata = {
        "status": "completed", "created_utc": stamp, "method": method, "feature_id": FEATURE_ID,
        "processing_spec": spec,
        "model_type": "RandomForestClassifier", "model_parameters": model.get_params(),
        "model_class_order": model.classes_.tolist(), "metric_class_order": CLASSES,
        "counts": counts, "split_id": split_id, "seed": SEED,
        "validation_fraction_of_original_train": VALIDATION_FRACTION,
        "validation_stratification": "fruit + stage", "test_evaluated": evaluate_test,
        "source_images_modified": False, "duplicates_removed": False,
        "target": "image-level ripeness stage; fruit type is metadata, not a predicted output",
        "training_feature_seconds": train_feature_seconds, "model_fit_seconds": fit_seconds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "Pillow": PIL.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                     "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__,
                     "opencv": cv2.__version__},
        "platform": platform.platform(),
        "limitations": ["Supplied duplicates and labels retained by project decision; scores can be inflated or distorted.",
                        "Validation is a fixed sample from supplied Train, not a cleaned or group-independent split.",
                        "Source Test remains unchanged; only evaluate after method/parameter selection is complete.",
                        "Model scores are uncalibrated; batch-average timing is not single-image UI latency."],
    }
    write_csv(run/"split_manifest.csv", records, ["path","source_split","split","fruit","stage","raw_stage","sha256"])
    for split,rows in tables.items():
        write_csv(run/f"predictions_{split}.csv",rows,["path","fruit","true_stage","predicted_stage","correct"] + [f"score_{c}" for c in CLASSES])
        matrix = metrics[split]["confusion_matrix"]
        write_csv(run/f"confusion_matrix_{split}.csv",
                  [{"true_stage":c,**dict(zip(CLASSES,row))} for c,row in zip(CLASSES,matrix)],
                  ["true_stage",*CLASSES])
    if diagnostics:
        write_csv(run/"processing_diagnostics.csv",diagnostics,
                  ["split","path","foreground_fraction","mask_status","processing_seconds",
                   "otsu_threshold","foreground_polarity","kmeans_clusters","kmeans_sample_pixels",
                   "background_cluster","background_border_fraction","kmeans_iterations",
                   "grabcut_width","grabcut_height","grabcut_rect","grabcut_iterations","grabcut_status"])
    bundle = {"model": model,"method":method,"feature_id":FEATURE_ID,"split_id":split_id,
              "processing_spec":spec,"metadata":metadata}
    joblib.dump(bundle,run/"model.joblib",compress=3)
    (run/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (run/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return run, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data/raw"))
    parser.add_argument("--out",type=Path,default=Path("outputs/baseline"))
    parser.add_argument("--jobs",type=int,default=-1,help="CPU workers; use 2 if the laptop becomes busy")
    parser.add_argument("--evaluate-test",action="store_true",help="Final evaluation only, after settings/methods are frozen")
    args = parser.parse_args()
    try:
        run,metrics = train_baseline(args.data,args.out,evaluate_test=args.evaluate_test,jobs=args.jobs)
    except (ValueError,OSError) as exc:
        parser.exit(1,f"Error: {exc}\n")
    for split,m in metrics.items():
        print(f"{split.title()}: images={m['n_images']}, accuracy={m['accuracy']:.4f}, "
              f"balanced_accuracy={m['balanced_accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
    print(f"Results and model saved: {run.resolve()}")
    if not args.evaluate_test:
        print("Original Test not evaluated. Use validation for development; final test evaluation comes later.")


if __name__ == "__main__":
    main()
