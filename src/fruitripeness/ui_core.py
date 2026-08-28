"""Read-only run discovery, shared inference, and explicit report exports for the UI.

No training, source edits, label inference from filenames, or Test evaluation.
Tk widgets live separately so these behaviours can be tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import scipy
import sklearn

from .audit import EXTENSIONS
from .baseline import CLASSES, FEATURE_ID, feature_vector
from .experiment import METHOD_LABELS
from .processing import ProcessingResult, process_image, processing_spec

METHODS = tuple(METHOD_LABELS)
LABELS = {"baseline": "Baseline (reference)", **METHOD_LABELS}
SCORE_FIELDS = [f"score_{name}" for name in CLASSES]
RESULT_FIELDS = ["source", "sha256", "method", "run", "model_sha256", "split_id", "status", "error",
                 "predicted_stage", *SCORE_FIELDS, "processing_ms", "feature_ms", "prediction_ms",
                 "foreground_fraction", "mask_status"]
METRIC_FIELDS = ["method", "run", "split", "split_id", "scope", "n_images", "accuracy",
                 "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
CURRENT_VERSIONS = {"numpy": np.__version__, "Pillow": PIL.__version__,
                    "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                    "scipy": scipy.__version__, "opencv": cv2.__version__}


@dataclass
class Run:
    path: Path
    metadata: dict
    metrics: dict

    @property
    def method(self):
        return self.metadata["method"]


def validate_metrics(values: dict) -> None:
    if not isinstance(values, dict):
        raise ValueError("Metrics must be a JSON object")
    matrix = np.asarray(values["confusion_matrix"])
    if (values["confusion_matrix_labels"] != CLASSES or matrix.shape != (3, 3)
            or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0)
            or int(matrix.sum()) != values["n_images"] or values["n_images"] <= 0):
        raise ValueError("Invalid confusion matrix, class order or image count")
    for key in METRIC_FIELDS[6:]:
        if not np.isfinite(values[key]) or not 0 <= values[key] <= 1:
            raise ValueError(f"Invalid metric: {key}")
    if not np.isclose(np.trace(matrix)/matrix.sum(), values["accuracy"]):
        raise ValueError("Accuracy disagrees with confusion matrix")


def read_run(path: Path, expected_method: str | None = None) -> Run:
    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError("Metadata and metrics must be JSON objects")
    method = metadata.get("method")
    if method not in LABELS or (expected_method and method != expected_method):
        raise ValueError("Unexpected method in run")
    if metadata.get("status") != "completed" or metadata.get("feature_id") != FEATURE_ID:
        raise ValueError("Incomplete run or incompatible features")
    saved_spec = metadata.get("processing_spec", processing_spec("baseline") if method == "baseline" else None)
    if saved_spec != processing_spec(method):
        raise ValueError("Processing settings differ from this implementation")
    if metadata.get("metric_class_order") != CLASSES or not metadata.get("split_id"):
        raise ValueError("Missing split identity or incompatible metric class order")
    validate_metrics(metrics["validation"])
    if metrics["validation"]["n_images"] != metadata["counts"]["validation"]:
        raise ValueError("Validation count disagrees with metadata")
    for values in metrics["validation"].get("per_fruit", {}).values():
        validate_metrics(values)
    return Run(path, metadata, metrics)


def discover_runs(outputs: Path) -> tuple[dict[str, list[Run]], list[str]]:
    """JSON only: discovery never deserialises a model or opens dataset images."""
    found = {method: [] for method in LABELS}
    warnings = []
    outputs = Path(outputs)
    if not outputs.is_dir():
        return found, [f"Output folder not found: {outputs}"]
    for method in LABELS:
        for path in sorted((outputs / method).glob("*/metadata.json"), reverse=True):
            try:
                found[method].append(read_run(path.parent, method))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"{method}/{path.parent.name}: {exc}")
    return found, warnings


def ensure_compatible(runs: list[Run]) -> None:
    if not runs:
        raise ValueError("Select at least one completed run")
    reference = runs[0].metadata
    for run in runs[1:]:
        current = run.metadata
        for key in ["split_id", "feature_id", "counts", "seed", "model_type"]:
            if reference.get(key) != current.get(key):
                raise ValueError(f"Incompatible {run.method} run: {key} differs")
        for key in set(reference["model_parameters"]) | set(current["model_parameters"]):
            if key != "n_jobs" and reference["model_parameters"].get(key) != current["model_parameters"].get(key):
                raise ValueError(f"Incompatible model parameter: {key}")
        for key in ["numpy", "Pillow", "scikit-learn"]:
            if reference["versions"].get(key) != current["versions"].get(key):
                raise ValueError(f"Incompatible training environment: {key}")


def load_model(run: Run, *, trusted: bool = False) -> tuple[dict, str]:
    if not trusted:
        raise ValueError("Confirm these are your team's trusted model files before loading")
    # Version and JSON checks happen before deserialisation. Trust is still essential:
    # joblib/pickle can execute code and metadata checks cannot make it safe.
    for key, current in CURRENT_VERSIONS.items():
        saved = run.metadata["versions"].get(key)
        if saved is not None and saved != current:
            raise ValueError(f"{key} version differs: trained {saved}, installed {current}")
    blob = (run.path / "model.joblib").read_bytes()
    bundle = joblib.load(io.BytesIO(blob))
    for key in ["method", "feature_id", "split_id"]:
        if bundle.get(key) != run.metadata[key]:
            raise ValueError(f"Model and run metadata disagree: {key}")
    if bundle.get("processing_spec", processing_spec("baseline") if run.method == "baseline" else None) != processing_spec(run.method):
        raise ValueError("Model processing settings differ")
    if bundle.get("metadata") != run.metadata:
        raise ValueError("Model's embedded metadata differs from selected run")
    model = bundle["model"]
    if list(model.classes_) != run.metadata["model_class_order"] or set(model.classes_) != set(CLASSES):
        raise ValueError("Model class order differs")
    if model.n_features_in_ != 3072:
        raise ValueError("Model expects different input features")
    # Bound interactive prediction CPU use, without changing the saved model.
    model.set_params(n_jobs=1)
    return bundle, hashlib.sha256(blob).hexdigest()


def reserved_test_hashes(runs: list[Run]) -> set[str]:
    hashes = set()
    for run in runs:
        with (run.path / "split_manifest.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        test = [row for row in rows if row.get("split") == "test"]
        if len(test) != run.metadata["counts"]["test"] or any(not row.get("sha256") for row in test):
            raise ValueError("Missing/incomplete Test manifest; cannot enforce development holdout")
        hashes.update(row["sha256"] for row in test)
    return hashes


def collect_inputs(paths: list[Path], recursive: bool = False) -> tuple[list[Path], list[str]]:
    files, warnings, seen = [], [], set()
    for selected in paths:
        selected = Path(selected).expanduser()
        candidates = sorted(selected.rglob("*") if recursive else selected.iterdir()) if selected.is_dir() else [selected]
        for path in candidates:
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                warnings.append(f"Skipped link or missing file: {path}")
                continue
            if path.suffix.lower() not in EXTENSIONS:
                warnings.append(f"Unsupported file: {path}")
                continue
            resolved = path.resolve()
            # Do not follow linked parent directories out of a selected folder.
            if selected.is_dir() and not resolved.is_relative_to(selected.resolve()):
                warnings.append(f"Skipped external link: {path}")
                continue
            if resolved not in seen:
                files.append(resolved)
                seen.add(resolved)
    return files, warnings


def infer(image: Image.Image, bundle: dict) -> tuple[dict, ProcessingResult]:
    start = time.perf_counter()
    result = process_image(image, bundle["method"])
    processing_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    features = feature_vector(result.processed).reshape(1, -1)
    feature_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    scores = bundle["model"].predict_proba(features)[0]
    prediction_ms = (time.perf_counter() - start) * 1000
    classes = bundle["model"].classes_
    return {"predicted_stage": str(classes[scores.argmax()]),
            **{f"score_{c}": float(v) for c, v in zip(classes, scores)},
            "processing_ms": processing_ms, "feature_ms": feature_ms, "prediction_ms": prediction_ms,
            "foreground_fraction": result.details["foreground_fraction"],
            "mask_status": result.details["mask_status"]}, result


def run_batch(paths, runs, *, trusted, cancel, emit):
    """One worker; emit rows and only current-image previews. Never call Tk here."""
    ensure_compatible(runs)
    blocked = reserved_test_hashes(runs)
    models = {}
    for run in runs:
        if cancel.is_set():
            return
        emit("status", f"Loading {LABELS[run.method]}...")
        models[run.method] = load_model(run, trusted=trusted)
    for index, path in enumerate(paths):
        if cancel.is_set():
            return
        image, digest, error = None, "", ""
        try:
            blob = Path(path).read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest in blocked:
                raise ValueError("Reserved original-Test content: blocked during development (including identical copies)")
            with Image.open(io.BytesIO(blob)) as source:
                if source.width * source.height > 20_000_000:
                    raise ValueError("Image exceeds UI safety limit of 20 megapixels; resize a COPY first")
                source.load()
                image = source.copy()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        for run in runs:
            if cancel.is_set():
                return
            bundle, model_hash = models[run.method]
            row = {"source": str(path), "sha256": digest, "method": run.method, "run": str(run.path),
                   "model_sha256": model_hash, "split_id": run.metadata["split_id"], "status": "error", "error": error}
            preview = None
            if not error:
                try:
                    prediction, result = infer(image, bundle)
                    row.update(prediction, status="ok")
                    preview = prediction_card(row, result)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
            emit("result", (row, preview))
        emit("progress", (index + 1, len(paths)))


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def export_csv(path: Path, rows: list[dict], fields: list[str], protected: list[Path]) -> None:
    path = check_destination(path, protected)
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: safe_cell(row.get(key, "")) for key in fields} for row in rows)


def check_destination(path: Path, protected: list[Path]) -> Path:
    path = Path(path).resolve()
    if any(path.is_relative_to(Path(parent).resolve()) for parent in protected):
        raise ValueError("Choose an export destination outside all input folders and saved run folders")
    if path.exists():
        raise ValueError("File already exists; choose a new filename (exports never overwrite)")
    return path


def export_png(path: Path, image: Image.Image, protected: list[Path]) -> None:
    path = check_destination(path, protected)
    with path.open("xb") as f:
        image.save(f, format="PNG")


def prediction_card(row: dict, result: ProcessingResult) -> Image.Image:
    canvas = Image.new("RGB", (900, 420), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=15)
    draw.text((18, 12), f"{LABELS[row['method']]}  |  {row['predicted_stage'].upper()}", fill="#214737", font=font)
    panels = [("Original", result.original), ("Processed", result.processed),
              ("Foreground mask", Image.fromarray(result.mask.astype('uint8') * 255))]
    for i, (title, panel) in enumerate(panels):
        x = 18 + i * 294
        draw.text((x, 45), title, fill="#555555", font=small)
        thumb = ImageOps.contain(panel.convert('RGB'), (275, 240))
        canvas.paste(thumb, (x + (275-thumb.width)//2, 70 + (240-thumb.height)//2))
    scores = ' | '.join(f"{name}: {row['score_'+name]:.1%}" for name in CLASSES)
    draw.text((18, 321), f"Model scores (uncalibrated): {scores}", fill="black", font=small)
    draw.text((18, 345), f"Processing {row['processing_ms']:.1f} ms | Features {row['feature_ms']:.1f} ms | Prediction {row['prediction_ms']:.1f} ms", fill="black", font=small)
    draw.text((18, 369), f"Mask: {row['mask_status']} | Retained: {row['foreground_fraction']:.1%} | Accuracy: unknown (no label)", fill="black", font=small)
    draw.text((18, 393), f"Run: {Path(row['run']).name}", fill="#555555", font=small)
    return canvas


def comparison_sheet(source: str, cards: list[Image.Image]) -> Image.Image:
    if not cards:
        raise ValueError("No successful preview to export")
    cols = min(2, len(cards))
    canvas = Image.new("RGB", (cols*920, ((len(cards)+cols-1)//cols)*440+90), "#edf2ee")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "FRUIT RIPENESS / Image predictions - not dataset accuracy", fill="#214737", font=ImageFont.load_default(size=23))
    # Filename is display-only; full source path and hash are in CSV.
    draw.text((20, 49), Path(source).name[:130], fill="black", font=ImageFont.load_default(size=18))
    for i, card in enumerate(cards):
        canvas.paste(card, (10+(i%cols)*920, 80+(i//cols)*440))
    return canvas


def evaluation_rows(runs: list[Run], scope: str = "overall") -> list[dict]:
    ensure_compatible(runs)
    rows = []
    for run in runs:
        values = run.metrics["validation"]
        if scope != "overall":
            values = values["per_fruit"].get(scope)
            if values is None:
                raise ValueError(f"No saved {scope} metrics for {run.method}")
        rows.append({"method": run.method, "run": str(run.path), "split": "validation",
                     "split_id": run.metadata["split_id"], "scope": scope,
                     **{key: values[key] for key in METRIC_FIELDS[5:]}})
    return rows


def evaluation_sheet(runs: list[Run], scope: str = "overall") -> Image.Image:
    rows = evaluation_rows(runs, scope)
    canvas = Image.new("RGB", (1280, 220 + len(rows)*240), "white")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    draw.text((25, 18), f"VALIDATION COMPARISON / {scope}", fill="#214737", font=font)
    draw.text((25, 52), "Saved measurements. Development results, not final Test performance.", fill="black", font=small)
    draw.text((25, 82), f"Split ID: {runs[0].metadata['split_id']}", fill="black", font=small)
    draw.text((25, 112), "Confusion matrices: rows = true stage, columns = predicted stage. Counts, not percentages.", fill="black", font=small)
    for i, (run, row) in enumerate(zip(runs, rows)):
        y = 160+i*240
        draw.text((25, y), LABELS[run.method], fill="#214737", font=font)
        draw.text((25, y+36), f"n={row['n_images']}  |  Accuracy {row['accuracy']:.2%}", fill="black", font=small)
        draw.text((25, y+66), f"Balanced accuracy {row['balanced_accuracy']:.2%}  |  Macro F1 {row['macro_f1']:.4f}", fill="black", font=small)
        draw.text((25, y+96), f"Macro precision {row['macro_precision']:.4f}  |  Recall {row['macro_recall']:.4f}", fill="black", font=small)
        draw.text((25, y+131), f"Run: {run.path.name}", fill="#555555", font=small)
        values = run.metrics['validation'] if scope == 'overall' else run.metrics['validation']['per_fruit'][scope]
        matrix = np.asarray(values['confusion_matrix'])
        for c, name in enumerate(CLASSES):
            draw.text((820+c*132, y), name, fill="black", font=small)
        for r, name in enumerate(CLASSES):
            draw.text((710, y+38+r*52), name, fill="black", font=small)
            for c in range(3):
                x, cy = 810+c*132, y+30+r*52
                colour = '#d5e9dc' if r == c else '#f0f1f1'
                draw.rectangle((x, cy, x+121, cy+44), fill=colour)
                draw.text((x+45, cy+10), str(matrix[r,c]), fill="black", font=small)
        draw.line((25, y+219, 1255, y+219), fill='#dddddd')
    return canvas
