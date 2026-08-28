"""Read-only run discovery, shared inference, and explicit report exports for the UI.

No training, source edits, label inference from filenames, or Test evaluation.
Tk widgets live separately so these behaviours can be tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
LABELS = {
    'baseline': 'Raw image (baseline)',
    'hsv': 'Method 1 - HSV colour threshold',
    'otsu': 'Method 2 - Otsu threshold',
    'kmeans': 'Method 3 - K-means clustering',
    'grabcut': 'Method 4 - GrabCut',
    'watershed': 'Method 5 - Watershed',
    'hybrid': 'Hybrid A - HSV + GrabCut union',
    'hybrid_refined': 'Hybrid B - HSV-seeded GrabCut',
}
CLASSIFIER_LABELS = {
    'random_forest': 'Random Forest (per-method models)',
    'shared_cnn': 'CNN - MobileNetV2 (shared model)',
}
INDIVIDUAL_METHODS = ("hsv", "otsu", "kmeans", "grabcut", "watershed")
HYBRID_VARIANTS = ("hybrid", "hybrid_refined")


def run_label(name: str) -> str:
    """Human-readable UTC date; never change the actual folder/checkpoint ID."""
    try:
        return datetime.strptime(name, '%Y%m%dT%H%M%S_%fZ').strftime('%d %b %Y, %H:%M:%S UTC')
    except ValueError:
        return name


def choice_labels(keys, label):
    """Unique display labels mapped to exact IDs, even for same-second runs."""
    keys = list(keys)
    labels = [label(key) for key in keys]
    if len(set(labels)) != len(labels):
        labels = [f'{value} [{key}]' for key, value in zip(keys, labels)]
    return dict(zip(keys, labels))


def comparison_methods(hybrid_variant: str) -> list[str]:
    if hybrid_variant not in HYBRID_VARIANTS:
        raise ValueError("Select a supported hybrid version")
    return [*INDIVIDUAL_METHODS, hybrid_variant]


SCORE_FIELDS = [f"score_{name}" for name in CLASSES]
RESULT_FIELDS = ["source", "sha256", "method", "run", "model_sha256", "split_id", "status", "error",
                 "predicted_stage", *SCORE_FIELDS, "processing_ms", "feature_ms", "prediction_ms",
                 "foreground_fraction", "mask_status", "refined_review_flag"]
METRIC_FIELDS = ["method", "run", "split", "split_id", "scope", "n_images", "accuracy",
                 "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
RESULT_FIELDS.append('backend')
METRIC_VALUE_FIELDS = METRIC_FIELDS[5:].copy()
METRIC_FIELDS.append('backend')
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

    @property
    def backend(self):
        return self.metadata.get('backend', 'random_forest')

    @property
    def model_path(self):
        return self.path / ('model.pt' if self.backend == 'shared_cnn' else 'model.joblib')


def validate_metrics(values: dict) -> None:
    if not isinstance(values, dict):
        raise ValueError("Metrics must be a JSON object")
    matrix = np.asarray(values["confusion_matrix"])
    if (values["confusion_matrix_labels"] != CLASSES or matrix.shape != (3, 3)
            or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0)
            or int(matrix.sum()) != values["n_images"] or values["n_images"] <= 0):
        raise ValueError("Invalid confusion matrix, class order or image count")
    for key in METRIC_VALUE_FIELDS[1:]:
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


def discover_runs(outputs: Path, backend: str = 'random_forest') -> tuple[dict[str, list[Run]], list[str]]:
    """JSON only: discovery never deserialises a model or opens dataset images."""
    found = {method: [] for method in LABELS}
    warnings = []
    outputs = Path(outputs)
    if not outputs.is_dir():
        return found, [f"Output folder not found: {outputs}"]
    if backend == 'shared_cnn':
        from .cnn import read_shared
        for path in sorted((outputs/'shared_cnn').glob('*/metadata.json'), reverse=True):
            try:
                for run in read_shared(path.parent):
                    found[run.method].append(run)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                warnings.append(f'shared_cnn/{path.parent.name}: {exc}')
        return found, warnings
    if backend != 'random_forest':
        raise ValueError('Unknown classifier backend')
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
        if current.get('evaluation_split', 'validation') != reference.get('evaluation_split', 'validation'):
            raise ValueError('Do not mix validation and final Test results')
        if run.backend != runs[0].backend:
            raise ValueError('Do not mix Random Forest and shared CNN in one method comparison')
        if run.backend == 'shared_cnn' and (run.path != runs[0].path or current.get('model_sha256') != reference.get('model_sha256')):
            raise ValueError('All methods must use the SAME shared CNN checkpoint')
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
    if run.metadata.get('evaluation_only'):
        raise ValueError('Saved Test reports are evaluation-only, not training model runs')
    if not trusted:
        raise ValueError("Confirm these are your team's trusted model files before loading")
    if run.backend == 'shared_cnn':
        from .cnn import load_shared
        return load_shared(run)
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
    if bundle.get('backend') == 'shared_cnn':
        from .cnn import input_array, predict_scores
        features = input_array(result.processed)
    else:
        features = feature_vector(result.processed).reshape(1, -1)
    feature_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    if bundle.get('backend') == 'shared_cnn':
        scores = predict_scores(bundle['model'], features)
        classes = np.asarray(CLASSES)
    else:
        scores = bundle["model"].predict_proba(features)[0]
        classes = bundle["model"].classes_
    prediction_ms = (time.perf_counter() - start) * 1000
    return {"predicted_stage": str(classes[scores.argmax()]),
            **{f"score_{c}": float(v) for c, v in zip(classes, scores)},
            "processing_ms": processing_ms, "feature_ms": feature_ms, "prediction_ms": prediction_ms,
            "foreground_fraction": result.details["foreground_fraction"],
            "mask_status": result.details["mask_status"],
            "refined_review_flag": result.details.get("refined_review_flag", "")}, result


def run_batch(paths, runs, *, trusted, cancel, emit):
    """One worker; emit rows and only current-image previews. Never call Tk here."""
    ensure_compatible(runs)
    blocked = reserved_test_hashes(runs)
    models = {}
    shared = None
    for run in runs:
        if cancel.is_set():
            return
        emit("status", f"Loading {LABELS[run.method]}...")
        if run.backend == 'shared_cnn':
            if shared is None:
                shared = load_model(run, trusted=trusted)
            models[run.method] = ({**shared[0], 'method': run.method}, shared[1])
        else:
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
                   "model_sha256": model_hash, "split_id": run.metadata["split_id"], "status": "error", "error": error,
                   'backend': run.backend}
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
    assessment = 'Accuracy: unknown (no label)'
    if row.get('split') == 'test' and row.get('true_stage') in CLASSES:
        assessment = f"Test label: {row['true_stage']} | {'Match' if row['predicted_stage'] == row['true_stage'] else 'Mismatch'}"
    draw.text((18, 369), f"Mask: {row['mask_status']} | Retained: {row['foreground_fraction']:.1%} | {assessment}", fill="black", font=small)
    note = f"Run: {run_label(Path(row['run']).name)}"
    if row.get('backend') == 'shared_cnn':
        note += ' | MobileNetV2 CNN'
    else:
        note += ' | Random Forest'
    if row.get('refined_review_flag'):
        note += f" | Review: {row['refined_review_flag']}"
    draw.text((18, 393), note, fill="#555555", font=small)
    return canvas


def comparison_sheet(source: str, cards: list[Image.Image], *, split: str | None = None) -> Image.Image:
    if not cards:
        raise ValueError("No successful preview to export")
    cols = min(2, len(cards))
    canvas = Image.new("RGB", (cols*920, ((len(cards)+cols-1)//cols)*440+90), "#edf2ee")
    draw = ImageDraw.Draw(canvas)
    title = 'FINAL TEST / Image predictions - not overall accuracy' if split == 'test' else 'FRUIT RIPENESS / Image predictions - not dataset accuracy'
    draw.text((20, 12), title, fill="#214737", font=ImageFont.load_default(size=23))
    # Filename is display-only; full source path and hash are in CSV.
    draw.text((20, 49), Path(source).name[:130], fill="black", font=ImageFont.load_default(size=18))
    for i, card in enumerate(cards):
        canvas.paste(card, (10+(i%cols)*920, 80+(i//cols)*440))
    return canvas


def evaluation_rows(runs: list[Run], scope: str = "overall", *, split: str = 'validation') -> list[dict]:
    ensure_compatible(runs)
    if split not in {'validation', 'test'}:
        raise ValueError('Select validation or test')
    if split == 'test' and any(r.metadata.get('evaluation_split') != 'test' for r in runs):
        raise ValueError('Final Test results must come from a saved final-Test report')
    rows = []
    for run in runs:
        if split not in run.metrics:
            raise ValueError(f'No saved {split} results for {run.method}')
        values = run.metrics[split]
        if scope != "overall":
            values = values["per_fruit"].get(scope)
            if values is None:
                raise ValueError(f"No saved {scope} metrics for {run.method}")
        rows.append({"method": run.method, "run": str(run.path), "split": split,
                     "split_id": run.metadata["split_id"], "scope": scope,
                     'backend': run.backend, **{key: values[key] for key in METRIC_VALUE_FIELDS}})
    return rows


def evaluation_sheet(runs: list[Run], scope: str = "overall", *, split: str = 'validation') -> Image.Image:
    rows = evaluation_rows(runs, scope, split=split)
    canvas = Image.new("RGB", (1280, 220 + len(rows)*240), "white")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    backend_label = 'MobileNetV2 CNN (shared)' if runs[0].backend == 'shared_cnn' else 'Random Forest'
    title = 'FINAL TEST COMPARISON' if split == 'test' else 'VALIDATION COMPARISON'
    subtitle = 'Saved original-Test results. Frozen checkpoint; no retraining. Dataset duplicates/labels retained.' if split == 'test' else 'Saved measurements. Development results, not final Test performance.'
    draw.text((25, 18), f"{title} / {scope} / {backend_label}", fill="#214737", font=font)
    draw.text((25, 52), subtitle, fill="black", font=small)
    draw.text((25, 82), f"Split ID: {runs[0].metadata['split_id']}", fill="black", font=small)
    draw.text((25, 112), "Confusion matrices: rows = true stage, columns = predicted stage. Counts, not percentages.", fill="black", font=small)
    for i, (run, row) in enumerate(zip(runs, rows)):
        y = 160+i*240
        draw.text((25, y), LABELS[run.method], fill="#214737", font=font)
        draw.text((25, y+36), f"n={row['n_images']}  |  Accuracy {row['accuracy']:.2%}", fill="black", font=small)
        draw.text((25, y+66), f"Balanced accuracy {row['balanced_accuracy']:.2%}  |  Macro F1 {row['macro_f1']:.4f}", fill="black", font=small)
        draw.text((25, y+96), f"Macro precision {row['macro_precision']:.4f}  |  Recall {row['macro_recall']:.4f}", fill="black", font=small)
        draw.text((25, y+131), f"Run: {run_label(run.path.name)}", fill="#555555", font=small)
        values = run.metrics[split] if scope == 'overall' else run.metrics[split]['per_fruit'][scope]
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
