"""One shared MobileNetV2 CNN: frozen ImageNet backbone + trained ripeness head.

All raw/processed training views receive equal exposure. Original Test is never
decoded or evaluated. Torch is optional and imported only for CNN operations.
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
from threading import Lock

import numpy as np
from PIL import Image, ImageOps

from .audit import write_csv
from .baseline import CLASSES, SEED, assign_splits, collect_records, compute_metrics, resolve_dataset_root
from .experiment import METHOD_LABELS
from .processing import process_image, processing_spec

METHODS = ('baseline', *METHOD_LABELS)
FEATURE_ID = 'shared_mobilenet_v2_letterbox224_v1'
INPUT_SPEC = {'size': 224, 'resize': 'Pillow BILINEAR, preserve aspect ratio, round; centre black pad, no crop',
              'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'layout': 'CHW float32 RGB /255 then normalise'}
TORCH_VERSION = '2.10.0'
VISION_VERSION = '0.25.0'
_TORCH_LOCK = Lock()


def runtime():
    try:
        import torch
        import torchvision
    except (ImportError, OSError, RuntimeError) as exc:
        raise ValueError('CNN dependencies unavailable; install requirements-cnn.txt. ' + str(exc)) from exc
    if torch.__version__.split('+')[0] != TORCH_VERSION or torchvision.__version__.split('+')[0] != VISION_VERSION:
        raise ValueError('CNN dependency versions differ; install pinned requirements-cnn.txt')
    return torch, torchvision


def input_array(image: Image.Image) -> np.ndarray:
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    size = INPUT_SPEC['size']
    scale = size / max(rgb.size)
    dims = tuple(max(1, round(d*scale)) for d in rgb.size)
    resized = rgb.resize(dims, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), 'black')
    canvas.paste(resized, ((size-dims[0])//2, (size-dims[1])//2))
    array = np.asarray(canvas, dtype=np.float32)/np.float32(255)
    array = (array-np.asarray(INPUT_SPEC['mean'], dtype=np.float32))/np.asarray(INPUT_SPEC['std'], dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def make_model(*, pretrained: bool):
    torch, vision = runtime()
    weights = vision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = vision.models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.classifier = torch.nn.Linear(model.last_channel, len(CLASSES))
    return model.cpu().eval()


def backbone_features(model, arrays: np.ndarray) -> np.ndarray:
    torch, _ = runtime()
    with torch.inference_mode():
        x = model.features(torch.from_numpy(arrays))
        return torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1).cpu().numpy().copy()


def image_view(root: Path, row: dict, method: str):
    if row['split'] not in {'train', 'validation'} or row['source_split'] != 'train':
        raise ValueError('Only fitting/validation views from original Train are permitted')
    blob = (root/row['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['sha256']:
        raise ValueError(f"Source changed: {row['path']}")
    start = time.perf_counter()
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        result = process_image(image, method)
    seconds = time.perf_counter()-start
    return input_array(result.processed), {'split': row['split'], 'path': row['path'], 'method': method,
            **result.details, 'decode_and_processing_seconds': seconds}


def extract_features(model, root, rows, method, batch_size):
    chunks, diagnostics = [], []
    for offset in range(0, len(rows), batch_size):
        views = [image_view(root, row, method) for row in rows[offset:offset+batch_size]]
        chunks.append(backbone_features(model, np.stack([v[0] for v in views])))
        diagnostics.extend(v[1] for v in views)
        completed = min(offset+batch_size, len(rows))
        if offset == 0 or completed == len(rows) or completed//250 != offset//250:
            print(f'  {method}: {completed}/{len(rows)} images', flush=True)
    return np.concatenate(chunks), diagnostics


def scores_for_features(model, features, batch_size=256):
    torch, _ = runtime()
    with torch.inference_mode():
        return np.concatenate([model.classifier(torch.from_numpy(features[i:i+batch_size])).softmax(1).numpy()
                               for i in range(0, len(features), batch_size)])


def metrics_for_scores(rows, scores):
    if scores.shape != (len(rows), len(CLASSES)) or not np.isfinite(scores).all():
        raise ValueError('Invalid CNN score dimensions or nonfinite scores')
    predicted = np.asarray(CLASSES)[scores.argmax(1)]
    metrics = compute_metrics([r['stage'] for r in rows], predicted)
    metrics['per_fruit'] = {}
    for fruit in sorted({r['fruit'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['fruit'] == fruit]
        metrics['per_fruit'][fruit] = compute_metrics([rows[i]['stage'] for i in indices], predicted[indices])
    predictions = [{**{k: row[k] for k in ['path', 'fruit']}, 'true_stage': row['stage'],
                    'predicted_stage': str(pred), 'correct': bool(pred == row['stage']),
                    **{f'score_{c}': float(s) for c, s in zip(CLASSES, score)}}
                   for row, pred, score in zip(rows, predicted, scores)]
    return metrics, predictions


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def prepare_records(data, baseline_run):
    root = resolve_dataset_root(data)
    records = assign_splits(collect_records(root))
    split_id = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    reference = json.loads((Path(baseline_run)/'metadata.json').read_text(encoding='utf-8'))
    if reference.get('status') != 'completed' or reference.get('method') != 'baseline' or reference.get('split_id') != split_id:
        raise ValueError('Dataset/split differs from the completed baseline run; do not mix experiments')
    return root, records, split_id


def train(data, out, baseline_run, *, epochs=30, batch_size=32, threads=2, patience=5):
    if min(epochs, batch_size, threads, patience) < 1:
        raise ValueError('epochs, batch-size, threads and patience must be positive')
    root, records, split_id = prepare_records(data, baseline_run)
    if Path(out).resolve().is_relative_to(root):
        raise ValueError('Write CNN results outside source data')
    torch, vision = runtime()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    run = None
    try:
        print('Loading ImageNet MobileNetV2 weights (first use downloads approximately 14 MB)...', flush=True)
        model = make_model(pretrained=True)  # Fail visibly; never fall back to random weights.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run = Path(out).resolve()/stamp
        run.mkdir(parents=True, exist_ok=False)
        cache = run/'features'
        cache.mkdir()
        counts = dict(Counter(r['split'] for r in records))
        from .ui_core import CURRENT_VERSIONS
        metadata = {'status': 'building', 'method': 'shared_cnn', 'backend': 'shared_cnn',
                    'created_utc': stamp, 'feature_id': FEATURE_ID, 'input_spec': INPUT_SPEC,
                    'methods': list(METHODS), 'processing_specs': {m: processing_spec(m) for m in METHODS},
                    'model_type': 'MobileNetV2_frozen_backbone_trained_linear_head',
                    'model_parameters': {'weights': 'IMAGENET1K_V2', 'head': 'Linear(1280,3)',
                        'max_epochs': epochs, 'batch_size_features': batch_size, 'batch_size_head': 128,
                        'learning_rate': 0.001, 'weight_decay': 0.0001, 'optimizer': 'AdamW',
                        'patience': patience, 'selection': 'mean per-method validation macro F1; earliest tie',
                        'class_weight': 'inverse fitting-stage frequency', 'method_sampling': 'every fitting image x every method each epoch',
                        'augmentation': 'method views only; no random crops/colour transforms'},
                    'seed': SEED, 'counts': counts, 'split_id': split_id,
                    'model_class_order': CLASSES, 'metric_class_order': CLASSES,
                    'test_evaluated': False, 'source_images_modified': False, 'duplicates_removed': False,
                    'device': 'cpu', 'threads': threads, 'versions': {**CURRENT_VERSIONS,
                        'torch': str(torch.__version__), 'torchvision': str(vision.__version__), 'python': platform.python_version()},
                    'limitations': ['Supplied duplicates and labels retained; split is not group-independent.',
                        'Views are not independent new source images.', 'Frozen ImageNet backbone, not end-to-end CNN fine-tuning.',
                        'Validation selects one shared checkpoint; these are development scores, not final Test results.',
                        'Model scores are uncalibrated; segmentation masks remain heuristics.']}
        json_write(run/'metadata.json', metadata)
        write_csv(run/'split_manifest.csv', records, ['path','source_split','split','fruit','stage','raw_stage','sha256'])
        print(f'Dataset: {root}\nImages: {counts}; source images and labels retained\nRun: {run}', flush=True)
        sets = {s: [r for r in records if r['split'] == s] for s in ['train','validation']}
        diagnostics = []
        started = time.perf_counter()
        for split in sets:
            print(f'Extracting frozen CNN features: {split}; {len(METHODS)} views per source image...', flush=True)
            for method in METHODS:
                features, details = extract_features(model, root, sets[split], method, batch_size)
                np.save(cache/f'{split}_{method}.npy', features, allow_pickle=False)
                diagnostics.extend(details)
        metadata['feature_extraction_seconds'] = time.perf_counter()-started
        write_csv(run/'processing_diagnostics.csv', diagnostics,
                  ['split','path','method','foreground_fraction','mask_status','refined_review_flag','decode_and_processing_seconds'])
        del diagnostics
        x = np.concatenate([np.load(cache/f'train_{m}.npy', allow_pickle=False) for m in METHODS])
        labels = np.asarray([CLASSES.index(r['stage']) for r in sets['train']], dtype=np.int64)
        y = np.tile(labels, len(METHODS))
        validation = {m: np.load(cache/f'validation_{m}.npy', allow_pickle=False) for m in METHODS}
        weights = len(labels)/(len(CLASSES)*np.bincount(labels, minlength=len(CLASSES)))
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.0001)
        rng = np.random.default_rng(SEED)
        history, best_score, best_state, stale = [], -1.0, None, 0
        print(f'Training ONE shared CNN classifier on {len(y)} views of {len(labels)} fitting images...', flush=True)
        for epoch in range(1, epochs+1):
            order = rng.permutation(len(y))
            loss_sum = 0.0
            for offset in range(0, len(y), 128):
                idx = order[offset:offset+128]
                optimizer.zero_grad(set_to_none=True)
                logits = model.classifier(torch.from_numpy(x[idx]))
                loss = criterion(logits, torch.from_numpy(y[idx]))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())*len(idx)
            measures = {m: metrics_for_scores(sets['validation'], scores_for_features(model, validation[m]))[0] for m in METHODS}
            score = float(np.mean([v['macro_f1'] for v in measures.values()]))
            history.append({'epoch': epoch, 'train_loss': loss_sum/len(y), 'mean_validation_macro_f1': score,
                            **{f'{m}_macro_f1': measures[m]['macro_f1'] for m in METHODS}})
            write_csv(run/'history.csv', history, list(history[0]))
            print(f'Epoch {epoch}/{epochs}: loss={loss_sum/len(y):.4f}, mean validation macro F1={score:.4f}', flush=True)
            if score > best_score:
                best_score, stale = score, 0
                best_state = {k: v.detach().clone() for k,v in model.classifier.state_dict().items()}
                metadata['best_epoch'] = epoch
            else:
                stale += 1
                if stale >= patience:
                    break
        model.classifier.load_state_dict(best_state)
        model.eval()
        metrics, comparison = {}, []
        for method in METHODS:
            values, predictions = metrics_for_scores(sets['validation'], scores_for_features(model, validation[method]))
            metrics[method] = {'validation': values}
            write_csv(run/f'predictions_validation_{method}.csv', predictions,
                      ['path','fruit','true_stage','predicted_stage','correct', *[f'score_{c}' for c in CLASSES]])
            write_csv(run/f'confusion_matrix_validation_{method}.csv',
                      [{'true_stage': c, **dict(zip(CLASSES,row))} for c,row in zip(CLASSES,values['confusion_matrix'])], ['true_stage',*CLASSES])
            comparison.append({'method': method, 'backend': 'shared_cnn', 'split_id': split_id,
                               **{k: values[k] for k in ['n_images','accuracy','balanced_accuracy','macro_f1']}})
            print(f"{method}: accuracy={values['accuracy']:.4f}, balanced_accuracy={values['balanced_accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}", flush=True)
        metadata.update(status='completed', epochs_run=len(history), best_mean_validation_macro_f1=best_score)
        torch.save({'metadata': metadata, 'state_dict': model.state_dict()}, run/'model.pt')
        metadata['model_sha256'] = hashlib.sha256((run/'model.pt').read_bytes()).hexdigest()
        json_write(run/'metrics.json', metrics)
        write_csv(run/'comparison_validation.csv', comparison, list(comparison[0]))
        json_write(run/'metadata.json', metadata)  # Completion marker written LAST.
        print(f'CNN saved: {run}\nOriginal Test not evaluated. Same CNN weights used for every method.', flush=True)
        return run, metrics
    except BaseException:
        if run is not None:
            print(f'Incomplete run retained for diagnosis: {run}. UI ignores incomplete runs.', flush=True)
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def read_shared(path):
    """JSON-only discovery; no torch import, downloads or model deserialisation."""
    from .ui_core import Run, validate_metrics
    path = Path(path).resolve()
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    metrics = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
    if (metadata.get('status') != 'completed' or metadata.get('backend') != 'shared_cnn'
            or metadata.get('method') != 'shared_cnn' or metadata.get('feature_id') != FEATURE_ID
            or metadata.get('methods') != list(METHODS) or metadata.get('input_spec') != INPUT_SPEC
            or metadata.get('processing_specs') != {m: processing_spec(m) for m in METHODS}
            or metadata.get('model_class_order') != CLASSES or metadata.get('metric_class_order') != CLASSES
            or not metadata.get('split_id') or len(metadata.get('model_sha256','')) != 64
            or metadata.get('test_evaluated') is not False):
        raise ValueError('Incomplete/incompatible shared CNN run')
    runs = []
    for method in METHODS:
        values = metrics[method]['validation']
        validate_metrics(values)
        if values['n_images'] != metadata['counts']['validation']:
            raise ValueError('CNN validation count differs')
        for fruit_values in values.get('per_fruit', {}).values():
            validate_metrics(fruit_values)
        runs.append(Run(path, {**metadata, 'method': method}, metrics[method]))
    return runs


def load_shared(run):
    torch, vision = runtime()
    from .ui_core import CURRENT_VERSIONS
    for key, current in {**CURRENT_VERSIONS, 'torch': torch.__version__, 'torchvision': vision.__version__}.items():
        if run.metadata['versions'].get(key) != current:
            raise ValueError(f'CNN {key} version differs from training')
    blob = (run.path/'model.pt').read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != run.metadata['model_sha256']:
        raise ValueError('CNN checkpoint hash differs from selected run')
    checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    expected = {k:v for k,v in run.metadata.items() if k != 'model_sha256'}
    expected['method'] = 'shared_cnn'
    if checkpoint.get('metadata') != expected:
        raise ValueError('CNN embedded metadata differs from selected run')
    model = make_model(pretrained=False)  # Loading never downloads weights.
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return {'backend': 'shared_cnn', 'method': run.method, 'model': model}, digest


def predict_scores(model, array):
    torch, _ = runtime()
    with _TORCH_LOCK:
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with torch.inference_mode():
                scores = model(torch.from_numpy(array[None])).softmax(1).cpu().numpy()[0]
        finally:
            torch.set_num_threads(previous)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite CNN prediction')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--out', type=Path, default=Path('outputs/shared_cnn'))
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    try:
        train(args.data, args.out, args.baseline_run, epochs=args.epochs,
              batch_size=args.batch_size, threads=args.threads, patience=args.patience)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
