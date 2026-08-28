"""Evaluate ALL input variants on original Test with one frozen saved CNN.

No fitting, parameter selection, image exclusions or modifications to source runs.
Requires explicit frozen-settings and trusted-model confirmations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import cnn
from .audit import write_csv
from .baseline import CLASSES, assign_splits, collect_records, resolve_dataset_root
from .processing import processing_spec
from .ui_core import (Run, METRIC_FIELDS, RESULT_FIELDS, comparison_sheet, evaluation_rows,
                      evaluation_sheet, infer, load_model, prediction_card, validate_metrics)

MANIFEST_FIELDS = ['path','source_split','split','fruit','stage','raw_stage','sha256']
PREDICTION_FIELDS = ['path','fruit','true_stage','predicted_stage','correct','split',
                     *(field for field in RESULT_FIELDS if field != 'predicted_stage')]
REPORT_VERSION = 'frozen_shared_cnn_test_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dataset(root, run):
    """Hash all source bytes and compare the full saved manifest before ANY decode."""
    with (run.path/'split_manifest.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(MANIFEST_FIELDS):
            raise ValueError('Saved split manifest fields differ')
        saved = list(reader)
    records = assign_splits(collect_records(root))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if digest != run.metadata['split_id'] or saved != records:
        raise ValueError('Dataset or saved split manifest changed; final evaluation requires the original saved split')
    if dict(Counter(r['split'] for r in records)) != run.metadata['counts']:
        raise ValueError('Dataset counts differ from the saved run')
    return records


def read_test_image(root, row):
    if row['split'] != 'test' or row['source_split'] != 'test':
        raise ValueError('Final evaluator only accepts original-Test manifest rows')
    path = root/row['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source image is outside dataset')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != row['sha256']:
        raise ValueError(f"Test image changed: {row['path']}")
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        return image.copy()


def preview_paths(rows):
    """First path in each fruit/stage stratum, selected BEFORE any predictions."""
    selected = {}
    for row in sorted(rows, key=lambda r:r['path']):
        selected.setdefault((row['fruit'], row['stage']), row['path'])
    return set(selected.values())


def report_runs(path, metadata, metrics):
    if (metadata.get('status') != 'completed' or metadata.get('report_version') != REPORT_VERSION
            or metadata.get('method') != 'shared_cnn_final_test' or metadata.get('backend') != 'shared_cnn'
            or metadata.get('evaluation_split') != 'test' or metadata.get('evaluation_only') is not True
            or metadata.get('test_evaluated') is not True or metadata.get('frozen_confirmed') is not True
            or metadata.get('feature_id') != cnn.FEATURE_ID or metadata.get('input_spec') != cnn.INPUT_SPEC
            or metadata.get('methods') != list(cnn.METHODS) or set(metrics) != set(cnn.METHODS)
            or metadata.get('processing_specs') != {m:processing_spec(m) for m in cnn.METHODS}
            or metadata.get('model_class_order') != CLASSES or metadata.get('metric_class_order') != CLASSES
            or not metadata.get('split_id') or not metadata.get('source_run')
            or len(metadata.get('model_sha256','')) != 64):
        raise ValueError('Incomplete/incompatible final-Test report')
    runs = []
    for method in cnn.METHODS:
        if set(metrics[method]) != {'test'}:
            raise ValueError('Final report must contain Test metrics only')
        values = metrics[method]['test']
        validate_metrics(values)
        if values['n_images'] != metadata['counts']['test']:
            raise ValueError('Test count differs from saved manifest count')
        for fruit_values in values.get('per_fruit', {}).values():
            validate_metrics(fruit_values)
        if sum(v['n_images'] for v in values.get('per_fruit',{}).values()) != values['n_images']:
            raise ValueError('Per-fruit counts do not cover all Test images')
        runs.append(Run(Path(path).resolve(), {**metadata,'method':method}, metrics[method]))
    return runs


def read_report(path):
    """Read-only, no model loading or image decoding; works without torch installed."""
    path = Path(path).resolve()
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    blob = (path/'metrics.json').read_bytes()
    if hashlib.sha256(blob).hexdigest() != metadata.get('metrics_sha256'):
        raise ValueError('Test metrics hash differs from completed report')
    return report_runs(path, metadata, json.loads(blob))


def evaluate_final(data, model_run, out, *, confirm_frozen=False, trusted_model=False):
    if not confirm_frozen:
        raise ValueError('Confirm frozen settings with --confirm-frozen before opening Test images')
    if not trusted_model:
        raise ValueError('Confirm your own trusted checkpoint with --trusted-model')
    training_runs = cnn.read_shared(model_run)
    source_run = training_runs[0]
    root = resolve_dataset_root(data)
    out = Path(out).resolve()
    if out.is_relative_to(root) or out.is_relative_to(source_run.path):
        raise ValueError('Write Test reports outside source data and the saved model run')
    records = verify_dataset(root, source_run)
    test = [r for r in records if r['split']=='test']
    selected = preview_paths(test)
    protected = {name:sha(source_run.path/name) for name in ['metadata.json','metrics.json','split_manifest.csv','model.pt']}
    bundle, model_hash = load_model(source_run, trusted=True)  # Exactly once; never fit.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = out/stamp
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {**source_run.metadata, 'status':'building', 'method':'shared_cnn_final_test',
                'report_version':REPORT_VERSION, 'evaluation_split':'test', 'evaluation_only':True,
                'test_evaluated':True, 'frozen_confirmed':True, 'evaluation_created_utc':stamp,
                'source_run':str(source_run.path), 'source_file_hashes':protected, 'model_sha256':model_hash,
                'preview_selection':'first lexicographic Test path per fruit/stage; selected before prediction',
                'preview_paths':sorted(selected), 'test_policy':'All original Test images retained; no tuning/retraining',
                'evaluation_limitations':['Supplied duplicates/labels unchanged; Test is not guaranteed independent of fitting data.',
                    'Single frozen checkpoint, not evidence of statistical significance.',
                    'Do not tune methods or the model after inspecting these Test results.']}
    cnn.json_write(destination/'metadata.json', metadata)
    write_csv(destination/'split_manifest.csv', records, MANIFEST_FIELDS)
    previews = destination/'previews'
    previews.mkdir()
    all_predictions, preview_index = [], []
    scores = {m:[] for m in cnn.METHODS}
    print(f'Frozen CNN: {source_run.path}\nCheckpoint SHA-256: {model_hash}\nEvaluating {len(test)} original-Test images x {len(cnn.METHODS)} variants; no retraining...', flush=True)
    try:
        for i, record in enumerate(test):
            image = read_test_image(root, record)
            cards = []
            for method in cnn.METHODS:
                prediction, processed = infer(image, {**bundle,'method':method})
                row = {**prediction, 'path':record['path'], 'source':str(root/record['path']),
                       'sha256':record['sha256'], 'fruit':record['fruit'], 'true_stage':record['stage'],
                       'correct':prediction['predicted_stage']==record['stage'], 'split':'test',
                       'method':method, 'backend':'shared_cnn', 'run':str(source_run.path),
                       'model_sha256':model_hash, 'split_id':source_run.metadata['split_id'], 'status':'ok','error':''}
                all_predictions.append(row)
                scores[method].append([prediction[f'score_{c}'] for c in CLASSES])
                if record['path'] in selected:
                    cards.append(prediction_card(row, processed))
            if cards:
                filename = f'{i+1:04d}_{Path(record["path"]).stem}.png'
                comparison_sheet(record['path'], cards, split='test').save(previews/filename)
                preview_index.append({'path':record['path'],'true_stage':record['stage'],'preview':filename})
            if (i+1)%10 == 0 or i+1 == len(test):
                print(f'  Evaluated {i+1}/{len(test)} Test images across all variants', flush=True)
        # Abort if source/checkpoint inputs changed while the evaluator was running.
        if any(sha(source_run.path/name) != digest for name,digest in protected.items()):
            raise ValueError('Saved model/run files changed during evaluation')
        verify_dataset(root, source_run)
        metrics = {method:{'test':cnn.metrics_for_scores(test,np.asarray(scores[method]))[0]} for method in cnn.METHODS}
        write_csv(destination/'predictions_test_all.csv', all_predictions, PREDICTION_FIELDS)
        for method in cnn.METHODS:
            values = metrics[method]['test']
            write_csv(destination/f'predictions_test_{method}.csv', [r for r in all_predictions if r['method']==method], PREDICTION_FIELDS)
            write_csv(destination/f'confusion_matrix_test_{method}.csv',
                      [{'true_stage':c,**dict(zip(CLASSES,row))} for c,row in zip(CLASSES,values['confusion_matrix'])], ['true_stage',*CLASSES])
            print(f"{method}: accuracy={values['accuracy']:.4f}, balanced_accuracy={values['balanced_accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}", flush=True)
        write_csv(previews/'index.csv', preview_index, ['path','true_stage','preview'])
        cnn.json_write(destination/'metrics.json', metrics)
        metadata.update(status='completed',metrics_sha256=sha(destination/'metrics.json'))
        runs = report_runs(destination, metadata, metrics)
        comparison = []
        for scope in ['overall', *sorted({r['fruit'] for r in test})]:
            comparison.extend(evaluation_rows(runs, scope, split='test'))
            evaluation_sheet(runs, scope, split='test').save(destination/f'comparison_test_{scope}.png')
        write_csv(destination/'comparison_test.csv', comparison, METRIC_FIELDS)
        cnn.json_write(destination/'metadata.json', metadata)  # Completion marker LAST.
        print(f'Final Test report: {destination}\nAll methods evaluated using one unchanged checkpoint. Do not tune against Test results.', flush=True)
        return destination, metrics
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        cnn.json_write(destination/'metadata.json', metadata)
        print(f'Incomplete report retained: {destination}; no complete Test result claimed.', flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--run', type=Path, required=True, help='Completed outputs/shared_cnn/<timestamp>')
    parser.add_argument('--out', type=Path, default=Path('outputs/final_test'))
    parser.add_argument('--confirm-frozen', action='store_true', help='Confirm no further model/method tuning')
    parser.add_argument('--trusted-model', action='store_true', help='Confirm this checkpoint was generated by your team')
    args = parser.parse_args()
    try:
        evaluate_final(args.data,args.run,args.out,confirm_frozen=args.confirm_frozen,trusted_model=args.trusted_model)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    main()
