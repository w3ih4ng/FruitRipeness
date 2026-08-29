from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from test_baseline import make_dataset
from fruitripeness.baseline import predict_image, train_baseline
from fruitripeness.ui_core import (LABELS, METHODS, METRIC_FIELDS, RESULT_FIELDS, collect_inputs,
    comparison_sheet, discover_runs, ensure_compatible, evaluation_rows, evaluation_sheet,
    export_csv, export_png, infer, load_model, read_run, reserved_test_hashes, run_batch)


class UICoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.data = cls.base/'data'
        cls.outputs = cls.base/'outputs'
        make_dataset(cls.data)
        for i, path in enumerate(sorted(cls.data.rglob('*.png'))):
            with Image.open(path) as original:
                colour = original.getpixel((0, 0))
            image = Image.new('RGB', (40, 40), 'white')
            ImageDraw.Draw(image).rectangle((8, 8, 31, 31), fill=colour)
            image.putpixel((0, 0), (i, i, i))  # Distinct bytes across fit/Test fixtures.
            image.save(path)
        cls.runs = []
        with redirect_stdout(io.StringIO()):
            for method in LABELS:
                path, _ = train_baseline(cls.data, cls.outputs/method, jobs=1, trees=8, method=method)
                cls.runs.append(read_run(path))
        cls.sample = cls.base/'sample.png'
        image = Image.new('RGB', (60, 60), 'white')
        ImageDraw.Draw(image).ellipse((8, 7, 52, 53), fill=(190, 43, 20))
        image.save(cls.sample)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_discovery_and_evaluation_never_load_models(self):
        with patch('fruitripeness.ui_core.joblib.load', side_effect=AssertionError('unsafe load')):
            found, warnings = discover_runs(self.outputs)
            self.assertFalse(warnings)
            self.assertEqual(set(found), set(LABELS))
            self.assertTrue(all(len(runs) == 1 for runs in found.values()))
            rows = evaluation_rows(self.runs)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r['split'] == 'validation' for r in rows))
            self.assertEqual(rows[0]['accuracy'], self.runs[0].metrics['validation']['accuracy'])
            self.assertEqual(evaluation_sheet(self.runs).size, (1280, 2140))

    def test_discovery_skips_invalid_metadata_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'hsv'/'bad'
            folder.mkdir(parents=True)
            (folder/'metadata.json').write_text('broken')
            found, warnings = discover_runs(root)
            self.assertEqual(found['hsv'], [])
            self.assertEqual(len(warnings), 1)
        self.assertTrue(discover_runs(self.base/'missing')[1])

    def test_trust_required_before_deserialisation(self):
        with patch('fruitripeness.ui_core.joblib.load') as loader:
            with self.assertRaisesRegex(ValueError, 'trusted'):
                load_model(self.runs[0])
            loader.assert_not_called()

    def test_current_environment_mismatch_rejected_before_deserialisation(self):
        with patch.dict('fruitripeness.ui_core.CURRENT_VERSIONS', {'numpy': 'other'}):
            with patch('fruitripeness.ui_core.joblib.load') as loader:
                with self.assertRaisesRegex(ValueError, 'version differs'):
                    load_model(self.runs[0], trusted=True)
                loader.assert_not_called()

    def test_all_method_versions_predictions_exactly_match_existing_helper(self):
        before = self.sample.read_bytes()
        for run in self.runs:
            with self.subTest(method=run.method):
                bundle, digest = load_model(run, trusted=True)
                self.assertEqual(digest, hashlib.sha256((run.path/'model.joblib').read_bytes()).hexdigest())
                expected = predict_image(bundle, self.sample)
                with Image.open(self.sample) as image:
                    row, result = infer(image, bundle)
                self.assertEqual(row['predicted_stage'], expected['predicted_stage'])
                for name, score in expected['scores'].items():
                    self.assertEqual(row['score_'+name], score)
                self.assertEqual(result.original.size, (60, 60))
                self.assertGreaterEqual(row['processing_ms'], 0)
                self.assertGreaterEqual(row['prediction_ms'], 0)
        self.assertEqual(before, self.sample.read_bytes())

    def test_compatible_runs_and_incompatible_split_rejection(self):
        ensure_compatible(self.runs)
        altered = read_run(self.runs[1].path)
        altered.metadata['split_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'split_id'):
            ensure_compatible([self.runs[0], altered])
        with self.assertRaisesRegex(ValueError, 'at least one'):
            ensure_compatible([])

    def test_incompatible_model_parameters_rejected(self):
        altered = read_run(self.runs[1].path)
        altered.metadata['model_parameters']['n_estimators'] = 9
        with self.assertRaisesRegex(ValueError, 'model parameter'):
            ensure_compatible([self.runs[0], altered])

    def test_folder_recursion_duplicate_names_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'nested').mkdir()
            Image.new('RGB', (10, 10)).save(root/'same.png')
            Image.new('RGB', (10, 10)).save(root/'nested'/'same.png')
            (root/'notes.txt').write_text('not an image')
            flat, warnings = collect_inputs([root])
            self.assertEqual(len(flat), 1)
            self.assertEqual(len(warnings), 1)
            deep, warnings = collect_inputs([root], recursive=True)
            self.assertEqual(len(deep), 2)
            self.assertNotEqual(deep[0], deep[1])
            dedup, _ = collect_inputs([*deep, deep[0]])
            self.assertEqual(len(dedup), 2)  # Only same path repeated; never content deduplication.

    def test_missing_inputs_are_visible(self):
        files, warnings = collect_inputs([self.base/'does-not-exist.png'])
        self.assertEqual(files, [])
        self.assertEqual(len(warnings), 1)

    def test_batch_continues_after_corrupt_image_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp)/'bad.png'
            corrupt.write_bytes(b'not an image')
            events = []
            before = self.sample.read_bytes()
            run_batch([corrupt, self.sample], self.runs[1:], trusted=True, cancel=threading.Event(),
                      emit=lambda kind, data: events.append((kind, data)))
            rows = [data[0] for kind, data in events if kind == 'result']
            cards = [data[1] for kind, data in events if kind == 'result' and data[1]]
            self.assertEqual(len(rows), 14)
            self.assertTrue(all(r['status'] == 'error' for r in rows[:7]))
            self.assertTrue(all(r['status'] == 'ok' for r in rows[7:]))
            self.assertTrue(all('true_stage' not in row for row in rows))
            self.assertEqual(comparison_sheet(str(self.sample), cards).size, (1840, 1850))
            self.assertEqual(before, self.sample.read_bytes())
            self.assertEqual(corrupt.read_bytes(), b'not an image')

    def test_test_content_blocked_before_image_decode_even_if_renamed(self):
        with (self.runs[0].path/'split_manifest.csv').open() as f:
            row = next(r for r in csv.DictReader(f) if r['split'] == 'test')
        with tempfile.TemporaryDirectory() as tmp:
            renamed = Path(tmp)/'holiday.png'
            renamed.write_bytes((self.data/row['path']).read_bytes())
            self.assertIn(row['sha256'], reserved_test_hashes(self.runs))
            events = []
            with patch('fruitripeness.ui_core.Image.open', side_effect=AssertionError('Test decoded')):
                run_batch([renamed], [self.runs[1]], trusted=True, cancel=threading.Event(),
                          emit=lambda kind, data: events.append((kind, data)))
            result = next(data[0] for kind, data in events if kind == 'result')
            self.assertEqual(result['status'], 'error')
            self.assertIn('Reserved original-Test', result['error'])

    def test_cancel_before_load_and_between_methods(self):
        cancel = threading.Event()
        cancel.set()
        with patch('fruitripeness.ui_core.load_model') as loader:
            run_batch([self.sample], self.runs[1:], trusted=True, cancel=cancel, emit=lambda *args: None)
            loader.assert_not_called()
        cancel.clear()
        rows = []
        def emit(kind, data):
            if kind == 'result':
                rows.append(data[0])
                cancel.set()
        run_batch([self.sample, self.sample], self.runs[1:], trusted=True, cancel=cancel, emit=emit)
        self.assertEqual(len(rows), 1)

    def test_safe_exports_refuse_source_folders_existing_files_and_escape_formulas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input'
            source.mkdir()
            with self.assertRaisesRegex(ValueError, 'outside'):
                export_csv(source/'report.csv', [], RESULT_FIELDS, [source])
            target = root/'report.csv'
            export_csv(target, [{'source': '=HYPERLINK("bad")', 'method': 'hsv', 'status': 'error'}], RESULT_FIELDS, [source])
            with target.open(encoding='utf-8-sig') as f:
                row = next(csv.DictReader(f))
            self.assertTrue(row['source'].startswith("'="))
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_csv(target, [], RESULT_FIELDS, [])
            self.assertEqual(target.read_bytes(), before)
            png = root/'evaluation.png'
            image = evaluation_sheet(self.runs)
            export_png(png, image, [source])
            with Image.open(png) as saved:
                self.assertEqual(saved.size, image.size)

    def test_per_fruit_metrics_and_missing_scope(self):
        scope = next(iter(self.runs[0].metrics['validation']['per_fruit']))
        rows = evaluation_rows(self.runs, scope)
        self.assertEqual(rows[0]['scope'], scope)
        self.assertEqual(rows[0]['n_images'], self.runs[0].metrics['validation']['per_fruit'][scope]['n_images'])
        with self.assertRaisesRegex(ValueError, 'No saved'):
            evaluation_rows(self.runs, 'unknown-fruit')


class DisplayNameTests(unittest.TestCase):
    def test_friendly_names_cover_exact_existing_ids(self):
        from fruitripeness.ui_core import CLASSIFIER_LABELS, choice_labels
        self.assertEqual(set(LABELS), {'baseline', *METHODS})
        self.assertEqual(set(CLASSIFIER_LABELS), {'random_forest', 'shared_cnn'})
        self.assertEqual(len(set(LABELS.values())), len(LABELS))
        self.assertEqual(choice_labels(LABELS, LABELS.__getitem__), LABELS)
        self.assertIn('Hybrid A', LABELS['hybrid'])
        self.assertIn('Hybrid B', LABELS['hybrid_refined'])

    def test_run_date_is_utc_and_invalid_names_are_unchanged(self):
        from fruitripeness.ui_core import run_label
        self.assertEqual(run_label('20260828T085810_496281Z'), '28 Aug 2026, 08:58:10 UTC')
        self.assertEqual(run_label('custom-run'), 'custom-run')
        self.assertEqual(run_label('20261328T085810_496281Z'), '20261328T085810_496281Z')

    def test_same_second_runs_are_distinct_and_exact_ids_are_retained(self):
        from fruitripeness.ui_core import run_label, choice_labels
        names = ['20260828T085810_496281Z', '20260828T085810_496282Z']
        labels = choice_labels(names, run_label)
        self.assertEqual(list(labels), names)
        self.assertEqual(len(set(labels.values())), 2)
        for key, value in labels.items():
            self.assertIn(key, value)
        self.assertEqual(choice_labels(names[:1], run_label)[names[0]], run_label(names[0]))


if __name__ == '__main__':
    unittest.main()