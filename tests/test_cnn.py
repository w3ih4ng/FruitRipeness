"""Optional CNN integration tests. No network/downloads; real randomly initialised
MobileNetV2 is used ONLY in tests, never as a production training fallback.
"""
from contextlib import redirect_stdout
import copy
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
from fruitripeness import cnn
from fruitripeness.baseline import assign_splits, collect_records
from fruitripeness.ui_core import (Run, discover_runs, ensure_compatible, evaluation_rows,
    evaluation_sheet, infer, load_model, reserved_test_hashes, run_batch)


def fixture(root, reference):
    make_dataset(root)
    for i, path in enumerate(sorted(root.rglob('*.png'))):
        with Image.open(path) as image:
            colour = image.getpixel((0, 0))
        image = Image.new('RGB', (40, 40), 'white')
        ImageDraw.Draw(image).ellipse((7, 6, 32, 33), fill=colour)
        image.putpixel((0, 0), (i, 0, 0))
        image.save(path)
    records = assign_splits(collect_records(root))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    reference.mkdir()
    (reference/'metadata.json').write_text(json.dumps({'status':'completed','method':'baseline','split_id':digest}))
    return records


class CNNPreparationTests(unittest.TestCase):
    def test_letterbox_preserves_full_image_and_normalises_channels(self):
        image = Image.new('RGB', (100, 50), 'red')
        result = cnn.input_array(image)
        self.assertEqual(result.shape, (3,224,224))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        expected = (np.array([1,0,0])-cnn.INPUT_SPEC['mean'])/cnn.INPUT_SPEC['std']
        np.testing.assert_allclose(result[:,112,112], expected, rtol=1e-6)
        black = -np.array(cnn.INPUT_SPEC['mean'])/cnn.INPUT_SPEC['std']
        np.testing.assert_allclose(result[:,0,112], black, rtol=1e-6)
        self.assertEqual(image.size, (100,50))

    def test_original_test_and_changed_sources_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            rows = fixture(root, Path(tmp)/'reference')
            test = next(r for r in rows if r['split']=='test')
            with patch('fruitripeness.cnn.Image.open', side_effect=AssertionError('decoded')):
                with self.assertRaisesRegex(ValueError, 'original Train'):
                    cnn.image_view(root, test, 'baseline')
                fitting = next(r for r in rows if r['split']=='train')
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    cnn.image_view(root, {**fitting,'sha256':'bad'}, 'hsv')

    def test_baseline_split_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, reference = Path(tmp)/'data', Path(tmp)/'reference'
            fixture(root, reference)
            cnn.prepare_records(root, reference)
            (reference/'metadata.json').write_text(json.dumps({'status':'completed','method':'baseline','split_id':'bad'}))
            with self.assertRaisesRegex(ValueError, 'split differs'):
                cnn.prepare_records(root, reference)

    def test_invalid_training_options_fail_without_torch_or_data(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            cnn.train(Path('missing'), Path('out'), Path('reference'), epochs=0)


class CNNIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch, _ = cnn.runtime()
        except ValueError as exc:
            raise unittest.SkipTest(str(exc))
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.data, cls.reference = cls.base/'data', cls.base/'reference'
        cls.records = fixture(cls.data, cls.reference)
        cls.before = {r['path']: hashlib.sha256((cls.data/r['path']).read_bytes()).hexdigest() for r in cls.records}
        cls.calls = []
        real_view, real_model = cnn.image_view, cnn.make_model
        def tracked(root, row, method):
            cls.calls.append((row['path'], row['split'], method))
            return real_view(root, row, method)
        # Offline fixture: real convolution and training, no pretrained download.
        with patch('fruitripeness.cnn.make_model', side_effect=lambda **kw: real_model(pretrained=False)), \
             patch('fruitripeness.cnn.image_view', side_effect=tracked), redirect_stdout(io.StringIO()):
            cls.run_path, cls.metrics = cnn.train(cls.data, cls.base/'outputs'/'shared_cnn', cls.reference,
                                           epochs=2, batch_size=8, threads=2, patience=2)
        cls.runs = cnn.read_shared(cls.run_path)
        cls.sample = cls.base/'sample.png'
        image = Image.new('RGB',(60,60),'white')
        ImageDraw.Draw(image).ellipse((8,7,52,53), fill=(190,43,20))
        image.save(cls.sample)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_training_views_balanced_source_split_unchanged_and_test_not_decoded(self):
        self.assertEqual(len(self.calls), 36*len(cnn.METHODS))
        self.assertFalse(any(s=='test' for _,s,_ in self.calls))
        for row in self.records:
            self.assertEqual(hashlib.sha256((self.data/row['path']).read_bytes()).hexdigest(), self.before[row['path']])
            methods = [m for p,_,m in self.calls if p==row['path']]
            self.assertEqual(set(methods), set() if row['split']=='test' else set(cnn.METHODS))
        self.assertEqual(len(list(self.run_path.glob('model.*'))), 1)
        self.assertFalse(list(self.run_path.glob('*test*.csv')))

    def test_metadata_and_metrics_for_all_methods_use_one_checkpoint(self):
        self.assertEqual(set(self.metrics), set(cnn.METHODS))
        ensure_compatible(self.runs)
        self.assertEqual(len({r.metadata['model_sha256'] for r in self.runs}), 1)
        self.assertEqual(len({r.model_path for r in self.runs}), 1)
        rows = evaluation_rows(self.runs)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['backend']=='shared_cnn' for r in rows))
        self.assertEqual(evaluation_sheet(self.runs).size, (1280,2140))
        with (self.run_path/'history.csv').open(newline='') as stream:
            history = list(csv.DictReader(stream))
        best = max(history, key=lambda row: float(row['mean_validation_macro_f1']))
        self.assertEqual(self.runs[0].metadata['best_epoch'], int(best['epoch']))
        mean = np.mean([r.metrics['validation']['macro_f1'] for r in self.runs])
        self.assertAlmostEqual(mean, float(best['mean_validation_macro_f1']))

    def test_discovery_does_not_load_torch_model_or_download(self):
        with patch('fruitripeness.cnn.runtime', side_effect=AssertionError('Torch called')):
            found, errors = discover_runs(self.base/'outputs', 'shared_cnn')
            self.assertFalse(errors)
            self.assertTrue(all(len(v)==1 for v in found.values()))

    def test_loading_requires_trust_and_does_not_download(self):
        with patch('fruitripeness.cnn.load_shared') as loader:
            with self.assertRaisesRegex(ValueError,'trusted'):
                load_model(self.runs[0])
            loader.assert_not_called()
        with patch('torch.hub.download_url_to_file', side_effect=AssertionError('download')):
            bundle, digest = load_model(self.runs[0], trusted=True)
        self.assertEqual(digest, self.runs[0].metadata['model_sha256'])
        self.assertFalse(bundle['model'].training)

    def test_backbone_is_frozen_and_head_was_trained(self):
        self.torch.manual_seed(42)
        initial = cnn.make_model(pretrained=False)
        bundle, _ = load_model(self.runs[0], trusted=True)
        model = bundle['model']
        for key, value in initial.features.state_dict().items():
            self.assertTrue(self.torch.equal(value, model.features.state_dict()[key]), key)
        self.assertFalse(self.torch.equal(initial.classifier.weight, model.classifier.weight))

    def test_cached_feature_and_full_cnn_inference_match(self):
        old_threads = self.torch.get_num_threads()
        self.torch.set_num_threads(2)
        try:
            bundle, _ = load_model(self.runs[0], trusted=True)
            row = next(r for r in self.records if r['split']=='validation')
            for run in self.runs:
                features = np.load(self.run_path/'features'/f'validation_{run.method}.npy', allow_pickle=False)
                expected = cnn.scores_for_features(bundle['model'], features[:1])[0]
                with Image.open(self.data/row['path']) as image:
                    actual, _ = infer(image, {**bundle, 'method':run.method})
                np.testing.assert_allclose([actual[f'score_{c}'] for c in cnn.CLASSES], expected, atol=1e-6)
        finally:
            self.torch.set_num_threads(old_threads)

    def test_batch_loads_shared_weights_once_and_handles_corrupt_inputs(self):
        corrupt = self.base/'broken.png'
        corrupt.write_bytes(b'broken')
        events=[]
        with patch('fruitripeness.ui_core.load_model', wraps=load_model) as loader:
            run_batch([corrupt,self.sample], self.runs, trusted=True, cancel=threading.Event(),
                      emit=lambda k,v: events.append((k,v)))
        self.assertEqual(loader.call_count, 1)
        rows = [v[0] for k,v in events if k=='result']
        self.assertEqual(len(rows),16)
        self.assertTrue(all(r['status']=='error' for r in rows[:8]))
        self.assertTrue(all(r['status']=='ok' for r in rows[8:]))
        self.assertEqual(len({r['model_sha256'] for r in rows}),1)

    def test_test_guard_prevents_decode_in_shared_cnn_batch(self):
        row = next(r for r in self.records if r['split']=='test')
        self.assertIn(row['sha256'], reserved_test_hashes(self.runs))
        events=[]
        with patch('fruitripeness.ui_core.Image.open', side_effect=AssertionError('Test decoded')):
            run_batch([self.data/row['path']], self.runs[:1], trusted=True, cancel=threading.Event(),
                      emit=lambda k,v: events.append((k,v)))
        result=next(v[0] for k,v in events if k=='result')
        self.assertIn('Reserved original-Test', result['error'])

    def test_mixed_backends_and_different_shared_checkpoints_are_rejected(self):
        other = copy.deepcopy(self.runs[1])
        other.path = self.base/'different'
        with self.assertRaisesRegex(ValueError, 'SAME'):
            ensure_compatible([self.runs[0],other])
        other = copy.deepcopy(self.runs[1])
        other.metadata['backend']='random_forest'
        with self.assertRaisesRegex(ValueError, 'mix'):
            ensure_compatible([self.runs[0],other])

    def test_hash_version_and_embedded_metadata_mismatches_are_rejected(self):
        altered = copy.deepcopy(self.runs[0])
        altered.metadata['model_sha256']='0'*64
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            load_model(altered, trusted=True)
        altered = copy.deepcopy(self.runs[0])
        altered.metadata['versions']['torch']='wrong'
        with self.assertRaisesRegex(ValueError, 'version differs'):
            load_model(altered, trusted=True)
        altered = copy.deepcopy(self.runs[0])
        altered.metadata['best_epoch']=99
        with self.assertRaisesRegex(ValueError, 'embedded metadata'):
            load_model(altered, trusted=True)

    def test_cancel_before_model_load(self):
        cancel=threading.Event();cancel.set()
        with patch('fruitripeness.ui_core.load_model') as loader:
            run_batch([self.sample],self.runs,trusted=True,cancel=cancel,emit=lambda *args:None)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
