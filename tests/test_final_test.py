from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from test_cnn import fixture
from fruitripeness import cnn, final_test
from fruitripeness.ui_core import ensure_compatible, evaluation_rows, evaluation_sheet, load_model, run_batch


class FinalTestGuardTests(unittest.TestCase):
    def test_confirmations_required_before_reading_or_loading(self):
        with patch('fruitripeness.cnn.read_shared') as loader:
            with self.assertRaisesRegex(ValueError, 'confirm-frozen'):
                final_test.evaluate_final(Path('missing'),Path('model'),Path('out'))
            with self.assertRaisesRegex(ValueError, 'trusted-model'):
                final_test.evaluate_final(Path('missing'),Path('model'),Path('out'),confirm_frozen=True)
            loader.assert_not_called()

    def test_preview_selection_is_fixed_not_based_on_predictions(self):
        rows=[{'path':name,'fruit':fruit,'stage':stage} for name,fruit,stage in
              [('z','apple','ripe'),('a','apple','ripe'),('b','banana','ripe'),('c','apple','unripe')]]
        self.assertEqual(final_test.preview_paths(rows),{'a','b','c'})

    def test_reader_rejects_non_test_rows_without_decoding(self):
        with patch('fruitripeness.final_test.Image.open',side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError,'original-Test'):
                final_test.read_test_image(Path('missing'),{'split':'validation','source_split':'train'})


class FinalTestIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cnn.runtime()
        except ValueError as exc:
            raise unittest.SkipTest(str(exc))
        cls.temp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.temp.name)
        cls.data=cls.base/'data'
        cls.reference=cls.base/'reference'
        cls.records=fixture(cls.data,cls.reference)
        real_model=cnn.make_model
        with patch('fruitripeness.cnn.make_model',side_effect=lambda **kw:real_model(pretrained=False)),redirect_stdout(io.StringIO()):
            cls.model_run,_=cnn.train(cls.data,cls.base/'outputs'/'shared_cnn',cls.reference,epochs=1,batch_size=8)
        cls.before={str(p):final_test.sha(p) for p in [*cls.data.rglob('*.png'),*cls.model_run.glob('*.*')]}
        cls.decode_paths=[]
        original=final_test.read_test_image
        def tracked(root,row):
            cls.decode_paths.append(row['path'])
            return original(root,row)
        with patch('fruitripeness.cnn.train',side_effect=AssertionError('retrained')), \
             patch('torch.optim.AdamW',side_effect=AssertionError('optimiser created')), \
             patch('fruitripeness.final_test.read_test_image',side_effect=tracked), \
             patch('fruitripeness.final_test.load_model',wraps=load_model) as loader,redirect_stdout(io.StringIO()):
            cls.report,cls.metrics=final_test.evaluate_final(cls.data,cls.model_run,cls.base/'outputs'/'final_test',confirm_frozen=True,trusted_model=True)
        cls.load_count=loader.call_count
        cls.runs=final_test.read_report(cls.report)
        cls.training_runs=cnn.read_shared(cls.model_run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_test_image_all_eight_methods_one_checkpoint_no_source_changes(self):
        expected={r['path'] for r in self.records if r['split']=='test'}
        self.assertEqual(set(self.decode_paths),expected)
        self.assertEqual(len(self.decode_paths),len(expected))
        self.assertEqual(self.load_count,1)
        self.assertEqual(set(self.metrics),set(cnn.METHODS))
        with (self.report/'predictions_test_all.csv').open(newline='') as stream:
            reader=csv.DictReader(stream)
            self.assertEqual(len(reader.fieldnames),len(set(reader.fieldnames)))
            rows=list(reader)
        self.assertEqual(len(rows),6*8)
        for method in cnn.METHODS:
            self.assertEqual({r['path'] for r in rows if r['method']==method},expected)
            self.assertEqual(self.metrics[method]['test']['n_images'],6)
        self.assertTrue(all(r['split']=='test' and r['status']=='ok' for r in rows))
        self.assertEqual(len({r['model_sha256'] for r in rows}),1)
        for path,digest in self.before.items():
            self.assertEqual(final_test.sha(path),digest,path)

    def test_all_report_exports_exist_and_split_labels_are_test(self):
        with (self.report/'comparison_test.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),24)  # Overall and two fruit scopes x eight variants.
        self.assertEqual({r['split'] for r in rows},{'test'})
        self.assertEqual({r['backend'] for r in rows},{'shared_cnn'})
        self.assertEqual(len(list((self.report/'previews').glob('*.png'))),6)
        with Image.open(self.report/'comparison_test_overall.png') as image:
            self.assertEqual(image.size,(1280,2140))
        for method in cnn.METHODS:
            self.assertTrue((self.report/f'predictions_test_{method}.csv').is_file())
            self.assertTrue((self.report/f'confusion_matrix_test_{method}.csv').is_file())

    def test_readonly_report_view_needs_no_torch_and_cannot_be_loaded_as_model(self):
        with patch('fruitripeness.cnn.runtime',side_effect=AssertionError('torch loaded')):
            runs=final_test.read_report(self.report)
            self.assertEqual(len(evaluation_rows(runs,split='test')),8)
            self.assertEqual(evaluation_sheet(runs,'apple',split='test').size,(1280,2140))
        with self.assertRaisesRegex(ValueError,'evaluation-only'):
            load_model(runs[0],trusted=True)
        with self.assertRaisesRegex(ValueError,'No saved validation'):
            evaluation_rows(runs)
        with self.assertRaisesRegex(ValueError,'mix validation'):
            ensure_compatible([self.training_runs[0],runs[1]])

    def test_development_guard_still_blocks_test_after_final_evaluation(self):
        source=next(r for r in self.records if r['split']=='test')
        events=[]
        with patch('fruitripeness.ui_core.Image.open',side_effect=AssertionError('decoded')):
            run_batch([self.data/source['path']],self.training_runs[:1],trusted=True,
                      cancel=threading.Event(),emit=lambda k,v:events.append((k,v)))
        row=next(v[0] for k,v in events if k=='result')
        self.assertIn('Reserved original-Test',row['error'])

    def test_changed_dataset_rejected_before_model_load_or_test_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data'
            shutil.copytree(self.data,data)
            source=next(r for r in self.records if r['split']=='train')
            (data/source['path']).write_bytes(b'changed')
            with patch('fruitripeness.final_test.load_model',side_effect=AssertionError('loaded')), \
                 patch('fruitripeness.final_test.read_test_image',side_effect=AssertionError('decoded')):
                with self.assertRaisesRegex(ValueError,'Dataset or saved split manifest changed'):
                    final_test.evaluate_final(data,self.model_run,Path(tmp)/'out',confirm_frozen=True,trusted_model=True)

    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied=Path(tmp)/'run'
            shutil.copytree(self.model_run,copied)
            saved=(copied/'split_manifest.csv').read_text()
            (copied/'split_manifest.csv').write_text(saved.replace(',test,',',train,',1))
            altered=copy.deepcopy(self.training_runs[0]);altered.path=copied
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                final_test.verify_dataset(self.data,altered)

    def test_unsafe_output_paths_rejected(self):
        for out in [self.data/'report',self.model_run/'report']:
            with self.assertRaisesRegex(ValueError,'outside'):
                final_test.evaluate_final(self.data,self.model_run,out,confirm_frozen=True,trusted_model=True)

    def test_tampered_metrics_and_incomplete_reports_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied=Path(tmp)/'report';shutil.copytree(self.report,copied)
            metrics=json.loads((copied/'metrics.json').read_text())
            metrics['baseline']['test']['accuracy']=0.123
            (copied/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'metrics hash'):
                final_test.read_report(copied)
            metadata=json.loads((self.report/'metadata.json').read_text())
            metadata['status']='building'
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                final_test.report_runs(copied,metadata,self.metrics)

    def test_processing_failure_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            with patch('fruitripeness.final_test.infer',side_effect=ValueError('simulated processing failure')):
                with self.assertRaisesRegex(ValueError,'simulated'):
                    final_test.evaluate_final(self.data,self.model_run,Path(tmp),confirm_frozen=True,trusted_model=True)
            paths=list(Path(tmp).glob('*/metadata.json'))
            self.assertEqual(len(paths),1)
            self.assertEqual(json.loads(paths[0].read_text())['status'],'failed')
            self.assertFalse((paths[0].parent/'comparison_test.csv').exists())


if __name__ == '__main__':
    unittest.main()
