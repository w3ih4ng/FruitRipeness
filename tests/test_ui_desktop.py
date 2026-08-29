"""Desktop integration checks. Skip the class when no graphical display exists."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from test_baseline import make_dataset
from fruitripeness.baseline import train_baseline


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
        except ImportError as exc:
            raise unittest.SkipTest(f'Tkinter unavailable: {exc}')
        try:
            root = tk.Tk()
            root.withdraw()
            root.destroy()
        except tk.TclError as exc:
            raise unittest.SkipTest(f'Graphical Tk display unavailable: {exc}')
        cls.tk = tk
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        data = cls.base/'data'
        make_dataset(data)
        with redirect_stdout(io.StringIO()):
            train_baseline(data, cls.base/'outputs'/'hsv', jobs=1, trees=8, method='hsv')
        cls.sample = cls.base/'inputs'/'sample.png'
        cls.sample.parent.mkdir()
        cls.original = Image.new('RGB', (50, 50), 'white')
        ImageDraw.Draw(cls.original).ellipse((8, 8, 42, 42), fill=(203, 30, 25))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        from fruitripeness.ui import App
        self.original.save(self.sample)
        self.root = self.tk.Tk()
        self.root.withdraw()
        self.app = App(self.root, self.base/'outputs')
        self.errors = []
        self.dialog = patch('fruitripeness.ui.messagebox.showerror', side_effect=lambda *a, **kw: self.errors.append(a))
        self.dialog.start()

    def tearDown(self):
        self.dialog.stop()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.root.destroy()

    def wait_worker(self):
        deadline = time.monotonic()+15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy, 'Background worker did not finish')

    def run_one(self):
        self.app.set_inputs([self.sample])
        self.app.start(['hsv'])
        self.wait_worker()
        self.assertEqual(len(self.app.rows), 1)
        self.assertEqual(self.app.rows[0]['status'], 'ok')

    def test_widgets_discovery_and_missing_model_state(self):
        self.assertEqual(len(self.app.tabs.tabs()), 5)
        self.assertFalse(hasattr(self.app, 'models_tab'))
        self.assertFalse(hasattr(self.app, 'trust'))
        self.assertEqual(self.app.hybrid_variant.get(), 'hybrid')
        with patch.object(self.app, 'start') as start:
            self.app.hybrid_variant.set('hybrid_refined')
            next(b for b in self.app.buttons if b.cget('text') == 'Compare all six').invoke()
            start.assert_called_with(['hsv', 'otsu', 'kmeans', 'grabcut', 'watershed', 'hybrid_refined'])
            next(b for b in self.app.buttons if b.cget('text') == 'Compare hybrid versions').invoke()
            start.assert_called_with(['hybrid', 'hybrid_refined'])
        self.assertEqual(len(self.app.selected_runs(['hsv'])), 1)
        self.assertEqual(self.app.selected_runs(['hybrid'], required=False), [])
        for label in [
            'Export report PDF',
            'Export validation PDF',
            'Export Test PDF',
            'Export surface PDF',
            'Start live camera',
            'Choose video',
            'Process and export annotated video',
            'Export frame results CSV',
        ]:
            self.assertTrue(any(b.cget('text') == label for b in self.app.buttons))
        self.assertEqual(str(self.app.camera_stop_button.cget('state')), 'disabled')
        self.assertEqual(str(self.app.video_process_button.cget('state')), 'disabled')

    def test_classifier_switch_updates_automatic_model_selection(self):
        self.app.backend.set('shared_cnn')
        self.app.change_backend()
        self.assertEqual(str(self.app.cnn_box.cget('state')), 'readonly')
        self.assertEqual(self.app.selected_runs(['hsv'], required=False), [])
        self.app.set_busy(True)
        self.assertEqual(str(self.app.backend_box.cget('state')), 'disabled')
        self.app.set_busy(False)
        self.app.backend.set('random_forest')
        self.app.change_backend()
        self.assertEqual(len(self.app.selected_runs(['hsv'])), 1)
        self.assertEqual(str(self.app.cnn_box.cget('state')), 'disabled')

    def test_prediction_starts_without_confirmation_and_missing_run_is_clear(self):
        self.app.set_inputs([self.sample])
        self.app.start(['hsv'])
        self.wait_worker()
        self.assertFalse(self.errors)
        self.assertEqual(self.app.rows[0]['status'], 'ok')
        self.app.start(['hsv', 'hybrid'])
        self.assertFalse(self.app.busy)
        self.assertIn('no compatible saved model', str(self.errors[-1]))

    def test_worker_populates_table_and_preview(self):
        self.run_one()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.app.table.get_children()), 1)
        self.assertEqual(self.app.card_source, str(self.sample.resolve()))
        self.assertEqual(set(self.app.cards), {'hsv'})
        self.assertTrue(self.app.preview_canvas.find_all())

    def test_preview_rejects_changed_source(self):
        self.run_one()
        self.app.table.selection_set('0')
        Image.new('RGB', (50, 50), 'blue').save(self.sample)
        self.app.preview_selected()
        self.wait_worker()
        self.assertTrue(self.errors)
        self.assertIn('changed', str(self.errors[-1]))

    def test_validation_view_and_prediction_exports(self):
        self.app.load_evaluation()
        self.assertFalse(self.errors)
        self.assertEqual(self.app.eval_rows[0]['split'], 'validation')
        self.assertTrue(self.app.eval_canvas.find_all())
        self.run_one()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.app, 'destination', return_value=str(Path(tmp)/'results.csv')):
                self.app.save_results()
            with patch.object(self.app, 'destination', return_value=str(Path(tmp)/'preview.png')):
                self.app.save_preview()
            self.assertTrue((Path(tmp)/'results.csv').is_file())
            with Image.open(Path(tmp)/'preview.png') as image:
                self.assertEqual(image.size, (920, 530))
        self.assertFalse(self.errors)

    def test_friendly_selectors_preserve_internal_method_and_model_ids(self):
        from fruitripeness.ui_core import LABELS, CLASSIFIER_LABELS, run_label
        self.assertEqual(self.app.backend_box.get(), CLASSIFIER_LABELS['random_forest'])
        self.app.method_box.set(LABELS['hsv'])
        self.assertEqual(self.app.method.get(), 'hsv')
        with patch.object(self.app, 'start') as start:
            next(b for b in self.app.buttons if b.cget('text') == 'Run selected method').invoke()
            start.assert_called_with(['hsv'])
        self.app.hybrid_variant.set('hybrid_refined')
        self.assertEqual(self.app.hybrid_box.get(), LABELS['hybrid_refined'])
        selected = self.app.selected_runs(['hsv'])[0]
        self.assertEqual(self.app.run_vars['hsv'].get(), selected.path.name)
        self.run_one()
        self.assertEqual(self.app.table.item('0', 'values')[1], LABELS['hsv'])
        self.assertEqual(self.app.rows[0]['method'], 'hsv')

    def test_same_second_run_selection_and_widget_cleanup(self):
        from fruitripeness.ui import NamedCombobox
        from fruitripeness.ui_core import run_label
        keys = ['20260828T085810_496281Z', '20260828T085810_496282Z']
        var = self.tk.StringVar(master=self.root, value=keys[0])
        box = NamedCombobox(self.root, textvariable=var, values=keys, label=run_label)
        box.current(1)
        self.assertEqual(var.get(), keys[1])
        box.set_choices(keys[:1])
        self.assertEqual(box.get(), '')  # No silent fallback to a different model.
        var.set(keys[0])
        self.assertEqual(box.get(), run_label(keys[0]))
        box.destroy()
        self.assertEqual(var.trace_info(), [])

    def test_saved_test_tab_is_separate_and_exports_test_labels(self):
        from fruitripeness.ui_core import Run
        original=self.app.selected_runs(['hsv'])[0]
        report_run=Run(self.base/'final_report',
            {**original.metadata,'backend':'shared_cnn','source_run':str(original.path),
             'evaluation_only':True,'evaluation_split':'test'},
            {'test':original.metrics['validation']})
        with patch('fruitripeness.final_test.read_report',return_value=[report_run]):
            self.app.load_test_report(self.base/'final_report')
        self.assertFalse(self.errors)
        self.assertEqual(self.app.test_rows[0]['split'],'test')
        self.assertEqual(self.app.eval_rows,[])
        self.assertEqual(self.app.backend.get(),'random_forest')
        self.assertTrue(self.app.test_canvas.find_all())
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'test.csv'
            with patch.object(self.app,'destination',return_value=str(target)):
                self.app.save_test_metrics()
            self.assertIn('test',target.read_text(encoding='utf-8-sig'))
        self.assertFalse(self.errors)

    def test_surface_tab_analyzes_without_needing_a_saved_model(self):
        self.app.set_inputs([self.sample])
        self.app.start_surface()
        self.wait_worker()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.app.surface_rows), 1)
        row = self.app.surface_rows[0]
        self.assertEqual(row['status'], 'ok')
        self.assertIn(row['blemish_status'], {'graded', 'too_small_foreground'})
        self.assertIn(row['quality_grade'], {'Good', 'Acceptable', 'Poor', 'Not graded'})
        self.assertGreaterEqual(row['objects_detected'], 1)
        self.assertTrue(self.app.surface_canvas.find_all())
        self.assertEqual(len(self.app.surface_table.get_children()), 1)

    def test_surface_report_exports_and_source_unchanged(self):
        before = self.sample.read_bytes()
        self.app.set_inputs([self.sample])
        self.app.start_surface()
        self.wait_worker()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.app, 'destination', return_value=str(Path(tmp)/'surface.csv')):
                self.app.save_surface_results()
            with patch.object(self.app, 'destination', return_value=str(Path(tmp)/'surface.png')):
                self.app.save_surface_preview()
            self.assertIn('blemish_fraction', (Path(tmp)/'surface.csv').read_text(encoding='utf-8-sig'))
            self.assertTrue((Path(tmp)/'surface.png').is_file())
        self.assertEqual(self.sample.read_bytes(), before)
        self.assertFalse(self.errors)


if __name__ == '__main__':
    unittest.main()
