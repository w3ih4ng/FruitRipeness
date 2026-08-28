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
        self.app.trust.set(True)
        self.app.set_inputs([self.sample])
        self.app.start(['hsv'])
        self.wait_worker()
        self.assertEqual(len(self.app.rows), 1)
        self.assertEqual(self.app.rows[0]['status'], 'ok')

    def test_widgets_discovery_and_missing_model_state(self):
        self.assertEqual(len(self.app.tabs.tabs()), 3)
        self.assertEqual(len(self.app.run_boxes), 7)
        self.assertEqual(len(self.app.selected_runs(['hsv'])), 1)
        self.assertIn('Not trained', self.app.model_notes['hybrid'].cget('text'))

    def test_trust_gate_and_compare_all_missing_runs(self):
        self.app.set_inputs([self.sample])
        self.app.start(['hsv'])
        self.assertFalse(self.app.busy)
        self.assertTrue(self.errors)
        self.app.trust.set(True)
        self.app.start(['hsv', 'hybrid'])
        self.assertFalse(self.app.busy)
        self.assertIn('not trained', str(self.errors[-1]))

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


if __name__ == '__main__':
    unittest.main()
