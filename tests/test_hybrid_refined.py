from contextlib import redirect_stdout
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw

from test_baseline import make_dataset
from fruitripeness.baseline import feature_vector, image_features, predict_image, train_baseline
from fruitripeness.experiment import compare_runs, export_previews, render_preview
from fruitripeness.processing import _refined_holes, process_image, processing_spec
from fruitripeness.ui_core import comparison_methods


def tray_image():
    image = Image.new('RGB', (200, 200), (170, 175, 178))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 194, 194), outline=(170, 120, 55), width=3)
    truth = Image.new('L', image.size)
    truth_draw = ImageDraw.Draw(truth)
    for box, colour in [((25, 35, 80, 115), (200, 45, 30)), ((110, 85, 168, 170), (50, 175, 30))]:
        draw.ellipse(box, fill=colour)
        truth_draw.ellipse(box, fill=255)
    return image, np.asarray(truth) > 0


class RefinedHybridTests(unittest.TestCase):
    def test_synthetic_tray_ring_is_not_flood_filled_and_fruits_retained(self):
        image, truth = tray_image()
        before = image.tobytes()
        old = process_image(image, 'hybrid')
        new = process_image(image, 'hybrid_refined')
        old_false = int((old.mask & ~truth).sum())
        new_false = int((new.mask & ~truth).sum())
        self.assertLess(new_false, old_false * 0.1)
        self.assertGreater((new.mask & truth).sum() / (new.mask | truth).sum(), 0.90)
        self.assertTrue(new.mask[70, 50])
        self.assertTrue(new.mask[120, 140])
        self.assertFalse(new.mask[170, 40])
        self.assertEqual(before, image.tobytes())
        np.testing.assert_array_equal(np.asarray(new.processed)[new.mask], np.asarray(image)[new.mask])
        self.assertFalse(np.asarray(new.processed)[~new.mask].any())

    def test_all_hues_are_allowed_and_small_pale_patch_is_preserved(self):
        for colour in [(200, 40, 30), (30, 190, 40), (230, 205, 30), (110, 65, 30)]:
            with self.subTest(colour=colour):
                image = Image.new('RGB', (100, 100), (180, 180, 180))
                draw = ImageDraw.Draw(image)
                draw.ellipse((20, 20, 80, 80), fill=colour)
                draw.ellipse((46, 46, 54, 54), fill=(235, 235, 235))
                result = process_image(image, 'hybrid_refined')
                self.assertTrue(result.mask[50, 50])
                self.assertFalse(result.mask[10, 10])
                self.assertEqual(result.processed.getpixel((50, 50)), (235, 235, 235))

    def test_small_holes_fill_but_large_tray_interior_does_not(self):
        mask = np.ones((100, 100), dtype=bool)
        mask[10:15, 10:15] = False
        mask[30:80, 30:80] = False
        result = _refined_holes(mask, processing_spec('hybrid_refined'))
        self.assertTrue(result[12, 12])
        self.assertFalse(result[50, 50])

    def test_tiny_constant_and_greyscale_cases_are_flagged_without_fallback(self):
        for image, status in [(Image.new('RGB', (1, 1), 'red'), 'too_small'),
                              (Image.new('RGB', (50, 50), 'red'), 'constant_working_image')]:
            with self.subTest(status=status):
                result = process_image(image, 'hybrid_refined')
                self.assertEqual(result.details['refined_status'], status)
                self.assertEqual(result.details['mask_status'], 'empty')
                self.assertFalse(np.asarray(result.processed).any())
        image = Image.new('RGB', (60, 60), 'white')
        ImageDraw.Draw(image).ellipse((10, 10, 50, 50), fill='grey')
        result = process_image(image, 'hybrid_refined')
        self.assertEqual(result.details['refined_status'], 'no_colour_seeds')
        self.assertEqual(result.details['refined_review_flag'], 'no_foreground')

    def test_working_size_cap_and_original_rgb_alignment(self):
        image, _ = tray_image()
        image = image.resize((600, 400))
        result = process_image(image, 'hybrid_refined')
        self.assertEqual(result.markers.shape, (133, 200))
        self.assertEqual(result.mask.shape, (400, 600))
        self.assertEqual(result.component_masks['colour_candidate'].shape, result.mask.shape)
        np.testing.assert_array_equal(np.asarray(result.processed)[result.mask], np.asarray(image)[result.mask])

    def test_deterministic_after_other_rng_calls_and_restores_threads(self):
        image, _ = tray_image()
        previous = cv2.getNumThreads()
        first = process_image(image, 'hybrid_refined')
        cv2.setRNGSeed(9876)
        repeat = process_image(image, 'hybrid_refined')
        np.testing.assert_array_equal(first.mask, repeat.mask)
        np.testing.assert_array_equal(first.markers, repeat.markers)
        self.assertEqual(first.details, repeat.details)
        self.assertEqual(cv2.getNumThreads(), previous)
        with patch('fruitripeness.processing.cv2.grabCut', side_effect=cv2.error('simulated')):
            with self.assertRaisesRegex(ValueError, 'Refined hybrid GrabCut failed'):
                process_image(image, 'hybrid_refined')
        self.assertEqual(cv2.getNumThreads(), previous)

    def test_initial_labels_survive_grabcut_mutation_and_preview_is_four_panels(self):
        image, _ = tray_image()
        def all_background(bgr, mask, rect, bg, fg, iterations, mode):
            self.assertEqual(mode, cv2.GC_INIT_WITH_MASK)
            mask[:] = cv2.GC_BGD
        with patch('fruitripeness.processing.cv2.grabCut', side_effect=all_background):
            result = process_image(image, 'hybrid_refined')
        self.assertTrue(np.any(result.markers == cv2.GC_FGD))
        self.assertFalse(result.mask.any())
        self.assertEqual(result.details['refined_status'], 'empty_result')
        preview = render_preview(image, 'Refined hybrid', 'synthetic.png', method='hybrid_refined')
        self.assertEqual(preview.size, (1330, 460))

    def test_spec_independence_old_model_mismatch_and_opencv_check(self):
        spec = processing_spec('hybrid_refined')
        spec['background_rgb'][0] = 123
        self.assertEqual(processing_spec('hybrid_refined')['background_rgb'], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            predict_image({'method': 'hybrid_refined', 'feature_id': 'rgb_pixels_32x32_v1',
                           'processing_spec': processing_spec('hybrid')}, Path('unused.png'))
        with patch('fruitripeness.processing.cv2.__version__', 'other'):
            with self.assertRaisesRegex(ValueError, 'OpenCV version differs'):
                process_image(tray_image()[0], 'hybrid_refined')

    def test_ui_comparison_always_five_individual_methods_and_one_selected_hybrid(self):
        old = comparison_methods('hybrid')
        new = comparison_methods('hybrid_refined')
        self.assertEqual(len(old), 6)
        self.assertEqual(old[:5], new[:5])
        self.assertEqual(new[-1], 'hybrid_refined')
        self.assertNotIn('hybrid', new)
        with self.assertRaises(ValueError):
            comparison_methods('baseline')

    def test_end_to_end_same_split_no_test_pixels_and_saved_inference_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            make_dataset(root)
            for path in root.rglob('*.png'):
                with Image.open(path) as source:
                    colour = source.getpixel((0, 0))
                image = Image.new('RGB', (50, 50), 'white')
                ImageDraw.Draw(image).ellipse((8, 8, 41, 41), fill=colour)
                image.save(path)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')}
            with redirect_stdout(io.StringIO()):
                baseline, _ = train_baseline(root, Path(tmp)/'baseline', trees=8, jobs=1)
                with patch('fruitripeness.baseline.process_image', wraps=process_image) as processor:
                    run, metrics = train_baseline(root, Path(tmp)/'refined', trees=8, jobs=1, method='hybrid_refined')
                self.assertEqual(processor.call_count, 36)
            self.assertEqual(set(metrics), {'validation'})
            self.assertFalse((run/'predictions_test.csv').exists())
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')})
            comparison = compare_runs(baseline, run)
            self.assertEqual(comparison['hybrid_refined']['accuracy'], metrics['validation']['accuracy'])
            self.assertGreaterEqual(export_previews(root, run), 6)
            with (run/'processing_diagnostics.csv').open() as f:
                diagnostics = list(csv.DictReader(f))
            self.assertEqual(len(diagnostics), 36)
            self.assertTrue(all(row['refined_status'] == 'completed' for row in diagnostics))
            bundle = joblib.load(run/'model.joblib')
            with (run/'predictions_validation.csv').open() as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                path = root/row['path']
                with Image.open(path) as source:
                    manual = feature_vector(process_image(source, 'hybrid_refined').processed)
                np.testing.assert_array_equal(manual, image_features(path, method='hybrid_refined'))
                prediction = predict_image(bundle, path)
                self.assertEqual(prediction['predicted_stage'], row['predicted_stage'])
                for name, score in prediction['scores'].items():
                    self.assertAlmostEqual(score, float(row['score_'+name]))


if __name__ == '__main__':
    unittest.main()
