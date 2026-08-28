from contextlib import redirect_stdout
import csv
import io
import json
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
from fruitripeness.processing import ProcessingResult, process_image, processing_spec


class HybridTests(unittest.TestCase):
    def test_union_matches_component_methods_and_preserves_source_rgb(self):
        image = Image.new('RGB', (100, 80), 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 15, 45, 65), fill=(205, 30, 20))
        draw.rectangle((55, 20, 80, 60), fill=(90, 90, 90))
        before = image.tobytes()
        hsv = process_image(image, 'hsv')
        grabcut = process_image(image, 'grabcut')
        result = process_image(image, 'hybrid')
        repeat = process_image(image, 'hybrid')
        np.testing.assert_array_equal(result.mask, hsv.mask | grabcut.mask)
        np.testing.assert_array_equal(result.mask, repeat.mask)
        self.assertEqual(result.details, repeat.details)
        self.assertEqual(image.tobytes(), before)
        np.testing.assert_array_equal(np.asarray(result.processed)[result.mask], np.asarray(image)[result.mask])
        self.assertFalse(np.asarray(result.processed)[~result.mask].any())
        for name, component in [('hsv', hsv), ('grabcut', grabcut)]:
            np.testing.assert_array_equal(result.component_masks[name], component.mask)
            self.assertGreaterEqual(result.details['foreground_fraction'], component.details['foreground_fraction'])
        self.assertAlmostEqual(result.details['hybrid_disagreement_fraction'], (hsv.mask ^ grabcut.mask).mean())
        self.assertAlmostEqual(result.details['hybrid_mask_jaccard'],
                               (hsv.mask & grabcut.mask).sum() / result.mask.sum())

    def test_disjoint_masks_are_unioned_without_extra_cleanup(self):
        image = Image.new('RGB', (10, 10), 'red')
        masks = [np.zeros((10, 10), dtype=bool) for _ in range(2)]
        masks[0][2, 2] = True
        masks[1][7, 7] = True
        components = [ProcessingResult(image, image, mask,
                      {'foreground_fraction': float(mask.mean()), 'mask_status': 'nonempty'}) for mask in masks]
        with patch('fruitripeness.processing.process_image', side_effect=components) as component_call:
            result = process_image(image, 'hybrid')
        self.assertEqual([call.args[1] for call in component_call.call_args_list], ['hsv', 'grabcut'])
        self.assertEqual(result.mask.sum(), 2)  # A second morphology pass would remove these.
        self.assertEqual(result.details['hybrid_mask_jaccard'], 0)
        self.assertAlmostEqual(result.details['hybrid_disagreement_fraction'], 0.02)

    def test_both_empty_is_black_not_original_and_no_image_is_dropped(self):
        for size in [(1, 1), (50, 50)]:
            with self.subTest(size=size):
                result = process_image(Image.new('RGB', size, 'white'), 'hybrid')
                self.assertEqual(result.details['mask_status'], 'empty')
                self.assertEqual(result.details['hybrid_mask_jaccard'], 1.0)
                self.assertEqual(result.details['hybrid_disagreement_fraction'], 0.0)
                self.assertEqual(result.processed.size, size)
                self.assertFalse(np.asarray(result.processed).any())

    def test_one_empty_keeps_other_mask_not_a_hidden_baseline_fallback(self):
        image = Image.new('RGB', (50, 50), 'red')
        result = process_image(image, 'hybrid')
        self.assertEqual(result.details['hybrid_grabcut_fraction'], 0)
        self.assertEqual(result.details['hybrid_hsv_fraction'], 1)
        self.assertEqual(result.details['mask_status'], 'near_full')
        self.assertEqual(result.details['hybrid_mask_jaccard'], 0)
        self.assertEqual(result.details['grabcut_status'], 'constant_working_image')
        np.testing.assert_array_equal(result.mask, result.component_masks['hsv'])

    def test_nested_settings_are_independent_and_mismatches_rejected(self):
        original = processing_spec('hybrid')
        spec = processing_spec('hybrid')
        spec['components']['hsv']['background_rgb'][0] = 255
        spec['components']['grabcut']['iterations'] = 99
        self.assertEqual(processing_spec('hybrid'), original)
        self.assertEqual(processing_spec('hsv')['background_rgb'], [0, 0, 0])
        self.assertEqual(processing_spec('grabcut')['iterations'], 5)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            predict_image({'method': 'hybrid', 'feature_id': 'rgb_pixels_32x32_v1',
                           'processing_spec': spec}, Path('unused.png'))
        with patch('fruitripeness.processing.cv2.__version__', 'other'):
            with self.assertRaisesRegex(ValueError, 'OpenCV version differs'):
                process_image(Image.new('RGB', (50, 50)), 'hybrid')

    def test_component_failure_is_reported_not_silently_ignored(self):
        image = Image.new('RGB', (50, 50), 'white')
        ImageDraw.Draw(image).rectangle((10, 10, 39, 39), fill='red')
        with patch('fruitripeness.processing.cv2.grabCut', side_effect=cv2.error('simulated failure')):
            with self.assertRaisesRegex(ValueError, 'GrabCut failed'):
                process_image(image, 'hybrid')

    def test_preview_contains_actual_component_and_union_masks(self):
        image = Image.new('RGB', (100, 100), 'white')
        ImageDraw.Draw(image).ellipse((20, 20, 80, 80), fill='red')
        result = process_image(image, 'hybrid')
        preview = render_preview(image, 'Hybrid - HSV + GrabCut Union', 'sample.png', method='hybrid')
        self.assertEqual(preview.size, (1660, 460))
        for panel, mask in [(1, result.component_masks['hsv']), (2, result.component_masks['grabcut']),
                            (3, result.mask)]:
            expected = Image.fromarray(mask.astype('uint8') * 255).convert('RGB').resize(
                (300, 300), Image.Resampling.BICUBIC)
            x = 20 + panel * 330
            self.assertEqual(preview.crop((x, 80, x + 300, 380)).tobytes(), expected.tobytes())

    def test_training_comparison_exports_and_saved_inference_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            make_dataset(root)
            for path in root.rglob('*.png'):
                with Image.open(path) as source:
                    colour = source.getpixel((0, 0))
                image = Image.new('RGB', (40, 40), 'white')
                ImageDraw.Draw(image).rectangle((8, 8, 31, 31), fill=colour)
                image.save(path)
            with redirect_stdout(io.StringIO()):
                baseline, _ = train_baseline(root, Path(tmp) / 'baseline', jobs=1, trees=8)
                with patch('fruitripeness.baseline.process_image', wraps=process_image) as process:
                    run, metrics = train_baseline(root, Path(tmp) / 'hybrid', jobs=1, trees=8, method='hybrid')
                self.assertEqual(process.call_count, 36)  # Top-level fit/validation calls only.
            self.assertNotIn('test', metrics)
            self.assertFalse((run / 'predictions_test.csv').exists())
            comparison = compare_runs(baseline, run)
            self.assertEqual(comparison['hybrid']['accuracy'], metrics['validation']['accuracy'])
            self.assertAlmostEqual(comparison['hybrid_minus_baseline']['accuracy'],
                                   comparison['hybrid']['accuracy'] - comparison['baseline']['accuracy'])
            with (run / 'processing_diagnostics.csv').open(newline='') as f:
                diagnostics = list(csv.DictReader(f))
            self.assertEqual(len(diagnostics), 36)
            self.assertNotIn('test', {row['split'] for row in diagnostics})
            self.assertTrue(all(row['hybrid_mask_jaccard'] for row in diagnostics))
            self.assertTrue(all(row['grabcut_status'] for row in diagnostics))
            self.assertGreaterEqual(export_previews(root, run), 6)
            with Image.open(next((run / 'previews').glob('*.png'))) as preview:
                self.assertEqual(preview.size, (1660, 460))
            for filename in ['metadata.json', 'model.joblib']:
                saved = (json.loads((run / filename).read_text()) if filename.endswith('.json')
                         else joblib.load(run / filename))
                self.assertEqual(saved['processing_spec'], processing_spec('hybrid'))
            bundle = joblib.load(run / 'model.joblib')
            with (run / 'predictions_validation.csv').open(newline='') as f:
                predictions = list(csv.DictReader(f))
            for row in predictions:
                path = root / row['path']
                with Image.open(path) as source:
                    manual = feature_vector(process_image(source, 'hybrid').processed)
                np.testing.assert_array_equal(manual, image_features(path, method='hybrid'))
                predicted = predict_image(bundle, path)
                self.assertEqual(predicted['predicted_stage'], row['predicted_stage'])
                for name, score in predicted['scores'].items():
                    self.assertAlmostEqual(score, float(row['score_' + name]))


if __name__ == '__main__':
    unittest.main()
