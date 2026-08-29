import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fruitripeness.calibration import CalibrationResult, measure, pixels_per_unit_from_reference
from fruitripeness.object_detection import detect_objects, draw_detections
from fruitripeness.preprocessing import contrast_stretch, denoise, enhance
from fruitripeness.processing import process_image


def noisy_apple() -> Image.Image:
    rng = np.random.default_rng(7)
    image = Image.new("RGB", (120, 120), (170, 170, 170))
    ImageDraw.Draw(image).ellipse((15, 15, 105, 105), fill=(150, 60, 40))  # low-contrast, dull colour
    pixels = np.asarray(image).astype(np.int16)
    pixels = np.clip(pixels + rng.integers(-25, 26, pixels.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


class PreprocessingTests(unittest.TestCase):
    def test_median_denoise_reduces_salt_and_pepper(self):
        rng = np.random.default_rng(3)
        clean = np.full((60, 60, 3), 120, dtype=np.uint8)
        salt_pepper = clean.copy()
        coords = rng.choice(60*60, 200, replace=False)
        salt_pepper.reshape(-1, 3)[coords] = rng.choice([0, 255], (200, 3))
        result = denoise(salt_pepper, {"denoise_method": "median", "median_kernel": 3})
        self.assertLess(np.abs(result.astype(int)-clean.astype(int)).mean(),
                        np.abs(salt_pepper.astype(int)-clean.astype(int)).mean())

    def test_contrast_stretch_expands_dull_range_without_new_extremes(self):
        dull = np.full((40, 40, 3), 128, dtype=np.uint8)
        dull[10:30, 10:30] = 140
        stretched = contrast_stretch(dull, {"stretch_low_percentile": 2.0, "stretch_high_percentile": 98.0})
        self.assertGreater(stretched.std(), dull.std())
        self.assertLessEqual(stretched.max(), 255)
        self.assertGreaterEqual(stretched.min(), 0)

    def test_constant_channel_is_left_unchanged_not_amplified(self):
        flat = np.full((20, 20, 3), 90, dtype=np.uint8)
        result = contrast_stretch(flat, {"stretch_low_percentile": 2.0, "stretch_high_percentile": 98.0})
        np.testing.assert_array_equal(result, flat)

    def test_enhance_pipeline_improves_contrast_and_is_deterministic(self):
        image = noisy_apple()
        first, details1 = enhance(image)
        second, details2 = enhance(image)
        self.assertEqual(list(first.getdata()), list(second.getdata()))
        self.assertEqual(details1, details2)
        self.assertGreaterEqual(details1["std_after_stretch"], details1["std_before"]-1e-6)
        self.assertGreater(details1["mean_absolute_change"], 0)

    def test_invalid_kernel_rejected(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            denoise(np.zeros((10, 10, 3), dtype=np.uint8), {"denoise_method": "median", "median_kernel": 4})


class CalibrationTests(unittest.TestCase):
    def test_uncalibrated_measurement_matches_known_circle_geometry(self):
        mask = np.zeros((200, 200), dtype=bool)
        cv2.circle(mask.view(np.uint8), (100, 100), 50, 1, -1)
        result = measure(mask)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.unit, "px")
        self.assertAlmostEqual(result.equivalent_diameter_px, 100, delta=3)
        self.assertAlmostEqual(result.area_px, np.pi*50**2, delta=400)

    def test_reference_based_calibration_converts_to_physical_units(self):
        mask = np.zeros((200, 200), dtype=bool)
        cv2.circle(mask.view(np.uint8), (100, 100), 50, 1, -1)
        scale = pixels_per_unit_from_reference(reference_length_px=100, known_physical_length=2.0)  # 50 px/cm
        result = measure(mask, pixels_per_unit=scale, unit="cm")
        self.assertTrue(result.calibrated)
        self.assertAlmostEqual(result.equivalent_diameter_physical, 2.0, delta=0.1)

    def test_empty_mask_rejected_not_silently_zero(self):
        with self.assertRaisesRegex(ValueError, "empty mask"):
            measure(np.zeros((50, 50), dtype=bool))

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            pixels_per_unit_from_reference(0, 2.0)


class ObjectDetectionTests(unittest.TestCase):
    def test_single_fruit_detected_with_correct_bbox(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).ellipse((20, 20, 80, 80), fill="red")
        mask = process_image(image, "hsv").mask
        objects = detect_objects(mask)
        self.assertEqual(len(objects), 1)
        x, y, w, h = objects[0].bounding_box
        self.assertAlmostEqual(x, 20, delta=3)
        self.assertAlmostEqual(w, 60, delta=3)

    def test_two_separate_blobs_both_detected(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:60, 10:60] = True
        mask[30:70, 120:180] = True
        objects = detect_objects(mask)
        self.assertEqual(len(objects), 2)
        self.assertTrue(all(o.area_px > 0 for o in objects))

    def test_tiny_noise_components_are_filtered(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # real object, 400 px
        mask[5, 5] = True  # single-pixel noise
        objects = detect_objects(mask)
        self.assertEqual(len(objects), 1)

    def test_empty_mask_returns_no_objects(self):
        self.assertEqual(detect_objects(np.zeros((50, 50), dtype=bool)), [])

    def test_draw_detections_does_not_mutate_source(self):
        image = Image.new("RGB", (60, 60), "white")
        ImageDraw.Draw(image).ellipse((10, 10, 50, 50), fill="green")
        before = image.tobytes()
        mask = process_image(image, "hsv").mask
        draw_detections(image, detect_objects(mask))
        self.assertEqual(image.tobytes(), before)


if __name__ == "__main__":
    unittest.main()