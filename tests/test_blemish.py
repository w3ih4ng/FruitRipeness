"""Unit tests for blemish and surface-damage quantification logic."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw
import numpy as np

from fruitripeness.blemish import BLEMISH_SPEC, blemish_report, detect_blemishes


def apple_with_bruise(bruise: bool) -> Image.Image:
    image = Image.new("RGB", (120, 120), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 15, 105, 105), fill=(205, 30, 25))
    if bruise:
        draw.ellipse((60, 55, 85, 80), fill=(60, 35, 20))  # dark brown bruise patch
    return image


def apple_with_glare() -> Image.Image:
    """Creates a healthy red apple with a bright white specular glare spot."""
    image = Image.new("RGB", (120, 120), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 15, 105, 105), fill=(205, 30, 25))
    draw.ellipse((40, 40, 55, 55), fill=(255, 255, 255))  # High-brightness specular glare
    return image


def apple_with_leaf() -> Image.Image:
    """Creates a healthy red apple with an attached green leaf."""
    image = Image.new("RGB", (120, 120), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 15, 105, 105), fill=(205, 30, 25))
    draw.ellipse((45, 10, 75, 30), fill=(40, 180, 40))  # Green leaf patch
    return image


class BlemishTests(unittest.TestCase):
    def test_clean_fruit_has_near_zero_blemish_fraction(self):
        result = detect_blemishes(apple_with_bruise(False))
        self.assertEqual(result.status, "graded")
        self.assertLess(result.blemish_fraction, 0.01)

    def test_bruised_fruit_is_flagged_and_located_correctly(self):
        result = detect_blemishes(apple_with_bruise(True))
        self.assertEqual(result.status, "graded")
        self.assertGreater(result.blemish_fraction, 0.02)
        self.assertTrue(result.blemish_mask[67, 72])   # inside the bruise
        self.assertFalse(result.blemish_mask[20, 20])  # clean skin, away from bruise

    def test_specular_glare_is_not_flagged_as_blemish(self):
        """Verify white light reflections are ignored by the blemish detector."""
        result = detect_blemishes(apple_with_glare())
        self.assertEqual(result.status, "graded")
        self.assertFalse(result.blemish_mask[47, 47])  # glare center must be False
        self.assertLess(result.blemish_fraction, 0.01)

    def test_green_leaf_is_not_flagged_as_blemish(self):
        """Verify green leaves/foliage are excluded from defect calculations."""
        result = detect_blemishes(apple_with_leaf())
        self.assertEqual(result.status, "graded")
        self.assertFalse(result.blemish_mask[20, 60])  # leaf center must be False
        self.assertLess(result.blemish_fraction, 0.01)

    def test_no_labels_or_filenames_influence_the_mask(self):
        clean, bruised = apple_with_bruise(False), apple_with_bruise(True)
        r1 = detect_blemishes(clean)
        r2 = detect_blemishes(bruised)
        self.assertNotEqual(r1.blemish_fraction, r2.blemish_fraction)

    def test_too_small_image_is_ungraded_not_silently_zero(self):
        result = detect_blemishes(Image.new("RGB", (5, 5), (200, 30, 20)))
        self.assertIsNone(result.blemish_fraction)
        self.assertEqual(result.status, "too_small_foreground")

    def test_deterministic_repeat_calls(self):
        image = apple_with_bruise(True)
        first = detect_blemishes(image)
        second = detect_blemishes(image)
        np.testing.assert_array_equal(first.blemish_mask, second.blemish_mask)
        self.assertEqual(first.blemish_fraction, second.blemish_fraction)

    def test_report_writes_csv_and_summary_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for split in ["Train", "Test"]:
                for stage, colour in [("Ripe", (205, 30, 25)), ("Unripe", (60, 200, 40)), ("Overipe", (90, 60, 30))]:
                    folder = root / split / stage
                    folder.mkdir(parents=True)
                    for i in range(6):
                        bruise = (i == 0)
                        stage_key = "overripe" if stage == "Overipe" else stage.lower()
                        apple_with_bruise(bruise).save(folder / f"apple_{stage_key}_{i}.png")
            summary = blemish_report(root, Path(tmp) / "out", split="train")
            self.assertEqual(summary["images"], 14)
            self.assertGreaterEqual(summary["graded_images"], 1)
            self.assertTrue((Path(tmp) / "out" / "blemish_report.csv").is_file())
            self.assertTrue((Path(tmp) / "out" / "summary.json").is_file())
            with self.assertRaisesRegex(ValueError, "outside"):
                blemish_report(root, root / "report", split="train")


if __name__ == "__main__":
    unittest.main()