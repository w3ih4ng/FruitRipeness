
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image, ImageDraw

from fruitripeness.ui_core import SURFACE_RESULT_FIELDS, analyze_surface, run_surface_batch


def apple(bruise: bool) -> Image.Image:
    image = Image.new("RGB", (120, 120), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 15, 105, 105), fill=(205, 30, 25))
    if bruise:
        draw.ellipse((60, 55, 85, 80), fill=(60, 35, 20))
    return image


class AnalyzeSurfaceTests(unittest.TestCase):
    def test_contrast_stretch_does_not_wash_out_blemish_signal(self):
        """Regression: chaining full contrast-stretch into blemish grading zeroed the signal out."""
        details, _, _, _ = analyze_surface(apple(True))
        self.assertGreater(details["blemish_fraction"], 0.02)
        clean_details, _, _, _ = analyze_surface(apple(False))
        self.assertLess(clean_details["blemish_fraction"], 0.01)

    def test_display_panel_still_uses_contrast_enhanced_pixels(self):
        _, panel, _, _ = analyze_surface(apple(True))
        raw = apple(True)
        self.assertNotEqual(list(panel.getdata()), list(raw.getdata()))

    def test_single_fruit_detects_one_object_and_measures_size(self):
        details, _, _, objects = analyze_surface(apple(False))
        self.assertEqual(len(objects), 1)
        self.assertEqual(details["objects_detected"], 1)
        self.assertGreater(details["equivalent_diameter_px"], 0)
        self.assertFalse(details["calibrated"])  # no reference object supplied


class RunSurfaceBatchTests(unittest.TestCase):
    def test_batch_emits_result_and_progress_and_handles_corrupt_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp)/"apple.png"
            apple(True).save(good)
            corrupt = Path(tmp)/"bad.png"
            corrupt.write_bytes(b"not an image")
            events = []
            run_surface_batch([corrupt, good], cancel=threading.Event(),
                              emit=lambda kind, value: events.append((kind, value)))
            rows = [v[0] for k, v in events if k == "result"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(rows[1]["status"], "ok")
            self.assertGreater(rows[1]["blemish_fraction"], 0)
            self.assertTrue(all(field in SURFACE_RESULT_FIELDS for field in
                               ["blemish_fraction", "objects_detected", "equivalent_diameter_px"]))
            progress = [v for k, v in events if k == "progress"]
            self.assertEqual(progress[-1], (2, 2))

    def test_cancel_stops_batch_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                path = Path(tmp)/f"apple_{i}.png"
                apple(False).save(path)
                paths.append(path)
            cancel = threading.Event()
            events = []
            def emit(kind, value):
                events.append((kind, value))
                if kind == "result":
                    cancel.set()
            run_surface_batch(paths, cancel=cancel, emit=emit)
            self.assertEqual(len([v for k, v in events if k == "result"]), 1)


if __name__ == "__main__":
    unittest.main()