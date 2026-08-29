from contextlib import redirect_stdout
import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from test_baseline import make_dataset
from fruitripeness.baseline import train_baseline
from fruitripeness.experiment import export_previews, surface_overlay


def bruised_apple():
    image = Image.new("RGB", (100, 100), (175, 175, 175))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, 90, 90), fill=(205, 30, 25))
    draw.ellipse((55, 45, 78, 68), fill=(60, 35, 20))
    return image


class SurfaceOverlayTests(unittest.TestCase):
    def test_overlay_uses_same_mask_as_the_requested_method(self):
        image = bruised_apple()
        overlay, stats = surface_overlay(image, "hsv")
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(stats["blemish_status"], "graded")
        self.assertGreater(stats["blemish_pixels"], 0)
        self.assertGreaterEqual(stats["objects_detected"], 1)

    def test_export_previews_writes_surface_subfolder_and_index_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"data"
            make_dataset(root)
            for path in root.rglob("*.png"):
                with Image.open(path) as source:
                    colour = source.getpixel((0, 0))
                image = Image.new("RGB", (40, 40), "white")
                ImageDraw.Draw(image).rectangle((8, 8, 31, 31), fill=colour)
                image.save(path)
            with redirect_stdout(io.StringIO()):
                run, _ = train_baseline(root, Path(tmp)/"hsv", jobs=1, trees=8, method="hsv")
                count = export_previews(root, run)
            self.assertGreaterEqual(count, 6)
            surface_dir = run/"previews"/"surface"
            self.assertTrue(surface_dir.is_dir())
            main_pngs = sorted((run/"previews").glob("*.png"))
            surface_pngs = sorted(surface_dir.glob("*.png"))
            self.assertEqual(len(main_pngs), len(surface_pngs))  # one overlay per preview, same filenames
            self.assertEqual({p.name for p in main_pngs}, {p.name for p in surface_pngs})
            with (run/"previews"/"index.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                self.assertIn("blemish_fraction", row)
                self.assertIn("objects_detected", row)
                self.assertEqual(row["surface_file"], f"surface/{row['file']}")


if __name__ == "__main__":
    unittest.main()