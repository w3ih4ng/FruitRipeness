import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from fruitripeness.audit import audit_zip, duplicate_groups, parse_labels, perceptual_hash


class AuditTests(unittest.TestCase):
    def test_alias_and_optional_outer_folder(self):
        labels = parse_labels("dataset/Train/Overipe/apple_overripe_001.jpg")
        self.assertEqual(labels["stage"], "overripe")
        self.assertEqual(labels["raw_stage"], "overipe")
        self.assertEqual(parse_labels("Test/Unripe/banana_unripe_001.jpg")["stage"], "unripe")

    def test_rejects_disagreement_and_unknown_fruit(self):
        for name in ["Train/Ripe/apple_unripe_001.jpg", "Test/Ripe/pear_ripe_1.jpg", "flat.jpg"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_labels(name)

    def test_hash_stable_across_lossless_encoding(self):
        array = np.random.default_rng(42).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.assertEqual(perceptual_hash(image), perceptual_hash(Image.open(io.BytesIO(buffer.getvalue()))))

    def test_duplicate_groups_flag_cross_split_and_label_conflict(self):
        rows = [dict(path="a", source_split="train", fruit="apple", stage="ripe", hash="x"),
                dict(path="b", source_split="test", fruit="apple", stage="unripe", hash="x")]
        group = duplicate_groups(rows, "hash")[0]
        self.assertTrue(group["cross_split"])
        self.assertTrue(group["conflicting_labels"])

    def test_zip_audit_detects_corruption_and_decoded_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = np.random.default_rng(4).integers(0, 256, (40, 40, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            a, b = io.BytesIO(), io.BytesIO()
            image.save(a, format="PNG", compress_level=0)
            image.save(b, format="PNG", compress_level=9)
            source = root / "sample.zip"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("Train/Ripe/apple_ripe_1.png", a.getvalue())
                z.writestr("Test/Ripe/apple_ripe_2.png", b.getvalue())
                z.writestr("Train/Unripe/mango_unripe_1.jpg", b"not an image")
                z.writestr("notes.txt", "not a dataset image")
            original = source.read_bytes()
            report = audit_zip(source, root / "out")
            self.assertEqual(report["valid_images"], 2)
            self.assertEqual(report["invalid_images"], 1)
            self.assertEqual(report["exact_pixel_cross_split_groups"], 1)
            self.assertEqual(report["exact_byte_groups"], 0)
            self.assertEqual(report["near_duplicate_candidate_pairs"], 0)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(json.loads((root/"out"/"summary.json").read_text())["valid_images"], 2)

    def test_invalid_distance_rejected(self):
        with self.assertRaises(ValueError):
            audit_zip(Path("missing.zip"), Path("unused"), distance=9)


if __name__ == "__main__":
    unittest.main()
