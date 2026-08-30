import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fruitripeness.fruit_detection import (
    FRUIT_CLASSES,
    FruitDetection,
    analyse_fruits,
    per_fruit_card,
)


class FakeDetector:
    def detect(self, image):
        apple = np.zeros((image.height, image.width), dtype=bool)
        banana = np.zeros_like(apple)
        apple[5:25, 5:25] = True
        banana[10:30, 35:65] = True
        return [
            FruitDetection("apple", 0.91, (5, 5, 25, 25), apple),
            FruitDetection("banana", 0.82, (35, 10, 65, 30), banana),
        ]


def fake_infer(image, bundle):
    stage = "ripe" if image.width <= 20 else "unripe"
    scores = {"unripe": 0.8, "ripe": 0.1, "overripe": 0.1}
    if stage == "ripe":
        scores = {"unripe": 0.1, "ripe": 0.8, "overripe": 0.1}
    mask = np.ones((image.height, image.width), dtype=bool)
    return (
        {
            "predicted_stage": stage,
            **{f"score_{name}": value for name, value in scores.items()},
            "processing_ms": 2.0,
            "feature_ms": 1.0,
            "prediction_ms": 3.0,
            "foreground_fraction": 1.0,
            "mask_status": "nonempty",
            "refined_review_flag": "",
        },
        SimpleNamespace(processed=image.copy(), mask=mask),
    )


class FruitDetectionTests(unittest.TestCase):
    def test_five_project_fruits_are_explicit(self):
        self.assertEqual(
            FRUIT_CLASSES,
            ("apple", "banana", "mango", "orange", "tomato"),
        )

    def test_each_detected_fruit_is_classified_independently(self):
        image = Image.new("RGB", (80, 40), "white")
        before = image.tobytes()
        with patch("fruitripeness.ui_core.infer", side_effect=fake_infer) as infer:
            row, analysis = analyse_fruits(image, {"method": "hsv"}, FakeDetector())
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(row["objects_detected"], 2)
        self.assertEqual(row["predicted_stage"], "mixed")
        self.assertEqual(row["detected_fruits"], "apple, banana")
        exported = json.loads(row["object_results_json"])
        self.assertEqual([item["fruit"] for item in exported], ["apple", "banana"])
        self.assertNotIn("crop", exported[0])
        self.assertEqual(image.tobytes(), before)
        row["method_label"] = "HSV"
        self.assertGreater(per_fruit_card(row, analysis).height, 500)

    def test_no_detection_does_not_fall_back_to_whole_image(self):
        detector = SimpleNamespace(detect=lambda image: [])
        with patch("fruitripeness.ui_core.infer") as infer:
            row, analysis = analyse_fruits(
                Image.new("RGB", (80, 40), "white"),
                {"method": "hsv"},
                detector,
            )
        infer.assert_not_called()
        self.assertEqual(row["predicted_stage"], "none")
        self.assertEqual(row["objects_detected"], 0)
        self.assertEqual(analysis.objects, [])


if __name__ == "__main__":
    unittest.main()
