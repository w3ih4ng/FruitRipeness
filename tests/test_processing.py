from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image, ImageDraw

from test_baseline import make_dataset
from fruitripeness.baseline import feature_vector, image_features, predict_image, train_baseline
from fruitripeness.experiment import compare_runs, export_previews, render_preview
from fruitripeness.processing import process_image, processing_spec


class ProcessingTests(unittest.TestCase):
    def test_hsv_accepts_different_hues_and_preserves_original_pixels(self):
        for colour in [(220,30,20),(30,200,40),(240,220,30),(110,60,20)]:
            with self.subTest(colour=colour):
                image = Image.new("RGB",(100,100),(230,230,230))
                ImageDraw.Draw(image).rectangle((20,20,79,79),fill=colour)
                before = image.tobytes()
                result = process_image(image,"hsv")
                self.assertTrue(result.mask[50,50])
                self.assertFalse(result.mask[0,0])
                self.assertEqual(result.processed.getpixel((50,50)),colour)
                self.assertEqual(result.processed.getpixel((0,0)),(0,0,0))
                self.assertEqual(image.tobytes(),before)

    def test_cleanup_removes_speckles_keeps_multiple_regions_and_fills_holes(self):
        image = Image.new("RGB",(100,100),"white")
        draw=ImageDraw.Draw(image)
        draw.rectangle((10,10,39,39),fill=(200,30,20))
        draw.rectangle((60,60,89,89),fill=(30,200,20))
        draw.rectangle((20,20,24,24),fill="black")
        draw.point((90,10),fill="red")
        result=process_image(image,"hsv")
        self.assertTrue(result.mask[22,22])
        self.assertTrue(result.mask[75,75])
        self.assertFalse(result.mask[10,90])
        self.assertEqual(result.processed.getpixel((22,22)),(0,0,0))

    def test_empty_mask_is_flagged_without_fallback(self):
        result=process_image(Image.new("RGB",(50,50),"white"),"hsv")
        self.assertEqual(result.details["mask_status"],"empty")
        self.assertEqual(result.details["foreground_fraction"],0)
        self.assertFalse(np.asarray(result.processed).any())

    def test_fully_coloured_image_does_not_get_artificial_black_rim(self):
        result=process_image(Image.new("RGB",(50,50),(240,20,20)),"hsv")
        self.assertTrue(result.mask.all())
        self.assertEqual(result.details["mask_status"],"near_full")

    def test_feature_path_uses_same_processing_as_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"arbitrary_name.png"
            image=Image.new("RGB",(100,100),"white")
            ImageDraw.Draw(image).ellipse((20,20,80,80),fill=(200,50,10))
            image.save(path)
            manual=feature_vector(process_image(image,"hsv").processed)
            np.testing.assert_array_equal(manual,image_features(path,method="hsv"))
            np.testing.assert_array_equal(feature_vector(image),image_features(path))
            preview=render_preview(image,"Method 1 - HSV","arbitrary_name.png")
            self.assertEqual(preview.size,(1000,460))

    def test_unknown_method_and_model_setting_mismatch_fail(self):
        with self.assertRaises(ValueError):
            process_image(Image.new("RGB",(20,20)),"unknown")
        with self.assertRaisesRegex(ValueError,"settings differ"):
            predict_image({"method":"hsv","feature_id":"rgb_pixels_32x32_v1",
                           "processing_spec":{"version":"other"}},Path("unused.png"))
        spec=processing_spec("hsv");spec["background_rgb"][0]=255
        self.assertEqual(processing_spec("hsv")["background_rgb"],[0,0,0])

    def test_hsv_training_comparison_previews_and_legacy_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data";make_dataset(root)
            with redirect_stdout(io.StringIO()):
                baseline,_=train_baseline(root,Path(tmp)/"baseline",jobs=1,trees=8)
                with patch('fruitripeness.baseline.process_image',wraps=process_image) as process:
                    hsv,metrics=train_baseline(root,Path(tmp)/"hsv",jobs=1,trees=8,method="hsv")
                self.assertEqual(process.call_count,36)  # Original Train only, not Test.
            comparison=compare_runs(baseline,hsv)
            self.assertEqual(comparison["hsv"]["accuracy"],metrics["validation"]["accuracy"])
            self.assertEqual(comparison["hsv_minus_baseline"]["accuracy"],
                             comparison["hsv"]["accuracy"]-comparison["baseline"]["accuracy"])
            with (hsv/"processing_diagnostics.csv").open(newline="") as f:
                diagnostics=list(csv.DictReader(f))
            self.assertEqual(len(diagnostics),36)
            self.assertNotIn("test",{r['split'] for r in diagnostics})
            n=export_previews(root,hsv)
            self.assertEqual(n,6)
            self.assertEqual(len(list((hsv/"previews").glob("*.png"))),n)
            with (hsv/"predictions_validation.csv").open(newline="") as f:
                row=next(csv.DictReader(f))
            bundle=joblib.load(hsv/"model.joblib")  # Only our own model is loaded.
            self.assertEqual(predict_image(bundle,root/row["path"])["predicted_stage"],row["predicted_stage"])
            legacy=joblib.load(baseline/"model.joblib");legacy.pop("processing_spec")
            self.assertIn(predict_image(legacy,root/row["path"])["predicted_stage"],['unripe','ripe','overripe'])
            # Refuse a mismatched split, model setting, or feature environment.
            original=json.loads((hsv/"metadata.json").read_text())
            for key,value in [("split_id","changed"),("model_parameters",{**original['model_parameters'],"n_estimators":9}),
                              ("versions",{**original['versions'],"Pillow":"other"})]:
                altered={**original,key:value}
                (hsv/"metadata.json").write_text(json.dumps(altered))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    compare_runs(baseline,hsv)


if __name__ == "__main__":
    unittest.main()
