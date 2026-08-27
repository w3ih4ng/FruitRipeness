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
from fruitripeness.processing import kmeans_candidate_mask, otsu_threshold, process_image, processing_spec


class ProcessingTests(unittest.TestCase):
    def test_kmeans_retains_multiple_colours_and_preserves_original_rgb(self):
        for background in ['white', 'black']:
            with self.subTest(background=background):
                image=Image.new('RGB',(100,100),background)
                draw=ImageDraw.Draw(image)
                draw.rectangle((15,15,44,84),fill=(210,30,20))
                draw.rectangle((55,15,84,84),fill=(20,180,50))
                before=image.tobytes()
                result=process_image(image,'kmeans')
                self.assertEqual(result.details['kmeans_clusters'],3)
                self.assertTrue(result.mask[50,30])
                self.assertTrue(result.mask[50,70])
                self.assertFalse(result.mask[0,0])
                np.testing.assert_array_equal(np.asarray(result.processed)[result.mask],
                                              np.asarray(image)[result.mask])
                self.assertFalse(np.asarray(result.processed)[~result.mask].any())
                self.assertEqual(image.tobytes(),before)

    def test_kmeans_fewer_colours_and_tiny_images_do_not_fail(self):
        for size in [(1,1),(1,30),(50,50)]:
            with self.subTest(size=size):
                result=process_image(Image.new('RGB',size,(150,30,20)),'kmeans')
                self.assertEqual(result.details['kmeans_clusters'],1)
                self.assertEqual(result.details['kmeans_iterations'],0)
                self.assertEqual(result.details['mask_status'],'empty')
                self.assertFalse(result.mask.any())
        image=Image.new('RGB',(80,80),'white')
        ImageDraw.Draw(image).rectangle((20,20,59,59),fill='red')
        result=process_image(image,'kmeans')
        self.assertEqual(result.details['kmeans_clusters'],2)
        self.assertTrue(result.mask[40,40])
        self.assertFalse(result.mask[0,0])

    def test_kmeans_determinism_grid_limit_and_chunked_assignment(self):
        image=Image.fromarray(np.random.default_rng(123).integers(0,256,(103,137,3),dtype=np.uint8))
        spec=processing_spec('kmeans')
        first,details=kmeans_candidate_mask(image,spec)
        second,details2=kmeans_candidate_mask(image,{**spec,'assignment_chunk_size':97})
        np.testing.assert_array_equal(first,second)
        self.assertEqual(details,details2)
        self.assertEqual(details['kmeans_sample_pixels'],4096)
        self.assertEqual(first.shape,(103,137))
        self.assertGreaterEqual(details['background_border_fraction'],1/3)

    def test_kmeans_equal_border_and_area_tie_uses_centroid_order(self):
        # Two equal halves: canonical dark centroid is background, independent
        # of the arbitrary IDs returned by the underlying clustering fit.
        pixels=np.full((100,100,3),30,dtype=np.uint8);pixels[:,50:]=220
        result=process_image(Image.fromarray(pixels),'kmeans')
        self.assertEqual(result.details['background_cluster'],0)
        self.assertEqual(result.details['background_border_fraction'],0.5)
        self.assertFalse(result.mask[50,25])
        self.assertTrue(result.mask[50,75])
        # Equal frame counts but more bright pixels inside: whole-image count
        # must take priority over the lower canonical ID.
        pixels[5:95,30:50]=220
        result=process_image(Image.fromarray(pixels),'kmeans')
        self.assertEqual(result.details['background_cluster'],1)
        self.assertFalse(result.mask[50,40])
        self.assertTrue(result.mask[50,15])

    def test_kmeans_saved_settings_mismatch_is_rejected(self):
        spec=processing_spec('kmeans')
        spec['background_rgb'][0]=255
        self.assertEqual(processing_spec('kmeans')['background_rgb'],[0,0,0])
        with self.assertRaisesRegex(ValueError,'settings differ'):
            predict_image({'method':'kmeans','feature_id':'rgb_pixels_32x32_v1',
                           'processing_spec':spec},Path('unused.png'))

    def test_kmeans_end_to_end_exports_features_and_saved_model_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';make_dataset(root)
            for path in root.rglob('*.png'):
                with Image.open(path) as source:
                    colour=source.getpixel((0,0))
                image=Image.new('RGB',(40,40),'white')
                ImageDraw.Draw(image).rectangle((8,8,31,31),fill=colour)
                image.save(path)
            with redirect_stdout(io.StringIO()):
                baseline,_=train_baseline(root,Path(tmp)/'baseline',jobs=1,trees=8)
                with patch('fruitripeness.baseline.process_image',wraps=process_image) as process:
                    run,metrics=train_baseline(root,Path(tmp)/'kmeans',jobs=1,trees=8,method='kmeans')
                self.assertEqual(process.call_count,36)  # Test pixels never processed.
            comparison=compare_runs(baseline,run)
            self.assertEqual(comparison['kmeans']['accuracy'],metrics['validation']['accuracy'])
            self.assertAlmostEqual(comparison['kmeans_minus_baseline']['accuracy'],
                                   comparison['kmeans']['accuracy']-comparison['baseline']['accuracy'])
            with (run/'processing_diagnostics.csv').open(newline='') as f:
                diagnostics=list(csv.DictReader(f))
            self.assertEqual(len(diagnostics),36)
            self.assertTrue(all(r['kmeans_clusters']=='2' for r in diagnostics))
            self.assertTrue(all(float(r['background_border_fraction'])==1 for r in diagnostics))
            self.assertNotIn('test',{r['split'] for r in diagnostics})
            n=export_previews(root,run)
            self.assertGreaterEqual(n,6)
            with Image.open(next((run/'previews').glob('*.png'))) as preview:
                self.assertEqual(preview.size,(1000,460))
            with (run/'predictions_validation.csv').open(newline='') as f:
                row=next(csv.DictReader(f))
            path=root/row['path']
            with Image.open(path) as source:
                manual=feature_vector(process_image(source,'kmeans').processed)
            np.testing.assert_array_equal(manual,image_features(path,method='kmeans'))
            bundle=joblib.load(run/'model.joblib')  # Our own model only.
            predicted=predict_image(bundle,path)
            self.assertEqual(predicted['predicted_stage'],row['predicted_stage'])
            for c,score in predicted['scores'].items():
                self.assertAlmostEqual(score,float(row['score_'+c]))

    def test_otsu_threshold_matches_brute_force_variance_minimum(self):
        two_levels = np.tile(np.array([30,220], dtype=np.uint8),(25,25))
        self.assertEqual(otsu_threshold(two_levels),30)  # First threshold in the optimal plateau.
        gray = np.random.default_rng(7).integers(0,256,(19,23),dtype=np.uint8)
        def within_variance(t):
            low, high = gray[gray <= t], gray[gray > t]
            if not low.size or not high.size:
                return float("inf")
            return low.size*low.var()+high.size*high.var()
        threshold = otsu_threshold(gray)
        self.assertAlmostEqual(within_variance(threshold),min(within_variance(t) for t in range(255)),places=7)

    def test_otsu_constant_images_and_invalid_arrays(self):
        for value in [0,128,255]:
            with self.subTest(value=value):
                gray=np.full((40,40),value,dtype=np.uint8)
                self.assertIsNone(otsu_threshold(gray))
                result=process_image(Image.fromarray(gray),"otsu")
                self.assertEqual(result.details['mask_status'],'empty')
                self.assertEqual(result.details['foreground_polarity'],'none')
                self.assertFalse(np.asarray(result.processed).any())
        for bad in [np.zeros((5,5),dtype=float),np.zeros((5,5,3),dtype=np.uint8),np.zeros((0,0),dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                otsu_threshold(bad)

    def test_otsu_polarity_handles_dark_and_light_backgrounds_preserving_rgb(self):
        for background, polarity in [("white","dark"),("black","bright")]:
            with self.subTest(background=background):
                image=Image.new("RGB",(100,100),background)
                colour=(220,40,10)
                ImageDraw.Draw(image).rectangle((20,20,79,79),fill=colour)
                original=image.tobytes()
                result=process_image(image,"otsu")
                self.assertEqual(result.details['foreground_polarity'],polarity)
                self.assertTrue(result.mask[50,50])
                self.assertFalse(result.mask[0,0])
                self.assertEqual(result.processed.getpixel((50,50)),colour)
                self.assertEqual(image.tobytes(),original)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'unnamed.png';image.save(path)
                    np.testing.assert_array_equal(image_features(path,method='otsu'),feature_vector(result.processed))

    def test_otsu_equal_border_and_area_tie_chooses_bright(self):
        gray=np.full((100,100),30,dtype=np.uint8);gray[:,50:]=220
        result=process_image(Image.fromarray(gray),'otsu')
        self.assertEqual(result.details['foreground_polarity'],'bright')
        self.assertFalse(result.mask[50,25])
        self.assertTrue(result.mask[50,75])

    def test_otsu_end_to_end_exports_and_saved_model_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';make_dataset(root)
            # Put fixture colours inside a neutral background to exercise segmentation.
            for path in root.rglob('*.png'):
                with Image.open(path) as source:
                    colour=source.getpixel((0,0))
                image=Image.new('RGB',(40,40),'white')
                ImageDraw.Draw(image).rectangle((8,8,31,31),fill=colour)
                image.save(path)
            with redirect_stdout(io.StringIO()):
                baseline,_=train_baseline(root,Path(tmp)/'baseline',jobs=1,trees=8)
                with patch('fruitripeness.baseline.process_image',wraps=process_image) as process:
                    run,metrics=train_baseline(root,Path(tmp)/'otsu',jobs=1,trees=8,method='otsu')
                self.assertEqual(process.call_count,36)
            comparison=compare_runs(baseline,run)
            self.assertEqual(comparison['otsu']['accuracy'],metrics['validation']['accuracy'])
            self.assertNotIn('hsv',comparison)
            with (run/'processing_diagnostics.csv').open(newline='') as f:
                diagnostics=list(csv.DictReader(f))
            self.assertEqual(len(diagnostics),36)
            self.assertTrue(all(r['otsu_threshold'] for r in diagnostics))
            self.assertTrue(all(r['foreground_polarity']=='dark' for r in diagnostics))
            self.assertNotIn('test',{r['split'] for r in diagnostics})
            n=export_previews(root,run)
            self.assertGreaterEqual(n,6)
            with Image.open(next((run/'previews').glob('*.png'))) as preview:
                self.assertEqual(preview.size,(1000,460))
            with (run/'predictions_validation.csv').open(newline='') as f:
                row=next(csv.DictReader(f))
            bundle=joblib.load(run/'model.joblib')  # Our own model only.
            predicted=predict_image(bundle,root/row['path'])
            self.assertEqual(predicted['predicted_stage'],row['predicted_stage'])
            for c,score in predicted['scores'].items():
                self.assertAlmostEqual(score,float(row['score_'+c]))

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
