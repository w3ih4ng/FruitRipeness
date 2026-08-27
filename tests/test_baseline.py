from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from fruitripeness.baseline import (CLASSES, assign_splits, collect_records, compute_metrics,
    feature_vector, image_features, predict_image, resolve_dataset_root, train_baseline)


def make_dataset(root):
    for split,n in [("Train",6),("Test",1)]:
        for fruit in ["apple","banana"]:
            for stage,colour in zip(CLASSES,[(30,210,20),(240,210,30),(80,40,20)]):
                folder = root/split/("Overipe" if split == "Train" and stage == "overripe" else stage.title())
                folder.mkdir(parents=True,exist_ok=True)
                for i in range(n):
                    # Repeated images deliberately prove that no deduplication occurs.
                    Image.new("RGB",(40,40),colour).save(folder/f"{fruit}_{stage}_{i}.png")


class BaselineTests(unittest.TestCase):
    def test_root_detection_handles_single_wrapper_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root/"archive")
            self.assertEqual(resolve_dataset_root(root),root/"archive")
            make_dataset(root/"other")
            with self.assertRaises(ValueError):
                resolve_dataset_root(root)

    def test_shared_split_preserves_original_test_and_every_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root)
            records = collect_records(root)
            split = assign_splits(records)
            self.assertEqual(len(split),42)
            self.assertEqual(split,assign_splits(records))
            original_test = {r["path"] for r in records if r["source_split"] == "test"}
            self.assertEqual(original_test,{r["path"] for r in split if r["split"] == "test"})
            self.assertTrue(all(r["source_split"] == "train" for r in split if r["split"] == "validation"))
            expected = {(r["fruit"],r["stage"]) for r in records if r["source_split"] == "train"}
            self.assertEqual(expected,{(r["fruit"],r["stage"]) for r in split if r["split"] == "validation"})

    def test_rgb_feature_order_scale_and_size(self):
        result = feature_vector(Image.new("RGB",(100,70),(255,0,128)))
        self.assertEqual(result.shape,(3072,))
        self.assertEqual(result.dtype,np.float32)
        np.testing.assert_allclose(result.reshape(-1,3),np.tile([1,0,128/255],(1024,1)),rtol=1e-6)

    def test_metrics_have_explicit_class_order_and_axes(self):
        result = compute_metrics(["unripe","ripe","overripe"],["ripe","ripe","overripe"])
        self.assertEqual(result["confusion_matrix"],[[0,1,0],[0,1,0],[0,0,1]])
        self.assertAlmostEqual(result["accuracy"],2/3)
        self.assertEqual(result["per_class"]["unripe"]["support"],1)

    def test_training_default_does_not_decode_test_and_model_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);root=base/"data"
            make_dataset(root)
            before = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')}
            calls=[]

            def tracked(path,*args,**kwargs):
                calls.append(Path(path))
                return image_features(path,*args,**kwargs)

            with patch('fruitripeness.baseline.image_features',side_effect=tracked),redirect_stdout(io.StringIO()):
                run,metrics=train_baseline(root,base/"out",jobs=1,trees=8)
            self.assertEqual(set(metrics),{"validation"})
            self.assertFalse(any("Test" in p.parts for p in calls))
            metadata=json.loads((run/"metadata.json").read_text())
            self.assertFalse(metadata["test_evaluated"])
            self.assertEqual(sum(metadata["counts"].values()),42)
            # Only our own just-written model is loaded; never load untrusted joblib files.
            bundle=joblib.load(run/"model.joblib")
            sample=root/"Train"/"Ripe"/"apple_ripe_0.png"
            prediction=predict_image(bundle,sample)
            self.assertAlmostEqual(sum(prediction["scores"].values()),1)
            self.assertEqual(prediction["predicted_stage"],"ripe")
            with (run/"predictions_validation.csv").open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),metadata["counts"]["validation"])
            self.assertEqual(before,{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')})
            with redirect_stdout(io.StringIO()):
                final,final_metrics=train_baseline(root,base/"out",evaluate_test=True,jobs=1,trees=8)
            self.assertNotEqual(final,run)
            self.assertEqual(final_metrics["test"]["n_images"],6)
            self.assertTrue((final/"predictions_test.csv").exists())

    def test_invalid_image_fails_instead_of_silently_changing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data"
            make_dataset(root)
            (root/"Train"/"Ripe"/"apple_ripe_0.png").write_bytes(b"broken")
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"no image was silently skipped"):
                train_baseline(root,Path(tmp)/"out",jobs=1,trees=1)

    def test_source_folder_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_dataset(root)
            with self.assertRaisesRegex(ValueError,"outside"):
                train_baseline(root,root/"results",trees=1)


if __name__ == "__main__":
    unittest.main()
