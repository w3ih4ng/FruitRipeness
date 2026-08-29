from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

import cv2
import numpy as np

from fruitripeness.media import (
    VIDEO_RESULT_FIELDS,
    annotate_frame,
    bgr_to_pil,
    process_video,
    run_live_camera,
)


def predictor(image):
    mask = np.zeros((image.height, image.width), dtype=bool)
    mask[4:-4, 4:-4] = True
    return (
        {
            "predicted_stage": "ripe",
            "score_unripe": 0.1,
            "score_ripe": 0.8,
            "score_overripe": 0.1,
            "processing_ms": 2.0,
            "feature_ms": 1.0,
            "prediction_ms": 3.0,
            "foreground_fraction": float(mask.mean()),
            "mask_status": "nonempty",
            "refined_review_flag": "",
        },
        SimpleNamespace(mask=mask),
    )


class FakeCapture:
    def __init__(self, source, frames):
        self.source = source
        self.frames = [frame.copy() for frame in frames]
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, key):
        values = {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: 40,
            cv2.CAP_PROP_FRAME_HEIGHT: 30,
            cv2.CAP_PROP_FRAME_COUNT: 2,
        }
        return values.get(key, 0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, path, codec, fps, size):
        self.path = Path(path)
        self.path.write_bytes(b"video")
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class MediaTests(unittest.TestCase):
    def test_bgr_conversion_and_annotation_preserve_source(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        frame[:, :] = (10, 20, 200)
        image = bgr_to_pil(frame)
        before = image.tobytes()
        prediction, result = predictor(image)
        annotated, objects = annotate_frame(
            image,
            prediction,
            result,
            method_label="Method 1 - HSV",
        )
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(annotated.size, image.size)
        self.assertEqual(objects, 1)
        self.assertEqual(image.getpixel((0, 0)), (200, 20, 10))

    def test_live_camera_emits_frame_and_releases_capture(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        capture = FakeCapture(0, [frame, frame])
        cancel = threading.Event()
        events = []

        def emit(kind, value):
            events.append((kind, value))
            if kind == "camera_frame":
                cancel.set()

        run_live_camera(
            0,
            predict=predictor,
            method="hsv",
            method_label="HSV",
            cancel=cancel,
            emit=emit,
            analysis_interval=0,
            capture_factory=lambda source: capture,
        )
        frames = [value for kind, value in events if kind == "camera_frame"]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["predicted_stage"], "ripe")
        self.assertTrue(capture.released)

    def test_video_processes_each_frame_and_atomically_finishes(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        capture = FakeCapture("source", [frame, frame])
        writers = []

        def writer_factory(*args):
            writer = FakeWriter(*args)
            writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"unchanged source")
            before = source.read_bytes()
            output = Path(tmp) / "annotated.mp4"
            events = []
            rows = process_video(
                source,
                output,
                predict=predictor,
                method="hsv",
                method_label="HSV",
                cancel=threading.Event(),
                emit=lambda kind, value: events.append((kind, value)),
                capture_factory=lambda value: capture,
                writer_factory=writer_factory,
                fourcc_factory=lambda *value: 0,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(writers[0].frames), 2)
            self.assertTrue(output.is_file())
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(all(field in rows[0] for field in VIDEO_RESULT_FIELDS))
            self.assertEqual(events[-1], ("video_completed", str(output.resolve())))

    def test_cancelled_video_discards_incomplete_output(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        capture = FakeCapture("source", [frame, frame])
        cancel = threading.Event()

        def emit(kind, value):
            if kind == "video_frame":
                cancel.set()

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.avi"
            source.write_bytes(b"source")
            output = Path(tmp) / "annotated.avi"
            process_video(
                source,
                output,
                predict=predictor,
                method="hsv",
                method_label="HSV",
                cancel=cancel,
                emit=emit,
                capture_factory=lambda value: capture,
                writer_factory=FakeWriter,
                fourcc_factory=lambda *value: 0,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
