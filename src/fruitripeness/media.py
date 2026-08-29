"""Live-camera and uploaded-video processing for the desktop application.

The module is UI-independent and receives an already configured prediction
callback. It never trains a model, changes source media or reads dataset labels.
"""

from __future__ import annotations

from pathlib import Path
import math
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .object_detection import detect_objects, draw_detections


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
VIDEO_OUTPUT_EXTENSIONS = {".mp4", ".avi"}
MAX_FRAME_PIXELS = 20_000_000

VIDEO_RESULT_FIELDS = [
    "source",
    "output_video",
    "frame_index",
    "timestamp_seconds",
    "method",
    "predicted_stage",
    "score_unripe",
    "score_ripe",
    "score_overripe",
    "processing_ms",
    "feature_ms",
    "prediction_ms",
    "objects_detected",
]


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Video frame must be an 8-bit BGR image")
    if frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
        raise ValueError("Video frame exceeds the 20-megapixel safety limit")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def annotate_frame(
    original: Image.Image,
    prediction: dict,
    processing_result,
    *,
    method_label: str,
) -> tuple[Image.Image, int]:
    """Draw the frame-level prediction and segmentation-derived objects."""

    objects = detect_objects(processing_result.mask)
    canvas = draw_detections(original, objects)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, min(canvas.width, canvas.height) // 28))
    small = ImageFont.load_default(size=max(11, min(canvas.width, canvas.height) // 38))
    bar_height = max(58, min(92, canvas.height // 5))
    draw.rectangle((0, 0, canvas.width, bar_height), fill=(25, 55, 42))
    draw.text(
        (12, 8),
        f"{method_label} | {prediction['predicted_stage'].upper()}",
        fill="white",
        font=font,
    )
    draw.text(
        (12, bar_height // 2 + 5),
        (
            f"unripe {prediction['score_unripe']:.1%} | "
            f"ripe {prediction['score_ripe']:.1%} | "
            f"overripe {prediction['score_overripe']:.1%} | "
            f"objects {len(objects)}"
        ),
        fill=(216, 237, 224),
        font=small,
    )
    return canvas, len(objects)


def _open_capture(source, capture_factory):
    capture = capture_factory(source)
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        raise ValueError(f"Unable to open video source: {source}")
    return capture


def run_live_camera(
    camera_index: int,
    *,
    predict,
    method: str,
    method_label: str,
    cancel,
    emit,
    analysis_interval: float = 0.35,
    capture_factory=cv2.VideoCapture,
    clock=time.monotonic,
) -> None:
    """Read and analyse live frames until cancel is set."""

    if camera_index < 0:
        raise ValueError("Camera index must be zero or greater")
    if analysis_interval < 0:
        raise ValueError("analysis_interval cannot be negative")

    capture = _open_capture(int(camera_index), capture_factory)
    emit("status", f"Camera {camera_index} opened; analysing live frames...")
    last_analysis = -math.inf
    frame_index = 0
    failed_reads = 0

    try:
        while not cancel.is_set():
            ok, frame = capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 5:
                    raise ValueError("Camera stopped returning frames")
                cancel.wait(0.05)
                continue

            failed_reads = 0
            now = clock()
            if now - last_analysis < analysis_interval:
                continue

            last_analysis = now
            frame_index += 1
            original = bgr_to_pil(frame)
            prediction, result = predict(original)
            annotated, objects = annotate_frame(
                original,
                prediction,
                result,
                method_label=method_label,
            )
            emit(
                "camera_frame",
                {
                    "original": original,
                    "annotated": annotated,
                    "frame_index": frame_index,
                    "method": method,
                    "objects_detected": objects,
                    **prediction,
                },
            )
    finally:
        capture.release()


def _video_properties(capture) -> tuple[float, int, int, int]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0:
        raise ValueError("Video reports invalid frame dimensions")
    if width * height > MAX_FRAME_PIXELS:
        raise ValueError("Video frames exceed the 20-megapixel safety limit")
    return fps, width, height, total


def _temporary_video_path(target: Path) -> Path:
    return target.with_name(
        f".{target.stem}.{uuid.uuid4().hex}.partial{target.suffix.lower()}"
    )


def process_video(
    source: Path,
    output: Path,
    *,
    predict,
    method: str,
    method_label: str,
    cancel,
    emit,
    capture_factory=cv2.VideoCapture,
    writer_factory=cv2.VideoWriter,
    fourcc_factory=cv2.VideoWriter_fourcc,
) -> list[dict]:
    """Analyse every decoded frame and atomically create an annotated video."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_file():
        raise ValueError("Choose an existing video file")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video type; use MP4, AVI, MOV or MKV")
    if output.suffix.lower() not in VIDEO_OUTPUT_EXTENSIONS:
        raise ValueError("Annotated output must be MP4 or AVI")
    if output == source:
        raise ValueError("Annotated output cannot replace the source video")
    if output.exists():
        raise ValueError("Output video already exists; choose a new filename")

    output.parent.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(str(source), capture_factory)
    temporary = _temporary_video_path(output)
    writer = None
    rows: list[dict] = []

    try:
        fps, width, height, total = _video_properties(capture)
        codec = "mp4v" if output.suffix.lower() == ".mp4" else "MJPG"
        writer = writer_factory(
            str(temporary),
            fourcc_factory(*codec),
            fps,
            (width, height),
        )
        if writer is None or not writer.isOpened():
            raise ValueError(f"Unable to create annotated {output.suffix} video")

        frame_index = 0
        while not cancel.is_set():
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            original = bgr_to_pil(frame)
            prediction, result = predict(original)
            annotated, objects = annotate_frame(
                original,
                prediction,
                result,
                method_label=method_label,
            )
            writer.write(pil_to_bgr(annotated))
            row = {
                "source": str(source),
                "output_video": str(output),
                "frame_index": frame_index,
                "timestamp_seconds": (frame_index - 1) / fps,
                "method": method,
                "objects_detected": objects,
                **prediction,
            }
            rows.append(row)
            emit("video_frame", (row, annotated, total))

        if frame_index == 0 and not cancel.is_set():
            raise ValueError("The selected video contains no decodable frames")

        writer.release()
        writer = None
        capture.release()
        capture = None

        if cancel.is_set():
            if temporary.exists():
                temporary.unlink()
            emit("status", "Video processing cancelled; incomplete output was discarded.")
            return rows

        temporary.replace(output)
        emit("video_completed", str(output))
        return rows

    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if temporary.exists():
            temporary.unlink()
