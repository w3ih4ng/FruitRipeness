"""Per-fruit detection and ripeness inference for interactive inputs.

This module is deliberately separate from validation and final-Test code.  It
uses a pretrained open-vocabulary instance-segmentation model to localise the
five project fruits, then applies the selected project processing method and
the already-trained ripeness classifier to every detected fruit crop.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FRUIT_CLASSES = ("apple", "banana", "mango", "orange", "tomato")
DEFAULT_DETECTOR_MODEL = "yoloe-26n-seg.pt"
DEFAULT_CONFIDENCE = 0.20


@dataclass(frozen=True)
class FruitDetection:
    fruit: str
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass
class PerFruitAnalysis:
    original: Image.Image
    annotated: Image.Image
    objects: list[dict]
    detection_ms: float


class YOLOEFruitDetector:
    """Lazy YOLOE adapter restricted to the five assignment fruits."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        image_size: int = 640,
    ):
        if not 0 < confidence <= 1:
            raise ValueError("Fruit detection confidence must be in (0, 1]")
        self.model_name = str(
            model
            or os.environ.get("FRUIT_DETECTOR_MODEL")
            or DEFAULT_DETECTOR_MODEL
        )
        self.confidence = confidence
        self.image_size = image_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError(
                "Per-fruit detection is not installed. Run: "
                ".\\.venv\\Scripts\\python.exe -m pip install "
                "-r requirements-detector.txt"
            ) from exc
        try:
            model = YOLOE(self.model_name)
            model.set_classes(list(FRUIT_CLASSES))
        except Exception as exc:
            raise RuntimeError(
                "Unable to prepare the five-fruit detector. The first run "
                "needs internet access to download YOLOE and its text "
                "encoder. Run `python -m fruitripeness.fruit_detection "
                "--prepare` once, then restart the UI."
            ) from exc
        self._model = model
        return model

    def detect(self, image: Image.Image) -> list[FruitDetection]:
        original = image.convert("RGB")
        with self._lock:
            result = self._load().predict(
                source=original,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
            )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        polygons = result.masks.xy if result.masks is not None else []
        found: list[FruitDetection] = []

        for index, (coords, score, class_id) in enumerate(
            zip(boxes, scores, classes)
        ):
            fruit = str(
                names.get(class_id, class_id)
                if isinstance(names, dict)
                else names[class_id]
            ).lower().strip()
            if fruit not in FRUIT_CLASSES:
                continue
            x1, y1, x2, y2 = _clamp_box(coords, original.size)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            mask = np.zeros((original.height, original.width), dtype=bool)
            if index < len(polygons) and len(polygons[index]) >= 3:
                layer = Image.new("1", original.size, 0)
                points = [tuple(map(float, point)) for point in polygons[index]]
                ImageDraw.Draw(layer).polygon(points, fill=1)
                mask = np.asarray(layer, dtype=bool)
            else:
                mask[y1:y2, x1:x2] = True
            found.append(
                FruitDetection(fruit, float(score), (x1, y1, x2, y2), mask)
            )
        return found


_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def shared_fruit_detector() -> YOLOEFruitDetector:
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = YOLOEFruitDetector()
        return _DETECTOR


def _clamp_box(coords, size) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = map(float, coords)
    return (
        max(0, min(width - 1, int(np.floor(x1)))),
        max(0, min(height - 1, int(np.floor(y1)))),
        max(1, min(width, int(np.ceil(x2)))),
        max(1, min(height, int(np.ceil(y2)))),
    )


def isolated_crop(image: Image.Image, detection: FruitDetection) -> Image.Image:
    """Return one fruit crop, using its instance mask to remove hands/background."""
    x1, y1, x2, y2 = detection.box
    crop = image.convert("RGB").crop((x1, y1, x2, y2))
    mask = Image.fromarray(
        detection.mask[y1:y2, x1:x2].astype(np.uint8) * 255,
        mode="L",
    )
    isolated = Image.new("RGB", crop.size, "black")
    isolated.paste(crop, mask=mask)
    return isolated


def analyse_fruits(
    image: Image.Image,
    bundle: dict,
    detector=None,
    *,
    detections: list[FruitDetection] | None = None,
    detection_ms: float | None = None,
):
    """Detect and independently classify all supported fruit instances."""
    from .ui_core import infer

    original = image.convert("RGB")
    detector = detector or shared_fruit_detector()
    if detections is None:
        started = time.perf_counter()
        detections = detector.detect(original)
        detection_ms = (time.perf_counter() - started) * 1000
    elif detection_ms is None:
        detection_ms = 0.0
    objects: list[dict] = []

    for object_id, detection in enumerate(detections, start=1):
        crop = isolated_crop(original, detection)
        prediction, processing = infer(crop, bundle)
        objects.append(
            {
                "object_id": object_id,
                "fruit": detection.fruit,
                "detection_confidence": detection.confidence,
                "box": detection.box,
                "crop": crop,
                "processing": processing,
                **prediction,
            }
        )

    annotated = draw_fruit_results(original, objects)
    row = aggregate_predictions(objects, detection_ms)
    return row, PerFruitAnalysis(original, annotated, objects, detection_ms)


def aggregate_predictions(objects: list[dict], detection_ms: float) -> dict:
    if objects:
        stages = {item["predicted_stage"] for item in objects}
        stage = next(iter(stages)) if len(stages) == 1 else "mixed"
        scores = {
            name: float(np.mean([item[f"score_{name}"] for item in objects]))
            for name in ("unripe", "ripe", "overripe")
        }
    else:
        stage = "none"
        scores = {name: 0.0 for name in ("unripe", "ripe", "overripe")}

    return {
        "predicted_stage": stage,
        **{f"score_{name}": value for name, value in scores.items()},
        "processing_ms": sum(item.get("processing_ms", 0.0) for item in objects),
        "feature_ms": sum(item.get("feature_ms", 0.0) for item in objects),
        "prediction_ms": sum(item.get("prediction_ms", 0.0) for item in objects),
        "detection_ms": detection_ms,
        "objects_detected": len(objects),
        "detected_fruits": ", ".join(item["fruit"] for item in objects),
        "object_results_json": json.dumps(
            [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"crop", "processing"}
                }
                for item in objects
            ],
            separators=(",", ":"),
        ),
        "foreground_fraction": float(
            np.mean([item.get("foreground_fraction", 0.0) for item in objects])
        ) if objects else 0.0,
        "mask_status": "per-fruit" if objects else "no-supported-fruit",
        "refined_review_flag": "" if objects else "no_supported_fruit_detected",
    }


def draw_fruit_results(image: Image.Image, objects: list[dict]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(12, min(canvas.width, canvas.height) // 35)
    font = ImageFont.load_default(size=font_size)
    line = max(2, min(canvas.width, canvas.height) // 180)
    for item in objects:
        x1, y1, x2, y2 = item["box"]
        label = (
            f"{item['object_id']}: {item['fruit']} "
            f"{item['detection_confidence']:.0%} | "
            f"{item['predicted_stage'].upper()}"
        )
        draw.rectangle((x1, y1, x2, y2), outline="#00d6a3", width=line)
        box = draw.textbbox((x1, y1), label, font=font, stroke_width=1)
        top = max(0, y1 - (box[3] - box[1]) - 6)
        draw.rectangle((x1, top, min(canvas.width, x1 + box[2] - box[0] + 8), y1), fill="#173f31")
        draw.text((x1 + 4, top + 2), label, fill="white", font=font)
    if not objects:
        message = "No supported fruit detected (apple, banana, mango, orange or tomato)"
        draw.rectangle((0, 0, canvas.width, font_size + 16), fill="#6d2f2f")
        draw.text((8, 7), message, fill="white", font=font)
    return canvas


def per_fruit_card(row: dict, analysis: PerFruitAnalysis) -> Image.Image:
    """Show the overview, processed image and method mask without hiding them."""
    objects = analysis.objects
    title = ImageFont.load_default(size=23)
    text_font = ImageFont.load_default(size=15)

    if not objects:
        canvas = Image.new("RGB", (1100, 390), "white")
        draw = ImageDraw.Draw(canvas)
        _draw_card_header(draw, row, analysis, title, text_font)
        annotated = ImageOps.contain(analysis.annotated, (1060, 270))
        canvas.paste(annotated, ((1100 - annotated.width) // 2, 82))
        return canvas

    if len(objects) == 1:
        canvas = Image.new("RGB", (1100, 560), "white")
        draw = ImageDraw.Draw(canvas)
        _draw_card_header(draw, row, analysis, title, text_font)

        draw.text((20, 79), "Detection overview", fill="#555555", font=text_font)
        annotated = ImageOps.contain(analysis.annotated, (325, 410))
        canvas.paste(
            annotated,
            (20 + (325 - annotated.width) // 2, 108 + (410 - annotated.height) // 2),
        )
        _draw_object_card(
            canvas,
            draw,
            objects[0],
            box=(370, 78, 710, 450),
            panel_size=(205, 285),
            title_font=title,
            text_font=text_font,
        )
        return canvas

    cols = 2
    object_height = 270
    rows = (len(objects) + cols - 1) // cols
    canvas = Image.new("RGB", (1100, 250 + rows * object_height), "white")
    draw = ImageDraw.Draw(canvas)
    _draw_card_header(draw, row, analysis, title, text_font)
    annotated = ImageOps.contain(analysis.annotated, (1060, 150))
    canvas.paste(annotated, ((1100 - annotated.width) // 2, 82))

    for index, item in enumerate(objects):
        col = index % cols
        row_index = index // cols
        x = 20 + col * 540
        y = 250 + row_index * object_height
        _draw_object_card(
            canvas,
            draw,
            item,
            box=(x, y, 520, 250),
            panel_size=(155, 125),
            title_font=title,
            text_font=text_font,
        )
    return canvas


def _draw_card_header(draw, row, analysis, title_font, text_font) -> None:
    draw.text(
        (20, 12),
        f"{row['method_label']} | PER-FRUIT RESULTS",
        fill="#214737",
        font=title_font,
    )
    draw.text(
        (20, 46),
        f"Detected {len(analysis.objects)} fruit(s) in "
        f"{analysis.detection_ms:.1f} ms. Each fruit is processed separately.",
        fill="#333333",
        font=text_font,
    )


def _draw_object_card(
    canvas,
    draw,
    item,
    *,
    box,
    panel_size,
    title_font,
    text_font,
) -> None:
    x, y, width, height = box
    draw.rectangle((x, y, x + width, y + height), outline="#c9d7d0", width=2)
    heading = (
        f"Object {item['object_id']}: {item['fruit'].title()} "
        f"({item['detection_confidence']:.0%}) | "
        f"{item['predicted_stage'].upper()}"
    )
    draw.text((x + 10, y + 8), heading, fill="#214737", font=title_font)

    panels = [
        ("Original crop", item["crop"]),
        ("Processed", item["processing"].processed),
        (
            "Foreground mask",
            Image.fromarray(item["processing"].mask.astype(np.uint8) * 255),
        ),
    ]
    panel_width, panel_height = panel_size
    gap = max(8, (width - 20 - panel_width * 3) // 2)
    panel_top = y + 72
    for panel_index, (label, panel) in enumerate(panels):
        px = x + 10 + panel_index * (panel_width + gap)
        draw.text((px, y + 43), label, fill="#555555", font=text_font)
        thumb = ImageOps.contain(panel.convert("RGB"), (panel_width, panel_height))
        canvas.paste(
            thumb,
            (
                px + (panel_width - thumb.width) // 2,
                panel_top + (panel_height - thumb.height) // 2,
            ),
        )

    detail_y = panel_top + panel_height + 12
    scores = " | ".join(
        f"{name} {item['score_' + name]:.1%}"
        for name in ("unripe", "ripe", "overripe")
    )
    draw.text((x + 10, detail_y), scores, fill="#111111", font=text_font)
    draw.text(
        (x + 10, detail_y + 23),
        f"Processing {item['processing_ms']:.1f} ms | "
        f"CNN {item['prediction_ms']:.1f} ms | "
        f"Mask {item['mask_status']}",
        fill="#555555",
        font=text_font,
    )


def prepare_detector() -> None:
    detector = shared_fruit_detector()
    detector.detect(Image.new("RGB", (640, 480), "white"))
    print(f"Five-fruit detector ready: {detector.model_name}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Prepare five-fruit YOLOE detector")
    parser.add_argument("--prepare", action="store_true", help="download and initialise detector assets")
    args = parser.parse_args(argv)
    if not args.prepare:
        parser.error("use --prepare")
    prepare_detector()


if __name__ == "__main__":
    main()
