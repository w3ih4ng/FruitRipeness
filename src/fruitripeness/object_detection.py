"""Object detection using contours from a segmentation mask.

Finds every sufficiently large connected foreground object and returns:
- bounding box
- contour
- area
- centroid

This module performs detection only. Classification is handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw


OBJECT_DETECTION_SPEC = {
    "version": "external_contours_v1",
    "retrieval_mode": "cv2.RETR_EXTERNAL",
    "approximation": "cv2.CHAIN_APPROX_SIMPLE",
    "min_area_fraction": 0.005,
    "min_area_pixels": 16,
    "sort_order": "area descending",
}


@dataclass
class DetectedObject:
    contour: np.ndarray
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    centroid: tuple[float, float]


def detect_objects(
    mask: np.ndarray,
    spec: dict | None = None,
) -> list[DetectedObject]:
    """Detect objects from a binary foreground mask."""

    spec = spec or OBJECT_DETECTION_SPEC

    # Make sure mask is a valid binary image.
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D grayscale/binary image.")

    if not mask.any():
        return []

    # Convert to uint8 binary mask.
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Find external contours.
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Minimum object area.
    min_area = max(
        spec["min_area_pixels"],
        int(np.ceil(mask.size * spec["min_area_fraction"])),
    )

    objects: list[DetectedObject] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Bounding box.
        x, y, w, h = cv2.boundingRect(contour)

        # Centroid.
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2
            cy = y + h / 2

        objects.append(
            DetectedObject(
                contour=contour,
                bounding_box=(x, y, w, h),
                area_px=int(area),
                centroid=(cx, cy),
            )
        )

    # Largest object first.
    objects.sort(
        key=lambda obj: obj.area_px,
        reverse=True,
    )

    return objects


def draw_detections(
    image: Image.Image,
    objects: list[DetectedObject],
) -> Image.Image:
    """Draw bounding boxes and contours on the image."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for i, obj in enumerate(objects, start=1):

        x, y, w, h = obj.bounding_box

        # Bounding box.
        draw.rectangle(
            (x, y, x + w, y + h),
            outline=(255, 40, 40),
            width=3,
        )

        # Contour.
        points = [
            tuple(point)
            for point in obj.contour.reshape(-1, 2)
        ]

        if len(points) >= 2:
            draw.line(
                points + [points[0]],
                fill=(40, 200, 255),
                width=2,
            )

        # Object number.
        draw.text(
            (x + 5, y + 5),
            f"Object {i}",
            fill=(255, 255, 0),
        )

    return canvas