"""Image calibration: convert pixel measurements to physical units.

This dataset ships no reference object (ruler, checkerboard, coin) in any
frame, so measurements are pixel-based by default and explicitly marked
uncalibrated. If a reference object of KNOWN physical size is available
(e.g. a checkerboard square, a coin, a marked tray edge), pass its pixel
width via `pixels_per_unit_from_reference()` to get real-world measurements.
Say this plainly in your report: without a physical reference in-frame,
"calibration" can only be relative (px), not absolute (cm/mm).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CALIBRATION_SPEC = {
    "version": "pixel_measurement_v1",
    "perimeter_source": "cv2.findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE, largest external contour",
    "equivalent_diameter_formula": "2 * sqrt(area_px / pi)",
    "default_unit": "px",
    "calibrated_requires": "an explicit pixels_per_unit from a known-size reference object in frame",
}


@dataclass
class CalibrationResult:
    area_px: int
    perimeter_px: float
    equivalent_diameter_px: float
    bounding_box_px: tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float
    calibrated: bool
    unit: str
    pixels_per_unit: float | None
    area_physical: float | None
    equivalent_diameter_physical: float | None


def pixels_per_unit_from_reference(reference_length_px: float, known_physical_length: float) -> float:
    """Scale factor from a reference object of known real-world size in the same frame."""
    if reference_length_px <= 0 or known_physical_length <= 0:
        raise ValueError("Reference lengths must be positive")
    return reference_length_px / known_physical_length


def measure(mask: np.ndarray, *, pixels_per_unit: float | None = None, unit: str = "cm",
           spec: dict | None = None) -> CalibrationResult:
    """Pixel-space measurements from a foreground mask; physical units only if calibrated."""
    spec = spec or CALIBRATION_SPEC
    if not mask.any():
        raise ValueError("Cannot calibrate an empty mask; nothing to measure")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    area_px = int(mask.sum())  # exact pixel count; more reliable than contour polygon area
    perimeter_px = float(cv2.arcLength(largest, closed=True))
    equivalent_diameter_px = float(2*np.sqrt(area_px/np.pi))
    x, y, w, h = cv2.boundingRect(largest)
    calibrated = pixels_per_unit is not None
    area_physical = equivalent_diameter_physical = None
    if calibrated:
        if pixels_per_unit <= 0:
            raise ValueError("pixels_per_unit must be positive")
        area_physical = area_px / (pixels_per_unit**2)
        equivalent_diameter_physical = equivalent_diameter_px / pixels_per_unit
    return CalibrationResult(area_px, perimeter_px, equivalent_diameter_px, (x, y, w, h),
                             float(w/h) if h else 0.0, calibrated, unit if calibrated else "px",
                             pixels_per_unit, area_physical, equivalent_diameter_physical)