"""Deterministic image-only transforms shared by training, previews and inference.

These methods are foreground heuristics, not verified fruit detectors.
Ripeness labels and fruit names must never be passed to this module.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

HSV_SPEC = {
    "version": "hsv_sv_foreground_v1",
    "saturation_min": 0.20,
    "value_min": 0.10,
    "hue_range": "all",
    "opening_kernel": 3,
    "closing_kernel": 5,
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "fill_enclosed_holes": True,
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no image exclusion",
    "working_size": "original image resolution; dataset images are 300x300",
}

OTSU_SPEC = {
    "version": "otsu_gray_border_v1",
    "channel": "Pillow L grayscale, uint8",
    "threshold_rule": "maximise between-class variance; first maximiser",
    "border_fraction": 0.05,
    "foreground_rule": "class occupying fewer border pixels; tie chooses smaller class; final tie chooses bright",
    "constant_image_policy": "empty mask, no separation possible",
    "opening_kernel": 3,
    "closing_kernel": 5,
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "fill_enclosed_holes": True,
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no image exclusion",
    "working_size": "original image resolution; dataset images are 300x300",
}


def processing_spec(method: str) -> dict:
    if method == "baseline":
        return {"version": "identity_rgb_v1"}
    if method == "hsv":
        # Return independent JSON-compatible metadata; callers cannot mutate defaults.
        return {**HSV_SPEC, "background_rgb": list(HSV_SPEC["background_rgb"])}
    if method == "otsu":
        return {**OTSU_SPEC, "background_rgb": list(OTSU_SPEC["background_rgb"])}
    raise ValueError(f"Unsupported method: {method}")


@dataclass
class ProcessingResult:
    original: Image.Image
    processed: Image.Image
    mask: np.ndarray
    details: dict


def otsu_threshold(gray: np.ndarray) -> int | None:
    """256-bin Otsu threshold, with <= t dark and > t bright.

    None explicitly represents a constant image with no two-class separation.
    Implemented in NumPy to avoid requiring another image-processing dependency.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("Otsu expects a nonempty 2D uint8 grayscale image")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) < 2:
        return None
    weight = np.cumsum(hist)[:-1]
    moment = np.cumsum(hist * np.arange(256))[:-1]
    total = hist.sum()
    total_moment = np.dot(hist, np.arange(256))
    valid = (weight > 0) & (weight < total)
    score = np.full(255, -np.inf)
    score[valid] = (total*moment[valid]-weight[valid]*total_moment)**2 / (weight[valid]*(total-weight[valid]))
    return int(np.argmax(score))


def otsu_candidate_mask(rgb: Image.Image, spec: dict) -> tuple[np.ndarray, dict]:
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(gray)
    if threshold is None:
        return np.zeros(gray.shape, dtype=bool), {"otsu_threshold": None, "foreground_polarity": "none"}
    bright = gray > threshold
    # The class dominating the image frame is assumed to be background. This
    # assumption can fail when fruit touches the frame or backgrounds are complex.
    width = max(1, int(round(min(gray.shape)*spec["border_fraction"])))
    border = np.zeros(gray.shape, dtype=bool)
    border[:width, :] = border[-width:, :] = True
    border[:, :width] = border[:, -width:] = True
    bright_border = int(bright[border].sum())
    border_pixels = int(border.sum())
    if 2*bright_border == border_pixels:
        use_bright = int(bright.sum())*2 <= bright.size
    else:
        use_bright = 2*bright_border < border_pixels
    return (bright if use_bright else ~bright), {
        "otsu_threshold": threshold,
        "foreground_polarity": "bright" if use_bright else "dark",
    }


def process_image(image: Image.Image, method: str) -> ProcessingResult:
    spec = processing_spec(method)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if method == "baseline":
        mask = np.ones((rgb.height, rgb.width), dtype=bool)
        return ProcessingResult(rgb, rgb.copy(), mask,
                                {"foreground_fraction": 1.0, "mask_status": "not_segmented"})
    details = {}
    if method == "hsv":
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
        mask = (hsv[:, :, 1] >= spec["saturation_min"]) & (hsv[:, :, 2] >= spec["value_min"])
    else:
        mask, details = otsu_candidate_mask(rgb, spec)
    # Edge padding prevents artificial black rims on fully coloured images.
    pad = spec["closing_kernel"]
    padded = np.pad(mask, pad, mode="edge")
    opening = spec["opening_kernel"]
    closing = spec["closing_kernel"]
    padded = ndimage.binary_opening(padded, structure=np.ones((opening, opening), dtype=bool))
    padded = ndimage.binary_closing(padded, structure=np.ones((closing, closing), dtype=bool))
    mask = padded[pad:-pad, pad:-pad]
    if spec["fill_enclosed_holes"]:
        mask = ndimage.binary_fill_holes(mask)
    components, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())
    keep = sizes >= max(spec["min_component_pixels"], int(np.ceil(mask.size * spec["min_component_fraction"])))
    keep[0] = False
    mask = keep[components]
    pixels = np.asarray(rgb).copy()
    pixels[~mask] = spec["background_rgb"]
    fraction = float(mask.mean())
    status = "empty" if fraction == 0 else "near_full" if fraction > 0.98 else "nonempty"
    return ProcessingResult(rgb, Image.fromarray(pixels), mask,
                            {"foreground_fraction": fraction, "mask_status": status, **details})
