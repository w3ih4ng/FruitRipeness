"""Deterministic image-only transforms shared by training, previews and inference.

HSV v1 is a foreground heuristic, not a verified fruit detector. All hues are
accepted; ripeness labels and fruit names must never be passed to this module.
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


def processing_spec(method: str) -> dict:
    if method == "baseline":
        return {"version": "identity_rgb_v1"}
    if method == "hsv":
        # Return independent JSON-compatible metadata; callers cannot mutate defaults.
        return {**HSV_SPEC, "background_rgb": list(HSV_SPEC["background_rgb"])}
    raise ValueError(f"Unsupported method: {method}")


@dataclass
class ProcessingResult:
    original: Image.Image
    processed: Image.Image
    mask: np.ndarray
    details: dict


def process_image(image: Image.Image, method: str) -> ProcessingResult:
    spec = processing_spec(method)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if method == "baseline":
        mask = np.ones((rgb.height, rgb.width), dtype=bool)
        return ProcessingResult(rgb, rgb.copy(), mask,
                                {"foreground_fraction": 1.0, "mask_status": "not_segmented"})
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    mask = (hsv[:, :, 1] >= spec["saturation_min"]) & (hsv[:, :, 2] >= spec["value_min"])
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
                            {"foreground_fraction": fraction, "mask_status": status})
