"""Preprocessing: noise removal and contrast enhancement, applied to pixels only.

Deliberately kept separate from processing.py's frozen segmentation methods —
those specs are already locked in saved models/runs. This module is meant to
run BEFORE segmentation, as an optional, independently-toggleable step, so
existing reproducibility guarantees for hsv/otsu/kmeans/grabcut/watershed/
hybrid/hybrid_refined are untouched.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

PREPROCESS_SPEC = {
    "version": "median_denoise_percentile_stretch_v1",
    "denoise_method": "median",
    "median_kernel": 3,
    "gaussian_kernel": 3,
    "gaussian_sigma": 0.0,
    "stretch_low_percentile": 2.0,
    "stretch_high_percentile": 98.0,
    "stretch_channels": "per-channel RGB, independent percentiles",
    "flat_channel_policy": "channel left unchanged if low percentile == high percentile",
}


def denoise(rgb: np.ndarray, spec: dict) -> np.ndarray:
    if spec["denoise_method"] == "median":
        k = spec["median_kernel"]
        if k % 2 == 0 or k < 3:
            raise ValueError("median_kernel must be an odd integer >= 3")
        return cv2.medianBlur(rgb, k)
    if spec["denoise_method"] == "gaussian":
        k = spec["gaussian_kernel"]
        if k % 2 == 0 or k < 3:
            raise ValueError("gaussian_kernel must be an odd integer >= 3")
        return cv2.GaussianBlur(rgb, (k, k), spec["gaussian_sigma"])
    raise ValueError(f"Unsupported denoise_method: {spec['denoise_method']}")


def contrast_stretch(rgb: np.ndarray, spec: dict) -> np.ndarray:
    """Classic per-channel percentile linear stretch to the full 0-255 range."""
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        channel = out[:, :, c]
        low, high = np.percentile(channel, [spec["stretch_low_percentile"], spec["stretch_high_percentile"]])
        if high <= low:
            continue  # near-constant channel; stretching would amplify noise, not signal
        out[:, :, c] = np.clip((channel-low) * (255.0/(high-low)), 0, 255)
    return out.astype(np.uint8)


def enhance(image: Image.Image, spec: dict | None = None) -> tuple[Image.Image, dict]:
    """Denoise then contrast-stretch. Returns the processed image and before/after stats."""
    spec = spec or PREPROCESS_SPEC
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    before_std = float(rgb.std())
    denoised = denoise(rgb, spec)
    stretched = contrast_stretch(denoised, spec)
    details = {
        "denoise_method": spec["denoise_method"],
        "std_before": before_std,
        "std_after_denoise": float(denoised.std()),
        "std_after_stretch": float(stretched.std()),
        "mean_absolute_change": float(np.abs(stretched.astype(np.int16)-rgb.astype(np.int16)).mean()),
    }
    return Image.fromarray(stretched), details