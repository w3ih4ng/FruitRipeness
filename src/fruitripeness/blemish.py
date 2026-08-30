"""
Blemish / surface-damage quantification.

The selected segmentation method first identifies the fruit foreground.
Blemish detection then searches only inside the detected fruit area for
dark or colour-shifted surface regions.

This is a heuristic surface-quality detector, not a trained defect classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .audit import EXTENSIONS, write_csv
from .baseline import (
    assign_splits,
    collect_records,
    resolve_dataset_root,
)
from .processing import process_image


# ================================================================
# SPECIFICATION
# ================================================================

BLEMISH_SPEC = {
    "version": "lab_instance_aware_v2",

    # IMPORTANT:
    # This is only the fallback.
    # The UI-selected method is passed into detect_blemishes().
    "base_segmentation_method": "hybrid_refined",

    "colour_space": "OpenCV Lab + HSV",

    # Dark-region detection
    "lightness_darkness_threshold": 3.0,

    # Colour-shift detection
    "colour_shift_threshold": 3.5,

    # Robust statistics
    "robust_scale": (
        "median absolute deviation, "
        "normal-consistent x1.4826"
    ),

    "mad_floor": 2.0,

    # Morphology
    "opening_kernel": 3,
    "closing_kernel": 3,

    # Candidate regions
    "min_component_fraction": 0.0003,
    "min_component_pixels": 12,

    # Fruit requirements
    "min_foreground_pixels": 200,

    # Stem/calyx suppression
    "stem_max_area_fraction": 0.012,
    "stem_max_compactness": 0.70,
    "stem_min_centrality": 0.55,
    "stem_max_darkness": 8.0,

    # Avoid treating large dark shadows as tiny blemishes
    "max_blemish_component_fraction": 0.18,

    "too_small_policy": (
        "blemish_fraction is None; "
        "not zero, to avoid a false clean reading"
    ),

    "empty_foreground_policy": (
        "blemish_fraction is None; "
        "nothing to grade"
    ),
}


# These thresholds are a transparent project grading rule, not a biological
# or commercial standard. The dataset contains no verified surface-damage
# annotations from which a validated quality scale could be learned.
SURFACE_QUALITY_SPEC = {
    "version": "blemish_fraction_grade_v1",
    "good_max_fraction": 0.05,
    "acceptable_max_fraction": 0.15,
    "labels": ["Good", "Acceptable", "Poor", "Not graded"],
    "basis": "detected blemish pixels / detected fruit foreground pixels",
    "limitation": "heuristic project grade; not a commercial inspection standard",
}


def surface_quality_grade(
    blemish_fraction: float | None,
    *,
    status: str = "graded",
    spec: dict | None = None,
) -> str:
    """Convert a valid blemish fraction into a documented project grade."""

    spec = spec or SURFACE_QUALITY_SPEC

    if status != "graded" or blemish_fraction is None:
        return "Not graded"

    fraction = float(blemish_fraction)

    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("blemish_fraction must be between 0 and 1")

    if fraction <= float(spec["good_max_fraction"]):
        return "Good"

    if fraction <= float(spec["acceptable_max_fraction"]):
        return "Acceptable"

    return "Poor"


# ================================================================
# RESULT
# ================================================================

@dataclass
class BlemishResult:

    foreground_mask: np.ndarray

    blemish_mask: np.ndarray

    foreground_pixels: int

    blemish_pixels: int

    blemish_fraction: float | None

    status: str

    details: dict


# ================================================================
# ROBUST STATISTICS
# ================================================================

def _mad_z(
    values: np.ndarray,
    spec: dict,
) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    """

    if values.size == 0:
        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    median = np.median(values)

    mad = (
        np.median(
            np.abs(values - median)
        )
        * 1.4826
    )

    scale = max(
        float(mad),
        float(spec["mad_floor"]),
    )

    return (
        (values - median)
        / scale
    )


# ================================================================
# COMPONENT HELPERS
# ================================================================

def _component_properties(
    mask: np.ndarray,
    labels: np.ndarray,
    label_id: int,
    centroid_y: float,
    centroid_x: float,
):
    """
    Calculate useful geometric properties for one candidate component.
    """

    ys, xs = np.where(
        labels == label_id
    )

    area = len(xs)

    if area == 0:
        return None

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    perimeter_mask = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    perimeter_mask[
        labels == label_id
    ] = 255

    contours, _ = cv2.findContours(
        perimeter_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:

        perimeter = float(
            cv2.arcLength(
                max(
                    contours,
                    key=cv2.contourArea,
                ),
                True,
            )
        )

    else:

        perimeter = 0.0

    if perimeter > 0:

        compactness = (
            4.0
            * np.pi
            * area
            / (perimeter * perimeter)
        )

    else:

        compactness = 0.0

    component_cx = float(
        xs.mean()
    )

    component_cy = float(
        ys.mean()
    )

    fruit_height, fruit_width = (
        mask.shape
    )

    dx = (
        component_cx
        - centroid_x
    )

    dy = (
        component_cy
        - centroid_y
    )

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    fruit_radius = max(
        1.0,
        min(
            fruit_width,
            fruit_height,
        ) / 2.0,
    )

    centrality = 1.0 - min(
        distance / fruit_radius,
        1.0,
    )

    return {
        "area": area,
        "width": width,
        "height": height,
        "compactness": float(
            compactness
        ),
        "centrality": float(
            centrality
        ),
        "centroid_x": component_cx,
        "centroid_y": component_cy,
    }


# ================================================================
# STEM / CALYX SUPPRESSION
# ================================================================

def _looks_like_stem_cavity(
    properties: dict,
    darkness: float,
    foreground_pixels: int,
    spec: dict,
) -> bool:
    """
    Reject small, compact, centrally located dark regions.

    This is deliberately conservative. It does NOT remove every
    central dark region because real bruises can also occur near
    the fruit centre.
    """

    area_fraction = (
        properties["area"]
        / max(
            foreground_pixels,
            1,
        )
    )

    is_small = (
        area_fraction
        <= spec[
            "stem_max_area_fraction"
        ]
    )

    is_central = (
        properties["centrality"]
        >= spec[
            "stem_min_centrality"
        ]
    )

    is_compact = (
        properties["compactness"]
        >= spec[
            "stem_max_compactness"
        ]
    )

    is_dark = (
        darkness
        <= spec[
            "stem_max_darkness"
        ]
    )

    return (
        is_small
        and is_central
        and is_compact
        and is_dark
    )


# ================================================================
# MAIN BLEMISH MAP
# ================================================================

def blemish_map(
    rgb: Image.Image,
    foreground: np.ndarray,
    spec: dict,
) -> tuple[np.ndarray, dict]:
    """
    Detect suspicious dark / colour-shifted regions inside the fruit.

    The fruit foreground itself is NOT treated as a blemish.
    """

    foreground = (
        foreground.astype(bool)
    )

    foreground_pixels = int(
        foreground.sum()
    )

    details = {
        "foreground_pixels":
            foreground_pixels,
    }

    if (
        foreground_pixels
        < spec["min_foreground_pixels"]
    ):

        return (
            np.zeros(
                foreground.shape,
                dtype=bool,
            ),
            {
                **details,
                "blemish_status":
                    "too_small_foreground",
            },
        )

    np_rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    lab = cv2.cvtColor(
        np_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    hsv = cv2.cvtColor(
        np_rgb,
        cv2.COLOR_RGB2HSV,
    )

    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    # ------------------------------------------------------------
    # Fruit statistics
    # ------------------------------------------------------------

    foreground_L = L[
        foreground
    ]

    foreground_A = A[
        foreground
    ]

    foreground_B = B[
        foreground
    ]

    foreground_H = H[
        foreground
    ]

    foreground_S = S[
        foreground
    ]

    foreground_V = V[
        foreground
    ]

    dark_z = np.zeros_like(
        L,
        dtype=np.float32,
    )

    colour_z = np.zeros_like(
        L,
        dtype=np.float32,
    )

    dark_z_values = (
        -_mad_z(
            foreground_L,
            spec,
        )
    )

    colour_z_values = np.hypot(
        _mad_z(
            foreground_A,
            spec,
        ),
        _mad_z(
            foreground_B,
            spec,
        ),
    )

    dark_z[
        foreground
    ] = dark_z_values

    colour_z[
        foreground
    ] = colour_z_values

    # ------------------------------------------------------------
    # Candidate defect pixels
    # ------------------------------------------------------------

    candidate = (
        (
            dark_z
            > spec[
                "lightness_darkness_threshold"
            ]
        )
        |
        (
            colour_z
            > spec[
                "colour_shift_threshold"
            ]
        )
    )

    candidate &= foreground

    # ------------------------------------------------------------
    # Glare suppression
    # ------------------------------------------------------------

    glare = (
        (V > 210)
        & (S < 50)
    )

    candidate &= ~glare

    # ------------------------------------------------------------
    # Green foliage suppression
    # ------------------------------------------------------------

    leaf = (
        (H >= 35)
        & (H <= 85)
        & (S > 45)
    )

    candidate &= ~leaf

    # ------------------------------------------------------------
    # Morphological cleaning
    # ------------------------------------------------------------

    opening_kernel = int(
        spec["opening_kernel"]
    )

    closing_kernel = int(
        spec["closing_kernel"]
    )

    opened = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones(
            (
                opening_kernel,
                opening_kernel,
            ),
            dtype=np.uint8,
        ),
    ).astype(bool)

    closed = cv2.morphologyEx(
        opened.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones(
            (
                closing_kernel,
                closing_kernel,
            ),
            dtype=np.uint8,
        ),
    ).astype(bool)

    cleaned = (
        closed
        & foreground
    )

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------

    labels, component_count = (
        ndimage.label(
            cleaned,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
        )
    )

    if component_count == 0:

        return (
            np.zeros(
                foreground.shape,
                dtype=bool,
            ),
            {
                **details,
                "blemish_status":
                    "graded",
                "candidate_components":
                    0,
                "removed_components":
                    0,
            },
        )

    sizes = np.bincount(
        labels.ravel()
    )

    minimum_area = max(
        int(
            spec[
                "min_component_pixels"
            ]
        ),
        int(
            np.ceil(
                foreground_pixels
                * spec[
                    "min_component_fraction"
                ]
            )
        ),
    )

    # ------------------------------------------------------------
    # Fruit centroid
    # ------------------------------------------------------------

    ys, xs = np.where(
        foreground
    )

    fruit_centroid_y = float(
        ys.mean()
    )

    fruit_centroid_x = float(
        xs.mean()
    )

    # ------------------------------------------------------------
    # Filter components
    # ------------------------------------------------------------

    final_mask = np.zeros(
        foreground.shape,
        dtype=bool,
    )

    kept_components = 0
    removed_components = 0
    stem_removed = 0
    tiny_removed = 0
    large_removed = 0

    component_details = []

    for label_id in range(
        1,
        component_count + 1,
    ):

        area = int(
            sizes[label_id]
        )

        if area < minimum_area:

            tiny_removed += 1

            continue

        properties = (
            _component_properties(
                cleaned,
                labels,
                label_id,
                fruit_centroid_y,
                fruit_centroid_x,
            )
        )

        if properties is None:
            continue

        area_fraction = (
            properties["area"]
            / max(
                foreground_pixels,
                1,
            )
        )

        # Average darkness of this region
        component_darkness = float(
            np.mean(
                dark_z[
                    labels == label_id
                ]
            )
        )

        # Very large dark areas are more likely to be
        # shadows / segmentation problems than local blemishes.
        if (
            area_fraction
            > spec[
                "max_blemish_component_fraction"
            ]
        ):

            large_removed += 1

            continue

        # Stem / calyx cavity suppression
        if _looks_like_stem_cavity(
            properties,
            component_darkness,
            foreground_pixels,
            spec,
        ):

            stem_removed += 1

            component_details.append(
                {
                    "label": label_id,
                    "reason":
                        "stem_cavity_like",
                    **properties,
                }
            )

            continue

        final_mask[
            labels == label_id
        ] = True

        kept_components += 1

        component_details.append(
            {
                "label": label_id,
                "reason":
                    "kept",
                "darkness":
                    component_darkness,
                **properties,
            }
        )

    blemish_pixels = int(
        final_mask.sum()
    )

    details.update(
        {
            "blemish_pixels":
                blemish_pixels,

            "candidate_components":
                int(component_count),

            "kept_components":
                int(kept_components),

            "tiny_components_removed":
                int(tiny_removed),

            "large_components_removed":
                int(large_removed),

            "stem_cavity_components_removed":
                int(stem_removed),

            "component_details":
                component_details,

            "blemish_status":
                "graded",
        }
    )

    return (
        final_mask,
        details,
    )


# ================================================================
# PUBLIC DETECTOR
# ================================================================

def detect_blemishes(
    image: Image.Image,
    *,
    method: str = BLEMISH_SPEC[
        "base_segmentation_method"
    ],
    spec: dict | None = None,
) -> BlemishResult:
    """
    Segment the fruit using the requested method and then quantify
    suspicious surface regions inside that segmentation.

    The method argument comes directly from the UI Surface Analysis
    dropdown.
    """

    spec = spec or BLEMISH_SPEC

    rgb = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    # ------------------------------------------------------------
    # IMPORTANT:
    # The selected UI method is used here.
    # ------------------------------------------------------------

    segmentation = process_image(
        rgb,
        method,
    )

    foreground = (
        segmentation.mask
        .astype(bool)
    )

    blemish_mask, details = (
        blemish_map(
            rgb,
            foreground,
            spec,
        )
    )

    foreground_pixels = int(
        foreground.sum()
    )

    blemish_pixels = int(
        blemish_mask.sum()
    )

    if (
        details[
            "blemish_status"
        ]
        == "graded"
        and foreground_pixels > 0
    ):

        fraction = (
            blemish_pixels
            / foreground_pixels
        )

    else:

        fraction = None

    return BlemishResult(
        foreground_mask=foreground,
        blemish_mask=blemish_mask,
        foreground_pixels=
            foreground_pixels,
        blemish_pixels=
            blemish_pixels,
        blemish_fraction=
            fraction,
        status=
            details[
                "blemish_status"
            ],
        details={
            **details,

            "segmentation_method":
                method,

            "segmentation_foreground_fraction":
                segmentation.details.get(
                    "foreground_fraction",
                    None,
                ),

            "segmentation_mask_status":
                segmentation.details.get(
                    "mask_status",
                    "",
                ),
        },
    )


# ================================================================
# OVERLAY
# ================================================================

def blemish_overlay(
    image: Image.Image,
    result: BlemishResult,
) -> Image.Image:
    """
    Display detected blemishes in red.

    The fruit itself remains in its original colour.
    """

    rgb = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    pixels = np.asarray(
        rgb
    ).copy()

    # Red = suspected blemish only
    pixels[
        result.blemish_mask
    ] = [
        255,
        40,
        40,
    ]

    overlay = Image.fromarray(
        pixels
    )

    if result.status != "graded":

        draw = ImageDraw.Draw(
            overlay
        )

        draw.text(
            (4, 4),
            result.status.replace(
                "_",
                " ",
            ),
            fill=(
                255,
                255,
                0,
            ),
        )

    return overlay


# ================================================================
# REPORT
# ================================================================

def blemish_report(
    data: Path,
    out: Path,
    *,
    split: str = "validation",
    method: str = BLEMISH_SPEC[
        "base_segmentation_method"
    ],
) -> dict:
    """
    Grade every image in one dataset split.

    No trained classifier is required.
    """

    root = resolve_dataset_root(
        data
    )

    out = Path(
        out
    ).resolve()

    if out.is_relative_to(
        root
    ):

        raise ValueError(
            "Write blemish report outside "
            "the source dataset folder"
        )

    records = assign_splits(
        collect_records(root)
    )

    rows_meta = [
        r
        for r in records
        if r["split"] == split
    ]

    if not rows_meta:

        raise ValueError(
            f"No images found for split={split!r}"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    previews = (
        out
        / "previews"
    )

    previews.mkdir(
        exist_ok=True
    )

    rows = []

    graded_fractions = []

    for i, record in enumerate(
        rows_meta,
        1,
    ):

        path = (
            root
            / record["path"]
        )

        with Image.open(
            path
        ) as image:

            image.load()

            result = detect_blemishes(
                image,
                method=method,
            )

            row = {
                "path":
                    record["path"],

                "fruit":
                    record["fruit"],

                "stage":
                    record["stage"],

                "split":
                    split,

                "segmentation_method":
                    method,

                "foreground_pixels":
                    result.foreground_pixels,

                "blemish_pixels":
                    result.blemish_pixels,

                "blemish_fraction":
                    (
                        ""
                        if result.blemish_fraction
                        is None
                        else round(
                            result.blemish_fraction,
                            6,
                        )
                    ),

                "quality_grade":
                    surface_quality_grade(
                        result.blemish_fraction,
                        status=result.status,
                    ),

                "blemish_components":
                    result.details.get(
                        "kept_components",
                        0,
                    ),

                "stem_cavity_removed":
                    result.details.get(
                        "stem_cavity_components_removed",
                        0,
                    ),

                "status":
                    result.status,
            }

            rows.append(
                row
            )

            if (
                result.blemish_fraction
                is not None
            ):

                graded_fractions.append(
                    result.blemish_fraction
                )

            # Bounded preview export
            if i <= 12:

                preview_name = (
                    f"{i:03d}_"
                    f"{Path(record['path']).stem}"
                    ".png"
                )

                blemish_overlay(
                    image,
                    result,
                ).save(
                    previews
                    / preview_name
                )

        if i % 250 == 0:

            print(
                f"  Graded "
                f"{i}/{len(rows_meta)} images",
                flush=True,
            )

    write_csv(
        out
        / "blemish_report.csv",
        rows,
        [
            "path",
            "fruit",
            "stage",
            "split",
            "segmentation_method",
            "foreground_pixels",
            "blemish_pixels",
            "blemish_fraction",
            "quality_grade",
            "blemish_components",
            "stem_cavity_removed",
            "status",
        ],
    )

    # ------------------------------------------------------------
    # Per-stage statistics
    # ------------------------------------------------------------

    stages = sorted(
        {
            r["stage"]
            for r in rows
        }
    )

    per_stage = {}

    for stage in stages:

        values = [
            r["blemish_fraction"]
            for r in rows
            if (
                r["stage"] == stage
                and r[
                    "blemish_fraction"
                ] != ""
            )
        ]

        if values:

            per_stage[
                stage
            ] = float(
                np.mean(values)
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "split":
            split,

        "method":
            method,

        "spec":
            BLEMISH_SPEC,

        "surface_quality_spec":
            SURFACE_QUALITY_SPEC,

        "images":
            len(rows),

        "graded_images":
            len(graded_fractions),

        "ungraded_images":
            (
                len(rows)
                - len(graded_fractions)
            ),

        "mean_blemish_fraction":
            (
                float(
                    np.mean(
                        graded_fractions
                    )
                )
                if graded_fractions
                else None
            ),

        "median_blemish_fraction":
            (
                float(
                    np.median(
                        graded_fractions
                    )
                )
                if graded_fractions
                else None
            ),

        "per_stage_mean_blemish_fraction":
            per_stage,

        "limitations": [
            (
                "Heuristic colour/lightness "
                "outlier detection inside the "
                "segmentation mask."
            ),

            (
                "The detector estimates suspicious "
                "surface regions; it is not a verified "
                "bruise, mould, rot, or disease classifier."
            ),

            (
                "Segmentation errors propagate into "
                "blemish measurements."
            ),

            (
                "Stem/calyx suppression is heuristic "
                "and may occasionally remove a genuine "
                "small central blemish."
            ),
        ],
    }

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


# ================================================================
# COMMAND LINE
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/raw"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/blemish"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--method",
        default=BLEMISH_SPEC[
            "base_segmentation_method"
        ],
    )

    args = parser.parse_args()

    try:

        summary = blemish_report(
            args.data,
            args.out,
            split=args.split,
            method=args.method,
        )

    except (
        ValueError,
        OSError,
    ) as exc:

        parser.exit(
            1,
            f"Error: {exc}\n",
        )

    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k != "spec"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
