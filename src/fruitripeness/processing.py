"""Deterministic image-only transforms shared by training, previews and inference.

These methods are foreground heuristics, not verified fruit detectors.
Ripeness labels and fruit names must never be passed to this module.
"""
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

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

KMEANS_SPEC = {
    "version": "kmeans_rgb_border_v1",
    "colour_space": "RGB float64 / 255; Euclidean distance",
    "n_clusters": 3,
    "sample_axis_limit": 64,
    "sampling_rule": "uniform grid including endpoints; floor linspace coordinates; no interpolation",
    "init": "k-means++",
    "n_init": 3,
    "max_iter": 50,
    "tol": 0.0001,
    "random_state": 42,
    "algorithm": "lloyd",
    "fit_threads": 1,
    "assignment_chunk_size": 16384,
    "cluster_order": "ascending centroid R, then G, then B; nearest-centroid ties choose first",
    "border_fraction": 0.05,
    "background_rule": "most border pixels; tie chooses most whole-image pixels; final tie chooses first cluster",
    "few_colours_policy": "reduce k to unique sampled colours; one colour yields empty mask",
    "opening_kernel": 3,
    "closing_kernel": 5,
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "fill_enclosed_holes": True,
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no image exclusion",
    "working_size": "original image resolution; dataset images are 300x300",
}

GRABCUT_SPEC = {
    "version": "grabcut_auto_rect_v1",
    "opencv_version": "4.13.0",
    "initialisation": "GC_INIT_WITH_RECT; outer margin is definite background",
    "max_working_side": 160,
    "image_resize": "preserve aspect ratio, round dimensions, Pillow BILINEAR; never upscale",
    "border_fraction": 0.05,
    "margin_rule": "round fraction of each working dimension; minimum one pixel",
    "iterations": 5,
    "random_seed": 42,
    "opencv_threads": 1,
    "mask_resize": "Pillow NEAREST to original dimensions, then shared cleanup",
    "small_image_policy": "empty mask if either initial class has fewer than five pixels",
    "constant_image_policy": "empty mask if working image has one RGB colour",
    "opening_kernel": 3,
    "closing_kernel": 5,
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "fill_enclosed_holes": True,
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no image exclusion",
    "working_size": "GrabCut at max side 160; cleanup and RGB masking at original resolution",
}

WATERSHED_SPEC = {
    "version": "watershed_otsu_distance_markers_v1",
    "opencv_version": "4.13.0",
    "coarse_mask_version": "otsu_gray_border_v1",
    "border_fraction": 0.05,
    "max_working_side": 300,
    "image_resize": "preserve aspect ratio, round dimensions, Pillow BILINEAR; never upscale",
    "marker_opening_kernel": 3,
    "background_frame_fraction": 0.02,
    "background_frame_min_pixels": 2,
    "background_dilation_kernel": 3,
    "background_dilation_iterations": 3,
    "distance_transform": "SciPy exact Euclidean distance inside coarse foreground",
    "foreground_core_fraction": 0.50,
    "core_rule": "distance >= fraction of each coarse component's own maximum",
    "marker_labels": "0 unknown, 1 background, 2+ foreground cores; 8-connected cores",
    "watershed_input": "unblurred working BGR uint8; OpenCV colour-difference watershed",
    "boundary_policy": "exclude label -1 before shared cleanup; union foreground labels > 1",
    "mask_resize": "Pillow NEAREST to original dimensions, then shared cleanup",
    "small_image_policy": "empty mask if either working dimension is below five pixels",
    "opening_kernel": 3,
    "closing_kernel": 5,
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "fill_enclosed_holes": True,
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no image exclusion",
    "working_size": "watershed at max side 300; cleanup and RGB masking at original resolution",
}

# Guard OpenCV's seed/thread configuration for concurrent calls through this wrapper.
_GRABCUT_LOCK = Lock()

HYBRID_REFINED_SPEC = {
    "version": "hybrid_hsv_seeded_grabcut_v1",
    "opencv_version": "4.13.0",
    "max_working_side": 200,
    "image_resize": "preserve aspect ratio, round dimensions, Pillow BILINEAR; never upscale",
    "saturation_min": 0.30,
    "value_min": 0.10,
    "hue_range": "all; no fruit/stage-specific thresholds",
    "seed_opening_kernel": 5,
    "core_erosion_kernel": 3,
    "support_radius_fraction": 0.03,
    "support_radius_min_pixels": 3,
    "support_rule": "Euclidean distance to cleaned candidate <= radius; outside is hard background",
    "border_pixels": 1,
    "initialisation": "GC_INIT_WITH_MASK; eroded colour core definite FG, candidate probable FG, expansion probable BG, outside hard BG",
    "min_component_fraction": 0.005,
    "min_component_pixels": 16,
    "max_hole_fraction": 0.02,
    "max_hole_min_pixels": 9,
    "hole_policy": "fill only enclosed holes at most max(9, floor(2% working area)); never fill all holes",
    "closing_kernel": 3,
    "iterations": 5,
    "random_seed": 42,
    "opencv_threads": 1,
    "mask_resize": "Pillow NEAREST to original dimensions; no second original-resolution cleanup",
    "background_rgb": [0, 0, 0],
    "empty_mask_policy": "black image, flag for review, no original-image fallback or image exclusion",
    "working_size": "seed construction, GrabCut and cleanup at max side 200; original RGB retained",
}


def processing_spec(method: str) -> dict:
    if method == "baseline":
        return {"version": "identity_rgb_v1"}
    if method == "hsv":
        # Return independent JSON-compatible metadata; callers cannot mutate defaults.
        return {**HSV_SPEC, "background_rgb": list(HSV_SPEC["background_rgb"])}
    if method == "otsu":
        return {**OTSU_SPEC, "background_rgb": list(OTSU_SPEC["background_rgb"])}
    if method == "kmeans":
        return {**KMEANS_SPEC, "background_rgb": list(KMEANS_SPEC["background_rgb"])}
    if method == "grabcut":
        return {**GRABCUT_SPEC, "background_rgb": list(GRABCUT_SPEC["background_rgb"])}
    if method == "watershed":
        return {**WATERSHED_SPEC, "background_rgb": list(WATERSHED_SPEC["background_rgb"])}
    if method == "hybrid":
        return {
            "version": "hybrid_hsv_grabcut_union_v1",
            "combination": "logical OR of final cleaned HSV and GrabCut masks",
            "post_union_cleanup": "none; reuse each component's complete cleanup exactly once",
            "components": {name: processing_spec(name) for name in ("hsv", "grabcut")},
            "background_rgb": [0, 0, 0],
            "empty_mask_policy": "black image only if both component masks are empty; no image exclusion",
            "working_size": "component masks combined at original image resolution",
            "selection_basis": "HSV best processed accuracy; GrabCut best processed macro F1 on development validation",
        }
    if method == "hybrid_refined":
        return {**HYBRID_REFINED_SPEC, "background_rgb": list(HYBRID_REFINED_SPEC["background_rgb"])}
    raise ValueError(f"Unsupported method: {method}")


@dataclass
class ProcessingResult:
    original: Image.Image
    processed: Image.Image
    mask: np.ndarray
    details: dict
    # Optional INITIAL marker labels at working resolution; never classification labels.
    markers: np.ndarray | None = None
    # Final component masks for hybrid explanations; no ground-truth annotations.
    component_masks: dict[str, np.ndarray] | None = None


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


def kmeans_candidate_mask(rgb: Image.Image, spec: dict) -> tuple[np.ndarray, dict]:
    """Fit colour clusters independently per image; select background by its frame.

    Sampling bounds the clustering work without resizing/interpolating colours.
    Every original pixel is then assigned to a centroid. Cluster IDs are colour
    groups, never ripeness classes. This function consumes no labels or filenames.
    """
    pixels = np.asarray(rgb, dtype=np.uint8)
    height, width = pixels.shape[:2]
    limit = spec["sample_axis_limit"]
    ys = np.linspace(0, height-1, min(height, limit), dtype=int)
    xs = np.linspace(0, width-1, min(width, limit), dtype=int)
    sample = pixels[ys[:, None], xs].reshape(-1, 3).astype(np.float64) / 255.0
    clusters = min(spec["n_clusters"], len(np.unique(sample, axis=0)))
    details = {"kmeans_clusters": clusters, "kmeans_sample_pixels": len(sample)}
    if clusters == 1:
        return np.zeros((height, width), dtype=bool), {
            **details, "background_cluster": 0, "background_border_fraction": 1.0,
            "kmeans_iterations": 0,
        }
    estimator = KMeans(n_clusters=clusters, init=spec["init"], n_init=spec["n_init"],
                       max_iter=spec["max_iter"], tol=spec["tol"],
                       random_state=spec["random_state"], algorithm=spec["algorithm"])
    # Bound native threads for these small fits, including Windows/MKL builds.
    # Restore prior limits afterwards; Random Forest parallelism is unchanged.
    with threadpool_limits(limits=spec["fit_threads"]):
        estimator.fit(sample)
    centres = estimator.cluster_centers_
    centres = centres[np.lexsort((centres[:, 2], centres[:, 1], centres[:, 0]))]
    flat = pixels.reshape(-1, 3)
    labels = np.empty(len(flat), dtype=np.int32)
    chunk = spec["assignment_chunk_size"]
    for start in range(0, len(flat), chunk):
        values = flat[start:start+chunk].astype(np.float64) / 255.0
        distances = ((values[:, None, :] - centres[None, :, :])**2).sum(axis=2)
        labels[start:start+chunk] = np.argmin(distances, axis=1)
    labels = labels.reshape(height, width)
    frame = max(1, int(round(min(height, width)*spec["border_fraction"])))
    border = np.zeros((height, width), dtype=bool)
    border[:frame, :] = border[-frame:, :] = True
    border[:, :frame] = border[:, -frame:] = True
    border_counts = np.bincount(labels[border], minlength=clusters)
    total_counts = np.bincount(labels.ravel(), minlength=clusters)
    background = max(range(clusters), key=lambda c: (border_counts[c], total_counts[c], -c))
    return labels != background, {
        **details, "background_cluster": background,
        "background_border_fraction": float(border_counts[background] / border.sum()),
        "kmeans_iterations": int(estimator.n_iter_),
    }


def grabcut_candidate_mask(rgb: Image.Image, spec: dict) -> tuple[np.ndarray, dict]:
    """Automatic rectangle initialisation; no hand-drawn masks or labels.

    The working-size limit bounds graph construction for future folder input.
    Only the binary mask is enlarged; retained RGB values come from the original.
    Empty and too-small cases remain in the dataset with explicit diagnostics.
    Unexpected OpenCV errors stop the experiment rather than dropping an image.
    """
    if cv2.__version__ != spec["opencv_version"]:
        raise ValueError("GrabCut OpenCV version differs; install the project's pinned requirements")
    scale = min(1.0, spec["max_working_side"] / max(rgb.size))
    width, height = (max(1, int(round(d*scale))) for d in rgb.size)
    working = rgb.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(working, dtype=np.uint8)
    mx = max(1, int(round(width*spec["border_fraction"])))
    my = max(1, int(round(height*spec["border_fraction"])))
    rw, rh = max(0, width-2*mx), max(0, height-2*my)
    details = {
        "grabcut_width": width, "grabcut_height": height,
        "grabcut_rect": f"{mx},{my},{rw},{rh}", "grabcut_iterations": 0,
    }
    if rw*rh < 5 or width*height-rw*rh < 5:
        return np.zeros((rgb.height, rgb.width), dtype=bool), {
            **details, "grabcut_status": "too_small",
        }
    if np.all(pixels == pixels[0, 0]):
        return np.zeros((rgb.height, rgb.width), dtype=bool), {
            **details, "grabcut_status": "constant_working_image",
        }
    # OpenCV accepts BGR uint8. Never pass its recoloured/clustered values onward.
    bgr = np.ascontiguousarray(pixels[:, :, ::-1])
    labels = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    with _GRABCUT_LOCK:
        previous_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(spec["opencv_threads"])
            cv2.setRNGSeed(spec["random_seed"])
            cv2.grabCut(bgr, labels, (mx, my, rw, rh), background_model,
                        foreground_model, spec["iterations"], cv2.GC_INIT_WITH_RECT)
        except cv2.error as exc:
            raise ValueError(f"GrabCut failed for {width}x{height} working image: {exc}") from exc
        finally:
            cv2.setNumThreads(previous_threads)
    mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    enlarged = Image.fromarray(mask.astype(np.uint8)*255).resize(rgb.size, Image.Resampling.NEAREST)
    return np.asarray(enlarged) > 0, {
        **details, "grabcut_iterations": spec["iterations"],
        "grabcut_status": "completed" if mask.any() else "empty_result",
    }


def watershed_candidate_mask(rgb: Image.Image, spec: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Otsu-derived distance markers followed by colour-boundary flooding.

    Returns the candidate mask, scalar diagnostics, and initial marker labels for
    report previews. Marker counts are not fruit counts. No true labels are used.
    """
    if cv2.__version__ != spec["opencv_version"]:
        raise ValueError("Watershed OpenCV version differs; install the project's pinned requirements")
    scale = min(1.0, spec["max_working_side"] / max(rgb.size))
    width, height = (max(1, int(round(d*scale))) for d in rgb.size)
    working = rgb.resize((width, height), Image.Resampling.BILINEAR)
    markers = np.zeros((height, width), dtype=np.int32)
    empty = np.zeros((rgb.height, rgb.width), dtype=bool)
    details = {
        "watershed_width": width, "watershed_height": height,
        "watershed_markers": 0, "watershed_seed_fraction": 0.0,
        "watershed_background_fraction": 0.0, "watershed_boundary_fraction": 0.0,
        "otsu_threshold": None, "foreground_polarity": "none",
    }
    if min(width, height) < 5:
        return empty, {**details, "watershed_status": "too_small"}, markers
    coarse, threshold_details = otsu_candidate_mask(working, spec)
    details.update(threshold_details)
    kernel = spec["marker_opening_kernel"]
    padded = np.pad(coarse, kernel, mode="edge")
    padded = ndimage.binary_opening(padded, structure=np.ones((kernel, kernel), dtype=bool))
    coarse = padded[kernel:-kernel, kernel:-kernel]
    # A two-pixel minimum frame leaves background seeds inside OpenCV's outer
    # one-pixel boundary, which watershed always replaces with label -1.
    frame = max(spec["background_frame_min_pixels"],
                int(round(min(width, height)*spec["background_frame_fraction"])))
    border = np.zeros((height, width), dtype=bool)
    border[:frame, :] = border[-frame:, :] = True
    border[:, :frame] = border[:, -frame:] = True
    coarse[border] = False
    components, _ = ndimage.label(coarse, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())
    keep = sizes >= max(spec["min_component_pixels"], int(np.ceil(coarse.size*spec["min_component_fraction"])))
    keep[0] = False
    coarse = keep[components]
    if not coarse.any():
        return empty, {**details, "watershed_status": "no_coarse_foreground"}, markers
    components, count = ndimage.label(coarse, structure=np.ones((3, 3), dtype=bool))
    distance = ndimage.distance_transform_edt(coarse)
    maxima = ndimage.maximum(distance, labels=components, index=np.arange(count+1))
    cores = coarse & (distance >= spec["foreground_core_fraction"]*maxima[components])
    core_labels, n_cores = ndimage.label(cores, structure=np.ones((3, 3), dtype=bool))
    dilation = spec["background_dilation_kernel"]
    expanded = ndimage.binary_dilation(coarse, structure=np.ones((dilation, dilation), dtype=bool),
                                      iterations=spec["background_dilation_iterations"])
    background = ~expanded | border
    markers[background] = 1
    markers[cores] = core_labels[cores] + 1
    initial = markers.copy()  # OpenCV mutates the array passed to watershed.
    details.update({"watershed_markers": int(n_cores),
                    "watershed_seed_fraction": float(cores.mean()),
                    "watershed_background_fraction": float(background.mean())})
    bgr = np.ascontiguousarray(np.asarray(working, dtype=np.uint8)[:, :, ::-1])
    try:
        cv2.watershed(bgr, markers)
    except cv2.error as exc:
        raise ValueError(f"Watershed failed for {width}x{height} working image: {exc}") from exc
    mask = markers > 1
    enlarged = Image.fromarray(mask.astype(np.uint8)*255).resize(rgb.size, Image.Resampling.NEAREST)
    return np.asarray(enlarged) > 0, {
        **details, "watershed_boundary_fraction": float((markers == -1).mean()),
        "watershed_status": "completed" if mask.any() else "empty_result",
    }, initial


def _refined_components(mask: np.ndarray, spec: dict) -> np.ndarray:
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(spec["min_component_pixels"], int(np.ceil(mask.size*spec["min_component_fraction"])))
    keep[0] = False
    return keep[labels]


def _refined_holes(mask: np.ndarray, spec: dict) -> np.ndarray:
    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, _ = ndimage.label(holes, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    small = sizes <= max(spec["max_hole_min_pixels"], int(mask.size*spec["max_hole_fraction"]))
    small[0] = False
    return mask | small[labels]


def refined_hybrid(rgb: Image.Image, spec: dict) -> ProcessingResult:
    """Colour-seeded GrabCut with bounded growth, NOT semantic fruit recognition.

    This is a new method ID. The original HSV/GrabCut/union implementations and
    their saved models are untouched. All decisions depend only on image pixels.
    """
    if cv2.__version__ != spec["opencv_version"]:
        raise ValueError("Refined hybrid OpenCV version differs; install the pinned requirements")
    scale = min(1.0, spec["max_working_side"]/max(rgb.size))
    width, height = (max(1, int(round(d*scale))) for d in rgb.size)
    working = rgb.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(working, dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.uint8)
    candidate = np.zeros(labels.shape, dtype=bool)
    mask = candidate.copy()
    initial = labels.copy()
    details = {"refined_width": width, "refined_height": height,
               "refined_candidate_fraction": 0.0, "refined_core_fraction": 0.0,
               "refined_support_fraction": 0.0, "refined_iterations": 0}
    if min(width, height) < 7:
        status = "too_small"
    elif np.all(pixels == pixels[0, 0]):
        status = "constant_working_image"
    else:
        hsv = np.asarray(working.convert("HSV"), dtype=np.float32)/255.0
        colour = (hsv[:, :, 1] >= spec["saturation_min"]) & (hsv[:, :, 2] >= spec["value_min"])
        kernel = spec["seed_opening_kernel"]
        colour = ndimage.binary_opening(colour, structure=np.ones((kernel, kernel), dtype=bool))
        colour = _refined_components(colour, spec)
        candidate = _refined_holes(colour, spec)
        core_kernel = spec["core_erosion_kernel"]
        core = ndimage.binary_erosion(colour, structure=np.ones((core_kernel, core_kernel), dtype=bool))
        details["refined_candidate_fraction"] = float(candidate.mean())
        if not candidate.any() or int(core.sum()) < 5:
            status = "no_colour_seeds"
        else:
            radius = max(spec["support_radius_min_pixels"], int(round(max(width, height)*spec["support_radius_fraction"])))
            support = ndimage.distance_transform_edt(~candidate) <= radius
            border = spec["border_pixels"]
            support[:border, :] = support[-border:, :] = False
            support[:, :border] = support[:, -border:] = False
            labels[support] = cv2.GC_PR_BGD
            labels[candidate & support] = cv2.GC_PR_FGD
            labels[core & support] = cv2.GC_FGD
            initial = labels.copy()
            details.update(refined_core_fraction=float((labels == cv2.GC_FGD).mean()),
                           refined_support_fraction=float(support.mean()))
            bgr = np.ascontiguousarray(pixels[:, :, ::-1])
            with _GRABCUT_LOCK:
                previous_threads = cv2.getNumThreads()
                try:
                    cv2.setNumThreads(spec["opencv_threads"])
                    cv2.setRNGSeed(spec["random_seed"])
                    cv2.grabCut(bgr, labels, None, np.zeros((1, 65), dtype=np.float64),
                                np.zeros((1, 65), dtype=np.float64), spec["iterations"], cv2.GC_INIT_WITH_MASK)
                except cv2.error as exc:
                    raise ValueError(f"Refined hybrid GrabCut failed: {exc}") from exc
                finally:
                    cv2.setNumThreads(previous_threads)
            mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
            closing = spec["closing_kernel"]
            mask = ndimage.binary_closing(mask, structure=np.ones((closing, closing), dtype=bool))
            mask = _refined_holes(_refined_components(mask, spec), spec) & support
            details["refined_iterations"] = spec["iterations"]
            status = "completed" if mask.any() else "empty_result"
    enlarged = Image.fromarray(mask.astype(np.uint8)*255).resize(rgb.size, Image.Resampling.NEAREST)
    final = np.asarray(enlarged) > 0
    fraction = float(final.mean())
    output = np.asarray(rgb).copy()
    output[~final] = spec["background_rgb"]
    candidate_image = Image.fromarray(candidate.astype(np.uint8)*255).resize(rgb.size, Image.Resampling.NEAREST)
    return ProcessingResult(rgb, Image.fromarray(output), final, {
        **details, "refined_status": status, "foreground_fraction": fraction,
        "mask_status": "empty" if fraction == 0 else "near_full" if fraction > 0.98 else "nonempty",
        "refined_review_flag": "no_foreground" if fraction == 0 else "broad_colour_candidates" if details["refined_candidate_fraction"] > 0.85 else "inspect_mask",
    }, markers=initial, component_masks={"colour_candidate": np.asarray(candidate_image) > 0})


def process_image(image: Image.Image, method: str) -> ProcessingResult:
    spec = processing_spec(method)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if method == "hybrid_refined":
        return refined_hybrid(rgb, spec)
    if method == "baseline":
        mask = np.ones((rgb.height, rgb.width), dtype=bool)
        return ProcessingResult(rgb, rgb.copy(), mask,
                                {"foreground_fraction": 1.0, "mask_status": "not_segmented"})
    if method == "hybrid":
        # Reuse the frozen, complete component pipelines. Do not clean the union
        # again: it must be exactly the OR of the masks shown in the previews.
        hsv = process_image(rgb, "hsv")
        grabcut = process_image(rgb, "grabcut")
        mask = hsv.mask | grabcut.mask
        intersection = hsv.mask & grabcut.mask
        fraction = float(mask.mean())
        pixels = np.asarray(rgb).copy()
        pixels[~mask] = spec["background_rgb"]
        details = {
            "foreground_fraction": fraction,
            "mask_status": "empty" if fraction == 0 else "near_full" if fraction > 0.98 else "nonempty",
            "hybrid_hsv_fraction": hsv.details["foreground_fraction"],
            "hybrid_grabcut_fraction": grabcut.details["foreground_fraction"],
            "hybrid_hsv_status": hsv.details["mask_status"],
            "hybrid_grabcut_status": grabcut.details["mask_status"],
            "hybrid_intersection_fraction": float(intersection.mean()),
            "hybrid_disagreement_fraction": float((hsv.mask ^ grabcut.mask).mean()),
            # This is agreement BETWEEN heuristics, not segmentation accuracy.
            "hybrid_mask_jaccard": float(intersection.sum()/mask.sum()) if mask.any() else 1.0,
            **{key: value for key, value in grabcut.details.items() if key.startswith("grabcut_")},
        }
        return ProcessingResult(rgb, Image.fromarray(pixels), mask, details,
                                component_masks={"hsv": hsv.mask, "grabcut": grabcut.mask})
    details = {}
    markers = None
    if method == "hsv":
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
        mask = (hsv[:, :, 1] >= spec["saturation_min"]) & (hsv[:, :, 2] >= spec["value_min"])
    elif method == "otsu":
        mask, details = otsu_candidate_mask(rgb, spec)
    elif method == "kmeans":
        mask, details = kmeans_candidate_mask(rgb, spec)
    elif method == "grabcut":
        mask, details = grabcut_candidate_mask(rgb, spec)
    else:
        mask, details, markers = watershed_candidate_mask(rgb, spec)
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
                            {"foreground_fraction": fraction, "mask_status": status, **details}, markers=markers)
