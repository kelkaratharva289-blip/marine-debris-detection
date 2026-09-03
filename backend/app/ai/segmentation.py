"""Segmentation engine for side-scan sonar marine debris detection.

This module is an **optional** extension over the object detector.  When
enabled and trained weights are available it produces a pixel-level mask
for every detected object.  Two backends are supported:

* ``yolo-seg`` — Ultralytics YOLO instance-segmentation models (e.g. a
  custom ``yolov8n-seg`` trained on sonar data).
* ``unet``    — a generic U-Net (torch) classifier producing per-pixel
  class probabilities.

Both backends return a common, backend-agnostic result the rest of the
pipeline consumes:

    {
        "class_id":    int,
        "class_label": str,
        "class_name":  str,
        "confidence":  float,
        "bbox_x/y/w/h": float,            # normalised bbox [0, 1]
        "mask":        numpy bool array,  # (H_img, W_img) binary
        "polygon":     [[x, y], ...],     # normalised contour points [0, 1]
        "area":        float,             # mask area as a fraction of the image
    }

The module **never fabricates masks** — if the segmentation model is
disabled or cannot be loaded, the caller simply gets no segmentation
output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ai.inference import CLASS_DISPLAY_NAMES, _canonicalise_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------

def load_seg_model(model_path: str | Path, model_type: str = "yolo-seg") -> Any:
    """Load a segmentation model for the given backend type.

    Args:
        model_path: Path to the segmentation weights.  For ``yolo-seg`` this
            is an Ultralytics ``*.pt`` file.  For ``unet`` it is a PyTorch
            ``*.pt`` checkpoint exposing ``forward(x) -> (N, C, H, W)``.
        model_type: ``"yolo-seg"`` or ``"unet"``.

    Returns:
        A backend model object.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If the required framework is not installed.
    """
    model_type = model_type.lower()
    if model_type == "yolo-seg":
        return _load_ultralytics_seg(model_path)
    if model_type == "unet":
        return _load_unet(model_path)
    raise ValueError(f"Unknown segmentation model type: {model_type}")


def _load_ultralytics_seg(model_path: str | Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO-Seg inference. "
            "Install it with: pip install ultralytics"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Segmentation weights not found: {path}. "
            "Set SEGMENTATION_MODEL_PATH in your environment to a valid "
            "YOLO-Seg .pt file (e.g. yolov8n-seg.pt)."
        )

    logger.info("Loading YOLO-Seg model from %s", path)
    return YOLO(str(path))


def _load_unet(model_path: str | Path) -> Any:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for U-Net segmentation. "
            "Install it with: pip install torch"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"U-Net weights not found: {path}. "
            f"Set SEGMENTATION_MODEL_PATH in your environment to a valid "
            f".pt checkpoint."
        )

    # Import lazily and defer to the application to restore the model class.
    # A minimal loader is provided here; consumers may pass an already
    # restored model instead by calling run_unet directly.
    logger.info("U-Net checkpoint at %s (load handled by restore_model)", path)
    return str(path)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def run_segmentation(
    model: Any,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[str] | None = None,
    model_type: str = "yolo-seg",
) -> list[dict]:
    """Run instance segmentation on a preprocessed image.

    Args:
        model: Loaded segmentation model.
        image: Preprocessed sonar image, ``(H, W)`` or ``(H, W, 3)`` uint8.
        conf: Confidence threshold.
        iou: IoU (NMS) threshold.
        classes: Optional canonical class names to keep.
        model_type: ``"yolo-seg"`` or ``"unet"``.

    Returns:
        A list of segmentation dicts (see module docstring).  Empty when
        nothing is found.
    """
    if model_type.lower() == "unet":
        return run_unet(model, image, classes=classes)

    return run_yolo_seg(model, image, conf=conf, iou=iou, classes=classes)


def run_yolo_seg(
    model: Any,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[str] | None = None,
) -> list[dict]:
    """Run YOLO-Seg instance segmentation."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model.predict(
        source=image, conf=conf, iou=iou, verbose=False
    )
    if not results:
        return []

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []

    img_h, img_w = image.shape[:2]
    names: dict[int, str] = result.names
    detections: list[dict] = []

    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        raw_label = names.get(cls_id, f"class_{cls_id}")
        canonical = _canonicalise_label(raw_label)

        if classes is not None and canonical not in classes:
            continue

        mask = _yolo_mask_to_binary(result.masks.data[idx], img_h, img_w)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            _build_result(
                cls_id=cls_id,
                raw_label=raw_label,
                canonical=canonical,
                confidence=float(box.conf[0]),
                bbox=(_norm(x1, img_w), _norm(y1, img_h),
                      _norm(x2 - x1, img_w), _norm(y2 - y1, img_h)),
                mask=mask,
            )
        )

    return detections


def run_unet(
    model: Any,
    image: np.ndarray,
    classes: list[str] | None = None,
) -> list[dict]:
    """Run class-agnostic U-Net segmentation.

    Expects ``model`` to have a ``predict_mask(image) -> (H, W) uint8``
    method (0 = background, 1 = foreground). A foreground-connected
    components pass produces one result per object.

    This is intentionally defensive: without a U-Net checkpoint the caller
    should not enable the backend.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    predict = getattr(model, "predict_mask", None)
    if predict is None:
        raise RuntimeError(
            "U-Net model must expose a predict_mask(image) -> uint8 method."
        )

    mask = np.asarray(predict(image))
    if mask.ndim == 3:
        mask = mask[..., 0]

    binary = (mask > 0).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    img_h, img_w = image.shape[:2]
    detections: list[dict] = []

    for comp in range(1, n_comp):
        comp_mask = labels == comp
        if comp_mask.sum() < 1:
            continue

        ys, xs = np.nonzero(comp_mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        detections.append(
            _build_result(
                cls_id=0,
                raw_label="other_debris",
                canonical="other_debris",
                confidence=1.0,
                bbox=(_norm(x1, img_w), _norm(y1, img_h),
                      _norm(x2 - x1 + 1, img_w), _norm(y2 - y1 + 1, img_h)),
                mask=comp_mask,
            )
        )

    if classes is not None:
        detections = [d for d in detections if d["class_label"] in classes]

    return detections


# ---------------------------------------------------------------------------
# Mask / util helpers
# ---------------------------------------------------------------------------

def _build_result(
    cls_id: int,
    raw_label: str,
    canonical: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray,
) -> dict:
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    img_h, img_w = mask.shape[:2]
    return {
        "class_id": cls_id,
        "class_label": canonical,
        "class_name": CLASS_DISPLAY_NAMES.get(canonical, raw_label),
        "confidence": round(confidence, 4),
        "bbox_x": round(max(0.0, bbox_x), 4),
        "bbox_y": round(max(0.0, bbox_y), 4),
        "bbox_width": round(max(0.0, bbox_w), 4),
        "bbox_height": round(max(0.0, bbox_h), 4),
        # Full-resolution binary mask for overlay rendering / stats.
        "mask": mask,
        # Compact, JSON-serialisable normalized polygon for the dashboard.
        "polygon": mask_to_polygon(mask),
        "area": round(float(mask.mean()), 4),
    }


def _yolo_mask_to_binary(mask_data: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a YOLO mask (HxWxN) slice up to the full image resolution."""
    mask = np.asarray(mask_data, dtype=np.uint8)
    if mask.shape[0] != h or mask.shape[1] != w:
        # YOLO-Seg returns masks at the network stride; resize to image size.
        mask = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    return mask > 0.5


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0) -> list[list[float]]:
    """Extract a simplified normalised polygon from a binary mask.

    Args:
        mask: Binary boolean mask, ``(H, W)``.
        epsilon: Douglas-Peucker simplification tolerance in pixels.

    Returns:
        List of ``[x, y]`` points normalised to [0, 1] relative to the image
        dimensions, tracing the outer contour(s) of the mask.
    """
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return []

    binary = mask.astype(np.uint8)
    h, w = binary.shape

    # Find the largest connected contour for a clean outline.
    n_comp, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_contour: np.ndarray | None = None
    best_area = -1

    for comp in range(1, n_comp):
        comp_mask = (binary == comp).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_contour = cnt

    if best_contour is None or best_area <= 0:
        return []

    approx = cv2.approxPolyDP(best_contour, float(epsilon), True)
    points = approx.reshape(-1, 2)

    # Normalise and cap the number of points.
    poly = [
        (float(round(float(x) / w, 4)), float(round(float(y) / h, 4)))
        for x, y in points
    ]

    # Deduplicate near-identical consecutive points and cap length.
    deduped: list[list[float]] = []
    for p in poly:
        if not deduped or abs(p[0] - deduped[-1][0]) > 1e-4 or abs(p[1] - deduped[-1][1]) > 1e-4:
            deduped.append([p[0], p[1]])

    # Cap to a reasonable number for lightweight SVG / JSON payloads.
    return deduped[:256]


def _norm(value: float, dimension: float) -> float:
    return max(min(float(value) / max(float(dimension), 1e-9), 1.0), 0.0)
