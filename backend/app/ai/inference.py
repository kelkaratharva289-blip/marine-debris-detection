"""YOLO inference engine for side-scan sonar marine debris detection.

Wraps ``ultralytics.YOLO`` behind a thin interface that returns detections
in a uniform dict format consumed by the detection service layer.  The
module **never** fabricates results — if the model weights cannot be loaded
or inference fails, an exception is raised immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marine debris class definitions
# ---------------------------------------------------------------------------

MARINE_CLASSES: list[str] = [
    "ghost_net",
    "shipwreck",
    "pipe",
    "cylinder",
    "container",
    "other_debris",
]

# Human-readable aliases returned in detection payloads.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "ghost_net": "Ghost Net",
    "shipwreck": "Shipwreck",
    "pipe": "Pipe",
    "cylinder": "Cylinder",
    "container": "Container",
    "other_debris": "Other Debris",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo_model(model_path: str | Path) -> Any:
    """Load a YOLOv8 model from disk.

    Supports any Ultralytics-loadable artifact: a PyTorch ``.pt`` checkpoint,
    or an edge-deployed ``.onnx`` / ``.engine`` (TensorRT) export produced by
    :mod:`app.ai.export`. Ultralytics selects the matching backend
    (torch / ONNX Runtime / TensorRT) from the file suffix automatically.

    Args:
        model_path: Path to model weights.

    Returns:
        An ``ultralytics.YOLO`` model instance ready for prediction.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If the ultralytics package is not installed or model
            loading fails.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO inference. "
            "Install it with: pip install ultralytics"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. "
            "Set YOLO_MODEL_PATH in your environment to a valid .pt, .onnx "
            "or .engine file."
        )

    logger.info("Loading YOLO model from %s", path)
    model = YOLO(str(path))
    return model


def resolve_inference_model_path(
    model_path: str,
    onnx_path: str | None = None,
    engine_path: str | None = None,
    backend: str = "torch",
) -> str:
    """Resolve the deployment artifact for a configured inference backend.

    The configured ``backend`` determines preference order, but selection
    always degrades gracefully: if the preferred artifact is missing the
    next available artifact (or the source checkpoint) is returned. Never
    returns a path that does not exist; callers treat a missing artifact as
    'no model'.

    Args:
        model_path: PyTorch checkpoint (``YOLO_MODEL_PATH``).
        onnx_path: Exported ONNX artifact (``YOLO_ONNX_PATH``) or ``None``.
        engine_path: Exported TensorRT engine (``YOLO_ENGINE_PATH``) or
            ``None``.
        backend: ``"torch"``, ``"onnx"`` or ``"tensorrt"``.

    Returns:
        The ``str`` path of the artifact to load, or ``""`` when none of the
        configured files exist on disk.
    """
    if backend == "onnx":
        candidates = [onnx_path, engine_path, model_path]
    elif backend == "tensorrt":
        candidates = [engine_path, onnx_path, model_path]
    else:
        candidates = [model_path, onnx_path, engine_path]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return ""


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def run_inference(
    model: Any,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: list[str] | None = None,
    device: str | None = None,
) -> list[dict]:
    """Run YOLO inference on a single preprocessed image.

    The image should already be letterboxed to the model's expected input
    size and converted to uint8 (the standard output of the preprocessing
    pipeline).

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        image: Preprocessed sonar image, ``(H, W)`` or ``(H, W, 3)`` uint8.
        conf: Minimum confidence threshold for detections.
        iou: IoU threshold for non-max suppression.
        classes: Optional list of class names to keep.  If ``None``, all
            classes predicted by the model are returned.
        device: Inference device (``"cpu"``, ``"cuda:0"``).  ``None`` lets
            Ultralytics choose its default (GPU when available).

    Returns:
        A list of detection dicts, each containing:
            - ``class_id``:   int — model-internal class index
            - ``class_label``: str — canonical class name (snake_case)
            - ``class_name``: str — human-readable display name
            - ``confidence``:  float — detection confidence in [0, 1]
            - ``bbox_x``:      float — normalised x of top-left corner [0, 1]
            - ``bbox_y``:      float — normalised y of top-left corner [0, 1]
            - ``bbox_width``:  float — normalised width [0, 1]
            - ``bbox_height``: float — normalised height [0, 1]

        Returns an **empty list** when the model finds no objects above the
        confidence threshold — never fabricated detections.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if device:
        kwargs["device"] = device

    results = model.predict(source=image, **kwargs)

    if not results:
        return []

    detections: list[dict] = []
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    names: dict[int, str] = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        raw_label = names.get(cls_id, f"class_{cls_id}")
        canonical = _canonicalise_label(raw_label)

        if classes is not None and canonical not in classes:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        img_h, img_w = image.shape[:2]

        bbox_x = x1 / img_w
        bbox_y = y1 / img_h
        bbox_w = (x2 - x1) / img_w
        bbox_h = (y2 - y1) / img_h

        detections.append(
            {
                "class_id": cls_id,
                "class_label": canonical,
                "class_name": CLASS_DISPLAY_NAMES.get(canonical, raw_label),
                "confidence": round(float(box.conf[0]), 4),
                "bbox_x": round(max(0.0, bbox_x), 4),
                "bbox_y": round(max(0.0, bbox_y), 4),
                "bbox_width": round(max(0.0, bbox_w), 4),
                "bbox_height": round(max(0.0, bbox_h), 4),
            }
        )

    return detections


def _canonicalise_label(raw: str) -> str:
    """Normalise a class label to one of the canonical marine class slugs.

    YOLO models may produce arbitrary class names from training data.  This
    function maps common variants to the canonical set so the rest of the
    pipeline only deals with a fixed vocabulary.
    """
    normalised = raw.strip().lower().replace(" ", "_").replace("-", "_")

    alias_map = {
        "ghost_net": "ghost_net",
        "ghostnet": "ghost_net",
        "fishing_net": "ghost_net",
        "net": "ghost_net",
        "shipwreck": "shipwreck",
        "wreck": "shipwreck",
        "ship_wreck": "shipwreck",
        "pipe": "pipe",
        "pipeline": "pipe",
        "cylinder": "cylinder",
        "barrel": "cylinder",
        "container": "container",
        "box": "container",
        "cargo": "container",
        "other_debris": "other_debris",
        "debris": "other_debris",
        "unknown": "other_debris",
        "anomaly": "other_debris",
    }

    return alias_map.get(normalised, "other_debris")
