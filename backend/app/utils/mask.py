"""Mask overlay rendering for segmentation display.

Generates a standalone PNG (transparent background, filled polygon + box +
label) that the dashboard can serve as an image layer over the sonar scan.
The overlay is rendered at a fixed canvas resolution; polygon points are
normalised [0, 1] coordinates.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

CANVAS_SIZE = 1024


def render_mask_overlay(
    polygon: list,
    mask_area: float | None = None,
    class_label: str = "object",
    confidence: float | None = None,
    output_dir: str = "./uploads/masks",
    detection_id: str = "mask",
) -> str:
    """Render a mask overlay PNG and return its on-disk path.

    Args:
        polygon: Normalised ``[[x, y], ...]`` polygon points in [0, 1].
        mask_area: Mask area as a fraction of the image (optional).
        class_label: Human-readable class label for the overlay caption.
        confidence: Detection confidence in [0, 1] (optional).
        output_dir: Directory in which to store the PNG.
        detection_id: Identifier used to name the output file.

    Returns:
        Absolute path to the generated PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    overlay = _build_overlay(polygon, class_label, confidence)

    filename = f"{detection_id or 'mask'}.png"
    path = os.path.abspath(os.path.join(output_dir, filename))
    cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    return path


def _build_overlay(
    polygon: list, class_label: str, confidence: float | None
) -> np.ndarray:
    size = CANVAS_SIZE
    overlay = np.zeros((size, size, 4), dtype=np.uint8)

    if polygon and len(polygon) >= 3:
        pts = np.array(
            [[p[0] * size, p[1] * size] for p in polygon], dtype=np.int32
        ).reshape(-1, 1, 2)

        # Semi-transparent cyan fill.
        fill = overlay.copy()
        cv2.fillPoly(fill, [pts], color=(34, 211, 238, 110))
        overlay = fill

        # Bright contour.
        cv2.polylines(
            overlay, [pts], isClosed=True, color=(34, 211, 238, 255), thickness=3
        )

    # Bounding box + label caption.
    if polygon and len(polygon) >= 2:
        xs = [p[0] * size for p in polygon]
        ys = [p[1] * size for p in polygon]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cv2.rectangle(
            overlay, (x1, y1), (x2, y2), color=(34, 211, 238, 255), thickness=2
        )

        caption = class_label.replace("_", " ").title()
        if confidence is not None:
            caption += f" {confidence:.0%}"
        _draw_label(overlay, caption, (x1, max(0, y1 - 26)))

    return overlay


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = origin
    cv2.rectangle(
        image,
        (x, y),
        (x + tw + 8, y + th + baseline + 8),
        color=(34, 211, 238, 255),
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y + th + 4),
        font,
        scale,
        color=(3, 8, 15, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
