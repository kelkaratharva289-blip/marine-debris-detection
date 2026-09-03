"""Image resizing (with letterboxing) for sonar imagery."""

from __future__ import annotations

import cv2
import numpy as np


def resize(image: np.ndarray, size: int, keep_aspect: bool = False) -> np.ndarray:
    """Resize an image to a target square size.

    Args:
        image: Input image.
        size: Target width/height for the output (square for YOLO).
        keep_aspect: If True, preserve aspect ratio by fitting the image into a
            ``size x size`` canvas without distortion via letterboxing. If
            False, resize directly (which may distort aspect ratio).

    Returns:
        Resized image.
    """
    if keep_aspect:
        return letterbox(image, size)

    return cv2.resize(
        image,
        (int(size), int(size)),
        interpolation=cv2.INTER_AREA,
    )


def letterbox(
    image: np.ndarray,
    size: int = 640,
    fill: int = 0,
    align: str = "center",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fit an image into a ``size x size`` canvas while preserving aspect ratio.

    Padding is added with the ``fill`` color so the short dimension is centered.
    This matches YOLO's inference convention where boxes are computed in the
    resized coordinate space and the padding must be accounted for downstream.

    Args:
        image: Input image (grayscale or BGR).
        size: Target square canvas size.
        fill: Grayscale/BGR fill value for the letterbox padding (0-255).
        align: Padding alignment, ``"center"`` or ``"top_left"``.

    Returns:
        A tuple ``(canvas, params)`` where ``params`` is
        ``(scale, pad_x, pad_y, total_pad)`` needed to map normalized box
        coordinates back to the original image:
            ``scale``   - scale factor applied to the original image
            ``pad_x``   - horizontal padding on the top-left before the image
            ``pad_y``   - vertical padding on the top-left before the image
            ``total_pad`` - original image size if the image was not resized
                            (use the returned scale to guide de-annotation)
    """
    h, w = image.shape[:2]

    ratio = min(size / float(h), size / float(w))
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Preserve the input dtype so float images (e.g. normalized [0,1]) are not
    # truncated when written into the canvas.
    if len(image.shape) == 3:
        canvas = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)
    else:
        canvas = np.full((size, size), fill, dtype=image.dtype)

    if align == "center":
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
    else:
        pad_x = 0
        pad_y = 0

    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    params = (ratio, pad_x, pad_y, size)
    return canvas, params
