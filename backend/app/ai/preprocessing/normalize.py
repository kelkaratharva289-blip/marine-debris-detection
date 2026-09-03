"""Image normalization for sonar imagery."""

from __future__ import annotations

import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a normalized float image back to an 8-bit uint8 image.

    Args:
        image: Float image in the [0, 1] range.

    Returns:
        ``np.uint8`` image in the [0, 255] range.
    """
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def normalize_minmax(
    image: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """Normalize pixel intensities to a target ``[min_val, max_val]`` range.

    Uses the image-specific min/max (min-max normalization), preserving the
    full dynamic range of the scan regardless of its original scale (e.g.
    float sonar returns or 16-bit depth).

    Args:
        image: Input image of any numeric dtype.
        min_val: Lower bound of the output range.
        max_val: Upper bound of the output range.

    Returns:
        Normalized image as ``float32``.
    """
    arr = np.asarray(image, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())

    if hi - lo < 1e-9:
        # Degenerate (constant) image: map everything to the lower bound.
        return np.full_like(arr, min_val)

    scaled = (arr - lo) / (hi - lo)  # [0, 1]
    return scaled * (max_val - min_val) + min_val


def standardize(
    image: np.ndarray, mean: float = 0.0, std: float = 255.0
) -> np.ndarray:
    """Standardize an image by subtracting ``mean`` and dividing by ``std``.

    With defaults ``mean=0`` / ``std=255`` this rescales an 8-bit [0, 255]
    image to [0, 1], which is a common input format for CNNs / YOLO.

    Args:
        image: Input image.
        mean: Mean value to subtract.
        std: Standard deviation to divide by.

    Returns:
        Standardized ``float32`` image.
    """
    arr = np.asarray(image, dtype=np.float32)
    guard = float(std) if float(std) != 0 else 1.0
    return (arr - float(mean)) / guard
