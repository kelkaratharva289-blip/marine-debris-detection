"""Grayscale conversion for sonar imagery."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel grayscale image.

    Side-scan sonar returns are fundamentally single-channel acoustic intensity
    maps. If the input is already grayscale (shape ``(H, W)``) it is returned
    unchanged. Color images (BGR or RGB) are converted using the standard
    luma weights.

    Args:
        image: Input image. Grayscale ``(H, W)`` or multi-channel ``(H, W, C)``.

    Returns:
        Grayscale image with shape ``(H, W)``.
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]

        # cv2 assumes BGR ordering; handle a possible RGB input by
        # interpreting via cvtColor. For other channel counts (e.g. 4 with
        # alpha) drop the extra channels first.
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale: {image.shape}")


def ensure_3channel(image: np.ndarray) -> np.ndarray:
    """Expand a grayscale image to a 3-channel BGR image.

    Some downstream filters (e.g. bilateral) are faster/simpler on 1-channel,
    but other tools expect 3 channels. This helper provides a consistent way
    to go back to a 3-channel representation.

    Args:
        image: Grayscale ``(H, W)`` or any image.

    Returns:
        Three-channel BGR image ``(H, W, 3)``.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
