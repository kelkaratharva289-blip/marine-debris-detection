"""Noise reduction for sonar imagery."""

from __future__ import annotations

import cv2
import numpy as np


def reduce_noise(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma_color: float = 45.0,
    sigma_space: float = 15.0,
) -> np.ndarray:
    """Reduce speckle / sensor noise with a bilateral filter.

    Bilateral filtering smooths homogeneous regions while preserving edges —
    important for sonar where debris boundaries are subtle. The filter works on
    a single-channel image, so the input is reduced to grayscale first if needed.

    Args:
        image: Input image (grayscale or color).
        kernel_size: Diameter of each pixel neighborhood. Must be a positive
            odd integer (>= 1). ``1`` disables the spatial component.
        sigma_color: Filter sigma in color space. Larger values blur stronger.
        sigma_space: Filter sigma in coordinate space. Larger values blur
            stronger for farther pixels.

    Returns:
        Denoised grayscale image with shape ``(H, W)``.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = max(1, kernel_size)
    return cv2.bilateralFilter(
        image,
        d=kernel,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def reduce_noise_median(
    image: np.ndarray, kernel_size: int = 5
) -> np.ndarray:
    """Reduce salt-and-pepper noise with a median filter.

    A useful alternative when sonar noise is predominantly impulsive. The
    kernel size must be a positive odd integer.

    Args:
        image: Input image.
        kernel_size: Median filter kernel size (odd, >= 1).

    Returns:
        Noise-reduced grayscale image.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = max(1, kernel_size)
    if kernel % 2 == 0:
        kernel += 1
    return cv2.medianBlur(image, kernel)
