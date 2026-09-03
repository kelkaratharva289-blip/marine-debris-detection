"""CLAHE contrast enhancement for sonar imagery."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Sonar images often have uneven illumination and low local contrast, with
    dark seafloor and bright reflectors in the same frame. Global histogram
    equalization over-amplifies noise in flat areas; CLAHE operates on local
    tiles and clips the histogram to avoid over-enhancement.

    Args:
        image: Grayscale input image ``(H, W)``. Multi-channel images are
            converted to grayscale.
        clip_limit: Threshold for contrast limiting (typical 1.0-4.0).
        grid_size: Size of the tile grid, e.g. 8 divides the image into an
            8x8 grid of tiles.

    Returns:
        CLAHE-enhanced grayscale image with the same shape.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grids = max(1, int(grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(grids, grids),
    )
    return clahe.apply(image)
