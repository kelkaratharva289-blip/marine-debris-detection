"""Image loading utilities for sonar imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageSource = Union[str, Path, np.ndarray]


def load_image(source: ImageSource, color: bool = False) -> np.ndarray:
    """Load an image from a file path or accept an in-memory ndarray.

    Sonar scans are typically single-channel (grayscale) intensity maps, so the
    image is loaded in grayscale by default. Pass ``color=True`` to load as a
    three-channel BGR image instead.

    Args:
        source: Path to an image file, or an existing ``np.ndarray`` BGR/RGB
            image passed in-memory.
        color: If True, load as a 3-channel BGR image; otherwise grayscale.

    Returns:
        Loaded image as a ``np.ndarray``. Grayscale images are returned with
        shape ``(H, W)`` (single channel); color images as ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    if isinstance(source, np.ndarray):
        return _prepare_array(source, color)

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    read_flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), read_flag)

    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def _prepare_array(array: np.ndarray, color: bool) -> np.ndarray:
    """Normalize an in-memory ndarray to the expected cv2 dtype/channels."""
    arr = np.asarray(array)

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        # Grayscale already
        if color:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    if color:
        return arr

    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {arr.shape}")
