"""Sonar image preprocessing for YOLO-ready marine debris detection.

Exposes a high-level pipeline (:func:`preprocess`) plus each step as an
independent, importable function so consumers can compose their own flow.

Example:
    >>> from app.ai.preprocessing import preprocess
    >>> image = preprocess("scan.tif", return_param=False)

    >>> from app.ai.preprocessing.pipeline import preprocess_to_uint8
    >>> img8, params = preprocess_to_uint8("scan.tif", return_param=True)
"""

from app.ai.preprocessing.clahe import apply_clahe
from app.ai.preprocessing.config import DEFAULT_CONFIG, PreprocessConfig
from app.ai.preprocessing.denoise import reduce_noise, reduce_noise_median
from app.ai.preprocessing.grayscale import ensure_3channel, to_grayscale
from app.ai.preprocessing.loader import load_image
from app.ai.preprocessing.normalize import (
    normalize_minmax,
    standardize,
    to_uint8,
)
from app.ai.preprocessing.pipeline import (
    PreprocessingError,
    preprocess,
    preprocess_to_uint8,
)
from app.ai.preprocessing.resize import letterbox, resize

__all__ = [
    # pipeline
    "preprocess",
    "preprocess_to_uint8",
    "PreprocessingError",
    # config
    "PreprocessConfig",
    "DEFAULT_CONFIG",
    # steps
    "load_image",
    "to_grayscale",
    "ensure_3channel",
    "reduce_noise",
    "reduce_noise_median",
    "apply_clahe",
    "normalize_minmax",
    "standardize",
    "to_uint8",
    "resize",
    "letterbox",
]
