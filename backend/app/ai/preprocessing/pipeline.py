"""Sonar preprocessing pipeline orchestrator."""

from __future__ import annotations

from app.ai.preprocessing import grayscale, loader
from app.ai.preprocessing.clahe import apply_clahe
from app.ai.preprocessing.config import DEFAULT_CONFIG, PreprocessConfig
from app.ai.preprocessing.denoise import reduce_noise
from app.ai.preprocessing.normalize import normalize_minmax
from app.ai.preprocessing.resize import letterbox


class PreprocessingError(Exception):
    """Raised when the sonar preprocessing pipeline fails."""

    def __init__(self, message: str, source_error: Exception | None = None):
        super().__init__(message)
        self.source_error = source_error


def preprocess(
    source,
    config: PreprocessConfig | None = None,
    return_param: bool = False,
):
    """Run the full sonar preprocessing pipeline.

    Pipeline order:
        1. Load image (grayscale by default)
        2. Grayscale conversion
        3. Noise reduction (bilateral)
        4. CLAHE contrast enhancement
        5. Normalization (min-max to [0, 1])
        6. Resize to square (letterboxed)

    Args:
        source: Image file path or in-memory ``np.ndarray``.
        config: ``PreprocessConfig`` overrides. Defaults to ``DEFAULT_CONFIG``.
        return_param: If True, also return letterboxing parameters so detection
            boxes can be mapped back to the original image coordinates.

    Returns:
        If ``return_param`` is False: the preprocessed image as a float array in
            [0, 1] with shape ``(size, size)`` (grayscale) or ``(size, size, C)``.
        If ``return_param`` is True: a tuple ``(image, params)`` where ``params``
            is ``(scale, pad_x, pad_y, size)`` from :func:`letterbox`.

    Raises:
        PreprocessingError: If any stage fails.
    """
    cfg = config or DEFAULT_CONFIG

    try:
        # 1. Load
        img = loader.load_image(source, color=False)

        # 2. Grayscale
        if cfg.enable_grayscale:
            img = grayscale.to_grayscale(img)

        # 3. Noise reduction
        if cfg.enable_denoise:
            img = reduce_noise(
                img,
                kernel_size=cfg.denoise_kernel,
                sigma_color=cfg.denoise_sigma_color,
                sigma_space=cfg.denoise_sigma_space,
            )

        # 4. CLAHE contrast enhancement (requires uint8 intensity)
        if cfg.enable_clahe:
            img = apply_clahe(img, clip_limit=cfg.clahe_clip_limit, grid_size=cfg.clahe_grid_size)

        # 5. Normalization to [0, 1]
        if cfg.enable_normalize:
            img = normalize_minmax(img, min_val=cfg.normalize_min, max_val=cfg.normalize_max)

        # 6. Resize (letterboxed) — do this last so detections map cleanly
        if cfg.enable_resize:
            img, params = letterbox(img, size=cfg.target_size)
        else:
            params = (1.0, 0.0, 0.0, int(img.shape[1]))

        if return_param:
            return img, params

        return img

    except PreprocessingError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap any downstream failure
        raise PreprocessingError(
            f"Sonar preprocessing failed: {exc}", source_error=exc
        ) from exc


def preprocess_to_uint8(
    source,
    config: PreprocessConfig | None = None,
    return_param: bool = False,
):
    """Run the pipeline and return the result as an 8-bit uint8 image.

    Useful for saving debug visualizations or feeding models expecting [0, 255].

    Args:
        source: Image file path or in-memory ``np.ndarray``.
        config: Optional preprocessing configuration.
        return_param: Whether to also return letterboxing parameters.

    Returns:
        8-bit preprocessed image, or ``(image, params)`` if ``return_param``.
    """
    from app.ai.preprocessing.normalize import to_uint8

    cfg = config or DEFAULT_CONFIG
    if not cfg.enable_normalize:
        raise PreprocessingError(
            "preprocess_to_uint8 requires normalization to be enabled"
        )

    img, params = preprocess(source, config=cfg, return_param=True)
    img = to_uint8(img)

    if return_param:
        return img, params
    return img
