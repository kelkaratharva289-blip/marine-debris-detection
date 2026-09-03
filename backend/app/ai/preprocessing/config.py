"""Sonar image preprocessing configuration."""

from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    """Configuration for the sonar preprocessing pipeline.

    All parameters have sensible defaults tuned for side-scan sonar imagery
    (high dynamic range, speckle noise, low local contrast). Individual steps
    are skipped or adjusted by passing explicit None / defaults.
    """

    # Resizing
    target_size: int = 640

    # Noise reduction (bilateral filter)
    denoise_kernel: int = 5
    denoise_sigma_color: float = 45.0
    denoise_sigma_space: float = 15.0

    # CLAHE contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Normalization (output range)
    normalize_min: float = 0.0
    normalize_max: float = 1.0

    # Pipeline controls: set a flag to False to skip a step
    enable_grayscale: bool = True
    enable_denoise: bool = True
    enable_clahe: bool = True
    enable_normalize: bool = True
    enable_resize: bool = True

    @classmethod
    def from_settings(cls, settings) -> "PreprocessConfig":
        """Build a config from an app settings object (falls back to defaults)."""
        return cls(
            target_size=getattr(settings, "PREPROCESS_IMG_SIZE", 640),
            denoise_kernel=getattr(settings, "PREPROCESS_DENOISE_KERNEL", 5),
            denoise_sigma_color=getattr(
                settings, "PREPROCESS_DENOISE_SIGMA_COLOR", 45.0
            ),
            denoise_sigma_space=getattr(
                settings, "PREPROCESS_DENOISE_SIGMA_SPACE", 15.0
            ),
            clahe_clip_limit=getattr(
                settings, "PREPROCESS_CLAHE_CLIP_LIMIT", 2.0
            ),
            clahe_grid_size=getattr(settings, "PREPROCESS_CLAHE_GRID_SIZE", 8),
            normalize_min=getattr(settings, "PREPROCESS_NORMALIZE_MIN", 0.0),
            normalize_max=getattr(settings, "PREPROCESS_NORMALIZE_MAX", 1.0),
        )


DEFAULT_CONFIG = PreprocessConfig()
