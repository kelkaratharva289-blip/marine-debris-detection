"""Unit tests for the sonar preprocessing module."""

import numpy as np
import pytest

from app.ai.preprocessing import (
    DEFAULT_CONFIG,
    PreprocessConfig,
    apply_clahe,
    ensure_3channel,
    letterbox,
    load_image,
    normalize_minmax,
    preprocess,
    preprocess_to_uint8,
    reduce_noise,
    reduce_noise_median,
    resize,
    standardize,
    to_grayscale,
    to_uint8,
)


def make_grayscale(h=100, w=120):
    """Create a deterministic grayscale test image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def make_color(h=100, w=120):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# loader / grayscale
# ---------------------------------------------------------------------------


class TestGrayscale:
    def test_2d_passthrough(self):
        img = make_grayscale()
        assert to_grayscale(img).ndim == 2

    def test_color_to_gray(self):
        img = make_color()
        out = to_grayscale(img)
        assert out.ndim == 2
        assert out.shape[:2] == img.shape[:2]

    def test_ensure_3channel(self):
        img = make_grayscale()
        out = ensure_3channel(img)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_load_single_channel(self, tmp_path):
        p = tmp_path / "scan.png"
        img = make_grayscale()
        import cv2

        cv2.imwrite(str(p), img)
        loaded = load_image(str(p))
        assert loaded.ndim == 2
        assert loaded.shape == img.shape


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------


class TestDenoise:
    def test_shape_preserved(self):
        img = make_grayscale()
        out = reduce_noise(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_median_shape_preserved(self):
        img = make_grayscale()
        out = reduce_noise_median(img, kernel_size=3)
        assert out.shape == img.shape
        assert reduce_noise_median(img, kernel_size=4).shape == img.shape  # even -> odd


# ---------------------------------------------------------------------------
# clahe
# ---------------------------------------------------------------------------


class TestClahe:
    def test_shape_and_uint8(self):
        img = make_grayscale()
        out = apply_clahe(img, clip_limit=2.0, grid_size=8)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_color_input_converted(self):
        img = make_color()
        out = apply_clahe(img)
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_minmax_range(self):
        img = make_grayscale().astype(np.float32)
        out = normalize_minmax(img, 0.0, 1.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert out.dtype == np.float32

    def test_constant_image(self):
        img = np.full((10, 10), 7, dtype=np.uint8)
        out = normalize_minmax(img, 0.0, 1.0)
        assert np.allclose(out, 0.0)

    def test_standardize_255(self):
        img = make_grayscale().astype(np.float32)
        out = standardize(img, 0.0, 255.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_to_uint8(self):
        f = normalize_minmax(make_grayscale(), 0.0, 1.0)
        u = to_uint8(f)
        assert u.dtype == np.uint8
        assert u.max() <= 255


# ---------------------------------------------------------------------------
# resize / letterbox
# ---------------------------------------------------------------------------


class TestResize:
    def test_square_resize(self):
        img = make_grayscale(60, 90)
        out = resize(img, 128)
        assert out.shape == (128, 128)

    def test_letterbox_fits_and_pads(self):
        img = make_grayscale(60, 90)
        canvas, params = letterbox(img, size=128)
        assert canvas.shape == (128, 128)
        scale, pad_x, pad_y, size = params
        assert size == 128
        # The landscape image fits with vertical padding only.
        assert pad_y > 0
        assert scale > 0

    def test_letterbox_grayscale(self):
        img = make_grayscale(100, 100)
        canvas, _ = letterbox(img, size=64)
        assert canvas.shape == (64, 64)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_preprocess_shape_and_range(self):
        img = make_grayscale(240, 320)
        out = preprocess(img, DEFAULT_CONFIG)
        assert out.shape == (DEFAULT_CONFIG.target_size, DEFAULT_CONFIG.target_size)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert out.dtype == np.float32

    def test_preprocess_returns_params(self):
        img = make_grayscale(240, 320)
        out, params = preprocess(img, DEFAULT_CONFIG, return_param=True)
        assert out.shape == (640, 640)
        assert len(params) == 4

    def test_preprocess_keeps_float_range_after_resize(self):
        # Normalization produces floats in [0,1]; letterbox resize must not
        # truncate them into uint8 buckets (regression guard).
        img = make_grayscale(240, 320)
        cfg = PreprocessConfig(target_size=128, enable_resize=True)
        out = preprocess(img, cfg)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_preprocess_to_uint8(self):
        img = make_grayscale(100, 200)
        out = preprocess_to_uint8(img, DEFAULT_CONFIG)
        assert out.dtype == np.uint8
        assert out.shape == (640, 640)

    def test_preprocess_custom_size(self):
        img = make_grayscale(100, 100)
        cfg = PreprocessConfig(target_size=224)
        out = preprocess(img, cfg)
        assert out.shape == (224, 224)

    def test_preprocess_from_path(self, tmp_path):
        p = tmp_path / "scan.png"
        import cv2

        cv2.imwrite(str(p), make_grayscale(100, 100))
        cfg = PreprocessConfig(target_size=256)
        out = preprocess(str(p), cfg)
        assert out.shape == (256, 256)

    def test_pipeline_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            preprocess(str(tmp_path / "nope.png"))

    def test_skip_steps(self):
        img = make_color(64, 64)
        cfg = PreprocessConfig(
            enable_grayscale=False,
            enable_denoise=False,
            enable_clahe=False,
            enable_normalize=False,
            enable_resize=False,
        )
        out = preprocess(img, cfg)
        # With all processing disabled and no resize, the pipeline should return
        # the image basically unchanged in shape (the loader reduces to
        # grayscale for sonar). dtype/shape should be preserved from load.
        assert out.shape == (64, 64)
        assert out.dtype == np.uint8


class TestConfigFromSettings:
    def test_from_settings_uses_defaults_when_missing(self):
        class Fake:
            pass

        cfg = PreprocessConfig.from_settings(Fake())
        assert cfg.target_size == 640

    def test_from_settings_uses_values(self):
        class Fake:
            PREPROCESS_IMG_SIZE = 512

        cfg = PreprocessConfig.from_settings(Fake())
        assert cfg.target_size == 512
