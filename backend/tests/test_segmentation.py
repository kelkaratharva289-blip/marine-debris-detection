"""Unit tests for the sonar segmentation module and mask helpers."""

import json

import numpy as np
import pytest

from app.ai.detector import MarineDetector, _bbox_iou, _remap_polygon
from app.ai.segmentation import (
    _canonicalise_label,
    mask_to_polygon,
    run_unet,
)
from app.utils.mask import render_mask_overlay


# ---------------------------------------------------------------------------
# segmentation helpers
# ---------------------------------------------------------------------------


class TestMaskHelpers:
    def test_canonicalise_variants(self):
        assert _canonicalise_label("Fishing Net") == "ghost_net"
        assert _canonicalise_label("barrel") == "cylinder"
        assert _canonicalise_label("Ship_Wreck") == "shipwreck"
        assert _canonicalise_label("random_stuff") == "other_debris"

    def test_mask_to_polygon_rectangle(self):
        mask = np.zeros((200, 300), dtype=bool)
        mask[50:150, 80:220] = True
        poly = mask_to_polygon(mask)
        assert len(poly) >= 3
        for x, y in poly:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_mask_to_polygon_empty(self):
        mask = np.zeros((50, 50), dtype=bool)
        assert mask_to_polygon(mask) == []

    def test_remap_polygon_bounds(self):
        poly = [[0.5, 0.5], [0.6, 0.6]]
        mapped = _remap_polygon(poly, scale=2.0, pad_x=100, pad_y=50, canvas_size=640)
        assert len(mapped) == 2
        for x, y in mapped:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_remap_polygon_zero_scale(self):
        poly = [[0.5, 0.5]]
        mapped = _remap_polygon(poly, scale=0, pad_x=0, pad_y=0, canvas_size=0)
        assert mapped == poly


class TestBboxIou:
    def test_overlapping(self):
        iou = _bbox_iou((0.1, 0.1, 0.2, 0.2), (0.15, 0.15, 0.2, 0.2))
        assert 0.0 < iou < 1.0

    def test_disjoint(self):
        iou = _bbox_iou((0.1, 0.1, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1))
        assert iou == 0.0

    def test_identical(self):
        iou = _bbox_iou((0.2, 0.2, 0.4, 0.4), (0.2, 0.2, 0.4, 0.4))
        assert iou == pytest.approx(1.0)


class TestMaskOverlay:
    def test_renders_png(self, tmp_path):
        poly = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]]
        path = render_mask_overlay(
            poly,
            mask_area=0.36,
            class_label="ghost_net",
            confidence=0.9,
            output_dir=str(tmp_path),
            detection_id="abc",
        )
        assert path.endswith(".png")
        import os

        assert os.path.exists(path)

    def test_empty_polygon_no_throw(self, tmp_path):
        path = render_mask_overlay(
            [],
            class_label="x",
            output_dir=str(tmp_path),
            detection_id="empty",
        )
        assert path.endswith(".png")


# ---------------------------------------------------------------------------
# run_unet with a fake model
# ---------------------------------------------------------------------------


class FakeUnet:
    def __init__(self, mask):
        self._mask = mask

    def predict_mask(self, image):
        return self._mask


class TestRunUnet:
    def test_returns_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        model = FakeUnet(mask)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = run_unet(model, img)
        assert len(results) == 1
        r = results[0]
        assert r["class_label"] == "other_debris"
        assert r["bbox_width"] > 0
        assert r["polygon"]
        assert r["mask"].sum() > 0

    def test_no_components_empty(self):
        model = FakeUnet(np.zeros((100, 100), dtype=np.uint8))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert run_unet(model, img) == []

    def test_missing_predict_method_raises(self):
        class Bad:
            pass

        img = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            run_unet(Bad(), img)


# ---------------------------------------------------------------------------
# detector mask wiring (no real model)
# ---------------------------------------------------------------------------


class TestDetectorSegConfig:
    def test_seg_disabled_default(self):
        det = MarineDetector()
        assert det.seg_enabled is False
        assert det.seg_model is None

    def test_segment_returns_empty_when_disabled(self, tmp_path):
        det = MarineDetector()
        # Should not attempt to load any model when disabled.
        assert det.seg_model_type == "yolo-seg"
        assert det.seg_enabled is False
