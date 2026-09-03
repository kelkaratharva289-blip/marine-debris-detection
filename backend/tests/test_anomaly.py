"""Unit tests for the anomaly classification layer."""

import numpy as np
import pytest

from app.ai.anomaly import (
    AnomalyFeatureConfig,
    classify_detection,
    _ai_score,
    _bounded,
    _combine,
    _shape_score,
    _texture_score,
    _shadow_score,
)


def _artificial_detection():
    """A bright pipe with a strong acoustic shadow (clearly artificial)."""
    img = np.full((200, 300), 80, np.uint8)
    img[60:110, 100:210] = 220   # bright sonar highlight
    img[110:180, 90:220] = 5     # deep acoustic shadow down-track
    mask = np.zeros((200, 300), dtype=bool)
    mask[60:110, 100:210] = True
    det = {
        "bbox_x": 0.33,
        "bbox_y": 0.30,
        "bbox_width": 0.37,
        "bbox_height": 0.25,
        "confidence": 0.9,
        "class_label": "pipe",
        "mask": mask,
    }
    return det, img


class TestAnomalyClassification:
    def test_artificial_label(self):
        det, img = _artificial_detection()
        result = classify_detection(det, img)
        assert result.label == "artificial"
        assert result.artificial_probability > result.natural_probability

    def test_probabilities_sum_to_one(self):
        det, img = _artificial_detection()
        result = classify_detection(det, img)
        assert abs(
            (result.natural_probability + result.artificial_probability) - 1.0
        ) < 1e-6
        assert 0.0 <= result.natural_probability <= 1.0
        assert 0.0 <= result.artificial_probability <= 1.0

    def test_natural_label(self):
        # Low-lying, low-contrast region: natural-leaning evidence.
        img = np.full((200, 300), 128, np.uint8)
        img[90:130, 130:180] = 140
        det = {
            "bbox_x": 0.43,
            "bbox_y": 0.45,
            "bbox_width": 0.17,
            "bbox_height": 0.2,
            "confidence": 0.35,
            "class_label": "ghost_net",
        }
        result = classify_detection(det, img)
        assert result.label == "natural"
        assert result.natural_probability > result.artificial_probability

    def test_uncertain_when_weak_evidence(self):
        # Tiny region, near-zero confidence, uniform scene: not enough info.
        img = np.full((100, 100), 100, np.uint8)
        det = {
            "bbox_x": 0.5,
            "bbox_y": 0.5,
            "bbox_width": 0.1,
            "bbox_height": 0.1,
            "confidence": 0.05,
            "class_label": "other_debris",
        }
        result = classify_detection(det, img)
        assert result.label == "uncertain"

    def test_to_dict_shape(self):
        det, img = _artificial_detection()
        d = classify_detection(det, img).to_dict()
        assert d["anomaly_class"] in ("natural", "artificial", "uncertain")
        assert "natural_probability" in d
        assert "artificial_probability" in d
        assert "anomaly_confidence" in d
        assert "anomaly_features" in d
        assert {0.0, 1.0}.isdisjoint({d["anomaly_class"]})

    def test_feature_scores_present(self):
        det, img = _artificial_detection()
        feats = classify_detection(det, img).features
        for group in ("shape", "texture", "acoustic_shadow", "ai"):
            assert group in feats
            assert -1.0 <= feats[group] <= 1.0


class TestWeightRenormalisation:
    def test_no_mask_renormalises_logical(self):
        det, img = _artificial_detection()
        det.pop("mask", None)
        result = classify_detection(det, img)
        assert result.label in ("natural", "artificial", "uncertain")
        assert -1.0 <= result.evidence <= 1.0

    def test_weighterenormalization_total(self):
        cfg = AnomalyFeatureConfig(
            weight_shape=0.5,
            weight_texture=0.0,
            weight_acoustic_shadow=0.0,
            weight_ai=0.0,
        )
        score = _combine({"shape": 0.8, "texture": -1.0, "acoustic_shadow": 0.0, "ai": 0.0}, cfg)
        assert abs(score - 0.8) < 1e-6


class TestHelpers:
    def test_bounded_clipping(self):
        assert _bounded(10.0, 0.0, 1.0) == 1.0
        assert _bounded(-10.0, 0.0, 1.0) == -1.0
        assert _bounded(0.5, 0.0, 1.0) == 0.0

    def test_ai_prior_artificial(self):
        s = _ai_score(0.9, "pipe")
        assert s > 0.5

    def test_ai_prior_neutral_at_low_confidence(self):
        s = _ai_score(0.0, "pipe")
        assert -0.5 <= s <= 0.5


class TestConfigurableShapeParams:
    """Verify that shape feature thresholds are configurable."""

    def test_wide_aspect_ratio_range_relaxes_artificiality(self):
        det, img = _artificial_detection()
        # Default config scores the aspect as artificial (0.37/0.25 ~ 1.48).
        default_result = classify_detection(det, img)

        # Widen the aspect ratio range so the same bbox is neutral.
        cfg = AnomalyFeatureConfig(aspect_ratio_min=1.0, aspect_ratio_max=10.0)
        custom_result = classify_detection(det, img, config=cfg)
        # The shape score should be less artificial with the wider range.
        assert custom_result.features["shape"] <= default_result.features["shape"]

    def test_custom_compactness_range(self):
        det, img = _artificial_detection()
        cfg = AnomalyFeatureConfig(compactness_min=0.0, compactness_max=1.0)
        result = classify_detection(det, img, config=cfg)
        # Should still classify, not crash.
        assert result.label in ("natural", "artificial", "uncertain")

    def test_custom_shape_weights_change_score(self):
        det, img = _artificial_detection()
        # Only shape matters.
        cfg_shape_only = AnomalyFeatureConfig(
            weight_shape=1.0, weight_texture=0.0,
            weight_acoustic_shadow=0.0, weight_ai=0.0,
        )
        # Only AI matters.
        cfg_ai_only = AnomalyFeatureConfig(
            weight_shape=0.0, weight_texture=0.0,
            weight_acoustic_shadow=0.0, weight_ai=1.0,
        )
        r_shape = classify_detection(det, img, config=cfg_shape_only)
        r_ai = classify_detection(det, img, config=cfg_ai_only)
        # Different feature mixes should produce different evidence.
        assert r_shape.evidence != r_ai.evidence


class TestConfigurableTextureParams:
    """Verify that texture feature parameters are configurable."""

    def test_wide_canny_range(self):
        det, img = _artificial_detection()
        cfg = AnomalyFeatureConfig(canny_low=10, canny_high=200)
        result = classify_detection(det, img, config=cfg)
        assert result.label in ("natural", "artificial", "uncertain")
        assert -1.0 <= result.features["texture"] <= 1.0

    def test_narrow_canny_range(self):
        det, img = _artificial_detection()
        cfg = AnomalyFeatureConfig(canny_low=100, canny_high=120)
        result = classify_detection(det, img, config=cfg)
        assert -1.0 <= result.features["texture"] <= 1.0

    def test_texture_uniform_dampen_affects_uniform_regions(self):
        img = np.full((100, 100), 100, np.uint8)
        det = {
            "bbox_x": 0.1,
            "bbox_y": 0.1,
            "bbox_width": 0.8,
            "bbox_height": 0.8,
            "confidence": 0.8,
            "class_label": "container",
        }
        cfg_no_dampen = AnomalyFeatureConfig(texture_uniform_dampen=1.0)
        cfg_strong_dampen = AnomalyFeatureConfig(texture_uniform_dampen=0.1)
        r_no = classify_detection(det, img, config=cfg_no_dampen)
        r_strong = classify_detection(det, img, config=cfg_strong_dampen)
        # Stronger dampen on a uniform region should pull toward neutral.
        assert abs(r_strong.features["texture"]) <= abs(r_no.features["texture"]) + 0.01


class TestConfigurableShadowParams:
    """Verify that acoustic shadow parameters are configurable."""

    def test_custom_shadow_search_multiplier(self):
        det, img = _artificial_detection()
        cfg = AnomalyFeatureConfig(shadow_search_multiplier=5.0)
        result = classify_detection(det, img, config=cfg)
        assert result.label in ("natural", "artificial", "uncertain")
        assert -1.0 <= result.features["acoustic_shadow"] <= 1.0

    def test_custom_shadow_neutral_gates(self):
        det, img = _artificial_detection()
        # Very strict neutral gate: almost any shadow counts.
        cfg = AnomalyFeatureConfig(
            shadow_neutral_frac_gate=0.0,
            shadow_neutral_contrast_gate=0.0,
        )
        result = classify_detection(det, img, config=cfg)
        assert -1.0 <= result.features["acoustic_shadow"] <= 1.0

    def test_shadow_sub_weights_sum_to_one(self):
        det, img = _artificial_detection()
        cfg = AnomalyFeatureConfig(
            shadow_fraction_weight=0.5,
            shadow_depth_weight=0.25,
            shadow_contrast_weight=0.25,
        )
        result = classify_detection(det, img, config=cfg)
        assert result.label in ("natural", "artificial", "uncertain")


class TestFromSettings:
    """Verify from_settings populates feature-level config."""

    def test_from_settings_uses_custom_values(self):
        class MockSettings:
            ANOMALY_WEIGHT_SHAPE = 0.10
            ANOMALY_WEIGHT_TEXTURE = 0.10
            ANOMALY_WEIGHT_ACOUSTIC_SHADOW = 0.40
            ANOMALY_WEIGHT_AI = 0.40
            ANOMALY_UNCERTAIN_THRESHOLD = 0.20
            ANOMALY_ASPECT_RATIO_MIN = 1.5
            ANOMALY_ASPECT_RATIO_MAX = 6.0
            ANOMALY_COMPACTNESS_MIN = 0.10
            ANOMALY_COMPACTNESS_MAX = 0.90
            ANOMALY_CONCAVITY_MIN = -0.5
            ANOMALY_CONCAVITY_MAX = 0.6
            ANOMALY_SHAPE_GEOMETRY_WEIGHT = 0.40
            ANOMALY_SHAPE_CONVEXITY_WEIGHT = 0.30
            ANOMALY_SHAPE_ASPECT_WEIGHT = 0.30
            ANOMALY_SHAPE_ASPECT_DAMPEN = 0.3
            ANOMALY_CANNY_LOW = 30
            ANOMALY_CANNY_HIGH = 180
            ANOMALY_TEXTURE_CONTRAST_MIN = 0.15
            ANOMALY_TEXTURE_CONTRAST_MAX = 3.0
            ANOMALY_TEXTURE_EDGE_MIN = 0.01
            ANOMALY_TEXTURE_EDGE_MAX = 0.5
            ANOMALY_TEXTURE_RELIABILITY_PIXELS = 500
            ANOMALY_TEXTURE_UNIFORM_CONTRAST_GATE = 0.40
            ANOMALY_TEXTURE_UNIFORM_SCORE_GATE = 0.60
            ANOMALY_TEXTURE_UNIFORM_DAMPEN = 0.30
            ANOMALY_SHADOW_SEARCH_MULTIPLIER = 3.0
            ANOMALY_SHADOW_SEARCH_MIN_PX = 8
            ANOMALY_SHADOW_FRACTION_MIN = 0.08
            ANOMALY_SHADOW_FRACTION_MAX = 0.7
            ANOMALY_SHADOW_DEPTH_MIN = 0.08
            ANOMALY_SHADOW_DEPTH_MAX = 0.85
            ANOMALY_SHADOW_STRIP_CONTRAST_MIN = 0.08
            ANOMALY_SHADOW_STRIP_CONTRAST_MAX = 0.85
            ANOMALY_SHADOW_FRACTION_WEIGHT = 0.50
            ANOMALY_SHADOW_DEPTH_WEIGHT = 0.25
            ANOMALY_SHADOW_CONTRAST_WEIGHT = 0.25
            ANOMALY_SHADOW_NEUTRAL_FRAC_GATE = 0.03
            ANOMALY_SHADOW_NEUTRAL_CONTRAST_GATE = 0.06

        cfg = AnomalyFeatureConfig.from_settings(MockSettings())
        assert cfg.weight_shape == 0.10
        assert cfg.aspect_ratio_min == 1.5
        assert cfg.aspect_ratio_max == 6.0
        assert cfg.compactness_min == 0.10
        assert cfg.compactness_max == 0.90
        assert cfg.canny_low == 30
        assert cfg.canny_high == 180
        assert cfg.shadow_search_multiplier == 3.0
        assert cfg.shadow_search_min_px == 8
        assert cfg.shadow_neutral_frac_gate == 0.03

    def test_from_settings_defaults(self):
        cfg = AnomalyFeatureConfig.from_settings(object())
        assert cfg.weight_shape == 0.25
        assert cfg.aspect_ratio_min == 1.0
        assert cfg.canny_low == 50
        assert cfg.shadow_search_multiplier == 2.5
