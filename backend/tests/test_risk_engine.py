"""Unit tests for the Marine Risk Scoring Engine."""

import pytest

from app.ai.risk_engine import (
    OBJECT_TYPE_RISK_PRIOR,
    RiskConfig,
    RiskLevel,
    compute_risk,
    level_for,
)


class TestRiskScoring:
    def test_high_factors_is_high_risk(self):
        # Confident, strongly-artificial, hazardous object type, large ->
        # high risk.
        r = compute_risk(
            confidence=0.9,
            object_type="container",
            estimated_size=0.4,
            artificial_probability=0.95,
            anomaly_class="artificial",
        )
        assert r.object_type_risk == pytest.approx(OBJECT_TYPE_RISK_PRIOR["container"])
        assert r.estimated_size == pytest.approx(1.0)
        assert r.risk_score >= 75
        assert r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_natural_is_low_risk(self):
        r = compute_risk(
            confidence=0.6,
            object_type="ghost_net",
            estimated_size=0.01,
            artificial_probability=0.2,
            anomaly_class="natural",
        )
        assert r.risk_score < 50
        assert r.risk_level == RiskLevel.LOW

    def test_risk_score_in_range(self):
        for conf in (0.1, 0.5, 0.9):
            r = compute_risk(
                confidence=conf,
                object_type="pipe",
                estimated_size=conf,
                artificial_probability=conf,
                anomaly_class="artificial",
            )
            assert 0.0 <= r.risk_score <= 100.0
            assert 0.0 <= r.final_confidence <= 1.0

    def test_object_type_raises_risk(self):
        # A hazardous object type raises risk vs. a benign one, all else equal.
        high = compute_risk(
            confidence=0.5,
            object_type="container",
            artificial_probability=0.5,
        )
        low = compute_risk(
            confidence=0.5,
            object_type="other_debris",
            artificial_probability=0.5,
        )
        assert high.risk_score > low.risk_score

    def test_larger_size_raises_risk(self):
        small = compute_risk(
            confidence=0.5,
            object_type="pipe",
            estimated_size=0.01,
            artificial_probability=0.5,
        )
        large = compute_risk(
            confidence=0.5,
            object_type="pipe",
            estimated_size=0.5,
            artificial_probability=0.5,
        )
        assert large.risk_score > small.risk_score

    def test_missing_inputs_do_not_crash(self):
        # Only confidence supplied — engine renormalises over available
        # evidence without fabricating missing factors.
        r = compute_risk(confidence=0.7)
        assert 0.0 <= r.risk_score <= 100.0
        assert 0.0 <= r.final_confidence <= 1.0

    def test_uncertain_penalised(self):
        cfg = RiskConfig(uncertain_penalty=0.5)
        a = compute_risk(
            confidence=0.8,
            object_type="container",
            artificial_probability=0.8,
            anomaly_class="artificial",
            config=cfg,
        )
        u = compute_risk(
            confidence=0.8,
            object_type="container",
            artificial_probability=0.8,
            anomaly_class="uncertain",
            config=cfg,
        )
        assert u.risk_score < a.risk_score

    def test_to_dict_keys(self):
        d = compute_risk(
            confidence=0.5,
            object_type="pipe",
            estimated_size=0.2,
            artificial_probability=0.5,
            anomaly_class="uncertain",
        ).to_dict()
        assert "object_type_risk" in d
        assert "estimated_size" in d
        assert "artificial_probability" in d
        assert "final_confidence" in d
        assert "risk_score" in d
        assert "risk_level" in d


class TestRiskLevels:
    def test_low_level(self):
        assert level_for(0) == RiskLevel.LOW
        assert level_for(49) == RiskLevel.LOW

    def test_medium_level(self):
        cfg = RiskConfig(medium_threshold=74.0)
        assert level_for(60, cfg) == RiskLevel.MEDIUM
        assert level_for(50, cfg) == RiskLevel.MEDIUM

    def test_high_level(self):
        cfg = RiskConfig(medium_threshold=74.0, high_threshold=89.0)
        assert level_for(75, cfg) == RiskLevel.HIGH
        assert level_for(89, cfg) == RiskLevel.HIGH

    def test_critical_level(self):
        assert level_for(90) == RiskLevel.CRITICAL
        assert level_for(100) == RiskLevel.CRITICAL

    def test_custom_thresholds(self):
        cfg = RiskConfig(
            low_threshold=30.0,
            medium_threshold=60.0,
            high_threshold=80.0,
        )
        assert level_for(40, cfg) == RiskLevel.MEDIUM
        assert level_for(85, cfg) == RiskLevel.CRITICAL


class TestConfigFromSettings:
    def test_from_settings_defaults(self):
        class FakeSettings:
            pass

        cfg = RiskConfig.from_settings(FakeSettings())
        assert cfg.weight_object_type == 0.25
        assert cfg.weight_confidence == 0.25
        assert cfg.weight_size == 0.25
        assert cfg.weight_artificial == 0.25
        assert cfg.low_threshold == 49.0
        assert cfg.medium_threshold == 74.0
        assert cfg.high_threshold == 89.0

    def test_from_settings_overrides(self):
        class FakeSettings:
            RISK_WEIGHT_OBJECT_TYPE = 0.1
            RISK_WEIGHT_CONFIDENCE = 0.3
            RISK_WEIGHT_SIZE = 0.3
            RISK_WEIGHT_ARTIFICIAL = 0.3
            RISK_SIZE_SMALL = 0.02
            RISK_SIZE_LARGE = 0.5
            RISK_HIGH_THRESHOLD = 85.0

        cfg = RiskConfig.from_settings(FakeSettings())
        assert cfg.weight_object_type == 0.1
        assert cfg.weight_confidence == 0.3
        assert cfg.weight_size == 0.3
        assert cfg.weight_artificial == 0.3
        assert cfg.size_small == 0.02
        assert cfg.size_large == 0.5
        assert cfg.high_threshold == 85.0
