"""Marine Risk Scoring Engine.

Computes a configurable **0-100 risk score** for each detected region by
fusing four interpretable factors:

* **Object type** — a per-class prior encoding how hazardous each debris
  class tends to be (containers, pipes, ships vs. nets / organic debris).
* **Confidence** — the detector's AI confidence: more certain detections
  carry more weight.
* **Estimated size** — the relative footprint of the object (derived from
  its bounding box area), normalised so larger objects score higher.
* **Artificial probability** — the anomaly classifier's belief that the
  region is anthropogenic (not natural seafloor).

The score is mapped to one of four levels:

* 0-49    ``low``
* 50-74   ``medium``
* 75-89   ``high``
* 90-100  ``critical``

Every weight and threshold is configurable via :class:`RiskConfig`. Inputs
that are unavailable are dropped and the remaining weights are
renormalised, so the engine never fabricates a factor for which no evidence
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    """Risk buckets with their configurable thresholds."""

    LOW = "low"                # 0-49
    MEDIUM = "medium"          # 50-74
    HIGH = "high"              # 75-89
    CRITICAL = "critical"      # 90-100


# Per-class risk prior in [0, 1]: how hazardous each marine debris class is
# considered to be. 0 = benign, 1 = maximally hazardous.
OBJECT_TYPE_RISK_PRIOR: dict[str, float] = {
    "ghost_net": 0.55,      # entangle wildlife but low physical hazard
    "shipwreck": 0.85,
    "pipe": 0.80,
    "cylinder": 0.70,
    "container": 0.90,
    "other_debris": 0.40,
}


@dataclass
class RiskConfig:
    """Configurable scoring formula for the Marine Risk Scoring Engine.

    Each weight contributes to the risk score; the engine renormalises over
    whichever inputs are provided, so weights need not sum to 1. The
    ``uncertain`` anomaly class is penalised so ambiguity never inflates
    risk.
    """

    weight_object_type: float = 0.25     # per-class hazard prior
    weight_confidence: float = 0.25      # detector AI confidence
    weight_size: float = 0.25            # estimated object size
    weight_artificial: float = 0.25      # anomaly artificiality probability
    uncertain_penalty: float = 0.30      # applied when anomaly_class == "uncertain"

    # Estimated-size mapping thresholds (bounding-box area as a fraction of
    # the image, roughly 0-1). A bbox fraction below ``size_small`` maps to
    # low risk, above ``size_large`` maps to high risk.
    size_small: float = 0.05
    size_large: float = 0.35

    low_threshold: float = 49.0          # < this            -> low
    medium_threshold: float = 74.0       # < this            -> medium
    high_threshold: float = 89.0         # >= this           -> high
                                         # otherwise          -> critical

    @classmethod
    def from_settings(cls, settings) -> "RiskConfig":
        return cls(
            weight_object_type=getattr(
                settings, "RISK_WEIGHT_OBJECT_TYPE", 0.25
            ),
            weight_confidence=getattr(settings, "RISK_WEIGHT_CONFIDENCE", 0.25),
            weight_size=getattr(settings, "RISK_WEIGHT_SIZE", 0.25),
            weight_artificial=getattr(settings, "RISK_WEIGHT_ARTIFICIAL", 0.25),
            uncertain_penalty=getattr(settings, "RISK_UNCERTAIN_PENALTY", 0.30),
            size_small=getattr(settings, "RISK_SIZE_SMALL", 0.05),
            size_large=getattr(settings, "RISK_SIZE_LARGE", 0.35),
            low_threshold=getattr(settings, "RISK_LOW_THRESHOLD", 49.0),
            medium_threshold=getattr(settings, "RISK_MEDIUM_THRESHOLD", 74.0),
            high_threshold=getattr(settings, "RISK_HIGH_THRESHOLD", 89.0),
        )


@dataclass
class RiskResult:
    """Output of the risk engine for a single detection."""

    object_type_risk: float          # 0-1 prior contribution
    estimated_size: float            # 0-1 normalised size
    artificial_probability: float    # anomaly classifier artificiality (0-1)
    final_confidence: float          # fused confidence (0-1)
    risk_score: float                # 0-100
    risk_level: RiskLevel            # low | medium | high | critical
    weights: dict = field(default_factory=dict)

    def to_dict(self, ai_confidence: float | None = None) -> dict:
        d = {
            "object_type_risk": round(float(self.object_type_risk), 4),
            "estimated_size": round(float(self.estimated_size), 4),
            "artificial_probability": round(float(self.artificial_probability), 4),
            "final_confidence": round(float(self.final_confidence), 4),
            "risk_score": round(float(self.risk_score), 2),
            "risk_level": self.risk_level.value,
        }
        if ai_confidence is not None:
            d["ai_confidence"] = round(float(ai_confidence), 4)
        return d


def compute_risk(
    confidence: float,
    object_type: str | None = None,
    estimated_size: float | None = None,
    artificial_probability: float | None = None,
    anomaly_class: str | None = None,
    config: RiskConfig | None = None,
) -> RiskResult:
    """Compute the 0-100 risk score for one detection.

    Args:
        confidence: Detector confidence in [0, 1].
        object_type: Canonical class label, e.g. ``"pipe"`` or
            ``"container"``. Used to look up the per-class hazard prior.
        estimated_size: Estimated object size as a fraction of the image in
            [0, 1] (typically ``bbox_width * bbox_height``). ``None`` if
            unknown.
        artificial_probability: Anomaly artificial probability in [0, 1].
        anomaly_class: Anomaly label ("natural" | "artificial" | "uncertain").
        config: Optional scoring configuration.

    Returns:
        :class:`RiskResult` with the contributing factors and a 0-100 risk
        score bucketed into a risk level.
    """
    cfg = config or RiskConfig()

    # Sanitise inputs so the formula can never produce out-of-range outputs.
    conf = _clamp01(confidence)
    art = (
        _clamp01(artificial_probability)
        if artificial_probability is not None
        else conf
    )
    obj_risk = _object_type_risk(object_type)
    obj_risk = max(0.0, min(1.0, obj_risk))

    # Estimated size: map bbox-area fraction [size_small, size_large] onto
    # [0, 1]. Guard against degenerate / missing values.
    size = _normalise_size(estimated_size, cfg)

    # Down-weight ambiguous (uncertain) regions so ambiguity never inflates
    # risk.
    effective_art = art
    if (anomaly_class or "").lower() == "uncertain":
        effective_art = max(0.0, art - _clamp01(cfg.uncertain_penalty))

    factors = {
        "object_type": obj_risk,
        "confidence": conf,
        "size": size,
    }
    weights = {
        "object_type": max(0.0, cfg.weight_object_type),
        "confidence": max(0.0, cfg.weight_confidence),
        "size": max(0.0, cfg.weight_size),
    }
    if artificial_probability is not None or art is not None:
        factors["artificial"] = effective_art
        weights["artificial"] = max(0.0, cfg.weight_artificial)

    risk_fraction = _weighted_mean(factors, weights)
    risk_score = float(_clamp01(risk_fraction) * 100.0)
    risk_level = level_for(risk_score, cfg)

    # Fused confidence: how much high-signal evidence is present (ignores the
    # "confidence" input itself standing in for uncertainty).
    fused_weights = {k: w for k, w in weights.items() if k != "confidence"}
    if fused_weights:
        fused_conf = _weighted_mean(
            {k: factors[k] for k in fused_weights}, fused_weights
        )
    else:
        fused_conf = _weighted_mean(factors, weights)

    return RiskResult(
        object_type_risk=round(float(obj_risk), 4),
        estimated_size=round(float(size), 4),
        artificial_probability=round(float(effective_art), 4),
        final_confidence=round(float(fused_conf), 4),
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _object_type_risk(object_type: str | None) -> float:
    """Return the 0-1 hazard prior for a canonical class label."""
    if not object_type:
        # Unknown / no object type: neutral prior (neither benign nor
        # hazardous) — do not assume.
        return 0.5
    return OBJECT_TYPE_RISK_PRIOR.get(object_type, 0.5)


def _normalise_size(size: float | None, cfg: RiskConfig) -> float:
    """Map an estimated-size fraction into [0, 1] using config thresholds.

    Values at/below ``size_small`` map to 0 (small/low risk), values at/above
    ``size_large`` map to 1 (large/high risk), linearly in between. Missing or
    invalid sizes map to a neutral 0.5 rather than being fabricated.
    """
    if size is None:
        return 0.5
    size = float(size)
    lo = max(0.0, min(cfg.size_small, cfg.size_large))
    hi = max(cfg.size_small, cfg.size_large)
    if hi <= lo:
        return 0.5
    return float(max(0.0, min(1.0, (size - lo) / (hi - lo))))


def _weighted_mean(factors: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of the active factors, renormalising over active weights."""
    active = [
        (factors[k], weights[k])
        for k in factors
        if weights.get(k, 0.0) > 0
    ]
    total = sum(w for _, w in active)
    if total <= 0:
        return 0.0
    return sum(v * w for v, w in active) / total


def level_for(risk_score: float, config: RiskConfig | None = None) -> RiskLevel:
    """Bucket a 0-100 risk score into a :class:`RiskLevel`.

    Thresholds are the upper bounds of each bucket: ``low_threshold`` is the
    max Low score, ``medium_threshold`` the max Medium score, and
    ``high_threshold`` the max High score; anything above ``high_threshold``
    is Critical. With defaults (49/74/89) this yields Low 0-49, Medium
    50-74, High 75-89, Critical 90-100.
    """
    cfg = config or RiskConfig()
    score = float(risk_score)
    if score <= cfg.low_threshold:
        return RiskLevel.LOW
    if score <= cfg.medium_threshold:
        return RiskLevel.MEDIUM
    if score <= cfg.high_threshold:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
