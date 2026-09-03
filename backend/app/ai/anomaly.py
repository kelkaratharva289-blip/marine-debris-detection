"""Anomaly classification layer for side-scan sonar detections.

Classifies each detected region as **Natural**, **Artificial**, or
**Uncertain** by fusing several *hand-crafted* sonar features with the
detector's AI confidence. No trained model is required — the classifier is
fully deterministic and, importantly, **never fabricates** evidence: if a
feature cannot be computed (e.g. no mask), its weight is renormalised over
the features that are available.

Feature groups (all configurable via :class:`AnomalyFeatureConfig`):

* **Shape** — compactness, convexity, aspect ratio and straight-edge
  fraction computed from the region mask (or its bounding box).
* **Texture** — intra-region contrast / edge density from the grayscale
  crop.
* **Acoustic shadow** — a dark, low-backscatter shadow cast beyond the
  bright sonar highlight. Artificial upright objects (pipes, cylinders,
  containers) cast a strong, well-defined shadow.
* **AI confidence** — the detector's confidence plus a per-class prior
  encoding how likely each debris class is to be artificial.

Each group produces a value in [-1, 1] (+1 strongly artificial, -1
strongly natural). The weighted sum maps to natural / artificial
probabilities via a softmax-like transform; when the two probabilities are
too close (margin < ``uncertain_threshold``) the region is labelled
Uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# Class prior: how "artificial" each marine debris class is expected to be.
# Values in [-1, 1]; +1 strongly artificial, -1 strongly natural, ~0 neutral.
CLASS_ARTIFICIALITY_PRIOR: dict[str, float] = {
    "ghost_net": -0.20,      # fishing nets are man-made but drift / tangle naturally
    "shipwreck": 0.60,
    "pipe": 0.90,
    "cylinder": 0.85,
    "container": 0.95,
    "other_debris": 0.10,
}


@dataclass
class AnomalyFeatureConfig:
    """Configuration for the anomaly classifier feature weights.

    Group weights control how much each feature group contributes to the
    final evidence score.  All weights are renormalised over whichever
    groups have non-zero weight, so they need not sum to 1.

    Feature-level parameters (shape, texture, shadow) control the internal
    thresholds of each scoring function, allowing tuning without code
    changes.
    """

    # --- group weights ---
    weight_shape: float = 0.25
    weight_texture: float = 0.20
    weight_acoustic_shadow: float = 0.25
    weight_ai: float = 0.30
    uncertain_threshold: float = 0.15

    # --- shape feature parameters ---
    aspect_ratio_min: float = 1.0
    aspect_ratio_max: float = 4.0
    compactness_min: float = 0.05
    compactness_max: float = 0.8
    concavity_min: float = -0.3
    concavity_max: float = 0.4
    shape_geometry_weight: float = 0.5
    shape_convexity_weight: float = 0.35
    shape_aspect_weight: float = 0.15
    shape_aspect_dampen: float = 0.5

    # --- texture feature parameters ---
    canny_low: int = 50
    canny_high: int = 150
    texture_contrast_min: float = 0.2
    texture_contrast_max: float = 2.5
    texture_edge_min: float = 0.02
    texture_edge_max: float = 0.4
    texture_reliability_pixels: int = 900
    texture_uniform_contrast_gate: float = 0.35
    texture_uniform_score_gate: float = 0.7
    texture_uniform_dampen: float = 0.4

    # --- acoustic shadow feature parameters ---
    shadow_search_multiplier: float = 2.5
    shadow_search_min_px: int = 4
    shadow_fraction_min: float = 0.05
    shadow_fraction_max: float = 0.6
    shadow_depth_min: float = 0.05
    shadow_depth_max: float = 0.9
    shadow_strip_contrast_min: float = 0.05
    shadow_strip_contrast_max: float = 0.9
    shadow_fraction_weight: float = 0.4
    shadow_depth_weight: float = 0.3
    shadow_contrast_weight: float = 0.3
    shadow_neutral_frac_gate: float = 0.02
    shadow_neutral_contrast_gate: float = 0.05

    @classmethod
    def from_settings(cls, settings) -> "AnomalyFeatureConfig":
        return cls(
            weight_shape=getattr(settings, "ANOMALY_WEIGHT_SHAPE", 0.25),
            weight_texture=getattr(settings, "ANOMALY_WEIGHT_TEXTURE", 0.20),
            weight_acoustic_shadow=getattr(
                settings, "ANOMALY_WEIGHT_ACOUSTIC_SHADOW", 0.25
            ),
            weight_ai=getattr(settings, "ANOMALY_WEIGHT_AI", 0.30),
            uncertain_threshold=getattr(
                settings, "ANOMALY_UNCERTAIN_THRESHOLD", 0.15
            ),
            # shape
            aspect_ratio_min=getattr(
                settings, "ANOMALY_ASPECT_RATIO_MIN", 1.0
            ),
            aspect_ratio_max=getattr(
                settings, "ANOMALY_ASPECT_RATIO_MAX", 4.0
            ),
            compactness_min=getattr(
                settings, "ANOMALY_COMPACTNESS_MIN", 0.05
            ),
            compactness_max=getattr(
                settings, "ANOMALY_COMPACTNESS_MAX", 0.8
            ),
            concavity_min=getattr(settings, "ANOMALY_CONCAVITY_MIN", -0.3),
            concavity_max=getattr(settings, "ANOMALY_CONCAVITY_MAX", 0.4),
            shape_geometry_weight=getattr(
                settings, "ANOMALY_SHAPE_GEOMETRY_WEIGHT", 0.5
            ),
            shape_convexity_weight=getattr(
                settings, "ANOMALY_SHAPE_CONVEXITY_WEIGHT", 0.35
            ),
            shape_aspect_weight=getattr(
                settings, "ANOMALY_SHAPE_ASPECT_WEIGHT", 0.15
            ),
            shape_aspect_dampen=getattr(
                settings, "ANOMALY_SHAPE_ASPECT_DAMPEN", 0.5
            ),
            # texture
            canny_low=getattr(settings, "ANOMALY_CANNY_LOW", 50),
            canny_high=getattr(settings, "ANOMALY_CANNY_HIGH", 150),
            texture_contrast_min=getattr(
                settings, "ANOMALY_TEXTURE_CONTRAST_MIN", 0.2
            ),
            texture_contrast_max=getattr(
                settings, "ANOMALY_TEXTURE_CONTRAST_MAX", 2.5
            ),
            texture_edge_min=getattr(
                settings, "ANOMALY_TEXTURE_EDGE_MIN", 0.02
            ),
            texture_edge_max=getattr(
                settings, "ANOMALY_TEXTURE_EDGE_MAX", 0.4
            ),
            texture_reliability_pixels=getattr(
                settings, "ANOMALY_TEXTURE_RELIABILITY_PIXELS", 900
            ),
            texture_uniform_contrast_gate=getattr(
                settings, "ANOMALY_TEXTURE_UNIFORM_CONTRAST_GATE", 0.35
            ),
            texture_uniform_score_gate=getattr(
                settings, "ANOMALY_TEXTURE_UNIFORM_SCORE_GATE", 0.7
            ),
            texture_uniform_dampen=getattr(
                settings, "ANOMALY_TEXTURE_UNIFORM_DAMPEN", 0.4
            ),
            # acoustic shadow
            shadow_search_multiplier=getattr(
                settings, "ANOMALY_SHADOW_SEARCH_MULTIPLIER", 2.5
            ),
            shadow_search_min_px=getattr(
                settings, "ANOMALY_SHADOW_SEARCH_MIN_PX", 4
            ),
            shadow_fraction_min=getattr(
                settings, "ANOMALY_SHADOW_FRACTION_MIN", 0.05
            ),
            shadow_fraction_max=getattr(
                settings, "ANOMALY_SHADOW_FRACTION_MAX", 0.6
            ),
            shadow_depth_min=getattr(
                settings, "ANOMALY_SHADOW_DEPTH_MIN", 0.05
            ),
            shadow_depth_max=getattr(
                settings, "ANOMALY_SHADOW_DEPTH_MAX", 0.9
            ),
            shadow_strip_contrast_min=getattr(
                settings, "ANOMALY_SHADOW_STRIP_CONTRAST_MIN", 0.05
            ),
            shadow_strip_contrast_max=getattr(
                settings, "ANOMALY_SHADOW_STRIP_CONTRAST_MAX", 0.9
            ),
            shadow_fraction_weight=getattr(
                settings, "ANOMALY_SHADOW_FRACTION_WEIGHT", 0.4
            ),
            shadow_depth_weight=getattr(
                settings, "ANOMALY_SHADOW_DEPTH_WEIGHT", 0.3
            ),
            shadow_contrast_weight=getattr(
                settings, "ANOMALY_SHADOW_CONTRAST_WEIGHT", 0.3
            ),
            shadow_neutral_frac_gate=getattr(
                settings, "ANOMALY_SHADOW_NEUTRAL_FRAC_GATE", 0.02
            ),
            shadow_neutral_contrast_gate=getattr(
                settings, "ANOMALY_SHADOW_NEUTRAL_CONTRAST_GATE", 0.05
            ),
        )


@dataclass
class AnomalyResult:
    """Output of anomaly classification for a single detected region."""

    label: str                     # "natural" | "artificial" | "uncertain"
    natural_probability: float
    artificial_probability: float
    confidence: float              # strength of the winning hypothesis
    evidence: float                # signed score in [-1, 1] pre-labelling
    features: dict = field(default_factory=dict)  # per-group scores for UI

    def to_dict(self) -> dict:
        return {
            "anomaly_class": self.label,
            "natural_probability": round(float(self.natural_probability), 4),
            "artificial_probability": round(float(self.artificial_probability), 4),
            "anomaly_confidence": round(float(self.confidence), 4),
            "anomaly_evidence": round(float(self.evidence), 4),
            "anomaly_features": self.features,
        }


def classify_detection(
    detection: dict,
    image: np.ndarray,
    config: AnomalyFeatureConfig | None = None,
) -> AnomalyResult:
    """Classify a single detection dict.

    ``detection`` must contain the bbox keys (``bbox_x/y/width/height`` in
    the same coordinate space as ``image``), an optional ``mask`` (binary
    array matching ``image``), and ``confidence`` / ``class_label`` for the
    AI feature.

    Args:
        detection: A detection dict as produced by the YOLO pipeline.
        image: The (preprocessed) image the bbox refers to. Used to crop the
            region and its surroundings for texture / shadow analysis.
        config: Optional feature-weight configuration.

    Returns:
        :class:`AnomalyResult` with natural / artificial probabilities and a
        final label.
    """
    cfg = config or AnomalyFeatureConfig()
    mask = detection.get("mask")

    bbox = (
        detection["bbox_x"],
        detection["bbox_y"],
        detection["bbox_width"],
        detection["bbox_height"],
    )

    shape_score, shape_feats = _shape_score(detection, mask, cfg)
    texture_score, texture_feats = _texture_score(image, bbox, mask, cfg)
    shadow_score, shadow_feats = _shadow_score(image, bbox, mask, cfg)
    ai_score = _ai_score(
        detection.get("confidence", 0.5),
        detection.get("class_label", "other_debris"),
    )

    features = {
        "shape": round(float(shape_score), 4),
        "texture": round(float(texture_score), 4),
        "acoustic_shadow": round(float(shadow_score), 4),
        "ai": round(float(ai_score), 4),
        **_shape_feature_values(shape_feats),
        **_texture_feature_values(texture_feats),
        **_shadow_feature_values(shadow_feats),
    }

    evidence = _combine(
        {
            "shape": shape_score,
            "texture": texture_score,
            "acoustic_shadow": shadow_score,
            "ai": ai_score,
        },
        cfg,
    )

    return _to_result(evidence, cfg, features)


# ---------------------------------------------------------------------------
# Evidence combination
# ---------------------------------------------------------------------------

def _combine(
    scores: dict[str, float], cfg: AnomalyFeatureConfig
) -> float:
    """Weighted sum of per-group scores, renormalising over available groups.

    Missing / zero-weight groups are dropped and the remaining weights are
    renormalised to sum to 1 so the final evidence stays in [-1, 1].
    """
    weights = {
        "shape": cfg.weight_shape,
        "texture": cfg.weight_texture,
        "acoustic_shadow": cfg.weight_acoustic_shadow,
        "ai": cfg.weight_ai,
    }

    pairs = [(scores[k], weights[k]) for k in scores if weights[k] > 0]
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0

    evidence = sum(s * w for s, w in pairs) / total_w
    return float(max(-1.0, min(1.0, evidence)))


def _to_result(
    evidence: float, cfg: AnomalyFeatureConfig, features: dict
) -> AnomalyResult:
    # Map evidence in [-1, 1] to natural / artificial probabilities.
    # evidence = artificial - natural; at evidence=0 both are 0.5.
    artificial = (evidence + 1.0) / 2.0
    natural = 1.0 - artificial

    # Margin between hypotheses; low margin -> uncertain.
    margin = abs(evidence)
    if margin < cfg.uncertain_threshold:
        return AnomalyResult(
            label="uncertain",
            natural_probability=natural,
            artificial_probability=artificial,
            confidence=round(margin / max(cfg.uncertain_threshold, 1e-9), 4),
            evidence=evidence,
            features=features,
        )

    label = "artificial" if evidence > 0 else "natural"
    # Confidence scales with how far the margin is past the threshold.
    strength = min(1.0, margin / max(cfg.uncertain_threshold, 1e-9))
    return AnomalyResult(
        label=label,
        natural_probability=natural,
        artificial_probability=artificial,
        confidence=round(strength, 4),
        evidence=evidence,
        features=features,
    )


# ---------------------------------------------------------------------------
# Shape features
# ---------------------------------------------------------------------------

def _shape_score(
    detection: dict, mask, cfg: AnomalyFeatureConfig
) -> tuple[float, dict]:
    """Shape-based artificiality score from mask geometry or bbox ratios."""
    w = detection["bbox_width"]
    h = detection["bbox_height"]
    if w <= 0 or h <= 0:
        return 0.0, {}

    aspect = max(w, h) / max(min(w, h), 1e-9)
    aspect_score = _bounded(
        aspect, cfg.aspect_ratio_min, cfg.aspect_ratio_max,
        high_is_artificial=True,
    )

    if mask is not None and int(np.count_nonzero(mask)) > 0:
        contour_feats = _contour_geometry(mask)
        perimeter = contour_feats["perimeter"]
        hull_area = contour_feats["hull_area"]

        area = float(np.count_nonzero(mask))
        compactness = (
            (4.0 * np.pi * area / (perimeter * perimeter))
            if perimeter > 0
            else 0.0
        )
        convexity = (
            (area / hull_area) if hull_area > 0 and area <= hull_area else 1.0
        )

        compactness_score = _bounded(
            compactness, cfg.compactness_min, cfg.compactness_max,
            high_is_artificial=True,
        )
        convexity_score = _bounded(
            1.0 - convexity, cfg.concavity_min, cfg.concavity_max,
            high_is_artificial=True,
        )

        feats = {
            "compactness": round(compactness, 4),
            "convexity": round(convexity, 4),
            "aspect_ratio": round(aspect, 4),
            "perimeter": round(perimeter, 2),
        }

        score = (
            cfg.shape_geometry_weight * compactness_score
            + cfg.shape_convexity_weight * convexity_score
            + cfg.shape_aspect_weight * aspect_score
        )
        return _bounded(score, -1, 1), feats

    feats = {"aspect_ratio": round(aspect, 4), "compactness": None}
    return _bounded(
        cfg.shape_aspect_dampen * aspect_score, -1, 1
    ), feats


def _contour_geometry(mask: np.ndarray) -> dict:
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {"perimeter": 0.0, "hull_area": 0.0, "area": 0.0}
    # Largest contour by area.
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return {
        "perimeter": float(perimeter),
        "hull_area": float(hull_area),
        "area": float(cv2.contourArea(cnt)),
    }


# ---------------------------------------------------------------------------
# Texture features
# ---------------------------------------------------------------------------

def _texture_score(
    image: np.ndarray, bbox: tuple, mask, cfg: AnomalyFeatureConfig
) -> tuple[float, dict]:
    """Texture-based artificiality score from intra-region contrast."""
    crop = _crop_region(image, bbox)
    if crop is None or crop.size == 0:
        return 0.0, {}

    gray = _to_gray(crop)
    if mask is not None:
        m = _crop_mask(mask, bbox)
        if m is not None and int(np.count_nonzero(m)) > 0:
            vals = gray[m > 0]
        else:
            vals = gray.ravel()
    else:
        vals = gray.ravel()

    if vals.size == 0:
        return 0.0, {}

    mean = float(vals.mean())
    std = float(vals.std())
    contrast = std / max(mean, 1e-6)

    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    edge_density = float(edges.mean() / 255.0)

    contrast_score = _bounded(
        contrast, cfg.texture_contrast_min, cfg.texture_contrast_max,
        high_is_artificial=True,
    )
    edge_score = _bounded(
        edge_density, cfg.texture_edge_min, cfg.texture_edge_max,
        high_is_artificial=True,
    )

    feats = {
        "contrast": round(contrast, 4),
        "edge_density": round(edge_density, 4),
    }

    raw_score = 0.5 * contrast_score + 0.5 * edge_score

    reliability = float(min(1.0, max(vals.size, 1) / cfg.texture_reliability_pixels))
    if (
        strength_of(raw_score) > cfg.texture_uniform_score_gate
        and contrast < cfg.texture_uniform_contrast_gate
    ):
        reliability *= cfg.texture_uniform_dampen

    score = raw_score * reliability
    return _bounded(score, -1, 1), feats


# ---------------------------------------------------------------------------
# Acoustic shadow features
# ---------------------------------------------------------------------------

def _shadow_score(
    image: np.ndarray, bbox: tuple, mask, cfg: AnomalyFeatureConfig
) -> tuple[float, dict]:
    """Score based on a strong acoustic shadow cast by an elevated object.

    In side-scan sonar, an elevated target returns a bright highlight and
    casts a long, dark (low-backscatter) *acoustic shadow* on the far side
    (down-track). Artificial objects (pipes, cylinders, containers) stand
    proud of the seafloor and cast a pronounced, well-defined shadow,
    whereas natural low-lying debris / outcrops cast little.

    The shadow is searched in a strip that extends downward from the object
    bbox, i.e. in the typical shadow direction.
    """
    h_img, w_img = image.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(round(x * w_img))
    x2 = int(round((x + bw) * w_img))
    y1 = int(round(y * h_img))
    y2 = int(round((y + bh) * h_img))

    if x2 - x1 < cfg.shadow_search_min_px:
        return 0.0, {}

    highlight = _crop(image, x1, y1, x2, y2)
    if highlight is None:
        return 0.0, {}
    gray_hl = _to_gray(highlight)
    hl_mean = float(gray_hl.mean())

    shadow_span = max(
        int((y2 - y1) * cfg.shadow_search_multiplier),
        cfg.shadow_search_min_px,
    )
    sy1 = y2
    sy2 = min(h_img, y2 + shadow_span)
    if sy2 - sy1 < cfg.shadow_search_min_px:
        return 0.0, {}

    strip = _crop(image, x1, sy1, x2, sy2)
    if strip is None or strip.size == 0:
        return 0.0, {}

    gray = _to_gray(strip)
    h, w = gray.shape

    thresh_val, dark_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark_frac = float((dark_binary == 0).mean()) if gray.size else 0.0

    dark_mask = dark_binary == 0
    bright_mask = dark_binary == 255
    n_dark = int(np.count_nonzero(dark_mask))
    n_bright = int(np.count_nonzero(bright_mask))

    if n_dark == 0:
        feats_none = {
            "shadow_fraction": 0.0,
            "shadow_depth": 0.0,
            "shadow_contrast": 0.0,
            "highlight_mean": round(float(hl_mean), 2),
        }
        return 0.0, feats_none

    dark_mean = float(gray[dark_mask].mean())
    bright_mean = float(gray[bright_mask].mean()) if n_bright else float(gray.mean())

    shadow_depth = (hl_mean - dark_mean) / max(hl_mean + dark_mean, 1e-6)
    strip_contrast = (bright_mean - dark_mean) / max(bright_mean + dark_mean, 1e-6)

    fraction_score = _bounded(
        dark_frac, cfg.shadow_fraction_min, cfg.shadow_fraction_max,
        high_is_artificial=True,
    )
    depth_score = _bounded(
        shadow_depth, cfg.shadow_depth_min, cfg.shadow_depth_max,
        high_is_artificial=True,
    )
    contrast_score = _bounded(
        strip_contrast, cfg.shadow_strip_contrast_min,
        cfg.shadow_strip_contrast_max, high_is_artificial=True,
    )

    shadow_result = (
        cfg.shadow_fraction_weight * fraction_score
        + cfg.shadow_depth_weight * depth_score
        + cfg.shadow_contrast_weight * contrast_score
    )

    feats = {
        "shadow_fraction": round(float(dark_frac), 4),
        "shadow_depth": round(float(shadow_depth), 4),
        "shadow_contrast": round(float(strip_contrast), 4),
        "highlight_mean": round(float(hl_mean), 2),
    }

    if (
        dark_frac < cfg.shadow_neutral_frac_gate
        and strip_contrast < cfg.shadow_neutral_contrast_gate
    ):
        shadow_result = 0.0

    return _bounded(shadow_result, -1, 1), feats


def _crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# AI confidence feature
# ---------------------------------------------------------------------------

def _ai_score(confidence: float, class_label: str) -> float:
    """Combine the detector's confidence with a class artificiality prior.

    Higher detection confidence amplifies the class prior; low confidence
    pulls the score toward neutral (-> uncertain downstream).
    """
    prior = CLASS_ARTIFICIALITY_PRIOR.get(class_label, 0.0)
    # Confidence in [0, 1]; scale prior effect by (2*conf - 1) so confident
    # detections weight the prior more heavily.
    conf_factor = max(-1.0, min(1.0, 2.0 * float(confidence) - 1.0))
    return float(max(-1.0, min(1.0, prior * (0.5 + 0.5 * conf_factor))))


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _crop_region(image: np.ndarray, bbox: tuple):
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(round(x * w))
    y1 = int(round(y * h))
    x2 = int(round((x + bw) * w))
    y2 = int(round((y + bh) * h))
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return image[y1:y2, x1:x2]


def _crop_mask(mask: np.ndarray, bbox: tuple):
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x, y, bw, bh = bbox
    x1 = int(round(x * w))
    y1 = int(round(y * h))
    x2 = int(round((x + bw) * w))
    y2 = int(round((y + bh) * h))
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return mask[y1:y2, x1:x2]


def _to_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3 and crop.shape[2] == 1:
        return crop[:, :, 0]
    if crop.ndim == 3:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return np.asarray(crop)


def _bounded(
    value: float,
    lo: float,
    hi: float,
    high_is_artificial: bool = True,
) -> float:
    """Map a raw feature value into a [-1, 1] artificiality score.

    ``lo``/``hi`` define the feature range that maps to -1 / +1; values
    outside are clipped. ``high_is_artificial=True`` maps high feature
    values to +1 (strongly artificial).
    """
    if hi <= lo:
        return 0.0
    clamped = max(lo, min(hi, float(value)))
    t = (clamped - lo) / (hi - lo)          # [0, 1]
    score = 2.0 * t - 1.0                    # [-1, 1]
    if not high_is_artificial:
        score = -score
    return float(score)


def strength_of(score: float) -> float:
    """Absolute magnitude of a signed [-1, 1] score (its decisiveness)."""
    return float(abs(score))


def _shape_feature_values(feats: dict) -> dict:
    return {f"shape_{k}": v for k, v in feats.items()}


def _texture_feature_values(feats: dict) -> dict:
    return {f"texture_{k}": v for k, v in feats.items()}


def _shadow_feature_values(feats: dict) -> dict:
    return {f"shadow_{k}": v for k, v in feats.items()}
