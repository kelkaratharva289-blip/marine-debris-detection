from typing import Any

from pydantic import BaseModel, Field


class DetectionAnalysisResult(BaseModel):
    """Single detection returned by the direct image-analysis endpoint.

    Contains the object type, bounding box, detection confidence and the
    downstream anomaly classification + risk score for one detected region.
    """

    object: str                    # canonical class label (e.g. "pipe")
    object_name: str               # human-readable name (e.g. "Pipe")
    confidence: float = Field(ge=0.0, le=1.0)

    bbox_x: float = Field(ge=0.0, le=1.0)
    bbox_y: float = Field(ge=0.0, le=1.0)
    bbox_width: float = Field(ge=0.0, le=1.0)
    bbox_height: float = Field(ge=0.0, le=1.0)

    # Natural / Artificial / Uncertain anomaly classification.
    anomaly_type: str            # "natural" | "artificial" | "uncertain"
    artificial_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    # 0-100 risk score + Low / Medium / High / Critical level.
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_level: str | None = None

    # Geolocation derived only from real sonar / GPS metadata. Both are None
    # when no metadata exists, so the UI must show "Location unavailable".
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    geo_source: str | None = None  # "exif" | "sidecar" | "scan"
    geo_timestamp: str | None = None


class AnalysisResponse(BaseModel):
    """Result of analysing a single sonar image upload."""

    detections: list[DetectionAnalysisResult]
    count: int
    model: str                 # artifact/model used for inference
    processed: bool = True


class AnalysisError(BaseModel):
    """Uniform error payload for the analysis endpoint."""

    detail: str


def build_analysis_result(
    det: dict,
    anchor_lat: float | None = None,
    anchor_lon: float | None = None,
    geo_source: str | None = None,
    geo_timestamp: str | None = None,
) -> DetectionAnalysisResult:
    """Map a pipeline detection dict to an :class:`DetectionAnalysisResult`.

    Only outcomes produced by the real pipeline are emitted; missing anomaly,
    risk or metadata values are carried through as ``None`` rather than
    fabricated.

    When real GPS anchor coordinates are supplied, the detection's bbox centre
    is projected to a sub-degree offset near the anchor (same logic as the
    persisted scan pipeline). Without an anchor, lat/lon stay ``None``.
    """
    latitude: float | None = None
    longitude: float | None = None
    if anchor_lat is not None and anchor_lon is not None:
        latitude, longitude = _project_center(det, anchor_lat, anchor_lon)

    return DetectionAnalysisResult(
        object=det.get("class_label", "other_debris"),
        object_name=det.get("class_name", det.get("class_label", "")),
        confidence=_num(det.get("confidence"), 0.0),
        bbox_x=_num(det.get("bbox_x"), 0.0),
        bbox_y=_num(det.get("bbox_y"), 0.0),
        bbox_width=_num(det.get("bbox_width"), 0.0),
        bbox_height=_num(det.get("bbox_height"), 0.0),
        anomaly_type=det.get("anomaly_class", "uncertain"),
        artificial_probability=_opt_num(det.get("artificial_probability")),
        risk_score=_opt_num(det.get("risk_score")),
        risk_level=det.get("risk_level"),
        latitude=latitude,
        longitude=longitude,
        geo_source=geo_source if latitude is not None else None,
        geo_timestamp=geo_timestamp if latitude is not None else None,
    )


def _project_center(
    det: dict, anchor_lat: float, anchor_lon: float
) -> tuple[float, float]:
    """Project a detection bbox centre near the scan anchor point.

    Mirrors :func:`app.utils.geotag.project_bbox_center` so both the persisted
    and stateless analysis paths resolve detections to the same geometry. Only
    shifts an already-valid anchor by a bounded sub-degree offset.
    """
    from app.utils.geotag import project_bbox_center

    return project_bbox_center(det, anchor_lat, anchor_lon)


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Keep unused import available to callers that want to reference the error
# model type explicitly.
__all__ = [
    "DetectionAnalysisResult",
    "AnalysisResponse",
    "AnalysisError",
    "build_analysis_result",
]
