from datetime import datetime
from typing import Any
from uuid import UUID

import json

from pydantic import BaseModel, Field, field_validator, model_validator


def _resolve_object_type(data: dict) -> dict:
    """Ensure ``object_type`` is populated from ``class_label`` when absent."""
    data = dict(data)
    if not data.get("object_type"):
        data["object_type"] = data.get("class_label")
    return data


class DetectionCreate(BaseModel):
    """Payload for creating a detection.

    ``scan_id`` is required (a detection always belongs to a scan for the
    dashboard and reports). ``object_type`` maps to ``class_label``.
    """

    scan_id: UUID
    class_label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    object_type: str | None = None
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    severity: str = "low"
    mask_area: float | None = None  # normalised size (0-1)
    artificial_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_level: str | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    image_reference: str | None = None
    detected_at: datetime | None = None


class DetectionUpdate(BaseModel):
    """Partial update — every field is optional."""

    class_label: str | None = None
    object_type: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None
    severity: str | None = None
    mask_area: float | None = Field(default=None, ge=0.0, le=1.0)
    artificial_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    risk_level: str | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    image_reference: str | None = None
    detected_at: datetime | None = None


class DetectionRead(BaseModel):
    id: UUID
    scan_id: UUID
    class_label: str
    object_type: str | None = None
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    severity: str
    annotated_image_path: str | None = None
    mask_polygon: list[list[float]] | None = None
    mask_area: float | None = None
    mask_image_path: str | None = None
    anomaly_class: str | None = None
    natural_probability: float | None = None
    artificial_probability: float | None = None
    anomaly_confidence: float | None = None
    anomaly_features: dict[str, Any] | None = None
    ai_confidence: float | None = None
    final_confidence: float | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    geo_source: str | None = None
    detected_at: datetime | None = None
    image_reference: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _populate_object_type(cls, data):
        return _resolve_object_type(data)

    @field_validator("mask_polygon", mode="before")
    @classmethod
    def _parse_mask_polygon(cls, value: Any) -> Any:
        return _parse_json_value(value)

    @field_validator("anomaly_features", mode="before")
    @classmethod
    def _parse_anomaly_features(cls, value: Any) -> Any:
        return _parse_json_value(value)


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


class DetectionList(BaseModel):
    id: UUID
    class_label: str
    object_type: str | None = None
    confidence: float
    severity: str
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    mask_polygon: list[list[float]] | None = None
    mask_area: float | None = None
    mask_image_path: str | None = None
    anomaly_class: str | None = None
    natural_probability: float | None = None
    artificial_probability: float | None = None
    anomaly_confidence: float | None = None
    anomaly_features: dict[str, Any] | None = None
    ai_confidence: float | None = None
    final_confidence: float | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    geo_source: str | None = None
    detected_at: datetime | None = None
    image_reference: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _populate_object_type(cls, data):
        return _resolve_object_type(data)

    @field_validator("mask_polygon", mode="before")
    @classmethod
    def _parse_mask_polygon(cls, value: Any) -> Any:
        return _parse_json_value(value)

    @field_validator("anomaly_features", mode="before")
    @classmethod
    def _parse_anomaly_features(cls, value: Any) -> Any:
        return _parse_json_value(value)
