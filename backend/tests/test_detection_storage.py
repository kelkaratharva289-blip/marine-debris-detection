"""Tests for PostgreSQL/PostGIS detection storage schemas and helpers."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.detection import (
    DetectionCreate,
    DetectionUpdate,
    DetectionRead,
    DetectionList,
)

try:
    from app.utils.geo import geom_from_lat_lon
    HAS_GEO = True
except Exception:  # pragma: no cover - geoalchemy2 not installed locally
    HAS_GEO = False


class TestDetectionCreate:
    def test_create_requires_scan_and_class(self):
        with pytest.raises(ValidationError):
            DetectionCreate()

    def test_create_with_minimal_fields(self):
        payload = DetectionCreate(scan_id=uuid.uuid4(), class_label="pipe", confidence=0.9)
        assert payload.class_label == "pipe"
        assert payload.confidence == 0.9

    def test_create_accepts_object_type_alias(self):
        payload = DetectionCreate(
            scan_id=uuid.uuid4(),
            class_label="pipe",
            object_type="pipe",
            confidence=0.9,
        )
        assert payload.object_type == "pipe"

    def test_create_enforces_confidence_bounds(self):
        with pytest.raises(ValidationError):
            DetectionCreate(scan_id=uuid.uuid4(), class_label="x", confidence=1.5)

    def test_create_enforces_lat_lon_bounds(self):
        with pytest.raises(ValidationError):
            DetectionCreate(
                scan_id=uuid.uuid4(),
                class_label="x",
                confidence=0.5,
                latitude=120.0,
                longitude=0.0,
            )

    def test_create_all_fields(self):
        now = datetime.now(timezone.utc)
        payload = DetectionCreate(
            scan_id=uuid.uuid4(),
            class_label="container",
            confidence=0.8,
            risk_score=85.0,
            risk_level="high",
            artificial_probability=0.95,
            latitude=-33.0,
            longitude=151.0,
            mask_area=0.42,
            image_reference="uploads/a.png",
            detected_at=now,
        )
        assert payload.latitude == -33.0
        assert payload.risk_score == 85.0
        assert payload.mask_area == 0.42
        assert payload.image_reference == "uploads/a.png"


class TestDetectionUpdate:
    def test_update_all_fields_optional(self):
        payload = DetectionUpdate()
        assert payload.model_dump(exclude_unset=True) == {}

    def test_update_partial(self):
        payload = DetectionUpdate(risk_score=90.0, risk_level="critical")
        d = payload.model_dump(exclude_unset=True)
        assert "class_label" not in d
        assert d["risk_score"] == 90.0

    def test_update_bounds(self):
        with pytest.raises(ValidationError):
            DetectionUpdate(longitude=200.0)


class TestDetectionReadObjectType:
    def test_object_type_derived_from_class_label(self):
        obj = DetectionRead(
            id=uuid.uuid4(),
            scan_id=uuid.uuid4(),
            class_label="shipwreck",
            confidence=0.7,
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_width=0.2,
            bbox_height=0.2,
            severity="medium",
            created_at=datetime.now(timezone.utc),
        )
        assert obj.object_type == "shipwreck"

    def test_explicit_object_type_wins(self):
        obj = DetectionList(
            id=uuid.uuid4(),
            class_label="pipe",
            object_type="pipe",
            confidence=0.7,
            severity="low",
            created_at=datetime.now(timezone.utc),
        )
        assert obj.object_type == "pipe"


@pytest.mark.skipif(not HAS_GEO, reason="geoalchemy2 not installed")
class TestGeomHelper:
    def test_valid_pair(self):
        geom = geom_from_lat_lon(10.5, -20.25)
        assert geom is not None
        assert "POINT(-20.25 10.5)" in str(geom)

    def test_missing_returns_none(self):
        assert geom_from_lat_lon(None, 20.0) is None
        assert geom_from_lat_lon(10.0, None) is None

    def test_out_of_range_returns_none(self):
        assert geom_from_lat_lon(95.0, 0.0) is None
        assert geom_from_lat_lon(0.0, 181.0) is None
