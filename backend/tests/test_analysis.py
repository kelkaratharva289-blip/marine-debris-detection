"""Tests for the direct sonar image analysis endpoint and schemas."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas.analysis import (
    AnalysisResponse,
    DetectionAnalysisResult,
    build_analysis_result,
)


def _pipeline_detection() -> dict:
    """A realistic detection dict as produced by the YOLO + anomaly + risk
    pipeline."""
    return {
        "class_id": 4,
        "class_label": "pipe",
        "class_name": "Pipe",
        "confidence": 0.91,
        "bbox_x": 0.1,
        "bbox_y": 0.2,
        "bbox_width": 0.3,
        "bbox_height": 0.4,
        "severity": "high",
        "anomaly_class": "artificial",
        "natural_probability": 0.1,
        "artificial_probability": 0.9,
        "anomaly_confidence": 0.85,
        "risk_score": 82.0,
        "risk_level": "high",
    }


class TestBuildAnalysisResult:
    def test_maps_pipeline_fields(self):
        r = build_analysis_result(_pipeline_detection())
        assert r.object == "pipe"
        assert r.object_name == "Pipe"
        assert r.confidence == 0.91
        assert r.bbox_x == 0.1
        assert r.bbox_y == 0.2
        assert r.bbox_width == 0.3
        assert r.bbox_height == 0.4
        assert r.anomaly_type == "artificial"
        assert r.artificial_probability == 0.9
        assert r.risk_score == 82.0
        assert r.risk_level == "high"

    def test_missing_values_are_none_not_fabricated(self):
        det = {
            "class_label": "other_debris",
            "confidence": 0.6,
            "bbox_x": 0.0,
            "bbox_y": 0.0,
            "bbox_width": 0.1,
            "bbox_height": 0.1,
        }
        r = build_analysis_result(det)
        assert r.anomaly_type == "uncertain"  # default, not fabricated
        assert r.artificial_probability is None
        assert r.risk_score is None
        assert r.risk_level is None

    def test_missing_values_default_bbox_and_confidence(self):
        r = build_analysis_result({})
        assert r.object == "other_debris"
        assert r.confidence == 0.0
        assert r.bbox_x == 0.0

    def test_is_valid_pydantic_model(self):
        r = build_analysis_result(_pipeline_detection())
        assert DetectionAnalysisResult.model_validate(r).object == "pipe"

    def test_geo_fields_none_without_anchor(self):
        r = build_analysis_result(_pipeline_detection())
        assert r.latitude is None
        assert r.longitude is None
        assert r.geo_source is None
        assert r.geo_timestamp is None

    def test_geo_fields_projected_with_anchor(self):
        r = build_analysis_result(
            _pipeline_detection(),
            anchor_lat=25.7,
            anchor_lon=-79.3,
            geo_source="exif",
            geo_timestamp="2024-01-01T00:00:00+00:00",
        )
        # Projected near the anchor, never fabricated far away.
        assert r.latitude is not None
        assert r.longitude is not None
        assert abs(r.latitude - 25.7) < 0.01
        assert abs(r.longitude - (-79.3)) < 0.01
        assert r.geo_source == "exif"
        assert r.geo_timestamp is not None


class TestAnalysisResponseSchema:
    def test_response_shape(self):
        resp = AnalysisResponse(
            detections=[build_analysis_result(_pipeline_detection())],
            count=1,
            model="models/test.pt",
        )
        assert resp.count == 1
        assert resp.processed is True
        assert resp.detections[0].object == "pipe"

    def test_detection_requires_bounds(self):
        with pytest.raises(ValidationError):
            DetectionAnalysisResult(
                object="pipe",
                confidence=1.5,  # out of [0, 1]
                bbox_x=0,
                bbox_y=0,
                bbox_width=0,
                bbox_height=0,
            )


def _make_app():
    """Build a minimal FastAPI app wrapping the analysis router so that HTTP
    exception handlers (which live on the app, not the router) are active."""
    from fastapi import FastAPI

    from app.api.v1.analysis import router as analysis_router

    app = FastAPI()
    app.include_router(analysis_router)
    return app


client = TestClient(_make_app())


class TestAnalyzeEndpoint:
    def test_rejects_unsupported_content_type(self):
        resp = client.post(
            "/analyze",
            files={"file": ("scan.exe", b"MZ...", "application/x-msdownload")},
        )
        assert resp.status_code == 415

    def test_rejects_empty_file(self):
        # A valid image extension but no bytes decodes to nothing.
        resp = client.post(
            "/analyze",
            files={"file": ("scan.png", b"", "image/png")},
        )
        assert resp.status_code in (413, 422)

    def test_rejects_garbage_bytes_as_invalid_image(self):
        resp = client.post(
            "/analyze",
            files={"file": ("scan.png", b"not a real image", "image/png")},
        )
        # Should be surfaced as invalid/empty rather than 500.
        assert resp.status_code in (413, 422)
        assert "detail" in resp.json()

    def test_valid_image_returns_processed_or_model_error(self):
        # A real image lets the pipeline run; depending on whether weights
        # exist this returns either results or a 503 model-unavailable —
        # neither may be a 500.
        from PIL import Image
        import io

        img = Image.new("L", (64, 64), 128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/analyze",
            files={"file": ("scan.png", buf.getvalue(), "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert data["processed"] is True
            assert data["count"] >= 0
            assert isinstance(data["detections"], list)
        elif resp.status_code == 503:
            assert "detail" in resp.json()
        else:
            pytest.fail(f"Unexpected status {resp.status_code}: {resp.text}")
