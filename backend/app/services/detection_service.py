from uuid import UUID

from sqlalchemy.orm import Session

from app.models.detection import Detection
from app.models.scan import Scan
from app.ai.detector import MarineDetector
from app.core.config import settings
from app.utils.geo import geom_from_lat_lon
from app.utils.geotag import project_bbox_center, read_geotag


_detector: MarineDetector | None = None


def get_detector() -> MarineDetector:
    global _detector
    if _detector is None:
        _detector = MarineDetector()
    return _detector


def run_detection(scan_id: UUID, db: Session) -> list[Detection]:
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise ValueError("Scan not found")

    scan.status = "processing"
    db.commit()

    try:
        detections = _run_pipeline(scan_id, scan, db)
    except Exception:
        # Never leave the scan stuck in "processing" when the pipeline fails.
        db.rollback()
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan is not None:
            scan.status = "failed"
            db.commit()
        raise

    scan.status = "completed"
    db.commit()

    for det in detections:
        db.refresh(det)

    return detections


def _run_pipeline(scan_id: UUID, scan: Scan, db: Session) -> list[Detection]:
    detector = get_detector()
    results = detector.detect(scan.file_path)

    # Resolve the scan's real GPS metadata once; never fabricate coordinates.
    # If no source yields valid lat/lon, geotag.available stays False and the
    # persisted detection lat/lon remain None ("Location unavailable").
    geotag = read_geotag(
        file_path=scan.file_path,
        scan_latitude=scan.latitude,
        scan_longitude=scan.longitude,
        scan_timestamp=scan.created_at,
    )
    anchor_lat, anchor_lon = geotag.sanitize()

    detections = []
    for result in results:
        mask_polygon = result.get("polygon") or None

        lat, lon = None, None
        geo_source = None
        if anchor_lat is not None and anchor_lon is not None:
            lat, lon = project_bbox_center(
                result, anchor_lat, anchor_lon
            )
            geo_source = geotag.source

        detection = Detection(
            scan_id=scan_id,
            class_label=result["class_label"],
            confidence=result["confidence"],
            bbox_x=result["bbox_x"],
            bbox_y=result["bbox_y"],
            bbox_width=result["bbox_width"],
            bbox_height=result["bbox_height"],
            severity=result["severity"],
            mask_polygon=_serialize_polygon(mask_polygon),
            mask_area=result.get("mask_area"),
            annotated_image_path=result.get("annotated_image_path"),
            anomaly_class=result.get("anomaly_class"),
            natural_probability=result.get("natural_probability"),
            artificial_probability=result.get("artificial_probability"),
            anomaly_confidence=result.get("anomaly_confidence"),
            anomaly_features=_serialize_features(result.get("anomaly_features")),
            ai_confidence=result.get("ai_confidence"),
            final_confidence=result.get("final_confidence"),
            risk_score=result.get("risk_score"),
            risk_level=result.get("risk_level"),
            latitude=lat,
            longitude=lon,
            detected_at=geotag.timestamp,
            geo_source=geo_source,
            geom=geom_from_lat_lon(lat, lon),
            image_reference=scan.file_path,
        )
        if mask_polygon:
            detection.mask_image_path = _render_mask_overlay(
                detection, mask_polygon
            )
        db.add(detection)
        detections.append(detection)

    return detections


def _serialize_polygon(polygon) -> str | None:
    """Serialise a normalised polygon to a compact JSON string."""
    if not polygon:
        return None
    import json

    return json.dumps([[float(p[0]), float(p[1])] for p in polygon])


def _serialize_features(features) -> str | None:
    """Serialise the anomaly feature dict to a compact JSON string."""
    if not features:
        return None
    import json

    return json.dumps(features)


def _render_mask_overlay(detection: Detection, polygon: list) -> str | None:
    """Render an overlay image with the detection box + mask and save to disk.

    Returns the on-disk path of the generated PNG, or ``None`` on failure.
    """
    try:
        import json
        import os

        from app.utils.mask import render_mask_overlay as _render

        points = json.loads(
            detection.mask_polygon if detection.mask_polygon else "[]"
        )
        if not points:
            return None

        path = _render(
            polygon=points,
            mask_area=detection.mask_area,
            class_label=detection.class_label,
            confidence=detection.confidence,
            output_dir=settings.MASK_OUTPUT_DIR,
            detection_id=str(detection.id),
        )
        return path
    except Exception as exc:  # noqa: BLE001 - optional image generation
        import logging

        logging.getLogger(__name__).warning(
            "Could not render mask overlay: %s", exc
        )
        return None
