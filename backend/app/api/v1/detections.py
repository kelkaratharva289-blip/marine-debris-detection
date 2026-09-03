from uuid import UUID

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.detection import Detection
from app.models.scan import Scan
from app.schemas.detection import (
    DetectionCreate,
    DetectionList,
    DetectionRead,
    DetectionUpdate,
)
from app.services.detection_service import run_detection
from app.utils.geo import geom_from_lat_lon

router = APIRouter()


def _get_detection_or_404(detection_id: UUID, db: Session) -> Detection:
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection


def _apply_geotag(detection: Detection, latitude, longitude) -> Detection:
    """Keep the PostGIS geometry in sync with the lat/lon scalar fields."""
    detection.latitude = latitude
    detection.longitude = longitude
    detection.geom = geom_from_lat_lon(latitude, longitude)
    return detection


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@router.post("/", response_model=DetectionRead, status_code=201)
def create_detection(payload: DetectionCreate, db: Session = Depends(get_db)):
    """Create a new anomaly detection and store it in PostGIS."""
    data = payload.model_dump()
    class_label = data.pop("object_type", None) or data.get("class_label")
    data["class_label"] = class_label

    detection = Detection(**data)
    _apply_geotag(detection, payload.latitude, payload.longitude)
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[DetectionList])
def list_detections(
    skip: int = 0,
    limit: int = 200,
    scan_id: Optional[UUID] = None,
    risk_level: Optional[str] = None,
    anomaly_class: Optional[str] = None,
    bbox_min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    bbox_max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    bbox_min_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    bbox_max_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    db: Session = Depends(get_db),
):
    """List detections with optional filtering.

    Supports PostGIS spatial filtering via a lat/lon bounding box
    (``bbox_min_lat``, ``bbox_max_lat``, ``bbox_min_lon``, ``bbox_max_lon``).
    """
    query = db.query(Detection)
    if scan_id is not None:
        query = query.filter(Detection.scan_id == scan_id)
    if risk_level is not None:
        query = query.filter(Detection.risk_level == risk_level)
    if anomaly_class is not None:
        query = query.filter(Detection.anomaly_class == anomaly_class)

    if (
        bbox_min_lat is not None
        and bbox_max_lat is not None
        and bbox_min_lon is not None
        and bbox_max_lon is not None
    ):
        query = query.filter(
            Detection.latitude >= bbox_min_lat,
            Detection.latitude <= bbox_max_lat,
            Detection.longitude >= bbox_min_lon,
            Detection.longitude <= bbox_max_lon,
        )

    return (
        query.order_by(Detection.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{detection_id}", response_model=DetectionRead)
def get_detection(detection_id: UUID, db: Session = Depends(get_db)):
    return _get_detection_or_404(detection_id, db)


@router.get("/scan/{scan_id}", response_model=list[DetectionList])
def list_detections_for_scan(scan_id: UUID, db: Session = Depends(get_db)):
    detections = (
        db.query(Detection)
        .filter(Detection.scan_id == scan_id)
        .order_by(Detection.confidence.desc())
        .all()
    )
    return detections


@router.get("/{detection_id}/mask")
def get_detection_mask(detection_id: UUID, db: Session = Depends(get_db)):
    """Return the rendered segmentation mask overlay image for a detection.

    Returns a 404 when the detection has no segmentation mask available.
    """
    detection = _get_detection_or_404(detection_id, db)
    if not detection.mask_image_path or not os.path.exists(
        detection.mask_image_path
    ):
        raise HTTPException(
            status_code=404, detail="No segmentation mask available for this detection"
        )
    return FileResponse(detection.mask_image_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

@router.patch("/{detection_id}", response_model=DetectionRead)
def update_detection(
    detection_id: UUID, payload: DetectionUpdate, db: Session = Depends(get_db)
):
    """Partially update a detection's stored attributes."""
    detection = _get_detection_or_404(detection_id, db)

    data = payload.model_dump(exclude_unset=True)
    # Accept object_type as an alias for class_label.
    if "object_type" in data:
        data["class_label"] = data.pop("object_type")

    for field_name, value in data.items():
        setattr(detection, field_name, value)

    # Keep geometry in sync with the (possibly updated) lat/lon.
    if "latitude" in data or "longitude" in data:
        _apply_geotag(detection, detection.latitude, detection.longitude)

    db.commit()
    db.refresh(detection)
    return detection


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{detection_id}", status_code=204)
def delete_detection(detection_id: UUID, db: Session = Depends(get_db)):
    detection = _get_detection_or_404(detection_id, db)
    if detection.mask_image_path and os.path.exists(detection.mask_image_path):
        try:
            os.remove(detection.mask_image_path)
        except OSError:
            pass
    db.delete(detection)
    db.commit()
    return None


# ---------------------------------------------------------------------------
# Detection run
# ---------------------------------------------------------------------------

@router.post("/run/{scan_id}", response_model=list[DetectionRead])
def trigger_detection(scan_id: UUID, db: Session = Depends(get_db)):
    """Run the detection pipeline for a scan.

    Returns the list of generated detections. The scan must exist; pipeline
    errors (e.g. missing model weights outside of simulation mode) are mapped
    to a 502 so the UI can surface a friendly message.
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    try:
        detections = run_detection(scan_id, db)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        db.rollback()
        raise HTTPException(
            status_code=502,
            detail=f"Detection pipeline failed: {exc}",
        ) from exc
    return detections
