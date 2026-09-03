import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from app.core.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id"), nullable=False)

    class_label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)

    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)

    severity = Column(String(20), default="low")
    annotated_image_path = Column(String(1024), nullable=True)

    # Segmentation / mask data (optional). mask_polygon stores a JSON array
    # of normalised [x, y] points; mask_image_path points to a rendered
    # overlay PNG served by the API for dashboard display.
    mask_polygon = Column(Text, nullable=True)
    mask_area = Column(Float, nullable=True)
    mask_image_path = Column(String(1024), nullable=True)

    # Anomaly classification: Natural / Artificial / Uncertain.
    anomaly_class = Column(String(20), nullable=True)
    natural_probability = Column(Float, nullable=True)
    artificial_probability = Column(Float, nullable=True)
    anomaly_confidence = Column(Float, nullable=True)
    anomaly_features = Column(Text, nullable=True)  # JSON of per-group scores

    # Risk & confidence engine: fused confidence and a 0-100 risk score with
    # a bucketed risk level (low | medium | high | critical).
    ai_confidence = Column(Float, nullable=True)
    final_confidence = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)

    # Geotagging: real GPS metadata attached to the anomaly. Null lat/lon
    # means no GPS was available (the API/UI show "Location unavailable").
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    detected_at = Column(DateTime, nullable=True)
    geo_source = Column(String(20), nullable=True)  # exif | sidecar | scan

    # PostGIS point geometry (SRID 4326), kept in sync with latitude/longitude
    # so detections can be queried spatially (bbox, ST_DWithin, etc.).
    geom = Column(Geometry("POINT", srid=4326), nullable=True)

    # Reference to the source image for this detection (raw scan or overlay).
    image_reference = Column(String(1024), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    scan = relationship("Scan", back_populates="detections")
