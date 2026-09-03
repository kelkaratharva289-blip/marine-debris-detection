"""Detection stage: run real inference + anomaly + risk on the TEST split.

For every **real** test image, this stage:

1. Runs YOLO inference and records real detections with confidence.
2. Classifies each detection as Natural/Artificial/Uncertain (reusing the
   deterministic :mod:`app.ai.anomaly` classifier on the real image pixels).
3. Scores each detection with a 0-100 risk score + level (:mod:`app.ai.risk_engine`).
4. Saves a real annotated prediction image (bounding boxes drawn on the actual
   scanned pixels) and writes the full detection result set for reporting.

No detections are synthesised: an image with nothing above threshold yields an
empty detection list (which is recorded honestly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.ai.inference import CLASS_DISPLAY_NAMES, load_yolo_model, run_inference
from app.ai.preprocessing import PreprocessConfig
from app.ai.preprocessing.loader import load_image
from app.ai.preprocessing.pipeline import preprocess_to_uint8
from app.ai.anomaly import AnomalyFeatureConfig, classify_detection
from app.ai.dataset.constants import IMAGE_EXTENSIONS
from app.ai.risk_engine import RiskConfig, compute_risk
from app.training.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionRecord:
    """One real detection with anomaly + risk analysis."""

    image: str
    class_id: int
    class_label: str
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    anomaly_class: str
    natural_probability: float
    artificial_probability: float
    anomaly_confidence: float
    ai_confidence: float
    final_confidence: float
    risk_score: float
    risk_level: str

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "class_id": self.class_id,
            "class_label": self.class_label,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_width": self.bbox_w,
            "bbox_height": self.bbox_h,
            "anomaly_class": self.anomaly_class,
            "natural_probability": self.natural_probability,
            "artificial_probability": self.artificial_probability,
            "anomaly_confidence": self.anomaly_confidence,
            "ai_confidence": self.ai_confidence,
            "final_confidence": self.final_confidence,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
        }


@dataclass
class DetectOutcome:
    """Outcome of running the detection pipeline on the test split."""

    detections: list[DetectionRecord] = field(default_factory=list)
    annotated_images: list[str] = field(default_factory=list)
    images_processed: int = 0

    def as_dict(self) -> dict:
        return {
            "images_processed": self.images_processed,
            "total_detections": len(self.detections),
            "detections": [d.as_dict() for d in self.detections],
            "annotated_images": self.annotated_images,
        }


_COLORS = [
    (34, 211, 238),  # cyan
    (255, 99, 88),   # red
    (52, 211, 153),  # green
    (250, 204, 21),  # yellow
    (167, 139, 250), # violet
    (251, 146, 60),  # orange
]


def _draw_detection(image, det, names):
    """Draw a real detection box + label on the image (mutates in place)."""
    h, w = image.shape[:2]
    x1 = int(det["bbox_x"] * w)
    y1 = int(det["bbox_y"] * h)
    x2 = int((det["bbox_x"] + det["bbox_width"]) * w)
    y2 = int((det["bbox_y"] + det["bbox_height"]) * h)
    color = _COLORS[det["class_id"] % len(_COLORS)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{det['class_name']} {det['confidence']:.2f} risk={det['risk_score']:.0f}"
    cv2.putText(image, label, (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def run_detection_pipeline(config: TrainConfig) -> DetectOutcome:
    """Run detection + anomaly + risk on all test images and save overlays.

    Args:
        config: Resolved config (model from ``best_model_path``, thresholds).

    Returns:
        A :class:`DetectOutcome` with the real detection records and paths to
        the annotated images.
    """
    from ultralytics import YOLO

    model = YOLO(config.best_model_path)
    anomaly_cfg = AnomalyFeatureConfig()
    risk_cfg = RiskConfig()
    pre_cfg = PreprocessConfig(target_size=config.imgsz)

    images_dir = config.root / "test" / "images"
    out_images = config.out / "predictions"
    out_images.mkdir(parents=True, exist_ok=True)

    outcome = DetectOutcome()
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image in image_files:
        outcome.images_processed += 1
        img = load_image(str(image), color=False)
        canvas = preprocess_to_uint8(img, config=pre_cfg)

        raw = run_inference(
            model, canvas,
            conf=config.conf, iou=config.iou,
            classes=config.classes, device=config.device,
        )

        # Map canvas-normalised boxes back to original-image fractions by
        # undoing letterbox using the raw (pre-letterbox) canvas dimensions.
        h_raw, w_raw = img.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        detections: list[dict] = []
        for d in raw:
            cx = d["bbox_x"] * canvas_w
            cy = d["bbox_y"] * canvas_h
            cw = d["bbox_width"] * canvas_w
            ch = d["bbox_height"] * canvas_h

            # Remove letterbox padding via scale = min(canvas/w_raw, canvas/h_raw).
            scale = min(canvas_w / w_raw, canvas_h / h_raw)
            pad_x = (canvas_w - w_raw * scale) / 2.0
            pad_y = (canvas_h - h_raw * scale) / 2.0
            ox = (cx - pad_x) / scale
            oy = (cy - pad_y) / scale
            ow = cw / scale
            oh = ch / scale

            det = {
                "class_id": d["class_id"],
                "class_label": d["class_label"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox_x": max(0.0, min(ox / w_raw, 1.0)),
                "bbox_y": max(0.0, min(oy / h_raw, 1.0)),
                "bbox_width": max(0.0, min(ow / w_raw, 1.0)),
                "bbox_height": max(0.0, min(oh / h_raw, 1.0)),
            }
            detections.append(det)

        # Anomaly + risk in original-image space (boxes fraction of original).
        display = img.copy()
        for det in detections:
            try:
                anom = classify_detection(det, img, config=anomaly_cfg).to_dict()
                det.update(anom)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Anomaly failed for %s: %s", image, exc)
                det.update({
                    "anomaly_class": "uncertain",
                    "natural_probability": 0.5,
                    "artificial_probability": 0.5,
                    "anomaly_confidence": 0.0,
                })
            try:
                risk = compute_risk(
                    confidence=det["confidence"],
                    object_type=det.get("class_label"),
                    estimated_size=(
                        float(det["bbox_width"] * det["bbox_height"])
                        if det.get("bbox_width") is not None
                        and det.get("bbox_height") is not None
                        else None
                    ),
                    artificial_probability=det.get("artificial_probability"),
                    anomaly_class=det.get("anomaly_class"),
                    config=risk_cfg,
                )
                det.update(risk.to_dict(ai_confidence=det["confidence"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Risk failed for %s: %s", image, exc)
                det.update({"risk_score": 0.0, "risk_level": "low",
                            "final_confidence": det["confidence"]})

            _draw_detection(display, det, CLASS_DISPLAY_NAMES)
            outcome.detections.append(
                DetectionRecord(
                    image=image.name,
                    class_id=det["class_id"],
                    class_label=det["class_label"],
                    class_name=det["class_name"],
                    confidence=round(det["confidence"], 4),
                    bbox_x=round(det["bbox_x"], 4),
                    bbox_y=round(det["bbox_y"], 4),
                    bbox_w=round(det["bbox_width"], 4),
                    bbox_h=round(det["bbox_height"], 4),
                    anomaly_class=det.get("anomaly_class", "uncertain"),
                    natural_probability=round(det.get("natural_probability", 0.5), 4),
                    artificial_probability=round(det.get("artificial_probability", 0.5), 4),
                    anomaly_confidence=round(det.get("anomaly_confidence", 0.0), 4),
                    ai_confidence=round(det.get("ai_confidence", det["confidence"]), 4),
                    final_confidence=round(det.get("final_confidence", det["confidence"]), 4),
                    risk_score=round(det.get("risk_score", 0.0), 2),
                    risk_level=det.get("risk_level", "low"),
                )
            )

        annotated_path = out_images / f"{image.stem}_pred.png"
        cv2.imwrite(str(annotated_path), display)
        outcome.annotated_images.append(str(annotated_path))

    logger.info(
        "Detection complete: %d images, %d detections",
        outcome.images_processed, len(outcome.detections),
    )
    return outcome

