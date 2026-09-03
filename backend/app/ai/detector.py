"""Marine debris detector — orchestrates preprocessing and YOLO inference.

The detector loads a sonar scan, feeds it through the preprocessing
pipeline, and runs YOLOv8 inference.  Detection boxes are mapped from the
letterboxed coordinate space back to the original image.

If trained weights are unavailable, the detector raises a clear error
rather than generating placeholder detections.

Optionally, when ``SEGMENTATION_ENABLED``, the detector also runs an
instance-segmentation pass (YOLO-Seg or U-Net) to attach a pixel mask to
each detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ai.anomaly import AnomalyFeatureConfig, classify_detection
from app.ai.inference import (
    MARINE_CLASSES,
    load_yolo_model,
    resolve_inference_model_path,
    run_inference,
)
from app.ai.preprocessing import PreprocessConfig, preprocess_to_uint8
from app.ai.risk_engine import RiskConfig, compute_risk
from app.ai.segmentation import load_seg_model, run_segmentation
from app.core.config import settings

logger = logging.getLogger(__name__)


def _confidence_to_severity(confidence: float) -> str:
    """Map a confidence score to a human-readable severity label."""
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


class ModelNotAvailableError(Exception):
    """Raised when a YOLO / segmentation model weights cannot be loaded."""


class MarineDetector:
    """Sonar debris detector backed by YOLOv8.

    The detector chains two stages:

    1. **Preprocessing** — grayscale, denoise, CLAHE, normalise, letterbox
       resize to the model's expected input size.
    2. **Inference** — YOLOv8 forward pass with confidence / NMS filtering.

    Detection boxes are remapped from the letterboxed canvas back to the
    original image's coordinate space so callers can overlay them on the
    raw scan.

    When segmentation is enabled, an optional pass attaches a pixel mask to
    every detection (see :meth:`segment`).
    """

    def __init__(self) -> None:
        self.confidence_threshold: float = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.iou_threshold: float = settings.DETECTION_IOU_THRESHOLD
        self.preprocess_config: PreprocessConfig = PreprocessConfig.from_settings(
            settings
        )

        self.seg_enabled: bool = settings.SEGMENTATION_ENABLED
        self.seg_model_type: str = settings.SEGMENTATION_MODEL_TYPE
        self.seg_confidence: float = settings.SEGMENTATION_CONFIDENCE_THRESHOLD
        self.seg_iou: float = settings.SEGMENTATION_IOU_THRESHOLD

        self.anomaly_enabled: bool = settings.ANOMALY_ENABLED
        self.anomaly_config: AnomalyFeatureConfig = (
            AnomalyFeatureConfig.from_settings(settings)
        )
        self.risk_config: RiskConfig = RiskConfig.from_settings(settings)

        self._model = None
        self._seg_model = None
        self._seg_load_error = None

    @property
    def model(self):
        """Lazily load the configured inference model on first access.

        Prefers the edge-deployed artifact for ``INFERENCE_BACKEND`` when it
        exists on disk (see :func:`resolve_inference_model_path`), falling
        back to the PyTorch checkpoint. No artifact and no simulation mode
        raises ``ModelNotAvailableError``.

        Raises:
            ModelNotAvailableError: If no model artifact is available.
        """
        if self._model is None:
            model_path = resolve_inference_model_path(
                settings.YOLO_MODEL_PATH,
                onnx_path=getattr(settings, "YOLO_ONNX_PATH", None),
                engine_path=getattr(settings, "YOLO_ENGINE_PATH", None),
                backend=getattr(settings, "INFERENCE_BACKEND", "torch"),
            )
            if not model_path:
                raise ModelNotAvailableError(
                    "No inference artifact found. Add PyTorch weights, export "
                    "an ONNX/TensorRT model (python -m app.ai.export), or "
                    "enable SIMULATION_MODE."
                )
            self._model = load_yolo_model(model_path)
        return self._model

    @property
    def seg_model(self):
        """Lazily load the segmentation model.

        Returns ``None`` when segmentation is disabled or the weights are
        unavailable (log + no-op rather than hard-fail, since segmentation
        is an optional enhancement).
        """
        if not self.seg_enabled:
            return None
        if self._seg_model is None and self._seg_load_error is None:
            try:
                self._seg_model = load_seg_model(
                    settings.SEGMENTATION_MODEL_PATH,
                    model_type=self.seg_model_type,
                )
            except Exception as exc:  # noqa: BLE001 - optional feature
                logger.warning(
                    "Segmentation disabled: %s", exc
                )
                self._seg_load_error = exc
        return self._seg_model

    def preprocess(self, image_path: str) -> tuple:
        """Run the sonar preprocessing pipeline and return ``(image, params)``.

        ``params`` is ``(scale, pad_x, pad_y, size)`` from the letterbox
        step, used to map detection boxes back to the original image.

        Args:
            image_path: Path to the sonar image.

        Returns:
            Tuple ``(preprocessed_uint8_image, params)``.
        """
        img, params = preprocess_to_uint8(
            image_path, config=self.preprocess_config, return_param=True
        )
        return img, params

    def detect(self, image_path: str) -> list[dict]:
        """Run full detection on a sonar scan.

        Steps:
            1. Preprocess the image to YOLO-ready form.
            2. Run YOLO inference.
            3. Annotate each detection with a severity label.
            4. Map letterboxed boxes back to original coordinates.
            5. Optionally run segmentation to attach a pixel mask / polygon.

        Args:
            image_path: Path to the sonar image file.

        Returns:
            List of detection dicts, each containing:
                ``class_label``, ``class_name``, ``confidence``,
                ``severity``, ``bbox_x``, ``bbox_y``, ``bbox_width``,
                ``bbox_height`` and, when segmentation is enabled,
                ``mask`` (bool array) and ``polygon`` (normalised points).

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            ModelNotAvailableError: If YOLO weights are not on disk and
                simulation mode is disabled.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Scan file not found: {image_path}")

        prepared, params = self.preprocess(image_path)

        simulation = False
        if not self._model_available():
            if settings.SIMULATION_MODE:
                logger.warning(
                    "YOLO weights not found — running in SIMULATION_MODE "
                    "with placeholder detections."
                )
                detections = self._simulate()
                simulation = True
            else:
                raise ModelNotAvailableError(
                    "YOLO weights not found. Upload trained weights or enable "
                    "SIMULATION_MODE to run the pipeline in demo mode."
                )
        else:
            detections = self._predict(prepared)

        if self.seg_enabled:
            self._attach_masks(detections, prepared, params)

        if self.anomaly_enabled:
            self._classify_anomalies(detections, prepared)

        # Simulation boxes are already expressed as fractions of the original
        # image, so the letterbox remap must be skipped for them.
        if simulation:
            return detections

        return self._map_boxes_to_original(detections, params)

    def segment(self, image_path: str) -> list[dict]:
        """Run segmentation on a scan and return per-object masks.

        Convenience wrapper that preprocesses the image and returns results
        in original-image coordinates.  Masks are attached to a synthetic
        detections list.

        Args:
            image_path: Path to the sonar image file.

        Returns:
            List of segmentation dicts (class, bbox, confidence, mask,
            polygon, area) mapped back to the original image coordinates.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Scan file not found: {image_path}")
        if not self.seg_enabled:
            return []

        prepared, params = self.preprocess(image_path)
        seg = self._run_seg(prepared)
        for det in seg:
            det["severity"] = _confidence_to_severity(det["confidence"])
        return self._map_boxes_to_original(seg, params)

    def _model_available(self) -> bool:
        """Return whether a loadable inference artifact exists on disk."""
        return bool(
            resolve_inference_model_path(
                settings.YOLO_MODEL_PATH,
                onnx_path=getattr(settings, "YOLO_ONNX_PATH", None),
                engine_path=getattr(settings, "YOLO_ENGINE_PATH", None),
                backend=getattr(settings, "INFERENCE_BACKEND", "torch"),
            )
        )

    @staticmethod
    def _simulate() -> list[dict]:
        """Produce a small set of realistic placeholder detections.

        Used only in simulation/demo mode when trained weights are absent, so
        the whole pipeline (preprocess -> infer -> anomaly -> risk -> geotag
        -> DB) can be exercised end-to-end. Boxes are normalised fractions of
        the original image.
        """
        import random

        rng = random.Random(2024)
        classes = [
            "ghost_net",
            "shipwreck",
            "pipe",
            "cylinder",
            "container",
            "other_debris",
        ]
        detections = []
        for _ in range(3):
            det = {
                "class_id": classes.index("other_debris"),
                "class_label": rng.choice(classes),
                "class_name": "Other Debris",
                "confidence": round(rng.uniform(0.55, 0.95), 4),
                "bbox_x": round(rng.uniform(0.05, 0.7), 4),
                "bbox_y": round(rng.uniform(0.05, 0.7), 4),
                "bbox_width": round(rng.uniform(0.08, 0.25), 4),
                "bbox_height": round(rng.uniform(0.08, 0.25), 4),
                "severity": _confidence_to_severity(
                    round(rng.uniform(0.55, 0.95), 4)
                ),
            }
            detections.append(det)
        return detections

    def _attach_masks(
        self, detections: list[dict], preprocessed, params: tuple
    ) -> None:
        """Cross-reference detection boxes with segmentation masks.

        Segmentation output is overlapped with each detection's box; the
        most overlapping mask supplies the polygon for that detection.
        """
        if not self.seg_enabled or not detections:
            return

        seg = self._run_seg(preprocessed)
        for det in detections:
            det["mask"] = None
            det["polygon"] = []
            det["mask_area"] = None
            det["seg_backend"] = self.seg_model_type

        for s in seg:
            # Match each segmentation object to the detection whose box best
            # overlaps the segmentation's bounding box.
            best = None
            best_iou = 0.0
            for det in detections:
                iou = _bbox_iou(
                    (det["bbox_x"], det["bbox_y"],
                     det["bbox_width"], det["bbox_height"]),
                    (s["bbox_x"], s["bbox_y"],
                     s["bbox_width"], s["bbox_height"]),
                )
                if iou > best_iou:
                    best_iou = iou
                    best = det
            if best is not None and best_iou > 0.1:
                best["mask"] = s["mask"]
                best["polygon"] = s["polygon"]
                best["mask_area"] = s["area"]

    def _classify_anomalies(
        self, detections: list[dict], preprocessed
    ) -> None:
        """Annotate each detection with an anomaly classification + risk.

        Runs in the letterboxed (canvas) coordinate space so the bbox and
        any mask align with the preprocessed image. Anomaly and risk results
        are attached directly to the detection dicts as ``anomaly_*`` and
        ``risk_*`` / ``final_confidence`` keys.
        """
        for det in detections:
            if self.anomaly_enabled:
                try:
                    result = classify_detection(
                        det, preprocessed, config=self.anomaly_config
                    )
                    det.update(result.to_dict())
                except Exception as exc:  # noqa: BLE001 - don't fail detection
                    logger.warning("Anomaly classification failed: %s", exc)
                    det.update(
                        {
                            "anomaly_class": "uncertain",
                            "natural_probability": 0.5,
                            "artificial_probability": 0.5,
                            "anomaly_confidence": 0.0,
                            "anomaly_evidence": 0.0,
                            "anomaly_features": {},
                        }
                    )
            self._score_risk(det)

    def _score_risk(self, det: dict) -> None:
        """Compute a 0-100 risk score for one detection.

        Fuses object type (per-class hazard prior), detector confidence,
        estimated size (bbox area) and the anomaly's artificial probability.
        Missing inputs are simply not supplied, so the engine renormalises
        over the evidence that is actually available.
        """
        try:
            # Estimated size: bounding-box area as a fraction of the image.
            size = None
            if det.get("bbox_width") is not None and det.get("bbox_height") is not None:
                size = float(det["bbox_width"] * det["bbox_height"])

            risk = compute_risk(
                confidence=det.get("confidence", 0.5),
                object_type=det.get("class_label"),
                estimated_size=size,
                artificial_probability=det.get("artificial_probability"),
                anomaly_class=det.get("anomaly_class"),
                config=self.risk_config,
            )
            det.update(risk.to_dict(ai_confidence=det.get("confidence", 0.5)))
        except Exception as exc:  # noqa: BLE001 - don't fail detection
            logger.warning("Risk scoring failed: %s", exc)
            det.update(
                {
                    "object_type_risk": 0.0,
                    "estimated_size": None,
                    "artificial_probability": float(
                        det.get("artificial_probability", 0.5)
                    ),
                    "final_confidence": float(det.get("confidence", 0.5)),
                    "risk_score": 0.0,
                    "risk_level": "low",
                }
            )

    def _run_seg(self, preprocessed) -> list[dict]:
        """Run the configured segmentation backend on a preprocessed image."""
        model = self.seg_model
        if model is None:
            return []
        return run_segmentation(
            model,
            preprocessed,
            conf=self.seg_confidence,
            iou=self.seg_iou,
            classes=MARINE_CLASSES,
            model_type=self.seg_model_type,
        )

    def _predict(self, preprocessed_image) -> list[dict]:
        """Run YOLO inference on a preprocessed image.

        Returns detection dicts in the **letterboxed** coordinate space
        (normalised 0-1 fractions of the canvas).
        """
        results = run_inference(
            self.model,
            preprocessed_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=MARINE_CLASSES,
            device=getattr(settings, "INFERENCE_DEVICE", None),
        )

        for det in results:
            det["severity"] = _confidence_to_severity(det["confidence"])

        return results

    @staticmethod
    def _map_boxes_to_original(
        detections: list[dict], params: tuple
    ) -> list[dict]:
        """Map letterboxed box fractions back to original image coordinates.

        Args:
            detections: Boxes as normalised fractions of the letterboxed
                canvas.  Each must have ``bbox_x, bbox_y, bbox_width,
                bbox_height`` in [0, 1].
            params: ``(scale, pad_x, pad_y, canvas_size)`` from the
                letterbox step.

        Returns:
            Detections with boxes re-expressed as fractions of the original,
            un-padded image.
        """
        scale, pad_x, pad_y, canvas_size = params

        if canvas_size == 0 or scale == 0:
            return detections

        # Dimensions of the original (unpadded) image. Padding is applied to
        # the short edges, so subtract it from both sides before scaling back.
        orig_w = (canvas_size - 2 * pad_x) / scale
        orig_h = (canvas_size - 2 * pad_y) / scale

        output: list[dict] = []
        for det in detections:
            # Fraction-of-canvas -> absolute canvas pixels.
            cx = det["bbox_x"] * canvas_size
            cy = det["bbox_y"] * canvas_size
            cw = det["bbox_width"] * canvas_size
            ch = det["bbox_height"] * canvas_size

            # Subtract letterbox padding -> pixels in the resized image.
            rx = cx - pad_x
            ry = cy - pad_y

            # Scale back to original pixel space, then normalise.
            out = dict(det)
            out["bbox_x"] = round(max(0.0, rx / scale / orig_w), 4)
            out["bbox_y"] = round(max(0.0, ry / scale / orig_h), 4)
            out["bbox_width"] = round(max(0.0, cw / scale / orig_w), 4)
            out["bbox_height"] = round(max(0.0, ch / scale / orig_h), 4)

            # Remap any segmentation polygon from canvas-space back to
            # original-image space so the dashboard can overlay it accurately.
            if isinstance(det.get("polygon"), list) and det["polygon"]:
                out["polygon"] = _remap_polygon(
                    det["polygon"], scale, pad_x, pad_y, canvas_size
                )
            else:
                out.setdefault("polygon", [])

            output.append(out)

        return output


def _remap_polygon(
    polygon: list[list[float]],
    scale: float,
    pad_x: float,
    pad_y: float,
    canvas_size: float,
) -> list[list[float]]:
    """Map a normalised polygon from the canvas back to the original image."""
    if canvas_size == 0 or scale == 0:
        return polygon

    orig_w = (canvas_size - 2 * pad_x) / scale
    orig_h = (canvas_size - 2 * pad_y) / scale

    mapped: list[list[float]] = []
    for px, py in polygon:
        cx = px * canvas_size - pad_x
        cy = py * canvas_size - pad_y
        mx = max(0.0, min(cx / scale / orig_w, 1.0))
        my = max(0.0, min(cy / scale / orig_h, 1.0))
        mapped.append([round(mx, 4), round(my, 4)])
    return mapped


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two boxes given as (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    inter_x = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_y = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_x * inter_y

    area_a = max(aw, 0.0) * max(ah, 0.0)
    area_b = max(bw, 0.0) * max(bh, 0.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
