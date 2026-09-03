"""Evaluation pipeline: real metrics, confusion matrix, and visualizations.

Runs the trained YOLO model against a held-out split (default TEST) and
produces an evaluation artifact set composed **only** of real results:

* Precision, Recall, mAP50, mAP50-95 (+ per-class AP)
* Per-image inference time (mean / median / p95 / FPS), measured on real images
* A confusion matrix (raw counts + normalized PNG) from the actual
  predictions-vs-ground-truth matches
* Detection visualizations: predicted boxes drawn on the real scanned pixels

Nothing here is simulated, extrapolated, or fabricated. Empty splits or a
missing checkpoint are reported clearly rather than papered over.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ai.dataset.constants import IMAGE_EXTENSIONS
from app.training.config import TrainConfig
from app.training.evaluate import (
    ConfusionMatrixData,
    EvalSplit,
    TimingMetrics,
    build_detection_visualizations,
    extract_confusion_matrix,
    extract_metrics,
    measure_timing,
)
from app.training.report import write_detection_csv

logger = logging.getLogger(__name__)


@dataclass
class EvaluationOutcome:
    """All real results from evaluating the model on one split."""

    split: str
    metrics: EvalSplit
    timing: TimingMetrics
    confusion_matrix: ConfusionMatrixData
    visualization_paths: list[str] = field(default_factory=list)
    detections_csv: str | None = None
    n_detections: int = 0

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "metrics": self.metrics.as_dict(),
            "timing": self.timing.as_dict(),
            "confusion_matrix": self.confusion_matrix.as_dict(),
            "detection_visualizations": list(self.visualization_paths),
            "total_detections": self.n_detections,
            "detections_csv": self.detections_csv,
        }


def _split_dirs(config: TrainConfig, split: str) -> tuple[Path, Path]:
    split = split.lower()
    if split == "test":
        return config.root / "test" / "images", config.root / "test" / "labels"
    if split == "val":
        return config.root / "val" / "images", config.root / "val" / "labels"
    if split == "train":
        return config.root / "train" / "images", config.root / "train" / "labels"
    raise ValueError(f"Unknown split '{split}'; expected 'train', 'val' or 'test'.")


def run_evaluation(config: TrainConfig, split: str = "test", batches: int = 1) -> EvaluationOutcome:
    """Run the full evaluation pipeline on one real split.

    Args:
        config: Resolved :class:`TrainConfig` (model from ``best_model_path``).
        split: Which split to evaluate ("test" by default).
        batches: Passes over the split for stable timing measurements.

    Returns:
        An :class:`EvaluationOutcome` aggregating the real metrics, confusion
        matrix, timing, and visualization paths.

    Raises:
        FileNotFoundError: If the split folders or the trained model are missing.
        ValueError: If the split is empty or an unknown split name is given.
    """
    from ultralytics import YOLO

    images_dir, labels_dir = _split_dirs(config, split)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(
            f"No supported images found in {images_dir} for split '{split}'. "
            "Run the split stage first so the real data is partitioned."
        )

    model_path = Path(config.best_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {model_path}. "
            "Train the model first (e.g. python train_model.py)."
        )
    model = YOLO(config.best_model_path)

    det_metrics = model.val(
        data=str(config.root / "data.yaml"),
        split=split,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        conf=config.conf,
        iou=config.iou,
        project=str(config.out / "evaluation_runs"),
        name=split,
        exist_ok=True,
        verbose=False,
    )

    metrics = extract_metrics(det_metrics)
    metrics.n_images = len(image_files)
    metrics.n_gt_boxes = _count_gt_boxes(images_dir, labels_dir)

    cm = extract_confusion_matrix(det_metrics, names=config.classes, split=split, config=config)

    timing = measure_timing(model, images_dir, config, batches=batches)

    # Capture the real detections per image (same inference as validation) so
    # the visualizations and CSV reflect the actual scored predictions.
    detections_by_image = _capture_detections(model, images_dir, config)
    total_detections = sum(len(v) for v in detections_by_image.values())

    viz_dir = config.out / "evaluation" / "detections"
    visualization_paths = build_detection_visualizations(
        images_dir=images_dir,
        detections_by_image=detections_by_image,
        names=config.classes,
        save_dir=viz_dir,
        conf=config.conf,
    )

    detection_csv = None
    if total_detections:
        records = _flatten_records(detections_by_image, names=config.classes)
        detection_csv = str(config.out / "evaluation" / f"evaluation_detections_{split}.csv")
        write_detection_csv(records, Path(detection_csv))

    return EvaluationOutcome(
        split=split,
        metrics=metrics,
        timing=timing,
        confusion_matrix=cm,
        visualization_paths=visualization_paths,
        detections_csv=detection_csv,
        n_detections=total_detections,
    )


def save_evaluation_report(outcome: EvaluationOutcome, out_base: Path) -> Path:
    """Write the evaluation outcome to ``evaluation_report.json``.

    Args:
        outcome: The real evaluation results to serialise.
        out_base: Base output directory (``config.out``).

    Returns:
        Path to the written JSON report.
    """
    out_base.mkdir(parents=True, exist_ok=True)
    path = out_base / "evaluation" / f"evaluation_report_{outcome.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _count_gt_boxes(images_dir: Path, labels_dir: Path) -> int:
    from app.ai.dataset.constants import IMAGE_EXTENSIONS
    total = 0
    for ext in IMAGE_EXTENSIONS:
        for image in images_dir.glob(f"*{ext}"):
            label = labels_dir / f"{image.stem}.txt"
            if label.is_file():
                total += sum(1 for line in label.read_text(encoding="utf-8").splitlines()
                             if line.strip())
    return total


def _capture_detections(model, images_dir: Path, config: TrainConfig) -> dict[str, list[dict]]:
    """Run real YOLO inference on every split image and group detections.

    Detections are computed on the same preprocessed canvases used elsewhere in
    the pipeline and their normalized boxes are mapped back to the original
    image space (undoing letterbox padding). No detections are manufactured:
    an image with nothing above ``conf`` contributes an empty list.

    Returns a mapping of image filename -> list of detection dicts, each with
    ``class_id``, ``confidence`` and fraction ``bbox_x/y/w/h`` in original space.
    """
    from app.ai.preprocessing import PreprocessConfig
    from app.ai.preprocessing.loader import load_image
    from app.ai.preprocessing.pipeline import preprocess_to_uint8
    from app.ai.inference import run_inference

    pre_cfg = PreprocessConfig(target_size=config.imgsz)
    by_image: dict[str, list[dict]] = {}

    for image in sorted(p for p in images_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        img = load_image(str(image), color=False)
        canvas = preprocess_to_uint8(img, config=pre_cfg)
        raw = run_inference(
            model, canvas,
            conf=config.conf, iou=config.iou,
            classes=config.classes, device=config.device,
        )

        h_raw, w_raw = img.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        scale = min(canvas_w / w_raw, canvas_h / h_raw)
        pad_x = (canvas_w - w_raw * scale) / 2.0
        pad_y = (canvas_h - h_raw * scale) / 2.0

        dets: list[dict] = []
        for d in raw:
            cx = d["bbox_x"] * canvas_w
            cy = d["bbox_y"] * canvas_h
            cw = d["bbox_width"] * canvas_w
            ch = d["bbox_height"] * canvas_h
            dets.append({
                "class_id": d["class_id"],
                "confidence": d["confidence"],
                "bbox_x": max(0.0, min((cx - pad_x) / scale / w_raw, 1.0)),
                "bbox_y": max(0.0, min((cy - pad_y) / scale / h_raw, 1.0)),
                "bbox_width": max(0.0, min(cw / scale / w_raw, 1.0)),
                "bbox_height": max(0.0, min(ch / scale / h_raw, 1.0)),
            })
        by_image[image.name] = dets

    return by_image


def _flatten_records(detections_by_image: dict[str, list[dict]],
                     names: list[str]) -> list[dict]:
    """Flatten grouped detections into simple dict records for CSV output."""
    records: list[dict] = []
    for name, dets in detections_by_image.items():
        for d in dets:
            cid = int(d["class_id"])
            records.append({
                "image": name,
                "class_id": cid,
                "class_name": names[cid] if cid < len(names) else f"id{cid}",
                "confidence": round(float(d["confidence"]), 4),
                "bbox_x": round(float(d["bbox_x"]), 4),
                "bbox_y": round(float(d["bbox_y"]), 4),
                "bbox_width": round(float(d["bbox_width"]), 4),
                "bbox_height": round(float(d["bbox_height"]), 4),
            })
    return records


def print_evaluation(outcome: EvaluationOutcome) -> None:
    """Print a human-readable summary of the real evaluation results."""
    m = outcome.metrics
    t = outcome.timing
    print()
    print("=" * 56)
    print(f"EVALUATION - {outcome.split.upper()} SPLIT (real results)")
    print(f"  Precision      : {m.precision}")
    print(f"  Recall         : {m.recall}")
    print(f"  mAP@50         : {m.map50}")
    print(f"  mAP@50-95      : {m.map50_95}")
    print(f"  Images         : {m.n_images}")
    print(f"  GT boxes       : {m.n_gt_boxes}")
    print(f"  Detections     : {outcome.n_detections}")
    if m.per_class:
        print("  Per-class AP   :")
        for cid, ap in sorted(m.per_class.items()):
            print(f"    class {cid}: {ap}")
    print(f"  Inference      : {t.mean_ms} ms/image ({t.fps} fps) "
          f"[{t.samples} samples, p95 {t.p95_ms} ms]")
    if outcome.confusion_matrix.plot_path:
        print(f"  Confusion      : {outcome.confusion_matrix.plot_path}")
    if outcome.visualization_paths:
        print(f"  Visualizations : {len(outcome.visualization_paths)} "
              f"({outcome.visualization_paths[0]} ...)")
    print("=" * 56)
