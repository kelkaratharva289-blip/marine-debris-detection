"""Report generation: JSON/CSV outputs from real measured results.

Serialises the actual pipeline results (split counts, validation report,
training args, real P/R/mAP metrics, test inference timing, detections with
anomaly + risk) to human- and machine-readable JSON and CSV files under the
output directory.

The reports contain only values produced by the stages above â€” never
synthetic, extrapolated, or fallback data. Where a metric could not be
measured it is recorded as ``null`` with an explanation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.training.config import TrainConfig
from app.training.detect import DetectOutcome
from app.training.evaluate import EvalSplit, TimingMetrics
from app.training.split import SplitResult
from app.training.train import TrainOutcome


@dataclass
class PipelineReport:
    """Aggregates every stage result into a single report."""

    config: TrainConfig
    split: SplitResult
    train: TrainOutcome
    validation_metrics: EvalSplit
    test_metrics: EvalSplit
    test_timing: TimingMetrics
    detection: DetectOutcome


def _risk_distribution(detections) -> dict:
    dist: dict[str, int] = {}
    for d in detections:
        dist[d.risk_level] = dist.get(d.risk_level, 0) + 1
    return dist


def _anomaly_distribution(detections) -> dict:
    dist: dict[str, int] = {}
    for d in detections:
        dist[d.anomaly_class] = dist.get(d.anomaly_class, 0) + 1
    return dist


def build_summary_json(report: PipelineReport, generation_timestamp: str) -> dict:
    """Assemble the full JSON summary from real stage results."""
    split_counts = report.split.counts
    return {
        "generated_at": generation_timestamp,
        "config": report.config.as_dict(),
        "dataset": {
            "split_counts": dict(split_counts),
            "validation": report.split.validation.as_dict(),
        },
        "training": {
            "best_model": str(report.train.deployed_pt),
            "run_dir": str(report.train.best_pt),
            "args": report.train.args,
        },
        "validation_metrics": report.validation_metrics.as_dict(),
        "test_metrics": report.test_metrics.as_dict(),
        "test_inference_timing": report.test_timing.as_dict(),
        "detection_summary": {
            "images_processed": report.detection.images_processed,
            "total_detections": len(report.detection.detections),
            "annotated_images": report.detection.annotated_images,
            "risk_distribution": _risk_distribution(report.detection.detections),
            "anomaly_distribution": _anomaly_distribution(report.detection.detections),
        },
    }


def write_detection_csv(detections, path: Path) -> Path:
    """Write the real detection records to a CSV file."""
    if not detections:
        # Still write a CSV with headers so consumers see an empty report.
        headers = [
            "image", "class_id", "class_label", "class_name", "confidence",
            "bbox_x", "bbox_y", "bbox_width", "bbox_height",
            "anomaly_class", "natural_probability", "artificial_probability",
            "anomaly_confidence", "ai_confidence", "final_confidence",
            "risk_score", "risk_level",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
        return path

    record = detections[0].as_dict()
    fieldnames = list(record.keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            writer.writerow(d.as_dict())
    return path


def write_reports(report: PipelineReport) -> dict:
    """Write JSON + CSV reports and return their paths."""
    ts = datetime.now(timezone.utc).isoformat()
    out = report.config.out
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary_json(report, ts)
    summary_path = out / "report.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    detections_csv = out / "detections.csv"
    write_detection_csv(report.detection.detections, detections_csv)

    # Per-image detection count CSV (real counts per test image).
    per_image: dict[str, int] = {}
    for d in report.detection.detections:
        per_image[d.image] = per_image.get(d.image, 0) + 1
    per_image_path = out / "per_image_counts.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "detection_count"])
        for name in sorted(per_image):
            writer.writerow([name, per_image[name]])

    return {
        "report_json": str(summary_path),
        "detections_csv": str(detections_csv),
        "per_image_csv": str(per_image_path),
    }


def print_progress(stage: str, message: str = "") -> None:
    """Print a plain pipeline progress line."""
    print(f"[pipeline] {stage} {message}".rstrip())

