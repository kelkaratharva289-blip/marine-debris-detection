"""Orchestrator: chain every pipeline stage into a single run.

Stage order (matches the requested pipeline):

    Real Dataset
      -> 1. Validation (real images + YOLO labels checked)
      -> 2. Automatic Split (deterministic, stratified)
      -> 3. Preprocessing (real sonar enhancement on split images)
      -> 4. YOLO Training (saves best.pt)
      -> 5. Validation metrics (P / R / mAP50 / mAP50-95)
      -> 6. Testing metrics + inference timing
      -> 7. Detection (confidence) on the test split
      -> 8. Natural/Artificial analysis + Risk score
      -> 9. JSON / CSV reports

Every step runs on **real** data. No dummy, simulated, or fallback results
are produced anywhere in the chain.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.ai.dataset.constants import DEFAULT_CLASSES
from app.training.config import TrainConfig, save_config
from app.training.detect import run_detection_pipeline
from app.training.evaluate import evaluate_test, evaluate_validation
from app.training.preprocess import preprocess_dataset
from app.training.report import (
    PipelineReport,
    print_progress,
    write_reports,
)
from app.training.split import prepare_and_split
from app.training.train import train_yolo
from app.training.validate_dataset import summarize

logger = logging.getLogger(__name__)


def run_pipeline(config: TrainConfig, stream: bool = True) -> PipelineReport:
    """Execute the full training pipeline and return the aggregated report.

    Args:
        config: Resolved :class:`TrainConfig`.
        stream: If True, print stage progress lines to stdout.

    Returns:
        A :class:`PipelineReport` aggregating every stage's real results.
    """
    root = config.root
    root.mkdir(parents=True, exist_ok=True)
    config.out.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config for auditing before anything else.
    save_config(config, config.out / "training_config.json")

    # ---- 1. Validation + 2. Automatic split --------------------------------
    print_progress("validating real dataset")
    split = prepare_and_split(
        images_dir=config.images_dir,
        labels_dir=config.labels_dir,
        data_root=root,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
        stratify=config.stratify,
        num_classes=len(DEFAULT_CLASSES),
        classes=config.classes,
    )
    print_progress("split", str(split.counts))

    # Print validation summary so the user sees dataset quality.
    print_progress("validation report:")
    for line in summarize(split.validation).splitlines():
        print(f"  {line}")

    # Guard: ensure each split has at least 1 image for downstream stages.
    counts = split.counts
    for split_name in ("train", "val", "test"):
        if counts.get(split_name, 0) == 0:
            raise ValueError(
                f"The '{split_name}' split has 0 images. Adjust split ratios or "
                f"provide more images to ensure every split is non-empty."
            )

    # ---- 3. Preprocessing ---------------------------------------------------
    print_progress("preprocessing (real sonar enhancement)")
    preprocess_result = preprocess_dataset(root, imgsz=config.imgsz)
    for split_name, res in preprocess_result.items():
        n_ok = res.get("count", 0)
        n_fail = len(res.get("failed", []))
        if n_fail:
            logger.warning("%s: %d preprocessed, %d failed", split_name, n_ok, n_fail)

    # ---- 4. Training ---------------------------------------------------------
    print_progress("training YOLO (real data)")
    train_outcome = train_yolo(config)

    # ---- 5. Validation metrics ----------------------------------------------
    print_progress("validating (P/R/mAP)")
    val_metrics = evaluate_validation(config)

    # ---- 6. Test metrics + timing ---------------------------------------------
    print_progress("testing + inference timing")
    test_metrics, test_timing = evaluate_test(config)

    # ---- 7/8. Detection + anomaly + risk --------------------------------------
    print_progress("detection + anomaly + risk on test split")
    detection = run_detection_pipeline(config)

    # ---- 9. Reports -----------------------------------------------------------
    report = PipelineReport(
        config=config,
        split=split,
        train=train_outcome,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        test_timing=test_timing,
        detection=detection,
    )
    outputs = write_reports(report)
    print_progress("reports written", str(outputs))
    return report

