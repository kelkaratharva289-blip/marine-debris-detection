#!/usr/bin/env python
"""Evaluation-only entry point for the trained marine-debris YOLO model.

Run from the ``backend`` directory::

    python run_evaluation.py                          # evaluate on the test split
    python run_evaluation.py --split val             # evaluate on the validation split
    python run_evaluation.py --best-model models/my_model.pt --device cpu

Runs the trained model against a real held-out split and writes an evaluation
artifact set (all derived from actual inference, never fabricated):

* Precision, Recall, mAP50, mAP50-95 (and per-class AP)
* Per-image inference time (mean / median / p95 / FPS)
* A confusion matrix (raw counts + normalized PNG)
* Detection visualizations (predicted boxes on the real scans)
* evaluation_report.json and an evaluations detections CSV

Requires the split folders to exist (run ``train.py`` or ``train_model.py``
first) and a trained checkpoint (``models/marine_debris_yolov8.pt`` by default).
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.ai.dataset.constants import default_data_root
from app.training.config import build_config
from app.training.evaluation import (
    print_evaluation,
    run_evaluation,
    save_evaluation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_evaluation",
        description=(
            "Evaluate the trained YOLO model on a real split: P/R/mAP, mAP50-95, "
            "inference time, confusion matrix, and detection visualizations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(default_data_root()),
                        help="Dataset root with raw/ + train/val/test splits.")
    parser.add_argument("--best-model", default="models/marine_debris_yolov8.pt",
                        help="Path to the trained model weights (best.pt).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (px).")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda:0' (default: auto).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--timing-batches", type=int, default=1,
                        help="Passes over the split for stable timing.")
    parser.add_argument("--verbose", action="store_true", help="Verbose Ultralytics logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.verbose:
        logging.getLogger("app").setLevel(logging.DEBUG)

    config = build_config(
        data_root=args.data_root,
        best_model_path=args.best_model,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    try:
        outcome = run_evaluation(config, split=args.split, batches=args.timing_batches)
        report_path = save_evaluation_report(outcome, config.out)
        print_evaluation(outcome)
        print(f"[run_evaluation] report saved: {report_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[run_evaluation] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())