#!/usr/bin/env python
"""Testing-only entry point for the marine debris YOLO pipeline.

Run from the ``backend`` directory::

    python test_model.py                             # test default model
    python test_model.py --best-model models/my_model.pt
    python test_model.py --data-root backend/data --device cpu

Loads the trained model (default ``models/marine_debris_yolov8.pt``) and runs
genuine TEST-split inference: Precision / Recall / mAP50 / mAP50-95 plus
per-image inference time (ms, FPS). Requires the split folders to exist (run
``train.py`` or ``train_model.py`` first).

Only real data is used; a missing checkpoint or split is reported clearly.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.ai.dataset.constants import default_data_root
from app.training.config import build_config
from app.training.evaluate import evaluate_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_model",
        description=(
            "Evaluate the trained YOLO model on the TEST split: real P/R/mAP, "
            "mAP50-95 and per-image inference timing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(default_data_root()),
                        help="Dataset root with raw/ + train/val/test splits.")
    parser.add_argument("--best-model", default="models/marine_debris_yolov8.pt",
                        help="Path to the trained model weights (best.pt).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (px).")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda:0' (default: auto).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold.")
    parser.add_argument("--timing-batches", type=int, default=1,
                        help="Passes over the test set for stable timing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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
        metrics, timing = evaluate_test(config, batches=args.timing_batches)
    except Exception as exc:  # noqa: BLE001 - surface actionable error
        print(f"[test_model] ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    print("=" * 56)
    print("TEST RESULTS (real inference on the test split)")
    print(f"  Precision      : {metrics.precision}")
    print(f"  Recall         : {metrics.recall}")
    print(f"  mAP@50         : {metrics.map50}")
    print(f"  mAP@50-95      : {metrics.map50_95}")
    print(f"  Images         : {metrics.n_images}")
    print(f"  GT boxes       : {metrics.n_gt_boxes}")
    if metrics.per_class:
        print("  Per-class AP   :")
        for cid, ap in sorted(metrics.per_class.items()):
            print(f"    class {cid}: {ap}")
    print(f"  Inference      : {timing.mean_ms} ms/image ({timing.fps} fps) "
          f"[{timing.samples} samples, p95 {timing.p95_ms} ms]")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())