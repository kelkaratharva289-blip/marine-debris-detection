"""Command-line interface for the end-to-end training pipeline.

Run from the ``backend`` directory::

    python train.py                            # run with defaults
    python train.py --device cpu --epochs 50
    python train.py --epochs 60 --batch 8 --imgsz 640
    python train.py --config runs/config.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.ai.dataset.constants import default_data_root
from app.training.config import TrainConfig, build_config, load_config
from app.training.pipeline import run_pipeline
from app.training.validate_dataset import summarize

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "End-to-end real-data training pipeline for side-scan sonar "
            "marine debris detection: validate -> split -> preprocess -> "
            "train -> validate -> test -> detect -> anomaly -> risk -> reports. "
            "Place your real images in <data_root>/raw/images and labels in "
            "<data_root>/raw/labels, then run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # dataset
    parser.add_argument(
        "--data-root", default=str(default_data_root()),
        help="Dataset root containing raw/images and raw/labels (default: backend/data).",
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Override raw images directory (default: <data-root>/raw/images).",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Override raw labels directory (default: <data-root>/raw/labels).",
    )
    # split
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train fraction.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test fraction.")
    parser.add_argument("--split-seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified splits.")
    # training
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model/weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (px).")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda:0' (default: auto).")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stop patience (0=off).")
    parser.add_argument("--optimizer", default="auto", help="YOLO optimizer.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Explicit class vocabulary (label-id order), e.g. "
             "--classes ghost_net pipe other_debris. Default: inferred from the "
             "actual label files present in the dataset.",
    )
    # detection
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection IoU (NMS) threshold.")
    # output
    parser.add_argument(
        "--output-dir", default="",
        help="Output directory (default: <data-root>/training_outputs).",
    )
    parser.add_argument(
        "--best-model", default="models/marine_debris_yolov8.pt",
        help="Where to copy the best trained model (best.pt).",
    )
    # config
    parser.add_argument(
        "--config", default=None,
        help="Optional JSON config file overlaying these flags onto defaults.",
    )
    # behaviour
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch on real data for a fast smoke test.")
    parser.add_argument("--verbose", action="store_true", help="Verbose Ultralytics logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Start from defaults, overlay any JSON config, then overlay explicit flags.
    overrides: dict = {}
    if args.config:
        overrides.update(load_config(args.config))
    flag_values = {
        "data_root": args.data_root,
        "raw_images_dir": args.images_dir,
        "raw_labels_dir": args.labels_dir,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "stratify": not args.no_stratify,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
"cos_lr": args.cos_lr,
        "seed": args.seed,
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "output_dir": args.output_dir,
        "best_model_path": args.best_model,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
    }
    overrides.update(flag_values)
    config = build_config(**overrides)

    try:
        report = run_pipeline(config, stream=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[train] ERROR: {exc}", file=sys.stderr)
        return 1

    # Final summary from real results.
    v = report.validation_metrics
    t = report.test_metrics
    print()
    print("=" * 64)
    print("PIPELINE COMPLETE")
    print(f"  Best model : {report.train.deployed_pt}")
    print(f"  Val mAP50  : {v.map50}")
    print(f"  Val mAP50-95: {v.map50_95}")
    print(f"  Test mAP50 : {t.map50}")
    print(f"  Test mAP50-95: {t.map50_95}")
    print(f"  Test P/R   : {t.precision} / {t.recall}")
    print(f"  Inference  : {report.test_timing.mean_ms} ms/image "
          f"({report.test_timing.fps} fps)")
    print(f"  Detections : {len(report.detection.detections)} "
          f"on {report.detection.images_processed} test image(s)")
    print(f"  Reports    : {report.config.out}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())

