#!/usr/bin/env python
"""Training-only entry point for the marine debris YOLO pipeline.

Run from the ``backend`` directory::

    python train_model.py                      # defaults (infer classes)
    python train_model.py --epochs 150 --batch 32 --imgsz 640
    python train_model.py --device cpu
    python train_model.py --classes ghost_net pipe other_debris

Place your real dataset in ``backend/data/raw/images`` (images) and
``backend/data/raw/labels`` (YOLO ``.txt``). The class vocabulary is inferred
**only** from the actual label files present (e.g. Ghost Net, Shipwreck, Pipe,
Cylinder, Container, Other Debris — using the subset that really appears). The
best trained checkpoint is saved as ``best.pt`` and copied to
``models/marine_debris_yolov8.pt``.

Only real data is used. Run ``python train_model.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.ai.dataset.constants import DEFAULT_CLASSES, default_data_root
from app.training.config import TrainConfig, build_config
from app.training.preprocess import preprocess_dataset
from app.training.split import prepare_and_split
from app.training.train import train_yolo
from app.training.validate_dataset import summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_model",
        description=(
            "Train the YOLO detector on real side-scan sonar data: validate -> "
            "split -> preprocess -> train. Saves best.pt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=str(default_data_root()),
                        help="Dataset root containing raw/images + raw/labels.")
    parser.add_argument("--images-dir", default=None,
                        help="Override raw images directory.")
    parser.add_argument("--labels-dir", default=None,
                        help="Override raw labels directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train fraction.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test fraction.")
    parser.add_argument("--split-seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified split.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model/weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
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
        help="Explicit class vocabulary (default: inferred from real labels).",
    )
    parser.add_argument(
        "--best-model", default="models/marine_debris_yolov8.pt",
        help="Where to copy the best trained model (best.pt).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose Ultralytics logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config: TrainConfig = build_config(
        data_root=args.data_root,
        raw_images_dir=args.images_dir,
        raw_labels_dir=args.labels_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        stratify=not args.no_stratify,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        seed=args.seed,
        classes=args.classes,
        best_model_path=args.best_model,
        verbose=args.verbose,
    )

    try:
        config.root.mkdir(parents=True, exist_ok=True)
        config.out.mkdir(parents=True, exist_ok=True)

        print(f"[train_model] dataset classes: {config.classes}")
        print("[train_model] validating real dataset")
        split = prepare_and_split(
            config.images_dir, config.labels_dir, config.root,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.split_seed,
            stratify=config.stratify,
            num_classes=len(DEFAULT_CLASSES),
            classes=config.classes,
        )
        for line in summarize(split.validation).splitlines():
            print(f"  {line}")

        print("[train_model] preprocessing (real sonar enhancement)")
        preprocess_dataset(config.root, imgsz=config.imgsz)

        print(f"[train_model] training YOLO (classes={config.classes})")
        outcome = train_yolo(config)

        print(f"[train_model] best model saved: {outcome.deployed_pt}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[train_model] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())