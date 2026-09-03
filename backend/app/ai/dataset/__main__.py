"""Command-line interface for the YOLO sonar dataset pipeline.

Run from the ``backend`` directory::

    # Summarise what is currently in the raw pool (reads only, writes nothing)
    python -m app.ai.dataset inspect [--images ...] [--labels ...]

    # Build train/val/test splits + data.yaml from the raw pool
    python -m app.ai.dataset build [--root backend/data]
                                   [--ratios 0.7 0.15 0.15] [--seed 42]
                                   [--move] [--clean]

    # Write only the Ultralytics data.yaml for an existing split layout
    python -m app.ai.dataset yaml [--root ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.ai.dataset.builder import build_dataset, write_dataset_yaml
from app.ai.dataset.constants import DEFAULT_CLASSES, default_data_root
from app.ai.dataset.loader import load_dataset
from app.ai.dataset.split import SplitRatios

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.ai.dataset")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.ai.dataset",
        description=(
            "Load real sonar images + YOLO labels from the raw pool, validate them, "
            "and split into train/val/test folders for Ultralytics training."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_data_root(),
        help=(
            "Dataset root directory (default: backend/data). Must contain "
            "raw/images and raw/labels."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=len(DEFAULT_CLASSES),
        help=f"Number of classes to validate label ids against (default: {len(DEFAULT_CLASSES)}).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Summarise the raw pool (no writes).")
    inspect.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Override the labels directory (default: <root>/raw/labels).",
    )
    inspect.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override the images directory (default: <root>/raw/images).",
    )

    build = sub.add_parser("build", help="Build train/val/test splits + data.yaml.")
    build.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.70, 0.15, 0.15],
        help="Train/val/test fractions (default: 0.70 0.15 0.15).",
    )
    build.add_argument(
        "--seed", type=int, default=42, help="Deterministic split seed (default: 42)."
    )
    build.add_argument(
        "--move",
        action="store_true",
        help="Move files out of the raw pool instead of copying them.",
    )
    build.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing files inside train/val/test folders first.",
    )
    build.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use a plain shuffled split instead of class-stratified splitting.",
    )

    yaml_cmd = sub.add_parser("yaml", help="Write only data.yaml for an existing layout.")
    yaml_cmd.add_argument(
        "--relative-path",
        action="store_true",
        help="Write train/val/test as repo-relative paths.",
    )

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    images_dir = args.images_dir or root / "raw" / "images"
    labels_dir = args.labels_dir or root / "raw" / "labels"

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        return 1

    samples = []
    missing_labels = 0
    try:
        for _ in load_dataset(images_dir, labels_dir, args.num_classes):
            samples.append(_)
    except Exception as exc:  # noqa: BLE001 - surface validation errors clearly
        logger.error("Dataset validation failed: %s", exc)
        return 1

    unlabelled = [s for s in samples if not s.is_labelled]
    distribution: dict[int, int] = {}
    total_boxes = 0
    for sample in samples:
        total_boxes += len(sample.annotations)
        for class_id in sample.class_ids:
            distribution[class_id] = distribution.get(class_id, 0) + 1

    logger.info(
        "Images: %d  |  annotated: %d  |  boxes: %d  |  unlabelled: %d",
        len(samples),
        len(samples) - len(unlabelled),
        total_boxes,
        len(unlabelled),
    )
    for class_id in sorted(distribution):
        logger.info(
            "  class %d (%s): %d image(s)",
            class_id,
            DEFAULT_CLASSES[class_id] if class_id < len(DEFAULT_CLASSES) else "?",
            distribution[class_id],
        )
    if not samples:
        logger.warning(
            "No supported images found. Place real scans in %s", images_dir
        )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    train, val, test = args.ratios
    ratios = SplitRatios(train=train, val=val, test=test)
    try:
        result = build_dataset(
            root=args.root,
            ratios=ratios,
            seed=args.seed,
            move=args.move,
            clean=args.clean,
            stratify=not args.no_stratify,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("%s", result)
    return 0


def cmd_yaml(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if not root.is_dir():
        logger.error("Root not found: %s", root)
        return 1
    path = write_dataset_yaml(root, classes=DEFAULT_CLASSES)
    logger.info("Wrote %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "yaml":
        return cmd_yaml(args)
    _parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())