"""Build train/val/test folders from a real sonar image + label pool.

This module only ever moves or copies **real** files from the raw pool:
``raw/images`` and ``raw/labels``. It never synthesises images, writes fake
annotations, or fabricates anything. The output is a standard Ultralytics
layout::

    data/
    ├── data.yaml                 # generated so model.train(data=...) works
    ├── raw/
    │   ├── images/               # source pool (real sonar scans)
    │   ├── labels/               # source pool (real YOLO .txt labels)
    │   └── metadata/
    │       └── split_manifest.json
    ├── train/{images,labels}/
    ├── val/{images,labels}/
    └── test/{images,labels}/
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.ai.dataset.constants import (
    DEFAULT_CLASSES,
    default_data_root,
    split_dirs,
)
from app.ai.dataset.loader import Sample, load_dataset
from app.ai.dataset.split import SplitDataset, SplitRatios, split_dataset

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Summary of a successful build."""

    root: Path
    config_yaml: Path
    manifest: Path
    counts: dict[str, int]

    def __str__(self) -> str:
        line = ", ".join(f"{key}={value}" for key, value in self.counts.items())
        return f"Built {self.root} ({line}) - {self.config_yaml.name} written"


def build_dataset(
    root: Path | None = None,
    ratios: SplitRatios | None = None,
    seed: int = 42,
    move: bool = False,
    clean: bool = False,
    stratify: bool = True,
    required_classes: Sequence[str] | None = None,
    label_class_map: dict[int, int] | None = None,
) -> BuildResult:
    """Load the raw pool, split it, and materialise train/val/test folders.

    Args:
        root: Dataset root (default ``backend/data``). Must already contain
            ``raw/images`` and ``raw/labels``.
        ratios: Train/val/test fractions (default 70/15/15).
        seed: Deterministic split seed.
        move: If True, move files out of the raw pool; otherwise copy them.
        clean: If True, remove pre-existing files in the split folders first.
        stratify: If True (default), stratify splits by class composition.
        required_classes: Class vocabulary to write into ``data.yaml``
            (default: the canonical marine-debris classes).
        label_class_map: Optional mapping of original label class-id -> new
            compact id (index into ``required_classes``). When provided, every
            copied label file is rewritten with the mapped ids so YOLO's
            ``data.yaml`` indices line up with the labels in the split folders.
            When ``None``, label files are copied verbatim.

    Returns:
        A :class:`BuildResult` summarising the build.

    Raises:
        ValueError: If the raw pool directories do not exist or are empty.
    """
    root = (Path(root) if root is not None else default_data_root()).resolve()
    images_dir, labels_dir, metadata_dir = root / "raw" / "images", root / "raw" / "labels", root / "raw" / "metadata"

    if not images_dir.is_dir():
        raise ValueError(f"Raw images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"Raw labels directory not found: {labels_dir}")

    samples = load_dataset(images_dir, labels_dir)
    if not samples:
        raise ValueError(
            f"No supported images found in {images_dir}. "
            "Place real side-scan sonar images and matching YOLO labels here "
            "before building the dataset."
        )

    partition = split_dataset(samples, ratios=ratios, seed=seed, stratify=stratify)
    write_partition(
        partition,
        root=root,
        move=move,
        clean=clean,
        label_class_map=label_class_map,
    )
    config_yaml = write_dataset_yaml(root, classes=required_classes or DEFAULT_CLASSES)

    manifest = write_manifest(
        partition,
        root=root,
        move=move,
        seed=seed,
        ratios=ratios or SplitRatios(),
        classes=list(required_classes or DEFAULT_CLASSES),
    )

    logger.info("Dataset built: %s", partition.counts())
    return BuildResult(
        root=root,
        config_yaml=config_yaml,
        manifest=manifest,
        counts=partition.counts(),
    )


def write_partition(
    split: SplitDataset,
    root: Path,
    move: bool = False,
    clean: bool = False,
    label_class_map: dict[int, int] | None = None,
) -> None:
    """Copy (or move) each sample's image and label into its split folder.

    Args:
        split: The partitioned dataset.
        root: Dataset root containing ``raw/{images,labels}``.
        move: If True, move files out of the raw pool; otherwise copy them.
        clean: If True, remove pre-existing files in the split folders first.
        label_class_map: Optional original-id -> compact-id map applied to the
            first token of each label line when copying (see
            :func:`build_dataset`).
    """
    for name in ("train", "val", "test"):
        images_dir, labels_dir = split_dirs(root, name)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            _empty(images_dir)
            _empty(labels_dir)

    samples = [split.train, split.val, split.test]
    for name, split_coll in zip(("train", "val", "test"), samples):
        images_dir, labels_dir = split_dirs(root, name)
        for sample in split_coll:
            target_image = images_dir / sample.image_path.name
            _place(sample.image_path, target_image, move=move)
            if sample.label_path is not None:
                target_label = labels_dir / sample.label_path.name
                if label_class_map:
                    _place_remapped(
                        sample.label_path, target_label, label_class_map, move=move
                    )
                else:
                    _place(sample.label_path, target_label, move=move)
            else:
                logger.warning(
                    "Sample %s has no label file; its annotations are empty.", sample.stem
                )


def _place_remapped(
    source: Path, target: Path, class_map: dict[int, int], move: bool = False
) -> None:
    """Copy a label file, rewriting its leading class ids via ``class_map``.

    Rewrites every non-empty line so the first token becomes ``class_map[old]``.
    This keeps the label ids consistent with the compact vocabulary written to
    ``data.yaml``. Real coordinates are preserved untouched; nothing is invented.
    """
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        try:
            old_id = int(float(tokens[0]))
        except ValueError:
            lines.append(line)
            continue
        new_id = class_map.get(old_id, old_id)
        tokens[0] = str(new_id)
        lines.append(" ".join(tokens))

    payload = "\n".join(lines) + ("\n" if lines else "")
    target.write_text(payload, encoding="utf-8")
    if move:
        source.unlink(missing_ok=True)


def write_dataset_yaml(
    root: Path,
    classes: Sequence[str],
    write_relative_path: bool = True,
) -> Path:
    """Write an Ultralytics ``data.yaml`` describing the split layout.

    Args:
        root: Dataset root containing ``train/``, ``val/`` and ``test/``.
        classes: Class vocabulary, in label-index order.
        write_relative_path: If True, ``path`` is written as the absolute
            directory; ``train`` / ``val`` / ``test`` as relative subdirs.
            Ultralytics accepts both absolute and repo-relative paths.

    Returns:
        Path to the written ``data.yaml``.
    """
    root = Path(root).resolve()
    names = dict(enumerate(classes))
    payload = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(_to_yaml(payload), encoding="utf-8")
    return yaml_path


def write_manifest(
    split: SplitDataset,
    root: Path,
    move: bool,
    seed: int,
    ratios: SplitRatios,
    classes: list[str],
) -> Path:
    """Record exactly which image was assigned to which split (no data invented)."""
    metadata_dir = root / "raw" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    def relative_entries(samples: list[Sample]) -> list[str]:
        return [str(s.image_path.resolve().relative_to(root.resolve())) for s in samples]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "val": ratios.val,
            "test": ratios.test,
        },
        "move": move,
        "classes": classes,
        "splits": {
            "train": relative_entries(split.train),
            "val": relative_entries(split.val),
            "test": relative_entries(split.test),
        },
    }
    path = metadata_dir / "split_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _place(source: Path, target: Path, move: bool) -> None:
    """Copy or move a single real file; ``copy2`` preserves metadata."""
    if source == target:
        return
    if move:
        shutil.move(str(source), str(target))
    else:
        shutil.copy2(str(source), str(target))


def _empty(directory: Path) -> None:
    """Remove all files (not subdirectories) inside a split folder."""
    for child in directory.iterdir():
        if child.is_file():
            child.unlink()


def _to_yaml(payload: dict) -> str:
    """Render a small YAML document without a third-party YAML dependency.

    ``names`` is always written with explicit integer keys so Ultralytics
    reads the index ordering unambiguously.
    """
    lines = [f"path: {payload['path']}", f"train: {payload['train']}", f"val: {payload['val']}", f"test: {payload['test']}"]
    lines.append("names:")
    for key, value in payload["names"].items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"