"""Automatic, deterministic train/val/test splitting.

Reuses the existing :mod:`app.ai.dataset` pipeline for a stratified,
seed-reproducible split. It materialises the three split folders under
``<data_root>/{train,val,test}/{images,labels}`` and writes the canonical
``data.yaml`` consumed by Ultralytics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.ai.dataset.builder import BuildResult, build_dataset
from app.ai.dataset.split import SplitRatios
from app.training.validate_dataset import ValidationReport, validate_dataset

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Outcome of the automatic data split stage."""

    data_root: Path
    data_yaml: Path
    manifest: Path
    counts: dict
    validation: ValidationReport = field(default_factory=ValidationReport)

    def as_dict(self) -> dict:
        return {
            "data_root": str(self.data_root),
            "data_yaml": str(self.data_yaml),
            "manifest": str(self.manifest),
            "counts": dict(self.counts),
            "validation": self.validation.as_dict(),
        }


def prepare_and_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    data_root: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    num_classes: int | None = None,
    classes: Sequence[str] | None = None,
) -> SplitResult:
    """Validate the real pool, then build stratified train/val/test splits.

    Args:
        images_dir: Directory of real sonar images (``raw/images``).
        labels_dir: Directory of real YOLO labels (``raw/labels``).
        data_root: Dataset root that already contains ``raw/{images,labels}``
            and receives the generated ``train/val/test`` folders.
        train_ratio/val_ratio/test_ratio: Split fractions (must be positive).
        seed: Deterministic split seed.
        stratify: Stratify each split by class composition.
        num_classes: Class count for label validation.
        classes: Optional explicit class vocabulary (label-id order) written
            into ``data.yaml``. If ``None``, classes are inferred from the real
            labels so only the classes actually present are trained.

    Returns:
        A :class:`SplitResult` describing the built layout plus the
        validation report from the real data.

    Raises:
        FileNotFoundError: If the raw pool directories are missing.
        ValueError: If the pool is empty or ratios are invalid.
    """
    from app.ai.dataset.constants import infer_dataset_classes

    report = validate_dataset(
        images_dir,
        labels_dir,
        num_classes=num_classes,
        allow_missing_labels=False,
    )

    # Build the compact class vocabulary and an id-remap derived from the real
    # labels. Only classes actually present in the dataset are trained; their
    # label ids are remapped to compact 0..N-1 indices matching ``data.yaml``.
    inferred = infer_dataset_classes(labels_dir)
    class_map: dict[int, int] | None = None
    if inferred:
        resolved: list[str] = (
            list(classes)
            if classes
            else [name for _, name in inferred]
        )
        missing = [name for _, name in inferred if name not in resolved]
        if missing:
            raise ValueError(
                f"Labels use class(es) {missing} that are missing from the "
                f"requested vocabulary {resolved}. Include them or rename the "
                "label ids so every labelled class is represented."
            )
        class_map = {orig_id: resolved.index(name) for orig_id, name in inferred}
        classes = resolved

    result = build_dataset(
        root=Path(data_root),
        ratios=SplitRatios(train_ratio, val_ratio, test_ratio),
        seed=seed,
        move=False,
        clean=True,
        stratify=stratify,
        required_classes=classes,
        label_class_map=class_map,
    )

    logger.info("Split complete: %s", result.counts)
    return SplitResult(
        data_root=result.root,
        data_yaml=result.config_yaml,
        manifest=result.manifest,
        counts=result.counts,
        validation=report,
    )

