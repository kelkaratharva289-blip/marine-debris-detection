"""Configuration for the end-to-end training pipeline.

Encapsulates every tunable used across the pipeline: dataset root, split
ratios, YOLO model/epochs/batch/imgsz/device, detection thresholds, and
output directories. Config values can be supplied via command-line flags, an
optional JSON config file (overlaid onto defaults), or environment variables.

No data is invented here: the config only holds *hyper-parameters*. The
dataset (images + labels) must be supplied by the user.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.ai.dataset.constants import (
    DEFAULT_CLASSES,
    default_data_root,
    infer_dataset_classes,
)


@dataclass
class TrainConfig:
    """All configurable hyper-parameters for the training pipeline."""

    # ---- dataset ---------------------------------------------------------
    # Root directory containing raw/images and raw/labels (default backend/data).
    data_root: str = field(default_factory=lambda: str(default_data_root()))
    # The single folder users drop the real dataset into. If provided, it is
    # used directly as the raw pool for validation + splitting.
    raw_images_dir: str | None = None
    raw_labels_dir: str | None = None

    # ---- split -----------------------------------------------------------
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    stratify: bool = True

    # ---- YOLO hyper-parameters --------------------------------------------
    model: str = "yolov8n.pt"          # base model / pre-trained weights
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str | None = None          # None -> auto (GPU if available)
    workers: int = 8
    patience: int = 30                 # early-stopping patience (0 to disable)
    seed: int = 42
    optimizer: str = "auto"
    lr0: float | None = None
    cos_lr: bool = False

    # ---- detection --------------------------------------------------------
    conf: float = 0.25
    iou: float = 0.45

    # ---- output ------------------------------------------------------------
    # Where training artifacts / reports are written.
    output_dir: str = ""               # "" -> <data_root>/training_outputs
    best_model_path: str = "models/marine_debris_yolov8.pt"

    # ---- behaviour ---------------------------------------------------------
    # Copy a new epoch-sized sample before training for a cheap end-to-end
    # smoke run. Requires real input data; it only shrinks how much is used.
    dry_run: bool = False
    verbose: bool = False

    # Private override for the class vocabulary; do not set directly.
    _classes_override: list[str] | None = field(default=None, repr=False)

    # Derived/convenience helpers ------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.data_root).resolve()

    @property
    def images_dir(self) -> Path:
        if self.raw_images_dir:
            return Path(self.raw_images_dir).resolve()
        return self.root / "raw" / "images"

    @property
    def labels_dir(self) -> Path:
        if self.raw_labels_dir:
            return Path(self.raw_labels_dir).resolve()
        return self.root / "raw" / "labels"

    @property
    def out(self) -> Path:
        base = Path(self.output_dir).resolve() if self.output_dir else self.root / "training_outputs"
        return base

    @property
    def classes(self) -> list[str]:
        """Resolve the class vocabulary actually used for training.

        If an explicit override was set it is returned as-is. Otherwise the
        vocabulary is inferred from the real label files in the raw pool (via
        :func:`app.ai.dataset.constants.infer_dataset_classes`), falling back
        to the canonical set only when no inference is possible. The returned
        order is the label-id order used in ``data.yaml``.
        """
        if self._classes_override:
            return list(self._classes_override)
        try:
            inferred = infer_dataset_classes(self.labels_dir)
        except FileNotFoundError:
            inferred = []
        if inferred:
            return [name for _, name in inferred]
        return list(DEFAULT_CLASSES)

    @classes.setter
    def classes(self, value: list[str] | None) -> None:
        """Set an explicit class-vocabulary override (or ``None`` to infer)."""
        self._classes_override = list(value) if value else None

    def split_ratios(self) -> tuple[float, float, float]:
        return self.train_ratio, self.val_ratio, self.test_ratio

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("_classes_override", None)
        data["classes"] = self.classes
        data["num_classes"] = len(self.classes)
        return data


def save_config(config: TrainConfig, path: str | Path) -> Path:
    """Write the resolved config to ``path`` as JSON (useful for auditing)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_config(path: str | Path) -> dict:
    """Read a JSON config dict from ``path`` (values may override defaults)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_config(**overrides) -> TrainConfig:
    """Build a :class:`TrainConfig`, applying ``overrides`` on top of defaults."""
    current = TrainConfig()
    for key, value in overrides.items():
        if value is not None and hasattr(current, key):
            setattr(current, key, value)
    return current
