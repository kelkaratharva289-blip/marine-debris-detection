"""YOLO dataset pipeline for side-scan sonar marine debris detection.

The pipeline consumes a **real** pool of sonar images + YOLO annotations,
validates them, splits them deterministically into train/val/test, and writes
a standard Ultralytics dataset (``data.yaml`` plus split folders). It never
generates images or labels; unlabelled or malformed samples are reported, not
fabricated.

CLI usage from the ``backend`` directory::

    python -m app.ai.dataset inspect
    python -m app.ai.dataset build --ratios 0.7 0.15 0.15 --seed 42
    python -m app.ai.dataset yaml --root ../my_dataset

Public API::

    from app.ai.dataset import load_dataset, split_dataset, build_dataset
"""

from app.ai.dataset.builder import BuildResult, build_dataset, write_dataset_yaml
from app.ai.dataset.constants import (
    DEFAULT_CLASSES,
    IMAGE_EXTENSIONS,
    default_data_root,
)
from app.ai.dataset.loader import (
    Annotation,
    LabelParseError,
    Sample,
    load_dataset,
    load_sample,
    parse_yolo_label,
)
from app.ai.dataset.split import SplitDataset, SplitRatios, split_dataset

__all__ = [
    # loader
    "Annotation",
    "Sample",
    "LabelParseError",
    "parse_yolo_label",
    "load_sample",
    "load_dataset",
    # split
    "SplitRatios",
    "SplitDataset",
    "split_dataset",
    # builder
    "build_dataset",
    "write_dataset_yaml",
    "BuildResult",
    # constants
    "DEFAULT_CLASSES",
    "IMAGE_EXTENSIONS",
    "default_data_root",
]