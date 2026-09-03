"""End-to-end YOLO training pipeline for side-scan sonar marine debris.

The pipeline consumes a **real** pool of sonar images + YOLO annotations in
``backend/data/raw`` and drives it through:

    Real Dataset -> Validation -> Automatic Split -> Preprocessing
        -> YOLO Training -> Validation -> Testing -> Detection
        -> Confidence -> Natural/Artificial Analysis -> Risk Score
        -> JSON/CSV Results

It never generates, simulates, hardcodes, or fabricates any data, labels,
coordinates, or results. Every metric and detection is measured at runtime on
the real images and annotations supplied. The single entry point is the
top-level ``train.py`` script (``python train.py``).
"""

from app.training.config import TrainConfig, load_config, save_config
from app.training.pipeline import run_pipeline

__all__ = [
    "TrainConfig",
    "load_config",
    "save_config",
    "run_pipeline",
]
