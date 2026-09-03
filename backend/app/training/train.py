"""YOLO training on real sonar data.

Trains an Ultralytics YOLO model on the preprocessed split folders generated
by the split + preprocess stages, using only **real** images and labels. After
training, the best checkpoint (``best.pt``) is copied to the configured
``best_model_path`` so the FastAPI inference service and edge export can load
it directly. Training progress is streamed via Ultralytics.

No weights are fabricated: the model learns from the real data supplied. A
``dry_run`` mode runs a single epoch on real data for a quick end-to-end
smoke test, and emits no placeholder metrics.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.training.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainOutcome:
    """Result of a training run (real Ultralytics metrics, not estimates)."""

    run_dir: Path
    best_pt: Path          # Ultralytics run/best.pt
    deployed_pt: Path      # copy placed at best_model_path
    args: dict
    metrics: dict = field(default_factory=dict)


def _resolve_device(device: str | None, model: str) -> str | None:
    """Reject an explicit CUDA device when CUDA is unavailable.

    Returns the device to pass to Ultralytics. ``None`` lets Ultralytics pick
    automatically (GPU if available, else CPU).
    """
    if not device:
        return None
    low = device.strip().lower()
    if low.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Requested device '{device}' but CUDA is unavailable. "
                f"Use device='cpu' or omit --device. ({exc})"
            ) from exc
    return device


def train_yolo(config: TrainConfig) -> TrainOutcome:
    """Run Ultralytics YOLO training on real data and deploy ``best.pt``.

    Args:
        config: Resolved :class:`TrainConfig` with split folders ready.

    Returns:
        A :class:`TrainOutcome` recording the run directory, the Ultralytics
        best checkpoint, its deployed copy, and the validated metrics.
    """
    from ultralytics import YOLO

    device = _resolve_device(config.device, config.model)

    data_yaml = config.root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Fresh run directory under outputs, stamped and cleaned so we always
    # rebuild from real data rather than caching a stale run.
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "yolo_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=1 if config.dry_run else config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        device=device,
        workers=config.workers,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        seed=config.seed,
        verbose=config.verbose,
    )
    if config.patience and config.patience > 0 and not config.dry_run:
        train_kwargs["patience"] = config.patience
    if config.optimizer:
        train_kwargs["optimizer"] = config.optimizer
    if config.lr0 is not None and config.lr0 > 0:
        train_kwargs["lr0"] = config.lr0
    if config.cos_lr:
        train_kwargs["cos_lr"] = True

    results = model.train(**train_kwargs)

    # Locate Ultralytics' own best.pt (the checkpoint with best val metrics).
    candidate = run_dir / "train" / "weights" / "best.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Training finished but best.pt was not produced at {candidate}. "
            "Inspect the training output."
        )

    deployed = Path(config.best_model_path)
    deployed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(candidate), str(deployed))
    # Persistent alias so the best trained model is saved as best.pt too.
    best_alias = deployed.parent / "best.pt"
    shutil.copy2(str(candidate), str(best_alias))
    logger.info(
        "Deployed best model to %s (alias %s)",
        deployed,
        best_alias,
    )

    return TrainOutcome(
        run_dir=candidate.parent,
        best_pt=candidate,
        deployed_pt=deployed,
        args=dict(train_kwargs),
    )

