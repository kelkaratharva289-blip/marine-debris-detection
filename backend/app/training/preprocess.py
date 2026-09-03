"""Preprocessing stage: apply the real sonar enhancement pipeline.

For every **real** image across the train/val/test splits, this stage runs
the production sonar preprocessing pipeline (grayscale -> denoise -> CLAHE ->
normalise -> letterbox) and writes the enhanced image back into the split
folder, so the model is trained on the **same** representation the serving
pipeline produces at inference time.

Annotations are invariant to this stage: YOLO boxes are stored as normalised
fractions, and letterboxing pads (does not rescale) content while preserving
normalised coordinates, so the original ``.txt`` labels stay valid unmodified.

Nothing is generated here — input images are real sonar scans, and the output
is a real preprocessing of them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from app.ai.preprocessing import PreprocessConfig, preprocess_to_uint8
from app.ai.preprocessing.loader import load_image
from app.ai.dataset.constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_split(
    split_dir: str | Path,
    imgsz: int = 640,
    config: PreprocessConfig | None = None,
) -> dict:
    """Preprocess all real images inside one split folder in place.

    Args:
        split_dir: A ``train``/``val``/``test`` directory containing
            ``images/`` (and sibling labels left untouched).
        imgsz: Letterbox size for the model input.
        config: Optional :class:`PreprocessConfig` overrides.

    Returns:
        A dict ``{"count": int, "failed": list[str]}`` reporting how many
        images were processed and any that could not be (with reasons).
    """
    split = Path(split_dir)
    images_dir = split / "images"
    cfg = config or PreprocessConfig(target_size=imgsz)

    images = _image_files(images_dir)
    processed = 0
    failed: list[str] = []

    for image in images:
        try:
            img = load_image(str(image), color=False)
            enhanced = preprocess_to_uint8(img, config=cfg)
            # Enhanced output is grayscale (single channel). Convert to the
            # PNG/BGR form Ultralytics reads unambiguously.
            if enhanced.ndim == 2:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            target = image.with_suffix(".png")
            written = cv2.imwrite(str(target), enhanced)
            if not written:
                raise RuntimeError("cv2.imwrite returned False")
            if target != image:
                image.unlink(missing_ok=True)
            processed += 1
        except Exception as exc:  # noqa: BLE001 - report, don't fabricate
            failed.append(f"{image}: {exc}")
            logger.error("Preprocessing failed for %s: %s", image, exc)

    logger.info("Preprocessed %d image(s) in %s", processed, split_dir)
    if failed:
        logger.warning("%d image(s) failed preprocessing", len(failed))
    return {"count": processed, "failed": failed}


def preprocess_dataset(
    data_root: str | Path,
    imgsz: int = 640,
    config: PreprocessConfig | None = None,
) -> dict:
    """Preprocess all real images across the ``train``/``val``/``test`` splits."""
    root = Path(data_root)
    summary = {}
    for name in ("train", "val", "test"):
        summary[name] = preprocess_split(root / name, imgsz=imgsz, config=config)
    return summary
