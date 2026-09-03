"""Load and validate YOLO-format annotations for side-scan sonar imagery.

Each annotation is stored in a ``.txt`` file next to its image. Every
non-empty line describes a single bounding box in YOLO's normalized format::

    <class_id> <x_center> <y_center> <width> <height>

``class_id`` is the zero-based index into the class list and the four numeric
fields are fractions of the source image width/height. Extra trailing tokens
(e.g. segmentation polygon points) are tolerated but ignored.

No image content is read or synthesised here — the loader only checks that the
images referenced by a dataset exist and that their label files parse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.ai.dataset.constants import (
    IMAGE_EXTENSIONS,
    LABEL_EXTENSION,
    is_supported_image,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Annotation:
    """A single validated YOLO bounding-box annotation (normalized units)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Sample:
    """An image and its (possibly empty) set of validated annotations."""

    image_path: Path
    label_path: Path | None
    annotations: tuple[Annotation, ...]

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Sorted tuple of classes present, used for stratified grouping."""
        return tuple(sorted({a.class_id for a in self.annotations}))

    @property
    def is_labelled(self) -> bool:
        return len(self.annotations) > 0


class LabelParseError(ValueError):
    """Raised when a YOLO label file cannot be parsed or validated."""

    def __init__(self, path: Path, line_no: int | None, message: str):
        self.path = Path(path)
        self.line_no = line_no
        detail = f"{line_no}" if line_no is not None else "<file>"
        super().__init__(f"{self.path}: line {detail}: {message}")


def parse_yolo_label(
    path: str | Path,
    num_classes: int | None = None,
) -> tuple[Annotation, ...]:
    """Parse a YOLO ``.txt`` annotation file.

    Args:
        path: Path to the label file.
        num_classes: Number of classes; the class vocabulary defaults to the
            canonical marine-debris set. Class ids outside ``[0, num_classes)``
            are rejected.

    Returns:
        A tuple of validated :class:`Annotation` objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        LabelParseError: If any line is malformed or out of range.
    """
    from app.ai.dataset.constants import DEFAULT_CLASSES

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")

    n_classes = num_classes if num_classes is not None else len(DEFAULT_CLASSES)
    annotations: list[Annotation] = []

    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue

        tokens = raw.split()
        if len(tokens) < 5:
            raise LabelParseError(
                p, line_no, f"expected at least 5 values, got {len(tokens)}"
            )

        try:
            class_id = int(float(tokens[0]))
        except ValueError as exc:
            raise LabelParseError(
                p, line_no, f"class id is not an integer: {tokens[0]!r}"
            ) from exc

        if not (0 <= class_id < n_classes):
            raise LabelParseError(
                p, line_no, f"class id {class_id} out of range [0, {n_classes})"
            )

        values: list[float] = []
        for token in tokens[1:5]:
            try:
                value = float(token)
            except ValueError as exc:
                raise LabelParseError(
                    p, line_no, f"not a float: {token!r}"
                ) from exc
            if not _isfinite(value):
                raise LabelParseError(p, line_no, f"non-finite value: {token!r}")
            values.append(value)

        x_center, y_center, width, height = values

        if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0):
            raise LabelParseError(
                p, line_no, "x_center/y_center must be within [0, 1]"
            )
        if width <= 0.0 or height <= 0.0:
            raise LabelParseError(p, line_no, "width/height must be > 0")
        if width > 1.0 or height > 1.0:
            raise LabelParseError(
                p, line_no, "width/height must not exceed 1 (normalized units)"
            )

        annotations.append(
            Annotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return tuple(annotations)


def iter_images(images_dir: str | Path) -> Iterator[Path]:
    """Yield supported image files under ``images_dir`` in sorted order."""
    base = Path(images_dir)
    if not base.exists():
        return
    for path in sorted(base.iterdir()):
        if path.is_file() and is_supported_image(path):
            yield path


def find_label(image_path: str | Path, labels_dir: str | Path) -> Path | None:
    """Resolve the sibling label file for an image, if one exists."""
    candidate = Path(labels_dir) / f"{Path(image_path).stem}{LABEL_EXTENSION}"
    return candidate if candidate.is_file() else None


def load_sample(
    image_path: str | Path,
    labels_dir: str | Path,
    num_classes: int | None = None,
    require_label: bool = True,
) -> Sample:
    """Load a single sample (an image plus its parsed annotations).

    Args:
        image_path: Path to the image.
        labels_dir: Directory containing YOLO label files.
        num_classes: Class count for validation (default: canonical set).
        require_label: If True, raise when no label file exists for the image.
            If False, unlabelled images yield ``Sample(annotations=())``.

    Returns:
        A :class:`Sample`.

    Raises:
        FileNotFoundError: If the image is missing.
        LabelParseError: If the label file exists but is malformed.
    """
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    label = find_label(image, labels_dir)
    annotations: tuple[Annotation, ...] = ()
    if label is not None:
        annotations = parse_yolo_label(label, num_classes=num_classes)
    elif require_label:
        raise FileNotFoundError(
            f"Label file missing for image: {image} "
            f"(expected {labels_dir / (image.stem + LABEL_EXTENSION)})"
        )

    return Sample(image_path=image, label_path=label, annotations=annotations)


def load_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int | None = None,
    require_label: bool = True,
    skip_unlabelled: bool = False,
) -> list[Sample]:
    """Load all samples from a raw ``images/`` + ``labels/`` pair.

    Args:
        images_dir: Directory of source images.
        labels_dir: Directory of YOLO label files.
        num_classes: Class count for validation (default: canonical set).
        require_label: If True, raise when an image has no label file.
        skip_unlabelled: If True, drop unlabelled images instead of raising.

    Returns:
        Loaded samples, sorted by image path.

    Raises:
        LabelParseError: If any label file is malformed.
        FileNotFoundError: If ``require_label`` is True and a label is missing
            (unless ``skip_unlabelled`` is True).
    """
    samples: list[Sample] = []
    skipped = 0
    for image in iter_images(images_dir):
        try:
            sample = load_sample(
                image,
                labels_dir,
                num_classes=num_classes,
                require_label=require_label,
            )
        except FileNotFoundError:
            if skip_unlabelled:
                skipped += 1
                continue
            raise
        samples.append(sample)

    if skipped:
        logger.warning("Skipped %d unlabelled image(s).", skipped)

    return samples


def _isfinite(value: float) -> bool:
    """True when the value is neither NaN nor infinite."""
    return value == value and abs(value) != float("inf")