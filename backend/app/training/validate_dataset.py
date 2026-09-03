"""Dataset validation for the training pipeline.

Consumes the **real** pool of sonar images + YOLO annotations and validates
each one before training:

* **Images** — the file must exist and be *decodable* as an image (a real
  ``cv2.imread`` attempt, not just a filename extension check). Corrupt or
  undecodable files are reported and can be excluded, never synthesised.
* **Labels** — every YOLO ``.txt`` is parsed and range-checked using the
  existing :mod:`app.ai.dataset` loader. Missing or malformed labels are
  reported with a clear, actionable message.

The module only ever reads and classifies real data; it produces a
:class:`ValidationReport` that the pipeline can act on. Nothing is invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2

from app.ai.dataset.constants import is_supported_image
from app.ai.dataset.loader import LabelParseError, load_sample

logger = logging.getLogger(__name__)


@dataclass
class ImageIssue:
    """A problem found with a single real image file."""

    image: str
    reason: str


@dataclass
class LabelIssue:
    """A problem found with a single real label file (or its absence)."""

    image: str
    label: str | None
    reason: str


@dataclass
class ValidationReport:
    """Aggregate result of validating the real dataset pool."""

    images_total: int = 0
    images_valid: int = 0
    images_invalid: int = 0
    labels_total: int = 0
    boxes_total: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)
    image_issues: list[ImageIssue] = field(default_factory=list)
    label_issues: list[LabelIssue] = field(default_factory=list)
    valid_image_paths: list[str] = field(default_factory=list)

    @property
    def missing_labels(self) -> list[LabelIssue]:
        return [
            i for i in self.label_issues if i.label is None or i.reason.startswith("missing")
        ]

    @property
    def corrupted_labels(self) -> list[LabelIssue]:
        return [
            i for i in self.label_issues if not (
                i.label is None or i.reason.startswith("missing")
            )
        ]

    def as_dict(self) -> dict:
        return {
            "images_total": self.images_total,
            "images_valid": self.images_valid,
            "images_invalid": self.images_invalid,
            "labels_ok": self.labels_total,
            "boxes_total": self.boxes_total,
            "class_counts": {str(k): v for k, v in sorted(self.class_counts.items())},
            "missing_labels": len(self.missing_labels),
            "corrupted_labels": len(self.corrupted_labels),
            "image_issues": [i.__dict__ for i in self.image_issues],
            "label_issues": [i.__dict__ for i in self.label_issues],
        }


def _decodable(image_path: Path) -> bool:
    """Return True only when ``image_path`` decodes as a real image."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        return img is not None and img.size > 0
    except Exception:  # noqa: BLE001 - any decode failure is reported
        return False


def validate_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int | None = None,
    require_all_labels: bool = True,
    allow_missing_labels: bool = False,
) -> ValidationReport:
    """Validate every image + label in the raw pool.

    Args:
        images_dir: Directory of real sonar images.
        labels_dir: Directory of YOLO-format ``.txt`` labels.
        num_classes: Number of classes to range-check label ids against
            (default: the canonical marine-debris class set).
        require_all_labels: If True, an image with no label file is recorded
            as a ``label_issue``. If False, only *present-but-corrupt* labels
            are flagged and unlabelled images are allowed through.
        allow_missing_labels: If False (default), a missing label is treated
            as a hard error (raises). Set True to tolerate and report it.

    Returns:
        A :class:`ValidationReport` summarising valid/invalid images and
        labels plus any issues found.

    Raises:
        FileNotFoundError: If the images/labels directory do not exist.
        LabelParseError: If a label is corrupt AND ``allow_missing_labels``
            is False (corrupt labels always raise; callers may set the flag
            to downgrade missing, not corrupt, labels).
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {lbl_dir}")

    report = ValidationReport()

    image_files = sorted(
        p for p in img_dir.iterdir() if p.is_file() and is_supported_image(p)
    )
    report.images_total = len(image_files)

    for image in image_files:
        # 1. Image must be a real, decodeable image.
        if not _decodable(image):
            report.images_invalid += 1
            report.image_issues.append(
                ImageIssue(str(image), "image cannot be decoded (corrupt/unsupported)")
            )
            continue

        report.images_valid += 1
        report.valid_image_paths.append(str(image))

        # 2. Label must exist and parse cleanly.
        label_path = lbl_dir / f"{image.stem}.txt"
        if not label_path.exists():
            if require_all_labels or not allow_missing_labels:
                if not allow_missing_labels:
                    raise FileNotFoundError(
                        f"Label file missing for image: {image} "
                        f"(expected {label_path}). Provide real YOLO labels "
                        "or set allow_missing_labels to tolerate/report them."
                    )
            report.label_issues.append(
                LabelIssue(str(image), None, "missing label file")
            )
            continue

        try:
            sample = load_sample(
                image, lbl_dir, num_classes=num_classes, require_label=True
            )
        except LabelParseError as exc:
            report.label_issues.append(
                LabelIssue(str(image), str(label_path), f"corrupt/malformed: {exc}")
            )
            # Continue to find all corrupt labels, then raise collectively below.
            continue

        for ann in sample.annotations:
            report.boxes_total += 1
            report.class_counts[ann.class_id] = report.class_counts.get(ann.class_id, 0) + 1
        if sample.annotations:
            report.labels_total += 1

    if report.corrupted_labels:
        first = report.corrupted_labels[0]
        raise LabelParseError(
            first.image, None,
            f"{len(report.corrupted_labels)} corrupt label(s) found. "
            f"First: {first.reason}"
        )

    return report


def summarize(report: ValidationReport) -> str:
    """Render a concise human-readable summary of a validation report."""
    lines = [
        f"Images: {report.images_total} "
        f"(valid {report.images_valid}, invalid {report.images_invalid})",
        f"Labeled images: {report.labels_total}  |  boxes: {report.boxes_total}",
        f"Missing labels: {len(report.missing_labels)}  "
        f"|  corrupt labels: {len(report.corrupted_labels)}",
    ]
    if report.class_counts:
        lines.append(
            "Class distribution: "
            + ", ".join(
                f"id {cid}={count}" for cid, count in sorted(report.class_counts.items())
            )
        )
    if report.image_issues:
        lines.append("Invalid images:")
        lines += [f"  - {issue.image}: {issue.reason}" for issue in report.image_issues[:20]]
    if report.label_issues:
        lines.append("Label issues:")
        lines += [
            f"  - {issue.image}: {issue.reason}" for issue in report.label_issues[:20]
        ]
    return "\n".join(lines)
