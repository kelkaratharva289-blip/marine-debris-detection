"""Constants and path helpers for the YOLO sonar dataset pipeline."""

from __future__ import annotations

from pathlib import Path

from app.ai.inference import MARINE_CLASSES

# Canonical marine-debris class vocabulary (used to name label ids). This is
# the *name lookup* for whatever ids appear in the dataset; it is NOT the set
# of classes actually trained. The classes used for training are inferred from
# the real label files via :func:`infer_dataset_classes`.
DEFAULT_CLASSES: tuple[str, ...] = tuple(MARINE_CLASSES)


def infer_dataset_classes(labels_dir: str | Path) -> list[tuple[int, str]]:
    """Infer the classes actually present in a real YOLO label folder.

    Scans every ``.txt`` label file, collects the unique class ids that
    appear in real annotations, and maps each id to its canonical name. Only
    classes backed by at least one real label are returned — vocabulary is
    driven entirely by the dataset, never the full canonical set.

    Args:
        labels_dir: Directory containing the YOLO ``.txt`` label files.

    Returns:
        A sorted list of ``(class_id, class_name)`` pairs present in the data.
        Empty if there are no labels.

    Raises:
        FileNotFoundError: If ``labels_dir`` does not exist.
    """
    from app.ai.dataset.loader import parse_yolo_label

    lbl_dir = Path(labels_dir)
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {lbl_dir}")

    present: dict[int, int] = {}
    for label_file in sorted(lbl_dir.glob(f"*{LABEL_EXTENSION}")):
        try:
            for ann in parse_yolo_label(label_file):
                present[ann.class_id] = present.get(ann.class_id, 0) + 1
        except ValueError:
            # By default the label id must be a valid marine-debris id; use the
            # canonical set for range checking. Malformed labels are reported by
            # the validation stage, so skip here rather than fail the inference.
            continue

    names = DEFAULT_CLASSES
    return [(cid, names[cid]) for cid in sorted(present) if cid < len(names)]


# Extensions accepted as side-scan sonar imagery.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)

# A label lives in a sibling ``.txt`` with the same base name as its image.
LABEL_EXTENSION = ".txt"


def default_data_root() -> Path:
    """Absolute default dataset root (``backend/data``)."""
    return (Path(__file__).resolve().parents[3] / "data").resolve()


def raw_subdirs(root: Path | None = None) -> tuple[Path, Path, Path]:
    """Return ``(images_dir, labels_dir, metadata_dir)`` for the raw pool."""
    root = Path(root) if root is not None else default_data_root()
    return root / "raw" / "images", root / "raw" / "labels", root / "raw" / "metadata"


def split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for a named split folder."""
    base = Path(root) / split
    return base / "images", base / "labels"


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS