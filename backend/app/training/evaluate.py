"""Evaluation: real validation/test metrics and inference timing.

After training, this stage runs the trained model against the held-out
VALIDATION and TEST splits and measures genuine metrics from actual
inference:

* Precision, Recall, mAP50, mAP50-95 (and per-class precision/recall/AP)
* Per-image inference time (ms) and FPS, measured on real images

The metrics are read from Ultralytics' ``DetMetrics`` / ``Metric`` results,
which are themselves computed from the real detections and ground-truth
labels. No number here is estimated or fabricated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ai.preprocessing import PreprocessConfig
from app.ai.preprocessing.loader import load_image
from app.ai.preprocessing.pipeline import preprocess_to_uint8
from app.ai.inference import run_inference
from app.training.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrixData:
    """A real confusion matrix plus the class names and a saved plot path.

    Rows are ground-truth classes, columns are predicted classes. For object
    detection the matrix is ``(nc+1) x (nc+1)`` where the extra last row/column
    is the background / unknown class (contains false negatives along the last
    row and false positives along the last column). Raw counts are stored so
    consumers can compute rates themselves; ``normalized`` and the saved PNG
    are derived from these real counts.
    """

    matrix: np.ndarray              # raw counts, shape (nc+1, nc+1)
    names: list[str]                # class names in id order (no background)
    plot_path: str | None           # saved confusion-matrix PNG (or None)

    @property
    def n_classes(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict:
        return {
            "n_classes": self.n_classes,
            "names": list(self.names),
            "matrix_raw": matrix_to_lists(self.matrix),
            "plot_path": self.plot_path,
        }


def matrix_to_lists(matrix: np.ndarray) -> list[list[int]]:
    """Convert an integer matrix to a JSON-friendly list-of-lists."""
    return [[int(v) for v in row] for row in matrix]


@dataclass
class EvalSplit:
    """Metrics for one split (validation or test) from real inference."""

    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    per_class: dict = field(default_factory=dict)
    n_images: int = 0
    n_gt_boxes: int = 0

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "images": self.n_images,
            "ground_truth_boxes": self.n_gt_boxes,
            "per_class": self.per_class,
        }


@dataclass
class TimingMetrics:
    """Measured per-image inference latency on real images."""

    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "fps": self.fps,
        }


def _count_images(images_dir: Path) -> int:
    from app.ai.dataset.constants import IMAGE_EXTENSIONS
    return sum(1 for p in images_dir.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _count_gt_boxes(images_dir: Path, labels_dir: Path) -> int:
    from app.ai.dataset.constants import IMAGE_EXTENSIONS
    total = 0
    for ext in IMAGE_EXTENSIONS:
        for image in images_dir.glob(f"*{ext}"):
            label = labels_dir / f"{image.stem}.txt"
            if label.is_file():
                total += sum(1 for line in label.read_text(encoding="utf-8").splitlines()
                             if line.strip())
    return total


def _collections_dir(root: Path) -> Path:
    """Compute Ultralytics' run/val collection dir from a data directory."""
    return root.parent.parent / "runs" / "detect" / "val"


def extract_metrics(det_metrics) -> EvalSplit:
    """Pull P/R/mAP from an Ultralytics results object (any version)."""
    # Normalise access to the outer morphological metrics object.
    box = getattr(det_metrics, "box", det_metrics)

    def _get(name, default=None):
        try:
            v = getattr(box, name)
            if v is None:
                return default
            return float(v)
        except (AttributeError, TypeError, ValueError):
            return default

    # NOTE: do not coalesce with ``or`` — a legitimate measured value of 0.0
    # is falsy and must be preserved, not replaced by None.
    precision = _get("mp")
    recall = _get("mr")
    map50 = _get("map50")
    map50_95 = _get("map")

    per_class = {}
    try:
        class_idx = list(box.ap_class_index)
        aps = [float(x) for x in box.ap] if box.ap is not None else []
        for cid, ap in zip(class_idx, aps):
            per_class[str(int(cid))] = round(ap, 4)
    except Exception:  # noqa: BLE001 - per-class is best-effort
        per_class = {}

    return EvalSplit(
        precision=round(precision, 4) if precision is not None else None,
        recall=round(recall, 4) if recall is not None else None,
        map50=round(map50, 4) if map50 is not None else None,
        map50_95=round(map50_95, 4) if map50_95 is not None else None,
        per_class=per_class,
    )


def evaluate_validation(config: TrainConfig, model=None) -> EvalSplit:
    """Evaluate the trained model on the VALIDATION split.

    Uses Ultralytics' ``model.val(data=...)`` which computes AP/AR from real
    predictions vs ground truth.
    """
    from ultralytics import YOLO

    model = model or YOLO(config.best_model_path)
    images_dir = config.root / "val" / "images"
    labels_dir = config.root / "val" / "labels"

    det_metrics = model.val(
        data=str(config.root / "data.yaml"),
        split="val",
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        conf=config.conf,
        iou=config.iou,
        project=str(config.out / "yolo_run"),
        name="val",
        exist_ok=True,
        verbose=False,
    )

    result = extract_metrics(det_metrics)
    result.n_images = _count_images(images_dir)
    result.n_gt_boxes = _count_gt_boxes(images_dir, labels_dir)
    return result


def evaluate_test(
    config: TrainConfig,
    model=None,
    batches: int = 1,
) -> tuple[EvalSplit, TimingMetrics]:
    """Evaluate the trained model on the TEST split with real timing.

    Returns ``(metrics, timing)`` where ``timing`` is the measured per-image
    inference latency across the test images (model forward pass + NMS on
    preprocessed canvases).
    """
    from ultralytics import YOLO

    model = model or YOLO(config.best_model_path)
    images_dir = config.root / "test" / "images"
    labels_dir = config.root / "test" / "labels"

    det_metrics = model.val(
        data=str(config.root / "data.yaml"),
        split="test",
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        conf=config.conf,
        iou=config.iou,
        project=str(config.out / "yolo_run"),
        name="test",
        exist_ok=True,
        verbose=False,
    )

    metrics = extract_metrics(det_metrics)
    metrics.n_images = _count_images(images_dir)
    metrics.n_gt_boxes = _count_gt_boxes(images_dir, labels_dir)

    timing = measure_timing(model, images_dir, config, batches=batches)
    return metrics, timing


def measure_timing(model, images_dir: Path, config: TrainConfig, batches: int = 1) -> TimingMetrics:
    """Measure real per-image inference time on the images in ``images_dir``."""
    from app.ai.dataset.constants import IMAGE_EXTENSIONS
    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        return TimingMetrics(0, 0.0, 0.0, 0.0, 0.0)

    times: list[float] = []
    # Warmup once.
    if images:
        warm = load_image(str(images[0]), color=False)
        canvas = preprocess_to_uint8(
            warm, config=PreprocessConfig(target_size=config.imgsz)
        )
        run_inference(model, canvas, conf=config.conf, iou=config.iou,
                      classes=config.classes, device=config.device)

    for _ in range(max(int(batches), 1)):
        for image in images:
            img = load_image(str(image), color=False)
            canvas = preprocess_to_uint8(
                img, config=PreprocessConfig(target_size=config.imgsz)
            )
            t0 = time.perf_counter()
            run_inference(model, canvas, conf=config.conf, iou=config.iou,
                          classes=config.classes, device=config.device)
            times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    mean = float(arr.mean())
    return TimingMetrics(
        samples=len(arr),
        mean_ms=round(mean, 3),
        median_ms=round(float(np.median(arr)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        fps=round(1000.0 / mean, 2) if mean > 0 else 0.0,
    )


def extract_confusion_matrix(det_metrics, names: list[str], split: str,
                             config: TrainConfig) -> ConfusionMatrixData:
    """Build a :class:`ConfusionMatrixData` from Ultralytics' real results.

    ``det_metrics.confusion_matrix`` is a :class:`ConfusionMatrix` holding the
    actual match counts produced while scoring the split: rows = ground truth,
    columns = predictions, plus a final background row/column. These raw counts
    (never estimated) are stored and rendered to a PNG.

    Args:
        det_metrics: Ultralytics ``DetMetrics`` (the object returned by
            ``model.val()``).
        names: The class names (id order) that the model was trained on.
        split: Label used in the plot title / filename ("val" or "test").
        config: Resolved config (used to locate the output directory).

    Returns:
        :class:`ConfusionMatrixData` with the raw integer matrix, class names,
        and the saved normalized plot path.
    """
    plots_dir = _evaluation_dir(config)
    cm_obj = getattr(det_metrics, "confusion_matrix", None)
    matrix = None
    if cm_obj is not None:
        matrix = getattr(cm_obj, "matrix", None)

    if matrix is None or not getattr(matrix, "size", 0):
        logger.warning("Confusion matrix unavailable for split %s", split)
        return ConfusionMatrixData(matrix=np.zeros((len(names) + 1, len(names) + 1)),
                                   names=list(names), plot_path=None)

    matrix = np.asarray(matrix, dtype=np.int64)
    nc = len(names) + 1
    if matrix.shape[0] != nc or matrix.shape[1] != nc:
        # Guard against a unpredictably sized matrix (e.g. an empty background).
        logger.warning("Unexpected confusion-matrix shape %s; skipping plot",
                       matrix.shape)
        return ConfusionMatrixData(matrix=matrix, names=list(names), plot_path=None)

    cm_data = ConfusionMatrixData(matrix=matrix, names=list(names), plot_path=None)
    try:
        path = _plot_confusion_matrix(cm_data, split, plots_dir, normalize=True)
        cm_data.plot_path = str(path)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.warning("Failed to render confusion matrix for %s: %s", split, exc)
    return cm_data


def _evaluation_dir(config: TrainConfig) -> Path:
    """Directory that holds evaluation artifacts (under the config output)."""
    out = (config.out if isinstance(config, TrainConfig) else Path(config))
    directory = out / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _plot_confusion_matrix(cm: ConfusionMatrixData, split: str,
                           plots_dir: Path, normalize: bool = True) -> Path:
    """Render the confusion matrix to a PNG from the real raw counts.

    Normalises each column (prediction) to a fraction of its total so the
    colouring reflects rates while the raw labelling option stays available.
    The last row/column is the background class (FN along the last row, FP along
    the last column), matching Ultralytics' convention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = cm.matrix.astype(np.float64)
    n = cm.n_classes
    labels = [*cm.names, "background"]

    if normalize:
        totals = matrix.sum(axis=0, keepdims=True)
        array = np.divide(matrix, totals, out=np.ones_like(matrix),
                          where=totals != 0)
    else:
        array = matrix

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(array, cmap="Blues", vmin=0.0, vmax=max(1.0, array.max()))
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix - {split.upper()} split (real detections)")

    thresh = (array.max() + 1.0) / 2.0 if not normalize else 0.5
    for i in range(n + 1):
        for j in range(n + 1):
            val = array[i, j]
            text = f"{int(cm.matrix[i, j])}"
            if normalize:
                text += f"\n({val:.2f})"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = plots_dir / f"confusion_matrix_{split}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_detection_visualizations(
    images_dir: Path,
    detections_by_image: dict[str, list[dict]],
    names: list[str],
    save_dir: Path,
    conf: float = 0.25,
) -> list[str]:
    """Overlay real detections on the actual images and save PNG visualizations.

    For each real image with at least one detection above ``conf``, boxes and
    class labels are drawn on the original scanned pixels (nothing is invented)
    and the result is written under ``save_dir`` as ``<stem>_det.png``.

    Args:
        images_dir: Directory containing the real source images.
        detections_by_image: Mapping of image filename -> list of detection
            dicts (each with ``class_id``, ``bbox_x/y/w/h`` fractions, and
            ``confidence``).
        names: Class names in id order (used for box labels / colours).
        save_dir: Where to write the annotated PNGs.
        conf: Confidence threshold; detections below it are dropped.

    Returns:
        Sorted list of absolute paths to the generated visualizations.
    """
    import cv2

    from app.ai.dataset.constants import IMAGE_EXTENSIONS

    save_dir.mkdir(parents=True, exist_ok=True)
    colors = _class_colors(len(names))
    written: list[str] = []

    for image_path in sorted(p for p in Path(images_dir).iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        dets = [d for d in detections_by_image.get(image_path.name, [])
                if float(d.get("confidence", 0.0)) >= conf]
        display = cv2.imread(str(image_path))
        if display is None:
            logger.warning("Could not read image for visualization: %s", image_path)
            continue
        h, w = display.shape[:2]
        for d in dets:
            x1 = int(float(d["bbox_x"]) * w)
            y1 = int(float(d["bbox_y"]) * h)
            x2 = int((float(d["bbox_x"]) + float(d["bbox_width"])) * w)
            y2 = int((float(d["bbox_y"]) + float(d["bbox_height"])) * h)
            cid = int(d["class_id"])
            if cid < len(names):
                label = f"{names[cid]} {float(d['confidence']):.2f}"
            else:
                label = f"id{cid} {float(d['confidence']):.2f}"
            color = colors[cid % len(colors)]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out_path = save_dir / f"{image_path.stem}_det.png"
        cv2.imwrite(str(out_path), display)
        written.append(str(out_path))

    return sorted(written)


def _class_colors(n: int) -> list[tuple[int, int, int]]:
    """Deterministic BGR colour palette for up to ``n`` classes."""
    import cv2
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        # Golden-ratio hue stepping produces well separated colours. We build a
        # 1x1 HSV->BGR patch and round each channel to a valid uint8.
        hue = int((i * 137) % 180)
        patch = np.zeros((1, 1, 3), dtype=np.uint8)
        patch[0, 0, 0] = hue
        patch[0, 0, 1] = 255
        patch[0, 0, 2] = 255
        rgb = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    if not palette:
        palette = [(0, 255, 0)]
    return palette

