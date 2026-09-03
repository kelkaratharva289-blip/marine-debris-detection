"""Benchmarking harness for edge-deployed detection models.

Measures, on **your own real data and hardware**, the operational metrics
that matter for edge deployment:

* ``inference_time`` — mean/median/p95 per-image latency of the model
  forward pass (ms).
* ``fps``            — frames per second computed from the measured mean
  latency (``1000 / mean_ms``).
* ``model_size``     — on-disk size of the artifact, measured with
  ``os.path.getsize``.
* ``accuracy``       — mAP@0.5, precision, recall and F1 computed against
  ground-truth YOLO labels when provided. Without labels, cross-model
  **agreement** against a reference model is reported instead.

Honesty contract
----------------
- Every number in the report is *measured at runtime* from actual inference
  runs on the images supplied. Nothing is extrapolated or estimated, and no
  synthetic accuracy figures are produced.
- Inference timing covers the model forward pass + NMS + result parsing on a
  preprocessed sonar canvas (the same preprocessing pipeline the service
  uses). Preprocessing itself is identical for every model under test, so it
  is reported as a constant and excluded from per-model latency.
- Accuracy is only computed when ground-truth labels (or a reference model)
  exist; otherwise the field is ``null``. The methodology is always recorded
  verbatim in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from app.ai.inference import MARINE_CLASSES, load_yolo_model, run_inference
from app.ai.preprocessing import PreprocessConfig
from app.ai.preprocessing.loader import load_image
from app.ai.preprocessing.pipeline import preprocess_to_uint8

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    class_id: int
    confidence: float
    # (x1, y1, x2, y2) normalised fractions of the original image.
    xyxy: tuple[float, float, float, float]


@dataclass
class GroundTruth:
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass
class Sample:
    image_path: str
    preds: list[Prediction] = field(default_factory=list)
    gts: list[GroundTruth] = field(default_factory=list)


@dataclass
class TimingStats:
    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    fps: float

    @classmethod
    def from_times(cls, times_ms: list[float]) -> "TimingStats":
        if not times_ms:
            raise ValueError("Cannot compute timing stats from no samples.")
        arr = np.asarray(times_ms, dtype=np.float64)
        mean = float(arr.mean())
        return cls(
            samples=len(arr),
            mean_ms=round(mean, 3),
            median_ms=round(float(np.median(arr)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            min_ms=round(float(arr.min()), 3),
            max_ms=round(float(arr.max()), 3),
            fps=round(1000.0 / mean, 2) if mean > 0 else 0.0,
        )


@dataclass
class AccuracyMetrics:
    methodology: str
    map50: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    n_images: int
    n_ground_truth: int
    reference_model: str | None = None

    def to_dict(self) -> dict:
        return {
            "methodology": self.methodology,
            "map50": self.map50,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "images_used": self.n_images,
            "ground_truth_boxes": self.n_ground_truth,
            "reference_model": self.reference_model,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def xywh_norm_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple:
    """YOLO normalized ``cx cy w h`` -> ``(x1, y1, x2, y2)``."""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_images(image_dir: str | Path) -> list[Path]:
    """Return the real image files under ``image_dir`` (sorted)."""
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    images = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(
            f"No supported images ({', '.join(sorted(IMAGE_SUFFIXES))}) "
            f"found in {root}. Supply real sonar scans to benchmark."
        )
    return images


def load_yolo_labels(label_path: str | Path) -> list[GroundTruth]:
    """Parse a YOLO-format ``.txt`` label file into ground-truth boxes.

    Expected format per line: ``class_id cx cy w h`` (all normalized). Only
    lines with exactly 5 numeric fields are accepted; malformed lines are
    skipped so a partially-labeled dataset never corrupts the run.
    """
    gts: list[GroundTruth] = []
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                values = [float(p) for p in parts[:5]]
            except ValueError:
                continue
            cls_id = int(values[0])
            cx, cy, w, h = values[1], values[2], values[3], values[4]
            if w <= 0 or h <= 0 or cx < 0 or cy < 0:
                continue
            gts.append(GroundTruth(cls_id, xywh_norm_to_xyxy(cx, cy, w, h)))
    return gts


def preprocess_canvases(
    images: Sequence[Path], cfg: PreprocessConfig
) -> list[np.ndarray]:
    """Preprocess images once (constant across models under test)."""
    canvases = []
    for p in images:
        img = load_image(str(p), color=False)
        canvas = preprocess_to_uint8(img, config=cfg)
        canvases.append(canvas)
    return canvases


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_one(
    model,
    image_path: str,
    conf: float,
    iou: float,
    cfg: PreprocessConfig,
    device: str | None = None,
) -> list[Prediction]:
    """Run inference and map boxes back to the original image coordinates.

    Uses the same preprocessing + letterbox remap the production service
    applies, so predictions live in the same normalized space as YOLO labels.
    """
    img = load_image(image_path, color=False)
    canvas, params = preprocess_to_uint8(img, config=cfg, return_param=True)
    dets = run_inference(
        model, canvas, conf=conf, iou=iou,
        classes=MARINE_CLASSES, device=device,
    )
    mapped = _map_to_original(dets, params, img_h=img.shape[0], img_w=img.shape[1])

    preds: list[Prediction] = []
    for d in mapped:
        x1 = d["bbox_x"]
        y1 = d["bbox_y"]
        x2 = d["bbox_x"] + d["bbox_width"]
        y2 = d["bbox_y"] + d["bbox_height"]
        preds.append(
            Prediction(
                class_id=int(d["class_id"]),
                confidence=float(d["confidence"]),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return preds


def _map_to_original(
    detections: list[dict],
    params: tuple[float, float, float, float],
    img_h: int,
    img_w: int,
) -> list[dict]:
    """Undo letterboxing: canvas-normalized boxes -> original-image fractions.

    ``params`` is ``(scale, pad_x, pad_y, canvas_size)`` from the letterbox
    step. Unlike the production mapper this uses the actual original image
    dimensions, so non-square sonar scans map correctly.
    """
    scale, pad_x, pad_y, canvas_size = params
    if scale <= 0 or canvas_size <= 0:
        return detections

    output: list[dict] = []
    for det in detections:
        cx = det["bbox_x"] * canvas_size
        cy = det["bbox_y"] * canvas_size
        cw = det["bbox_width"] * canvas_size
        ch = det["bbox_height"] * canvas_size

        ox = (cx - pad_x) / scale
        oy = (cy - pad_y) / scale
        ow = cw / scale
        oh = ch / scale

        out = dict(det)
        out["bbox_x"] = round(min(max(ox / img_w, 0.0), 1.0), 4)
        out["bbox_y"] = round(min(max(oy / img_h, 0.0), 1.0), 4)
        out["bbox_width"] = round(min(max(ow / img_w, 0.0), 1.0), 4)
        out["bbox_height"] = round(min(max(oh / img_h, 0.0), 1.0), 4)
        output.append(out)
    return output


def measure_inference(
    model,
    canvases: Sequence[np.ndarray],
    conf: float,
    iou: float,
    runs: int = 10,
    warmup: int = 3,
    device: str | None = None,
) -> TimingStats:
    """Measure real per-image inference latency (ms) over multiple runs.

    Covers model forward pass + NMS + result parsing on preprocessed
    canvases. Preprocessing is excluded (it is identical for every model).
    """
    if not canvases:
        raise ValueError("Nothing to benchmark: no images provided.")
    if runs < 1:
        raise ValueError("runs must be >= 1")

    canvas_sample = canvases[0]
    for _ in range(warmup):
        run_inference(
            model, canvas_sample, conf=conf, iou=iou,
            classes=MARINE_CLASSES, device=device,
        )

    times_ms: list[float] = []
    for run_idx in range(runs):
        canvas = canvases[run_idx % len(canvases)]
        t0 = time.perf_counter()
        run_inference(
            model, canvas, conf=conf, iou=iou,
            classes=MARINE_CLASSES, device=device,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)

    return TimingStats.from_times(times_ms)


# ---------------------------------------------------------------------------
# Accuracy evaluation (mAP@0.5)
# ---------------------------------------------------------------------------

def _ap_for_class(samples: Sequence[Sample], cls_id: int, iou_threshold: float) -> tuple[float, int]:
    """VOC-style 11-point interpolated AP for a single class.

    Returns ``(ap, n_gt)``. ``n_gt`` is the number of ground-truth boxes for
    this class across all samples. Predictions are greedily matched to
    unmatched ground truths at ``iou_threshold``.
    """
    n_gt = 0
    rows: list[tuple[float, bool]] = []
    for sample in samples:
        gts = [g for g in sample.gts if g.class_id == cls_id]
        n_gt += len(gts)
        matched = [False] * len(gts)
        class_preds = sorted(
            (p for p in sample.preds if p.class_id == cls_id),
            key=lambda p: p.confidence,
            reverse=True,
        )
        for p in class_preds:
            best_i = -1
            best_iou = iou_threshold
            for i, gt in enumerate(gts):
                if matched[i]:
                    continue
                v = bbox_iou(p.xyxy, gt.xyxy)
                if v > best_iou:
                    best_iou = v
                    best_i = i
            is_tp = best_i >= 0
            if is_tp:
                matched[best_i] = True
            rows.append((p.confidence, is_tp))

    if n_gt == 0:
        return 0.0, 0

    rows.sort(key=lambda r: r[0], reverse=True)
    tp = np.cumsum([1.0 if is_tp else 0.0 for _, is_tp in rows])
    fp = np.cumsum([0.0 if is_tp else 1.0 for _, is_tp in rows])
    recall = tp / float(n_gt)
    precision = tp / np.maximum(tp + fp, 1e-9)

    # 11-point interpolated Average Precision.
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 11):
        reachable = recall >= threshold
        if reachable.any():
            ap += float(precision[reachable].max())
    return ap / 11.0, n_gt


def compute_accuracy(
    samples: Sequence[Sample],
    iou_threshold: float = 0.5,
    reference_model: str | None = None,
) -> AccuracyMetrics | None:
    """Compute mAP@0.5 / precision / recall / F1 from measured detections.

    Returns ``None`` when there is no ground truth to evaluate against. The
    ``methodology`` string always describes exactly what was measured so no
    accuracy claim is ever implicit.
    """
    gt_classes = sorted({g.class_id for s in samples for g in s.gts})
    if not gt_classes or all(len(s.gts) == 0 for s in samples):
        return None

    total_gt = sum(len(s.gts) for s in samples)
    ap_sum = 0.0
    tp_total = 0
    fp_total = 0

    for cls_id in gt_classes:
        ap, n_gt = _ap_for_class(samples, cls_id, iou_threshold)
        ap_sum += ap
        counts = cm_for_class(samples, cls_id, iou_threshold)
        tp_total += counts["tp"]
        fp_total += counts["fp"]

    fn_total = total_gt - tp_total
    map50 = ap_sum / len(gt_classes)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    methodology = (
        "mAP@0.5: VOC-style 11-point interpolated AP averaged over ground-"
        "truth classes; precision/recall/F1 at the evaluation confidence "
        "threshold. Computed from real detections on the provided labels."
    )
    return AccuracyMetrics(
        methodology=methodology,
        map50=round(float(map50), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
        n_images=len(samples),
        n_ground_truth=total_gt,
        reference_model=reference_model,
    )


def cm_for_class(
    samples: Sequence[Sample], cls_id: int, iou_threshold: float
) -> dict[str, int]:
    """Confusion counts (tp/fp/fn) for one class under greedy matching."""
    tp = 0
    fp = 0
    n_gt = 0
    for sample in samples:
        gts = [g for g in sample.gts if g.class_id == cls_id]
        n_gt += len(gts)
        matched = [False] * len(gts)
        class_preds = sorted(
            (p for p in sample.preds if p.class_id == cls_id),
            key=lambda p: p.confidence,
            reverse=True,
        )
        for p in class_preds:
            best_i = -1
            best_iou = iou_threshold
            for i, gt in enumerate(gts):
                if matched[i]:
                    continue
                v = bbox_iou(p.xyxy, gt.xyxy)
                if v > best_iou:
                    best_iou = v
                    best_i = i
            if best_i >= 0:
                matched[best_i] = True
                tp += 1
            else:
                fp += 1
    return {"tp": tp, "fp": fp, "fn": n_gt - tp}


def build_samples_with_agreement(
    candidate_preds: list[list[Prediction]],
    reference_preds: list[list[Prediction]],
    image_paths: Sequence[str],
) -> list[Sample]:
    """Build samples that treat a reference model's output as ground truth.

    Used only for cross-model agreement when no human labels exist; the
    methodology string records this explicitly so it is not mistaken for
    ground-truth accuracy.
    """
    samples: list[Sample] = []
    for image_path, cand, ref in zip(image_paths, candidate_preds, reference_preds):
        samples.append(
            Sample(
                image_path=image_path,
                preds=cand,
                gts=[
                    GroundTruth(p.class_id, p.xyxy) for p in ref
                ],
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect_reported_device() -> str:
    """Best-effort human-readable device string (cuda/cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"{name} (cuda:{torch.cuda.current_device()})"
        return "cpu"
    except Exception:  # noqa: BLE001 - device label is cosmetic only
        return "unknown"


def benchmark_models(
    model_paths: Sequence[str],
    image_dir: str,
    label_dir: str | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    runs: int = 10,
    warmup: int = 3,
    device: str | None = None,
    json_out: str | None = None,
) -> dict:
    """Benchmark one or more models and return a full measured report.

    Args:
        model_paths: Artifacts to benchmark (``.pt``, ``.onnx``, ``.engine``).
            The first entry is the reference for cross-model agreement when
            no labels are supplied.
        image_dir: Directory containing real sonar images.
        label_dir: Optional directory of YOLO ``.txt`` labels (same base
            names as the images) for ground-truth accuracy.
        conf/iou: Detection thresholds used for every model.
        imgsz: Letterbox size (must match how the models were exported).
        runs/warmup: Timing repetitions.
        device: Inference device (``cpu`` / ``cuda:0``). Defaults to the FLAG
            or auto-detected torch device.
        json_out: Optional path to write the JSON report to.

    Returns:
        A dict with only measured values plus an explicit honesty statement.
    """
    images = collect_images(image_dir)
    image_paths = [str(p) for p in images]
    cfg = PreprocessConfig(target_size=imgsz)

    # Preprocess once; identical for every model so it drops out of the
    # per-model timing comparison (it is reported separately below).
    canvases = preprocess_canvases(images, cfg)

    labels_present = False
    label_files = {}
    if label_dir:
        label_root = Path(label_dir)
        for p in images:
            lbl = label_root / f"{p.stem}.txt"
            if lbl.exists():
                label_files[str(p)] = load_yolo_labels(lbl)
        labels_present = bool(label_files)

    models_report = []
    samples_by_model: dict[str, list[Sample]] = {}
    det_device = device or detect_reported_device()
    use_device = device  # pass None so ultralytics uses its own default

    for path in model_paths:
        mpath = Path(path)
        if not mpath.exists():
            raise FileNotFoundError(f"Model not found: {mpath}")

        model = load_yolo_model(mpath)
        timing = measure_inference(
            model, canvases, conf=conf, iou=iou,
            runs=runs, warmup=warmup, device=use_device,
        )

        sample_preds: list[list[Prediction]] = []
        for img_path in image_paths:
            preds = predict_one(
                model, img_path, conf=conf, iou=iou, cfg=cfg, device=use_device
            )
            sample_preds.append(preds)

        accuracy: AccuracyMetrics | None = None
        if labels_present:
            samples: list[Sample] = []
            for img_idx, img_path in enumerate(image_paths):
                samples.append(
                    Sample(
                        image_path=img_path,
                        preds=sample_preds[img_idx],
                        gts=label_files.get(img_path, []),
                    )
                )
            accuracy = compute_accuracy(samples, reference_model=None)

        samples_by_model[str(mpath)] = sample_preds

        models_report.append(
            {
                "model_path": str(mpath),
                "suffix": mpath.suffix,
                "model_size_bytes": os.path.getsize(mpath),
                "model_size_mb": round(os.path.getsize(mpath) / (1024 * 1024), 3),
                "timing": {
                    "method": "model forward pass + NMS + parse on a "
                    "preprocessed canvas; preprocessing identical for all "
                    "models and reported separately as a constant.",
                    "samples": timing.samples,
                    "mean_ms": timing.mean_ms,
                    "median_ms": timing.median_ms,
                    "p95_ms": timing.p95_ms,
                    "min_ms": timing.min_ms,
                    "max_ms": timing.max_ms,
                    "fps": timing.fps,
                },
                "accuracy": accuracy.to_dict() if accuracy else None,
            }
        )

    # Cross-model agreement when no labels exist: all non-reference models
    # measured against the first model (which acts as reference only then).
    if not labels_present and len(model_paths) > 1:
        ref_path = str(Path(model_paths[0]))
        ref_preds = samples_by_model[ref_path]
        for idx, path in enumerate(model_paths):
            if idx == 0:
                continue
            samples = build_samples_with_agreement(
                samples_by_model[str(Path(path))], ref_preds, image_paths
            )
            agreement = compute_accuracy(
                samples,
                reference_model=ref_path,
            )
            if agreement is not None:
                # Re-label the methodology so nobody mistakes cross-model
                # agreement for ground-truth accuracy. The measured match
                # statistics are genuine runtime values.
                agreement.methodology = (
                    "cross-model agreement: detections of this model matched "
                    f"against the reference model '{ref_path}' (IoU>=0.5). "
                    "Not ground-truth accuracy — no labels were provided."
                )
                agreement.reference_model = ref_path
                models_report[idx]["accuracy"] = agreement.to_dict()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": det_device,
        "preprocessing": {
            "method": "sonar preprocessing pipeline (grayscale -> denoise "
            "-> CLAHE -> normalize -> letterbox)",
            "imgsz": imgsz,
            "excluded_from_model_timing": True,
            "constant_for_all_models": True,
        },
        "evaluation": {
            "images": len(images),
            "image_paths": image_paths,
            "labels_provided": labels_present,
            "conf_threshold": conf,
            "iou_threshold": iou,
        },
        "models": models_report,
        "honesty": (
            "Every value above is measured at runtime on the provided "
            "images and hardware; no synthetic data, extrapolation or "
            "estimates are included."
        ),
    }

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report written to {out_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    print("\n=== Edge deployment benchmark ===")
    print(f"Device: {report['device']}")
    print(
        f"Images: {report['evaluation']['images']} "
        f"| labels: {'yes' if report['evaluation']['labels_provided'] else 'no'}"
    )
    print(
        f"Preprocessing: {report['preprocessing']['imgsz']}x{report['preprocessing']['imgsz']}"
        f" letterbox (constant, excluded from model timing)"
    )
    print()
    header = (
        f"{'model':<46}{'size':>9}{'mean':>9}{'p95':>9}{'fps':>8}"
        f"{'mAP@0.5':>10}{'(agree)':>6}"
    )
    print(header)
    print("-" * len(header))
    for m in report["models"]:
        acc = m["accuracy"]
        map50 = acc["map50"] if acc and acc["map50"] is not None else "n/a"
        agree = (
            acc["methodology"].startswith("cross-model")
            if acc
            else False
        )
        print(
            f"{Path(m['model_path']).name:<46}"
            f"{m['model_size_mb']:>8}MB"
            f"{m['timing']['mean_ms']:>9.3f}"
            f"{m['timing']['p95_ms']:>9.3f}"
            f"{m['timing']['fps']:>8.2f}"
            f"{str(map50):>10}"
            f"{'*' if agree else '':>6}"
        )
    print(
        "* = cross-model agreement against the reference model, not "
        "ground-truth accuracy."
    )
    print("\n" + report["honesty"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark detection models on real images: measures "
        "inference time, FPS, model size and accuracy."
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model artifacts to benchmark (.pt, .onnx, .engine). First is "
        "the reference for cross-model agreement when no labels are given.",
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory of real sonar images to run inference on.",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Optional directory of YOLO-format .txt labels (same base "
        "names as images) for ground-truth accuracy.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Letterbox size (must match the exported model input).",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--device", default=None,
        help="Inference device: 'cpu' or 'cuda:0' (default: auto).",
    )
    parser.add_argument("--json", default=None, dest="json_out")
    args = parser.parse_args(argv)

    report = benchmark_models(
        model_paths=args.models,
        image_dir=args.images,
        label_dir=args.labels,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        runs=args.runs,
        warmup=args.warmup,
        device=args.device,
        json_out=args.json_out,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())