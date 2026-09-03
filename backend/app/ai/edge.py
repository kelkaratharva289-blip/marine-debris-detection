"""One-command edge deployment workflow: export + benchmark.

Prepares a trained PyTorch model for edge deployment by exporting the
configured ONNX variants (FP32 baseline, plus optional FP16 and INT8) and an
optional TensorRT engine, then benchmarking **every** artifact — including the
source checkpoint — on your own real sonar images.

The returned report contains only *measured* values:

* ``model_size``      -> ``os.path.getsize`` on the artifact actually written.
* ``inference_time``  -> mean/median/p95 latency from real inference runs (ms).
* ``fps``             -> ``1000 / mean_ms``.
* ``accuracy``        -> mAP@0.5 / precision / recall / F1 against ground
                         truth, or cross-model agreement vs. the source
                         checkpoint when no labels are supplied.

Honesty contract
----------------
No speed or size improvement is claimed without a measurement. A variant that
cannot be produced because an optional dependency (``onnx``,
``onnxruntime``) or a CUDA GPU / TensorRT is unavailable is reported as
``skipped`` with the exact reason — no figures are invented for it. Compare
the measured numbers across the exported artifacts and the source checkpoint
to decide what to ship to the edge.

The heavy work is delegated to :mod:`app.ai.export` and
:mod:`app.ai.benchmark`; this module only orchestrates them and collects the
results into one report so the workflow is a single command.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

from app.ai.benchmark import benchmark_models
from app.ai.export import ExportResult, export_model


@dataclass
class SkippedVariant:
    """An export that could not be produced, with the reason why."""

    label: str
    reason: str


@dataclass
class EdgeDeployResult:
    """Outcome of one prepared (exported + benchmarked) variant."""

    label: str                # e.g. "fp32", "fp16", "int8", "engine", "pt"
    artifact_path: str
    export: ExportResult | None = None
    benchmark: dict | None = None  # per-model block from the benchmark report
    skipped: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "artifact_path": self.artifact_path,
            "skipped": self.skipped,
            "model_size_bytes": (
                self.benchmark["model_size_bytes"] if self.benchmark else None
            ),
            "model_size_mb": (
                self.benchmark["model_size_mb"] if self.benchmark else None
            ),
            "mean_ms": (
                self.benchmark["timing"]["mean_ms"] if self.benchmark else None
            ),
            "median_ms": (
                self.benchmark["timing"]["median_ms"] if self.benchmark else None
            ),
            "p95_ms": (
                self.benchmark["timing"]["p95_ms"] if self.benchmark else None
            ),
            "fps": self.benchmark["timing"]["fps"] if self.benchmark else None,
            "accuracy": self.benchmark["accuracy"] if self.benchmark else None,
        }


def prepare_edge_model(
    model_path: str | Path,
    image_dir: str | Path,
    label_dir: str | Path | None = None,
    half: bool = True,
    int8: bool = True,
    engine: bool = False,
    include_source: bool = True,
    formats: Sequence[str] = ("onnx",),
    imgsz: int = 640,
    dynamic: bool = False,
    opset: int = 17,
    runs: int = 10,
    warmup: int = 3,
    export_device: str | None = None,
    inference_device: str | None = None,
    output_dir: str | Path = "models",
    engine_device: int = 0,
    workspace: int = 4,
    json_out: str | Path | None = None,
) -> dict:
    """Export configured variants and benchmark them all on real images.

    Args:
        model_path: Trained PyTorch (``.pt``) checkpoint to edge-deploy.
        image_dir: Directory of real sonar images used for benchmarking.
        label_dir: Optional YOLO ``.txt`` labels for ground-truth accuracy.
        half: Export an FP16 ONNX artifact in addition to FP32.
        int8: Additionally dynamic-quantize the ONNX artifact to INT8.
        engine: Also build a TensorRT engine (requires GPU + TensorRT).
        include_source: Benchmark the source ``.pt`` checkpoint too (acts as
            the reference for cross-model agreement when no labels exist).
        formats: ONNX export format list (only ``"onnx"`` is supported here).
        imgsz: Input size for export and benchmarking.
        runs/warmup: Timing repetitions for the benchmark.
        inference_device: Device used to benchmark (``"cpu"`` / ``"cuda:0"``).
        output_dir: Where exported artifacts are written.
        json_out: Optional path to write the JSON report to.

    Returns:
        A dict report with ``variants`` (measured or skipped), the measured
        `benchmark` summary, an explicit ``honesty`` statement, and provenance.

    Raises:
        FileNotFoundError: If the source checkpoint or image dir is missing.
    """
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(
            f"Model weights not found: {source}. Provide a trained .pt file."
        )
    images = Path(image_dir)
    if not images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images}")

    # 1) Export the requested variants. Each fails independently: an optional
    #    dependency or GPU being unavailable is recorded as "skipped", never
    #    faked, so one missing variant does not abort the whole deployment.
    exported: list[tuple[str, ExportResult]] = []  # (label, result)
    skipped: list[SkippedVariant] = []

    exported_onnx = _run_export_fn(
        lambda: export_model(
            source,
            output_dir=output_dir,
            formats=formats,
            imgsz=imgsz,
            half=False,           # FP32 baseline first
            int8=False,
            dynamic=dynamic,
            opset=opset,
            device=export_device,
        ),
        "fp32",
        skipped,
    )
    if exported_onnx:
        exported.append(("fp32", exported_onnx[0]))

    if half:
        fp16 = _run_export_fn(
            lambda: export_model(
                source,
                output_dir=output_dir,
                formats=formats,
                imgsz=imgsz,
                half=True,
                int8=False,
                dynamic=dynamic,
                opset=opset,
                device=export_device,
            ),
            "fp16",
            skipped,
        )
        if fp16:
            exported.append(("fp16", fp16[0]))

    if int8:
        int8_res = _run_export_fn(
            lambda: _quantize_from_fp32(
                source, output_dir, formats, imgsz, dynamic, opset, export_device
            ),
            "int8",
            skipped,
        )
        if int8_res:
            exported.append(("int8", int8_res))

    if engine:
        engine_res = _run_export_fn(
            lambda: _export_engine(
                source, output_dir, imgsz, half, workspace, engine_device
            ),
            "engine",
            skipped,
        )
        if engine_res:
            exported.append(("engine", engine_res))

    # 2) Benchmark every artifact + the source checkpoint on real images.
    benchmark_paths: list[str] = []
    variant_labels: list[str] = []
    if include_source:
        benchmark_paths.append(str(source))
        variant_labels.append("pt")
    for label, res in exported:
        benchmark_paths.append(res.output_path)
        variant_labels.append(label)

    benchmark: dict | None = None
    benchmark_error: str | None = None
    if benchmark_paths:
        try:
            benchmark = benchmark_models(
                model_paths=benchmark_paths,
                image_dir=str(images),
                label_dir=str(label_dir) if label_dir else None,
                imgsz=imgsz,
                runs=runs,
                warmup=warmup,
                device=inference_device,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never faked
            logger.warning("Benchmark did not complete: %s", exc)
            benchmark_error = str(exc)

    # 3) Assemble the variant report. Measured benchmark blocks are matched to
    #    their artifact by the benchmark's own model_path order.
    variants: list[dict] = []
    if benchmark is not None:
        measured_paths = [Path(m["model_path"]) for m in benchmark["models"]]
        for label, res in exported:
            variants.append(
                _variant_dict(label, res, measured_paths, benchmark)
            )
        if include_source:
            variants.insert(
                0,
                _variant_dict("pt", None, measured_paths, benchmark),
            )
    else:
        # Benchmark unavailable: fall back to the measured export sizes only.
        for label, res in exported:
            variants.append(
                {
                    "label": label,
                    "artifact_path": res.output_path,
                    "skipped": None,
                    "model_size_bytes": res.size_bytes,
                    "model_size_mb": res.size_mb,
                    "mean_ms": None,
                    "median_ms": None,
                    "p95_ms": None,
                    "fps": None,
                    "accuracy": None,
                }
            )
        if include_source:
            variants.insert(
                0,
                {
                    "label": "pt",
                    "artifact_path": str(source),
                    "skipped": None,
                    "model_size_bytes": os.path.getsize(source),
                    "model_size_mb": round(
                        os.path.getsize(source) / (1024 * 1024), 3
                    ),
                    "mean_ms": None,
                    "median_ms": None,
                    "p95_ms": None,
                    "fps": None,
                    "accuracy": None,
                },
            )

    for s in skipped:
        variants.append(
            {
                "label": s.label,
                "artifact_path": None,
                "skipped": s.reason,
                "model_size_bytes": None,
                "model_size_mb": None,
                "mean_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "fps": None,
                "accuracy": None,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source),
        "image_dir": str(images),
        "benchmark_completed": benchmark is not None,
        "benchmark_error": benchmark_error,
        "benchmark": benchmark,
        "variants": variants,
        "honesty": (
            "No improvement is claimed without a measured result. Every size, "
            "latency, FPS and accuracy value above was measured at runtime on "
            "the provided images and hardware. Variants marked 'skipped' were "
            "not produced (missing optional dependency or GPU) and carry no "
            "figures. Choose what to ship from the measured numbers."
        ),
    }

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            import json

            json.dump(report, fh, indent=2)
        print(f"Report written to {out_path}")

    return report


# ---------------------------------------------------------------------------
# Per-variant export helpers
# ---------------------------------------------------------------------------

def _run_export_fn(fn, label: str, skipped: list):
    """Run an export closure, recording a skip (never raising) on failure.

    Any failure to produce a variant — missing optional dependency, wrong
    hardware, or an unreadable source checkpoint — is recorded as a skip with
    the reason so the remaining variants can still be prepared and measured.
    """
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001 - per-variant failure is a skip
        logger.warning("Variant %s skipped: %s", label, exc)
        skipped.append(SkippedVariant(label=label, reason=str(exc)))
        return []


def _quantize_from_fp32(
    source, output_dir, formats, imgsz, dynamic, opset, export_device,
) -> list[ExportResult]:
    """Export FP32 ONNX then dynamic-quantize it to INT8."""
    from app.ai.export import export_to_onnx, quantize_int8, ExportResult

    fp32 = export_to_onnx(
        source,
        output_dir=output_dir,
        imgsz=imgsz,
        half=False,
        dynamic=dynamic,
        opset=opset,
        device=export_device,
    )
    return [quantize_int8(fp32.output_path)]


def _export_engine(source, output_dir, imgsz, half, workspace, device) -> list[ExportResult]:
    """Build a TensorRT engine (requires GPU + TensorRT installed)."""
    from app.ai.export import export_to_tensorrt

    return [export_to_tensorrt(
        source,
        output_dir=output_dir,
        imgsz=imgsz,
        half=half,
        workspace=workspace,
        device=device,
    )]


def _variant_dict(
    label: str,
    res: ExportResult | None,
    measured_paths: list[str],
    benchmark: dict,
) -> dict:
    if res is None:
        # The source checkpoint should be the first measured model.
        m = benchmark["models"][0]
    else:
        match = [m for m in benchmark["models"] if m["model_path"] == res.output_path]
        m = match[0] if match else None
    if m is None:
        return {
            "label": label,
            "artifact_path": res.output_path if res else None,
            "skipped": "not present in the benchmark report",
            "model_size_bytes": None,
            "model_size_mb": None,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "fps": None,
            "accuracy": None,
        }
    return {
        "label": label,
        "artifact_path": m["model_path"],
        "skipped": None,
        "model_size_bytes": m["model_size_bytes"],
        "model_size_mb": m["model_size_mb"],
        "mean_ms": m["timing"]["mean_ms"],
        "median_ms": m["timing"]["median_ms"],
        "p95_ms": m["timing"]["p95_ms"],
        "fps": m["timing"]["fps"],
        "accuracy": m["accuracy"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    print("\n=== Edge deployment report ===")
    print(f"Source model : {report['source_model']}")
    print(f"Images       : {report['image_dir']}")
    if report["benchmark_completed"]:
        b = report["benchmark"]
        print(
            f"Device       : {b['device']} "
            f"(benchmark completed)"
        )
    else:
        print(f"Device       : benchmark not completed"
              f"{' - ' + report['benchmark_error'] if report['benchmark_error'] else ''}")
    print()
    header = f"{'variant':<12}{'size':>9}{'mean(ms)':>10}{'p95(ms)':>10}{'fps':>8}"
    print(header)
    print("-" * len(header))
    for v in report["variants"]:
        if v["skipped"]:
            print(f"{v['label']:<12}{'skipped':<9} {v['skipped']}")
            continue
        print(
            f"{v['label']:<12}"
            f"{v['model_size_mb'] if v['model_size_mb'] is not None else '-':>8}MB"
            f"{v['mean_ms'] if v['mean_ms'] is not None else '-':>10}"
            f"{v['p95_ms'] if v['p95_ms'] is not None else '-':>10}"
            f"{v['fps'] if v['fps'] is not None else '-':>8}"
        )
    print("\n" + report["honesty"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edge",
        description="Export + benchmark a trained PyTorch model for edge "
        "deployment (ONNX FP32/FP16/INT8, optional TensorRT). Every number "
        "reported is measured on the provided images and hardware.",
    )
    parser.add_argument("--model", required=True,
                        help="Trained PyTorch (.pt) checkpoint.")
    parser.add_argument("--images", required=True,
                        help="Directory of real sonar images to benchmark on.")
    parser.add_argument("--labels", default=None,
                        help="Optional YOLO .txt labels dir for ground-truth accuracy.")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-fp16", action="store_true", help="Skip FP16 ONNX export.")
    parser.add_argument("--no-int8", action="store_true", help="Skip INT8 quantization.")
    parser.add_argument("--engine", action="store_true",
                        help="Also build a TensorRT engine (needs GPU + TensorRT).")
    parser.add_argument("--no-source", action="store_true",
                        help="Do not benchmark the source .pt checkpoint.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--inference-device", default=None,
                        help="'cpu' or 'cuda:0' for the benchmark.")
    parser.add_argument("--export-device", default=None,
                        help="'cpu' or 'cuda:0' for ONNX export.")
    parser.add_argument("--json", default=None, dest="json_out")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = prepare_edge_model(
        model_path=args.model,
        image_dir=args.images,
        label_dir=args.labels,
        half=not args.no_fp16,
        int8=not args.no_int8,
        engine=args.engine,
        include_source=not args.no_source,
        imgsz=args.imgsz,
        runs=args.runs,
        warmup=args.warmup,
        export_device=args.export_device,
        inference_device=args.inference_device,
        output_dir=args.output_dir,
        json_out=args.json_out,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
