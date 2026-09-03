"""Model export utilities for edge deployment.

Converts a trained PyTorch (Ultralytics YOLO) checkpoint into deployment
artifacts and optionally applies precision optimisations:

* ``onnx``   — ONNX graph. FP32 weights by default; FP16 when ``half=True``.
* ``int8``   — INT8 dynamically-quantized ONNX (onnxruntime-quantization),
               applied as a further step on an FP32/FP16 ONNX export.
* ``engine`` — TensorRT engine (requires a CUDA GPU and TensorRT installed).

Honesty contract
----------------
Every file size in :class:`ExportResult` is measured with
``os.path.getsize`` on the artifact that was actually written.

This module makes **no latency or accuracy claims**. Speed, FPS, size and
accuracy must be verified on real data with :mod:`app.ai.benchmark`; the
exported artifact may be *smaller or larger* and *faster or slower* than the
source checkpoint depending on hardware and export options.

Optional dependencies (``onnx``, ``onnxruntime``, ``tensorrt``) are imported
lazily so the base inference pipeline never requires them. See
``requirements-edge.txt``.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

#: Export formats understood by the CLI.
EXPORT_FORMATS = ("onnx", "engine")


class ExportError(RuntimeError):
    """Raised when a model cannot be exported or optimized."""


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one export / optimisation step.

    ``size_bytes`` is always the measured on-disk size of ``output_path``.
    """

    source: str
    format: str        # "onnx" | "int8" | "engine"
    precision: str     # "fp32" | "fp16" | "int8"
    output_path: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 3)


def get_file_size(path: str | Path) -> int:
    """Measure the real on-disk size of a file in bytes."""
    return os.path.getsize(path)


def precision_label(half: bool, int8: bool) -> str:
    """Precision label for an export given the optimisation flags."""
    if int8:
        return "int8"
    if half:
        return "fp16"
    return "fp32"


# ---------------------------------------------------------------------------
# ONNX export (PyTorch -> ONNX)
# ---------------------------------------------------------------------------

def export_to_onnx(
    model_path: str | Path,
    output_dir: str | Path = "models",
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    opset: int = 17,
    device: str | None = None,
) -> ExportResult:
    """Export a PyTorch YOLO checkpoint to ONNX via Ultralytics.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        output_dir: Directory the ONNX artifact is written into.
        imgsz: Inference input size (square).
        half: Export FP16 weights (``onnx`` format only).
        dynamic: Use dynamic input axes (variable batch / input size).
        opset: ONNX opset version passed to the exporter.
        device: Export device (``"cpu"``, ``"cuda:0"``). Defaults to "cpu".

    Returns:
        :class:`ExportResult` describing the artifact actually written.

    Raises:
        ExportError: If the source weights are missing, the optional ``onnx``
            dependency is unavailable, or the exporter fails.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. "
            "Export requires a trained PyTorch (.pt) checkpoint."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ExportError(
            "ultralytics is required for PyTorch->ONNX export. "
            "Install it with: pip install ultralytics"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Exporting %s to ONNX (half=%s, dynamic=%s)", path, half, dynamic)

    model = YOLO(str(path))
    try:
        out = model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            opset=opset,
            device=device or "cpu",
        )
    except Exception as exc:  # noqa: BLE001 - surface exporter errors verbatim
        raise ExportError(f"ONNX export failed: {exc}") from exc

    onnx_path = Path(out)
    if not onnx_path.exists():
        raise ExportError(
            "Exporter reported success but no ONNX file was found at "
            f"{onnx_path}. Enable DEBUG and inspect the export log."
        )

    return ExportResult(
        source=str(path),
        format="onnx",
        precision=precision_label(half, int8=False),
        output_path=str(onnx_path),
        size_bytes=get_file_size(onnx_path),
    )


# ---------------------------------------------------------------------------
# INT8 (dynamic) quantization
# ---------------------------------------------------------------------------

def quantize_int8(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    per_channel: bool = False,
    weight_type: str = "int8",
) -> ExportResult:
    """Dynamically quantize an ONNX model's weights to INT8.

    Uses ``onnxruntime.quantization`` (dynamic quantization: activations are
    quantized per-input at runtime, weights statically to INT8). This is the
    dependency-free route to an INT8 artifact; static quantization with a
    calibration dataset can be added by swapping in a ``CalibrationDataReader``.

    Args:
        onnx_path: Source ONNX file (FP32/FP16 output of
            :func:`export_to_onnx`).
        output_path: Destination path. Defaults to ``<name>_int8.onnx`` next
            to the source.
        per_channel: Quantize weights per output channel instead of per
            tensor. Often improves accuracy.
        weight_type: ``"int8"`` or ``"uint8"``.

    Returns:
        :class:`ExportResult` for the quantized artifact.

    Raises:
        ExportError: If ``onnxruntime`` is unavailable or quantization fails.
    """
    source = Path(onnx_path)
    if not source.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {source}. Export the model to ONNX first."
        )
    if source.suffix.lower() != ".onnx":
        raise ExportError(
            f"INT8 quantization expects an .onnx input, got: {source}"
        )
    try:
        import onnxruntime
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ExportError(
            "onnxruntime is required for INT8 quantization. "
            "Install it with: pip install onnxruntime"
        ) from exc

    target = Path(output_path) if output_path else source.with_name(
        f"{source.stem}_int8.onnx"
    )
    wt = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8

    logger.info("Quantizing %s to INT8 (per_channel=%s)", source, per_channel)
    try:
        quantize_dynamic(
            model_input=str(source),
            model_output=str(target),
            weight_type=wt,
            per_channel=per_channel,
            extra_options={"EnableSubgraph": True},
        )
    except Exception as exc:  # noqa: BLE001 - surface quantizer errors verbatim
        raise ExportError(f"INT8 quantization failed: {exc}") from exc

    if not target.exists():
        raise ExportError(
            "Quantizer reported success but produced no file at " f"{target}."
        )

    return ExportResult(
        source=str(source),
        format="int8",
        precision="int8",
        output_path=str(target),
        size_bytes=get_file_size(target),
    )


# ---------------------------------------------------------------------------
# TensorRT engine export
# ---------------------------------------------------------------------------

def export_to_tensorrt(
    model_path: str | Path,
    output_dir: str | Path = "models",
    imgsz: int = 640,
    half: bool = True,
    workspace: int = 4,
    device: int = 0,
) -> ExportResult:
    """Export a PyTorch YOLO checkpoint to a TensorRT engine.

    Requires a CUDA-capable GPU and the TensorRT Python package (installed
    separately, e.g. ``tensorrt`` + matching CUDA/cuDNN). FP16 engines use
    TensorRT's native FP16 kernels when ``half=True``.

    Args:
        model_path: Path to the ``.pt`` checkpoint.
        output_dir: Directory the ``.engine`` artifact is written into.
        imgsz: Inference input size (square).
        half: Build an FP16 TensorRT engine.
        workspace: TensorRT workspace size in GB during build.
        device: CUDA device index used to build the engine.

    Returns:
        :class:`ExportResult` for the engine artifact.

    Raises:
        ExportError: If TensorRT is not installed, no CUDA GPU is available,
            or engine building fails.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. "
            "Export requires a trained PyTorch (.pt) checkpoint."
        )
    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        raise ExportError(
            "TensorRT is not installed. TensorRT engine export requires:\n"
            "  pip install tensorrt\n"
            "plus a matching CUDA / cuDNN installation on a GPU machine.\n"
            "Engine builds cannot run on CPU-only hosts."
        ) from exc

    import torch

    if not torch.cuda.is_available():
        raise ExportError(
            "No CUDA GPU detected — TensorRT engine export requires a "
            "CUDA-capable device."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ExportError(
            "ultralytics is required for TensorRT export."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "Building TensorRT engine (half=%s, workspace=%dGB, device=%d)",
        half, workspace, device,
    )

    model = YOLO(str(path))
    try:
        out = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            workspace=workspace,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - surface engine-builder errors
        raise ExportError(f"TensorRT engine export failed: {exc}") from exc

    engine_path = Path(out)
    if not engine_path.exists():
        raise ExportError(
            "Engine builder reported success but produced no file at "
            f"{engine_path}."
        )

    return ExportResult(
        source=str(path),
        format="engine",
        precision=precision_label(half, int8=False),
        output_path=str(engine_path),
        size_bytes=get_file_size(engine_path),
    )


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def export_model(
    model_path: str | Path,
    output_dir: str | Path = "models",
    formats: Sequence[str] = ("onnx",),
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    opset: int = 17,
    device: str | None = None,
    engine_device: int = 0,
    workspace: int = 4,
) -> list[ExportResult]:
    """Run one or more export / optimization steps and return all results.

    ``formats`` accepts ``"onnx"`` and ``"engine"``. When ``int8=True`` the
    ONNX artifact is additionally dynamic-quantized to INT8. Results are
    returned in the order the steps executed.
    """
    results: list[ExportResult] = []

    for fmt in formats:
        if fmt == "onnx":
            onnx_result = export_to_onnx(
                model_path,
                output_dir=output_dir,
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                opset=opset,
                device=device,
            )
            results.append(onnx_result)
            if int8:
                if half:
                    logger.warning(
                        "INT8 quantization on an FP16 ONNX model may be "
                        "lossy; consider exporting FP32 first."
                    )
                int8_result = quantize_int8(onnx_result.output_path)
                results.append(int8_result)
        elif fmt == "engine":
            results.append(
                export_to_tensorrt(
                    model_path,
                    output_dir=output_dir,
                    imgsz=imgsz,
                    half=half,
                    workspace=workspace,
                    device=engine_device,
                )
            )
        else:
            raise ExportError(f"Unknown export format: {fmt}")

    return results


def _print_results(results: list[ExportResult]) -> None:
    print("\nExport results (sizes measured on disk):")
    print(f"{'format':<8}{'precision':<10}{'bytes':>14}  path")
    print("-" * 90)
    for r in results:
        print(
            f"{r.format:<8}{r.precision:<10}{r.size_bytes:>14,}  "
            f"{r.output_path}"
        )
    print(
        "\nNo speed or accuracy claim is made here — measure the artifacts "
        "with:\n"
        "    python -m app.ai.benchmark\n"
        "on real images and your target hardware."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch YOLO model for edge deployment "
        "(ONNX / INT8 / TensorRT)."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the trained PyTorch (.pt) checkpoint.",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory to write exported artifacts (default: models).",
    )
    parser.add_argument(
        "--format", action="append", choices=list(EXPORT_FORMATS),
        default=["onnx"], dest="formats",
        help="Artifact format(s). May be repeated (default: onnx).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input size.")
    parser.add_argument(
        "--half", action="store_true",
        help="Export FP16 weights (ONNX) / FP16 engine (TensorRT).",
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Additionally dynamic-quantize ONNX weights to INT8.",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Use dynamic input axes for ONNX export.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--device", default=None,
        help="Export device for ONNX: 'cpu' or 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--engine-device", type=int, default=0,
        help="CUDA device index for TensorRT engine builds.",
    )
    parser.add_argument("--workspace", type=int, default=4, help="TRT workspace GB.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = export_model(
        model_path=args.model,
        output_dir=args.output_dir,
        formats=args.formats,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        opset=args.opset,
        device=args.device,
        engine_device=args.engine_device,
        workspace=args.workspace,
    )
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())