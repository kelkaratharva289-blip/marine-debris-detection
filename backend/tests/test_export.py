"""Unit tests for the edge-deployment export module."""

import pytest

from app.ai.export import (
    ExportError,
    get_file_size,
    precision_label,
    export_to_onnx,
    export_to_tensorrt,
    quantize_int8,
)
from app.ai.inference import resolve_inference_model_path


class TestResolveInferencePath:
    def test_prefers_onnx_for_onnx_backend(self, tmp_path):
        pt = tmp_path / "model.pt"
        onnx = tmp_path / "model.onnx"
        engine = tmp_path / "model.engine"
        pt.write_bytes(b"pt")
        onnx.write_bytes(b"onnx")
        engine.write_bytes(b"engine")
        assert (
            resolve_inference_model_path(str(pt), str(onnx), str(engine), "onnx")
            == str(onnx)
        )

    def test_prefers_engine_for_tensorrt_backend(self, tmp_path):
        pt = tmp_path / "model.pt"
        engine = tmp_path / "model.engine"
        pt.write_bytes(b"pt")
        engine.write_bytes(b"engine")
        assert (
            resolve_inference_model_path(str(pt), None, str(engine), "tensorrt")
            == str(engine)
        )

    def test_torch_backend_uses_checkpoint_first(self, tmp_path):
        pt = tmp_path / "model.pt"
        onnx = tmp_path / "model.onnx"
        pt.write_bytes(b"pt")
        onnx.write_bytes(b"onnx")
        assert resolve_inference_model_path(str(pt), str(onnx), None, "torch") == str(pt)

    def test_graceful_fallback_when_preferred_missing(self, tmp_path):
        pt = tmp_path / "model.pt"
        onnx = tmp_path / "model.onnx"
        pt.write_bytes(b"pt")
        onnx.write_bytes(b"onnx")
        # tensorrt preferred but no engine; must fall back to the ONNX export.
        assert (
            resolve_inference_model_path(str(pt), str(onnx), None, "tensorrt")
            == str(onnx)
        )

    def test_returns_empty_when_nothing_exists(self, tmp_path):
        assert resolve_inference_model_path(str(tmp_path / "missing.pt")) == ""


class TestPrecisionLabel:
    def test_fp32_default(self):
        assert precision_label(half=False, int8=False) == "fp32"

    def test_fp16_half(self):
        assert precision_label(half=True, int8=False) == "fp16"

    def test_int8_wins(self):
        assert precision_label(half=True, int8=True) == "int8"
        assert precision_label(half=False, int8=True) == "int8"


class TestFileSize:
    def test_measures_real_bytes(self, tmp_path):
        p = tmp_path / "model.onnx"
        p.write_bytes(b"x" * 4096)
        assert get_file_size(p) == 4096

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            get_file_size(tmp_path / "nope.onnx")


class TestExportToOnnx:
    def test_missing_weights_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_to_onnx(tmp_path / "missing.pt", output_dir=str(tmp_path))


class TestExportToTensorrt:
    def test_missing_weights_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_to_tensorrt(tmp_path / "missing.pt", output_dir=str(tmp_path))


class TestQuantizeInt8:
    def test_missing_onnx_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            quantize_int8(tmp_path / "missing.onnx")

    def test_rejects_non_onnx_before_importing(self, tmp_path):
        # A non-.onnx file must be rejected up-front, before any optional
        # dependency (onnxruntime) is imported.
        fake = tmp_path / "model.bin"
        fake.write_bytes(b"not an onnx")
        with pytest.raises(ExportError):
            quantize_int8(fake)