"""Tests for the edge-deployment orchestration (export + benchmark)."""

import pytest

import app.ai.edge as edge
from app.ai.export import ExportError, ExportResult


@pytest.fixture
def model_and_images(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake checkpoint weights")
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "scan.png").write_bytes(b"x")
    return pt, img_dir, tmp_path


class TestGuardRails:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            edge.prepare_edge_model(
                tmp_path / "missing.pt", tmp_path / "images"
            )

    def test_missing_images_raises(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"x")
        with pytest.raises(FileNotFoundError):
            edge.prepare_edge_model(pt, tmp_path / "no_images")


class TestSkippedVariantsOnMissingDeps:
    def test_export_failures_recorded_as_skipped_not_raised(
        self, model_and_images, monkeypatch
    ):
        pt, img_dir, tmp = model_and_images

        def fake_export_model(*args, **kwargs):
            raise ExportError("onnx is not installed")

        monkeypatch.setattr(edge, "export_model", fake_export_model)
        # Benchmark never runs because there is nothing to benchmark.
        report = edge.prepare_edge_model(
            pt, img_dir, half=True, int8=True, engine=False
        )
        labels = {v["label"] for v in report["variants"]}
        # fp32, fp16 and int8 all attempted and skipped.
        assert {"fp32", "fp16", "int8"} <= labels
        for v in report["variants"]:
            if v["label"] in ("fp32", "fp16", "int8"):
                assert v["skipped"] is not None
                assert v["mean_ms"] is None

    def test_honesty_statement_present(self, model_and_images, monkeypatch):
        pt, img_dir, tmp = model_and_images
        monkeypatch.setattr(edge, "export_model", lambda *a, **k: [])
        report = edge.prepare_edge_model(pt, img_dir, half=False, int8=False)
        assert "honesty" in report
        assert "measured" in report["honesty"]


class TestBenchmarkFailureFallback:
    def test_report_records_measured_sizes_when_benchmark_fails(
        self, model_and_images, monkeypatch
    ):
        pt, img_dir, tmp = model_and_images

        fp32 = ExportResult(
            source=str(pt), format="onnx", precision="fp32",
            output_path=str(tmp / "model.onnx"), size_bytes=1234,
        )

        def fake_export_model(*args, **kwargs):
            return [fp32]

        def fake_benchmark(*args, **kwargs):
            raise RuntimeError("inference backend unavailable")

        monkeypatch.setattr(edge, "export_model", fake_export_model)
        monkeypatch.setattr(edge, "benchmark_models", fake_benchmark)

        report = edge.prepare_edge_model(
            pt, img_dir, half=False, int8=False
        )
        assert report["benchmark_completed"] is False
        assert report["benchmark_error"] is not None
        fp32_v = next(v for v in report["variants"] if v["label"] == "fp32")
        assert fp32_v["model_size_bytes"] == 1234  # measured export size kept
        assert fp32_v["mean_ms"] is None
        # Source checkpoint size is measured too.
        src_v = next(v for v in report["variants"] if v["label"] == "pt")
        assert src_v["model_size_bytes"] == len(b"fake checkpoint weights")


class TestReportShape:
    def test_variants_include_pt_first_when_enabled(
        self, model_and_images, monkeypatch
    ):
        pt, img_dir, tmp = model_and_images
        monkeypatch.setattr(edge, "export_model", lambda *a, **k: [])
        benchmark = {
            "device": "cpu",
            "models": [
                {
                    "model_path": str(pt),
                    "model_size_bytes": 24,
                    "model_size_mb": 0.000,
                    "timing": {"mean_ms": 10.0, "median_ms": 10.0, "p95_ms": 12.0,
                               "min_ms": 9.0, "max_ms": 14.0, "samples": 10, "fps": 100.0},
                    "accuracy": None,
                }
            ],
        }

        def fake_benchmark(model_paths, **kwargs):
            # The orchestration passes the source image dir through; simulate
            # a successful benchmark returning blocks in the same order.
            return benchmark

        monkeypatch.setattr(edge, "benchmark_models", fake_benchmark)
        report = edge.prepare_edge_model(pt, img_dir, half=False, int8=False)
        assert report["benchmark_completed"] is True
        assert report["variants"][0]["label"] == "pt"
        assert report["variants"][0]["fps"] == 100.0
