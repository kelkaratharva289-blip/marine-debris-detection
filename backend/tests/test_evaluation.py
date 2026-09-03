"""Tests for the evaluation pipeline (metrics, confusion matrix, visualizations).

These verify the real-results machinery: confusion-matrix extraction/rendering
and detection overlay visualization. Only tiny placeholder files under
``tmp_path`` are used; the real ``backend/data`` pool is never touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.training.config import TrainConfig
from app.training.evaluate import (
    ConfusionMatrixData,
    build_detection_visualizations,
    extract_confusion_matrix,
    matrix_to_lists,
)


class _FakeCM:
    def __init__(self, matrix):
        self.matrix = matrix


class _FakeDetMetrics:
    def __init__(self, matrix):
        self.confusion_matrix = _FakeCM(matrix)


def _make_config(root: Path) -> TrainConfig:
    return TrainConfig(data_root=str(root), output_dir=str(root / "out"))


def _make_images(root: Path, n: int = 2) -> Path:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        src = images / f"img_{i}.png"
        from cv2 import imwrite
        imwrite(str(src), img)
    return images


def test_matrix_to_lists_converts_to_ints():
    result = matrix_to_lists(np.array([[0, 1], [2, 3]]))
    assert result == [[0, 1], [2, 3]]
    assert all(isinstance(v, int) for row in result for v in row)


def test_extract_confusion_matrix_builds_data_and_plot(tmp_path):
    # 2 classes -> 3x3 matrix (last row/col background). Real counts.
    matrix = np.array(
        [
            [1, 0, 0],  # GT ghost_net: 1 correct TP, 1 background FN implicitly omitted
            [0, 2, 1],  # GT pipe: 2 TP, 1 FN (model said background)
            [1, 0, 0],  # background GT: 1 FP into ghost_net
        ],
        dtype=np.int64,
    )
    metrics = _FakeDetMetrics(matrix)
    config = _make_config(tmp_path)

    cm = extract_confusion_matrix(metrics, names=["ghost_net", "pipe"], split="test",
                                  config=config)

    assert cm.n_classes == 2
    assert cm.names == ["ghost_net", "pipe"]
    assert cm.matrix.shape == (3, 3)
    assert cm.plot_path is not None
    plot = Path(cm.plot_path)
    assert plot.is_file()
    assert plot.name == "confusion_matrix_test.png"


def test_extract_confusion_matrix_handles_missing_matrix(tmp_path):
    class _NoCM:
        confusion_matrix = None

    cm = extract_confusion_matrix(_NoCM(), names=["ghost_net"], split="val",
                                  config=_make_config(tmp_path))
    assert cm.plot_path is None
    assert cm.matrix.shape == (2, 2)


def test_confusion_matrix_data_as_dict(tmp_path):
    data = ConfusionMatrixData(matrix=np.arange(9).reshape(3, 3),
                               names=["a", "b"], plot_path="p.png")
    out = data.as_dict()
    assert out["n_classes"] == 2
    assert out["names"] == ["a", "b"]
    assert out["matrix_raw"][2][2] == 8
    assert out["plot_path"] == "p.png"


def test_build_detection_visualizations_overlays_real_detections(tmp_path):
    images = _make_images(tmp_path)
    names = ["ghost_net", "pipe"]
    detections = {
        "img_0.png": [
            {"class_id": 0, "confidence": 0.9,
             "bbox_x": 0.1, "bbox_y": 0.1, "bbox_width": 0.3, "bbox_height": 0.3},
            {"class_id": 1, "confidence": 0.4,
             "bbox_x": 0.5, "bbox_y": 0.5, "bbox_width": 0.2, "bbox_height": 0.2},
        ],
        "img_1.png": [
            {"class_id": 0, "confidence": 0.2,   # below conf threshold -> dropped
             "bbox_x": 0.1, "bbox_y": 0.1, "bbox_width": 0.2, "bbox_height": 0.2},
        ],
    }
    save_dir = tmp_path / "viz"
    paths = build_detection_visualizations(images, detections, names, save_dir, conf=0.25)

    # img_0 has 2 detections >= 0.25; img_1 has none above threshold.
    assert len(paths) == 2
    assert (save_dir / "img_0_det.png").is_file()
    assert (save_dir / "img_1_det.png").is_file()

    # The overlay should differ from the blank source for img_0 (boxes drawn).
    from cv2 import imread
    out_img = imread(str(save_dir / "img_0_det.png"))
    assert out_img is not None
    # Some pixel differs from plain 128 (e.g. box/label colour).
    assert int(np.abs(out_img.astype(int) - 128).sum()) > 0


def test_build_detection_visualizations_empty_detections(tmp_path):
    images = _make_images(tmp_path, n=1)
    paths = build_detection_visualizations(images, {}, ["ghost_net"],
                                           tmp_path / "viz_empty", conf=0.25)
    # One image processed, but no overlays drawn (no detections to draw).
    assert len(paths) == 1
    assert (tmp_path / "viz_empty" / "img_0_det.png").is_file()
