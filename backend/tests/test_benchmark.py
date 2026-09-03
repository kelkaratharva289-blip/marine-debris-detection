"""Unit tests for the benchmarking harness pure helpers.

These tests verify the *measurement logic* with synthetic arrays — the
benchmark itself always runs on real images and never fabricates data.
"""

import numpy as np
import pytest

from app.ai.benchmark import (
    GroundTruth,
    Prediction,
    Sample,
    TimingStats,
    _map_to_original,
    bbox_iou,
    collect_images,
    compute_accuracy,
    load_yolo_labels,
    xywh_norm_to_xyxy,
)


def _sample(preds, gts=None):
    return Sample(image_path="scan.png", preds=preds, gts=gts or [])


class TestBboxIoU:
    def test_identical_boxes(self):
        box = (0.1, 0.1, 0.6, 0.6)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert bbox_iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.7, 0.7)) == 0.0

    def test_partial_overlap(self):
        # (0.1,0.1,0.5,0.5) area=0.16 and (0.3,0.3,0.6,0.6) area=0.09;
        # inter = 0.04, union = 0.21, iou = 0.04/0.21.
        iou = bbox_iou((0.1, 0.1, 0.5, 0.5), (0.3, 0.3, 0.6, 0.6))
        assert iou == pytest.approx(0.04 / 0.21)

    def test_zero_area(self):
        assert bbox_iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.5, 0.5)) == 0.0


class TestXYWHConversion:
    def test_centered_box(self):
        assert xywh_norm_to_xyxy(0.5, 0.5, 0.4, 0.2) == pytest.approx(
            (0.3, 0.4, 0.7, 0.6)
        )


class TestMapToOriginal:
    def test_non_square_image_remap(self):
        # Original 300(W)x600(H), letterboxed to a 640 canvas:
        # ratio = min(640/600, 640/300) = 1.0667 -> resized 320x640,
        # pad_x = 160, pad_y = 0.
        params = (1.0666666667, 160.0, 0.0, 640.0)
        # Canvas-center box (160+160, 0+160) = (320, 160)px, 320px square.
        dets = [
            {
                "class_id": 0,
                "bbox_x": 320.0 / 640.0,   # 0.5
                "bbox_y": 160.0 / 640.0,   # 0.25
                "bbox_width": 320.0 / 640.0,   # 0.5
                "bbox_height": 320.0 / 640.0,  # 0.5
            }
        ]
        mapped = _map_to_original(dets, params, img_h=600, img_w=300)
        # Canvas px -> original px: (320-160)/1.0667=150 -> /300 = 0.5
        # (160-0)/1.0667=150 -> /600 = 0.25; w=320/1.0667=300 -> /300=1.0
        # h=320/1.0667=300 -> /600=0.5
        assert mapped[0]["bbox_x"] == pytest.approx(0.5, abs=1e-2)
        assert mapped[0]["bbox_y"] == pytest.approx(0.25, abs=1e-2)
        assert mapped[0]["bbox_width"] == pytest.approx(1.0, abs=1e-2)
        assert mapped[0]["bbox_height"] == pytest.approx(0.5, abs=1e-2)


class TestTimingStats:
    def test_fps_from_mean(self):
        stats = TimingStats.from_times([10.0, 10.0, 10.0])
        assert stats.mean_ms == 10.0
        assert stats.fps == pytest.approx(100.0)

    def test_median_and_p95(self):
        times = [1.0, 2.0, 3.0, 100.0, 200.0]
        stats = TimingStats.from_times(times)
        assert stats.median_ms == 3.0
        assert stats.p95_ms == pytest.approx(np.percentile(times, 95), abs=1e-9)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            TimingStats.from_times([])


class TestCollectImages:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_images(tmp_path / "nope")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            collect_images(tmp_path)

    def test_finds_supported_images(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "readme.txt").write_text("not an image")
        images = collect_images(tmp_path)
        assert [p.name for p in images] == ["a.png", "b.jpg"]


class TestLoadYoloLabels:
    def test_parses_valid_line(self, tmp_path):
        p = tmp_path / "scan.txt"
        p.write_text("2 0.5 0.5 0.4 0.2\n")
        gts = load_yolo_labels(p)
        assert len(gts) == 1
        assert gts[0].class_id == 2
        assert gts[0].xyxy == pytest.approx((0.3, 0.4, 0.7, 0.6))

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "scan.txt"
        p.write_text("2 0.5 0.5 0.4 0.2\njunk line\n1 0.5\n")
        gts = load_yolo_labels(p)
        assert len(gts) == 1


class TestComputeAccuracy:
    def test_perfect_detection_is_map100(self):
        box = GroundTruth(0, (0.1, 0.1, 0.5, 0.5))
        pred = Prediction(class_id=0, confidence=0.9, xyxy=(0.1, 0.1, 0.5, 0.5))
        samples = [
            _sample([pred], [box]),
            _sample([pred], [box]),
        ]
        acc = compute_accuracy(samples)
        assert acc is not None
        # Every GT matched perfectly across both scene classes present.
        assert acc.map50 == pytest.approx(1.0)
        assert acc.n_ground_truth == 2

    def test_no_type_mismatch_not_counted(self):
        # Class mismatch must not be treated as a true positive.
        box = GroundTruth(1, (0.1, 0.1, 0.5, 0.5))
        pred = Prediction(class_id=0, confidence=0.9, xyxy=(0.1, 0.1, 0.5, 0.5))
        acc = compute_accuracy([_sample([pred], [box])])
        assert acc is not None
        assert acc.map50 == 0.0
        assert acc.recall == 0.0

    def test_no_ground_truth_returns_none(self):
        pred = Prediction(class_id=0, confidence=0.9, xyxy=(0.1, 0.1, 0.5, 0.5))
        assert compute_accuracy([_sample([pred])]) is None

    def test_no_predictions_scores_zero(self):
        box = GroundTruth(0, (0.1, 0.1, 0.5, 0.5))
        acc = compute_accuracy([_sample([], [box])])
        assert acc is not None
        assert acc.map50 == 0.0
        assert acc.recall == 0.0