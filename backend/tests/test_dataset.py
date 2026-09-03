"""Unit tests for the YOLO dataset pipeline (loader, split, builder).

Test fixtures are tiny placeholder files written to pytest's ``tmp_path`` —
never to the real ``backend/data`` pool — so the raw/published dataset folders
always stay empty until real sonar scans are placed there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.dataset import (
    Annotation,
    LabelParseError,
    build_dataset,
    default_data_root,
    load_dataset,
    parse_yolo_label,
    split_dataset,
    write_dataset_yaml,
)
from app.ai.dataset.split import SplitRatios
from app.ai.dataset.builder import SplitDataset


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode, encoding=None if isinstance(content, bytes) else "utf-8") as fh:
        fh.write(content)
    return path


def make_image(path: Path) -> Path:
    """Write a minimal non-empty file that the loader accepts by extension."""
    return _write(path, b"\xff\xd8\xff\xe0 example sonar scan (not decoded here)")


def make_pool(root: Path, n: int, labelled: int) -> Path:
    """Create a raw pool of ``n`` images; the first ``labelled`` get a label."""
    images = root / "raw" / "images"
    labels = root / "raw" / "labels"
    for i in range(n):
        make_image(images / f"scan_{i:02d}.png")
        if i < labelled:
            class_id = i % 6
            _write(
                labels / f"scan_{i:02d}.txt",
                f"{class_id} 0.5 0.5 0.2 0.2\n",
            )
    return root


class TestDefaults:
    def test_default_root_is_backend_data(self):
        root = default_data_root()
        assert root.name == "data"
        assert root.parent.name == "backend"
        assert (root / "raw" / "images").is_dir()
        assert (root / "raw" / "labels").is_dir()


# ---------------------------------------------------------------------------
# parse_yolo_label
# ---------------------------------------------------------------------------


class TestParseYoloLabel:
    def test_parses_valid_lines(self, tmp_path):
        label = _write(tmp_path / "a.txt", "2 0.5 0.5 0.2 0.1\n3 0.25 0.75 0.4 0.3\n")
        result = parse_yolo_label(label, num_classes=6)
        assert result == (
            Annotation(class_id=2, x_center=0.5, y_center=0.5, width=0.2, height=0.1),
            Annotation(class_id=3, x_center=0.25, y_center=0.75, width=0.4, height=0.3),
        )

    def test_ignores_blank_lines_and_extra_tokens(self, tmp_path):
        label = _write(tmp_path / "b.txt", "\n\n5 0.5 0.5 0.1 0.1 3.2 4.1\n\n")
        result = parse_yolo_label(label, num_classes=6)
        assert len(result) == 1
        assert result[0].class_id == 5
        assert result[0].width == 0.1

    def test_rejects_too_few_tokens(self, tmp_path):
        label = _write(tmp_path / "c.txt", "0 0.5 0.5\n")
        with pytest.raises(LabelParseError, match="expected at least 5"):
            parse_yolo_label(label)

    def test_rejects_non_integer_class(self, tmp_path):
        label = _write(tmp_path / "d.txt", "net 0.5 0.5 0.2 0.2\n")
        with pytest.raises(LabelParseError, match="not an integer"):
            parse_yolo_label(label)

    def test_rejects_out_of_range_class(self, tmp_path):
        label = _write(tmp_path / "e.txt", "9 0.5 0.5 0.2 0.2\n")
        with pytest.raises(LabelParseError, match="out of range"):
            parse_yolo_label(label, num_classes=6)

    def test_rejects_non_float_coord(self, tmp_path):
        label = _write(tmp_path / "f.txt", "0 x 0.5 0.2 0.2\n")
        with pytest.raises(LabelParseError, match="not a float"):
            parse_yolo_label(label)

    def test_rejects_center_outside_unit_square(self, tmp_path):
        label = _write(tmp_path / "g.txt", "0 1.5 0.5 0.2 0.2\n")
        with pytest.raises(LabelParseError, match="within \\[0, 1\\]"):
            parse_yolo_label(label)

    def test_rejects_zero_or_oversize_box(self, tmp_path):
        zero = _write(tmp_path / "h.txt", "0 0.5 0.5 0 0.2\n")
        with pytest.raises(LabelParseError, match="width/height must be > 0"):
            parse_yolo_label(zero)
        wide = _write(tmp_path / "i.txt", "0 0.5 0.5 1.4 0.2\n")
        with pytest.raises(LabelParseError, match="must not exceed 1"):
            parse_yolo_label(wide)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_yolo_label(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_pairs_images_with_labels(self, tmp_path):
        root = make_pool(tmp_path, n=3, labelled=3)
        samples = load_dataset(root / "raw" / "images", root / "raw" / "labels")
        assert len(samples) == 3
        assert all(sample.is_labelled for sample in samples)
        assert samples[0].annotations[0].class_id == 0

    def test_unlabelled_skip(self, tmp_path):
        root = make_pool(tmp_path, n=4, labelled=2)
        samples = load_dataset(
            root / "raw" / "images",
            root / "raw" / "labels",
            skip_unlabelled=True,
        )
        assert len(samples) == 2

    def test_missing_label_raises_by_default(self, tmp_path):
        root = make_pool(tmp_path, n=4, labelled=2)
        with pytest.raises(FileNotFoundError, match="Label file missing"):
            load_dataset(root / "raw" / "images", root / "raw" / "labels")

    def test_malformed_label_raises(self, tmp_path):
        root = make_pool(tmp_path, n=1, labelled=1)
        _write(root / "raw" / "labels" / "scan_00.txt", "0 0.5\n")
        with pytest.raises(LabelParseError):
            load_dataset(root / "raw" / "images", root / "raw" / "labels")


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def _samples(self, tmp_path, classes_by_image: list[list[int]]) -> list:
        labels = tmp_path / "labels"
        samples = []
        for i, classes in enumerate(classes_by_image):
            image = make_image(tmp_path / "images" / f"img_{i:02d}.png")
            label = labels / f"img_{i:02d}.txt"
            _write(
                label,
                "".join(f"{c} 0.5 0.5 0.2 0.2\n" for c in classes),
            )
            from app.ai.dataset import load_sample

            samples.append(load_sample(image, labels, num_classes=6))
        return samples

    def test_deterministic_reproducibility(self, tmp_path):
        classes_by_image = [[i % 3] for i in range(60)]
        samples = self._samples(tmp_path, classes_by_image)
        a = split_dataset(samples, ratios=SplitRatios(0.7, 0.15, 0.15), seed=7)
        b = split_dataset(samples, ratios=SplitRatios(0.7, 0.15, 0.15), seed=7)
        assert sorted(s.stem for s in a.train) == sorted(s.stem for s in b.train)
        assert sorted(s.stem for s in a.val) == sorted(s.stem for s in b.val)

    def test_different_seed_differs(self, tmp_path):
        classes_by_image = [[i % 3] for i in range(60)]
        samples = self._samples(tmp_path, classes_by_image)
        a = split_dataset(samples, seed=1)
        b = split_dataset(samples, seed=2)
        assert sorted(s.stem for s in a.train) != sorted(s.stem for s in b.train)

    def test_counts_sum_and_ratios(self, tmp_path):
        classes_by_image = [[i % 3] for i in range(100)]
        samples = self._samples(tmp_path, classes_by_image)
        split = split_dataset(samples, ratios=SplitRatios(0.7, 0.2, 0.1), seed=42)
        assert len(split.train) == 70
        assert len(split.val) == 20
        assert len(split.test) == 10
        assert len(split) == 100

    def test_class_stratification_spreads_classes(self, tmp_path):
        # 300 single-class images; every class must appear in every split.
        classes_by_image = [[i % 3] for i in range(300)]
        samples = self._samples(tmp_path, classes_by_image)
        split = split_dataset(samples, ratios=SplitRatios(0.7, 0.15, 0.15), seed=3)
        for partition in (split.train, split.val, split.test):
            seen = {s.annotations[0].class_id for s in partition}
            assert seen == {0, 1, 2}, f"class mix missing in a split: {seen}"

    def test_empty_input(self, tmp_path):
        split = split_dataset([])
        assert isinstance(split, SplitDataset)
        assert len(split) == 0


# ---------------------------------------------------------------------------
# builder / yaml
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_build_dataset_full_pipeline(self, tmp_path):
        root = make_pool(tmp_path, n=10, labelled=10)
        result = build_dataset(
            root=root,
            ratios=SplitRatios(0.7, 0.15, 0.15),
            seed=42,
        )

        assert result.counts["total"] == 10
        assert (
            result.counts["train"] + result.counts["val"] + result.counts["test"]
            == result.counts["total"]
        )

        # Every sample is materialised exactly once across the three splits.
        placed = sum(
            len(list((root / name / "images").glob("*"))) for name in ("train", "val", "test")
        )
        assert placed == 10

        # Labels follow their images.
        for name in ("train", "val", "test"):
            img_dir = root / name / "images"
            lbl_dir = root / name / "labels"
            for image in img_dir.glob("*"):
                assert (lbl_dir / f"{image.stem}.txt").is_file()

        assert result.config_yaml.is_file()
        assert result.manifest.is_file()

    def test_build_requires_nonempty_pool(self, tmp_path):
        with pytest.raises(ValueError, match="Raw images directory"):
            build_dataset(root=tmp_path)
        (tmp_path / "raw").mkdir(parents=True)
        (tmp_path / "raw" / "images").mkdir()
        (tmp_path / "raw" / "labels").mkdir()
        with pytest.raises(ValueError, match="No supported images"):
            build_dataset(root=tmp_path)

    def test_move_removes_from_raw(self, tmp_path):
        root = make_pool(tmp_path, n=6, labelled=6)
        build_dataset(root=root, seed=1, move=True)
        assert len(list((root / "raw" / "images").glob("*"))) == 0

    def test_build_refuses_missing_raw_dirs(self, tmp_path):
        with pytest.raises(ValueError, match="Raw images directory"):
            build_dataset(root=tmp_path)


class TestWriteYaml:
    def test_yaml_names_index_order(self, tmp_path):
        path = write_dataset_yaml(tmp_path, classes=["ghost_net", "pipe"])
        text = path.read_text(encoding="utf-8")
        assert "train: train/images" in text
        assert "val: val/images" in text
        assert "test: test/images" in text
        assert "names:" in text
        assert "0: ghost_net" in text
        assert "1: pipe" in text