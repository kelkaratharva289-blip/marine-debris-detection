"""Tests for the train/val/test split stage and label-id remapping.

These cover the critical invariant that class ids in the split label folders
match the compact vocabulary written to ``data.yaml`` — derived **only** from
the actual label files present in the raw pool. Fixtures are tiny placeholder
files under ``tmp_path``; the real ``backend/data`` pool is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.dataset.constants import infer_dataset_classes
from app.training.split import prepare_and_split


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_pool(root: Path) -> Path:
    """A tiny pool whose labels use sparse original ids 0, 2 and 5 only.

    This mirrors the real-world case where only a *subset* of the canonical
    vocabulary (ghost_net=0, pipe=2, other_debris=5) actually appears in the
    dataset. A valid remap must compress them to 0/1/2 in the split folders.
    """
    ids = [0, 2, 5, 0, 2, 5, 0, 2, 5, 5, 2, 0]
    for i, cid in enumerate(ids):
        (root / "raw" / "images").mkdir(parents=True, exist_ok=True)
        (root / "raw" / "images" / f"s{i:02d}.png").write_bytes(
            b"\xff\xd8\xff\xe0 placeholder"
        )
        _write_text(
            root / "raw" / "labels" / f"s{i:02d}.txt",
            f"{cid} 0.5 0.5 0.2 0.2\n",
        )
    return root


def _split_label_ids(root: Path, split: str) -> set[int]:
    ids: set[int] = set()
    for txt in (root / split / "labels").glob("*.txt"):
        ids.add(int(txt.read_text().split()[0]))
    return ids


def test_infer_dataset_classes_derives_from_real_labels(tmp_path):
    _make_pool(tmp_path)
    pairs = infer_dataset_classes(tmp_path / "raw" / "labels")
    assert pairs == [(0, "ghost_net"), (2, "pipe"), (5, "other_debris")]


def test_prepare_and_split_remaps_label_ids_to_compact(tmp_path):
    _make_pool(tmp_path)
    result = prepare_and_split(
        tmp_path / "raw" / "images",
        tmp_path / "raw" / "labels",
        tmp_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        num_classes=6,
        classes=None,
    )
    assert result.counts["total"] == 12

    yaml_text = (tmp_path / "data.yaml").read_text()
    assert "0: ghost_net" in yaml_text
    assert "1: pipe" in yaml_text
    assert "2: other_debris" in yaml_text
    assert "shipwreck" not in yaml_text

    for split in ("train", "val", "test"):
        ids = _split_label_ids(tmp_path, split)
        assert ids, f"{split} split has labels"
        assert ids <= {0, 1, 2}, f"{split} ids must be compact 0..2, got {ids}"


def test_prepare_and_split_honours_explicit_class_order(tmp_path):
    _make_pool(tmp_path)
    prepare_and_split(
        tmp_path / "raw" / "images",
        tmp_path / "raw" / "labels",
        tmp_path,
        num_classes=6,
        classes=["other_debris", "ghost_net", "pipe"],
    )
    yaml_text = (tmp_path / "data.yaml").read_text()
    assert "0: other_debris" in yaml_text
    assert "1: ghost_net" in yaml_text
    assert "2: pipe" in yaml_text
    for split in ("train", "val", "test"):
        ids = _split_label_ids(tmp_path, split)
        assert ids <= {0, 1, 2}, f"{split} ids must match data.yaml order, got {ids}"


def test_prepare_and_split_rejects_vocabulary_missing_a_labelled_class(tmp_path):
    _make_pool(tmp_path)
    with pytest.raises(ValueError, match="missing from the requested vocabulary"):
        prepare_and_split(
            tmp_path / "raw" / "images",
            tmp_path / "raw" / "labels",
            tmp_path,
            num_classes=6,
            classes=["ghost_net"],
        )