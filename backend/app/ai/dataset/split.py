"""Deterministic, stratification-aware train/val/test splitting.

Samples are grouped by their class composition (the sorted tuple of class ids
present in an image) and each group is shuffled with a seed derived from the
split seed and the group key. Each split then draws a share of every group that
is proportional to that group's size, so:

* global train/val/test counts match the configured ratios **exactly**, and
* every split keeps a similar class mix (stratification).

The same seed always reproduces the same split. No files are written here —
:func:`split_dataset` returns the partition and the builder copies/moves real
files into the split folders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from app.ai.dataset.loader import Sample


@dataclass(frozen=True)
class SplitRatios:
    """Train/val/test ratios; values are normalized internally."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def normalize(self) -> tuple[float, float, float]:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Split ratios must be positive and sum to > 0")
        return self.train / total, self.val / total, self.test / total


@dataclass
class SplitDataset:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": len(self),
        }


def _group_seed(base_seed: int, key: Sequence[int]) -> int:
    """Per-group RNG seed derived from the base seed and the class key."""
    return (base_seed + sum(c * 31 for c in key) * 1000003) & 0xFFFFFFFF


def _allocate(total: int, *fractions: float) -> list[int]:
    """Split an integer count across fractions using the largest-remainder rule.

    Guarantees the returned counts sum to ``total``.
    """
    raw = [total * f for f in fractions]
    counts = [int(v) for v in raw]
    remainders = [v - int(v) for v in raw]
    remaining = total - sum(counts)
    for _ in range(remaining):
        bucket = max(range(len(remainders)), key=lambda i: remainders[i])
        counts[bucket] += 1
        remainders[bucket] = 0.0
    return counts


def _distribute(total: int, remaining: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    """Allocate ``total`` items across groups proportionally to group size.

    Uses largest remainders so the allocations sum exactly to ``total`` and
    never exceed any group's remaining size.
    """
    base_total = sum(remaining.values())
    if base_total <= 0:
        return {key: 0 for key in remaining}

    take: dict[tuple[int, ...], int] = {}
    rem_bits: dict[tuple[int, ...], float] = {}
    allocated = 0
    for key, size in remaining.items():
        exact = total * size / base_total
        base = int(exact)
        take[key] = base
        rem_bits[key] = exact - base
        allocated += base

    for _ in range(total - allocated):
        candidates = sorted(
            (k for k in rem_bits if take[k] < remaining[k]),
            key=lambda k: rem_bits[k],
            reverse=True,
        )
        if not candidates:
            break
        take[candidates[0]] += 1
        rem_bits[candidates[0]] = 0.0

    return take


def split_dataset(
    samples: Sequence[Sample],
    ratios: SplitRatios | None = None,
    seed: int = 42,
    stratify: bool = True,
) -> SplitDataset:
    """Partition samples into train/val/test splits.

    Args:
        samples: The full dataset to partition.
        ratios: Desired train/val/test fractions (default 70/15/15).
        seed: Deterministic RNG seed; the same seed reproduces the same split.
        stratify: If True (default), draw proportionally from each class
            composition so every split keeps a similar class mix. If False, a
            plain shuffled split is produced.

    Returns:
        A :class:`SplitDataset` containing the three partitions.
    """
    if not samples:
        return SplitDataset(train=[], val=[], test=[])

    r_train, r_val, r_test = (ratios or SplitRatios()).normalize()
    global_counts = _allocate(len(samples), r_train, r_val, r_test)

    if not stratify:
        pool = list(samples)
        random.Random(seed).shuffle(pool)
        slices: list[list[Sample]] = []
        cursor = 0
        for count in global_counts:
            slices.append(pool[cursor : cursor + count])
            cursor += count
        return SplitDataset(train=slices[0], val=slices[1], test=slices[2])

    groups: dict[tuple[int, ...], list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.class_ids, []).append(sample)

    remaining = {key: len(bucket) for key, bucket in groups.items()}
    cursors = {key: 0 for key in groups}
    for key, bucket in groups.items():
        random.Random(_group_seed(seed, key)).shuffle(bucket)

    partitions: list[list[Sample]] = []
    for count in global_counts:
        take = _distribute(count, remaining)
        part: list[Sample] = []
        for key, qty in take.items():
            bucket = groups[key]
            cursor = cursors[key]
            part.extend(bucket[cursor : cursor + qty])
            cursors[key] = cursor + qty
            remaining[key] -= qty
        partitions.append(part)

    return SplitDataset(
        train=partitions[0],
        val=partitions[1],
        test=partitions[2],
    )