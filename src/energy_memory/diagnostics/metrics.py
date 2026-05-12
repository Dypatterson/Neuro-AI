"""Small metrics for recall experiments."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def recall_at_k(expected: T, ranked_items: Iterable[tuple[T, float]], k: int) -> float:
    return 1.0 if expected in [item for item, _ in list(ranked_items)[:k]] else 0.0


def temporal_association_score(expected: set[T], ranked_items: Iterable[tuple[T, float]], k: int) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for item, _ in list(ranked_items)[:k] if item in expected)
    return hits / min(k, len(expected))

