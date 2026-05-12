"""Aggregation helpers for Phase 2 retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Dict, Sequence


@dataclass(frozen=True)
class RetrievalAggregate:
    accuracy: float
    stdev: float
    lower_ci: float
    upper_ci: float
    effective_n: int
    mean_gap: float
    mean_entropy: float
    mean_energy: float
    mean_top_score: float
    cap_coverage_error: Dict[float, float]
    metastable_rate: float


def summarize_binary_outcomes(
    outcomes: Sequence[int],
    gaps: Sequence[float],
    entropies: Sequence[float],
    energies: Sequence[float],
    top_scores: Sequence[float],
) -> RetrievalAggregate:
    n = len(outcomes)
    if n == 0:
        raise ValueError("cannot summarize zero outcomes")
    accuracy = sum(outcomes) / n
    lower_ci, upper_ci = wilson_interval(sum(outcomes), n)
    return RetrievalAggregate(
        accuracy=accuracy,
        stdev=pstdev(outcomes) if n > 1 else 0.0,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        effective_n=n,
        mean_gap=mean(gaps) if gaps else 0.0,
        mean_entropy=mean(entropies) if entropies else 0.0,
        mean_energy=mean(energies) if energies else 0.0,
        mean_top_score=mean(top_scores) if top_scores else 0.0,
        cap_coverage_error={theta: 1.0 - cap_coverage(top_scores, theta) for theta in (0.3, 0.5, 0.7)},
        metastable_rate=meta_stable_rate(top_scores, threshold=0.95),
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


def cap_coverage(top_scores: Sequence[float], threshold: float) -> float:
    if not top_scores:
        return 0.0
    return sum(score >= threshold for score in top_scores) / len(top_scores)


def meta_stable_rate(top_scores: Sequence[float], threshold: float = 0.95) -> float:
    if not top_scores:
        return 0.0
    return sum(score < threshold for score in top_scores) / len(top_scores)


def build_frequency_buckets(counts: Dict[str, int]) -> Dict[str, str]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return {}
    bucket_size = max(1, math.ceil(len(ranked) / 4))
    labels = ["q1_most_frequent", "q2", "q3", "q4_least_frequent"]
    buckets: Dict[str, str] = {}
    for index, (token, _) in enumerate(ranked):
        bucket_index = min(index // bucket_size, len(labels) - 1)
        buckets[token] = labels[bucket_index]
    return buckets
