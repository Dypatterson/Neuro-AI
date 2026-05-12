"""Synthetic worlds used by the MVP 0 experiment scripts."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from energy_memory.diagnostics import temporal_association_score
from energy_memory.memory import TemporalAssociationMemory
from energy_memory.substrate import FHRR, Vector

TEMPORAL_STREAM = [
    "candle",
    "table",
    "book",
    "ink",
    "window",
    "stair",
    "slip",
    "doctor",
    "bandage",
    "crutch",
    "rain",
    "windowpane",
    "letter",
    "lamp",
    "silence",
    "garden",
    "gate",
    "key",
    "path",
    "bell",
]

DISTRACTOR_STREAM = [
    "candle",
    "table",
    "stair",
    "slip",
    "doctor",
    "bandage",
    "garden",
    "ladder",
    "paint",
    "window",
    "rain",
    "ramp",
    "cart",
    "letter",
    "step",
    "dance",
    "music",
    "escalator",
    "mall",
    "coffee",
]

MOBILITY_FAMILY = ["stair", "ladder", "ramp", "step", "escalator"]


def expected_neighbors(stream: Sequence[str], index: int, window: int) -> set[str]:
    return {
        stream[j]
        for j in range(max(0, index - window), min(len(stream), index + window + 1))
        if j != index
    }


def random_vectors(substrate: FHRR, labels: Sequence[str]) -> Dict[str, Vector]:
    return {label: substrate.random_vector() for label in labels}


def distractor_vectors(substrate: FHRR, labels: Sequence[str], family_noise: float) -> Dict[str, Vector]:
    vectors = random_vectors(substrate, labels)
    family_base = substrate.random_vector()
    for label in MOBILITY_FAMILY:
        vectors[label] = substrate.perturb(family_base, noise=family_noise)
    return vectors


def build_memory(
    substrate: FHRR,
    stream: Sequence[str],
    vectors: Dict[str, Vector],
    window: int,
    shuffle: bool = False,
    seed: int = 0,
) -> TemporalAssociationMemory[str]:
    labels = list(stream)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(labels)
    memory = TemporalAssociationMemory[str](substrate, window=window)
    memory.store_sequence(labels, [vectors[label] for label in labels])
    return memory


def mean_temporal_recall(
    substrate: FHRR,
    stream: Sequence[str],
    vectors: Dict[str, Vector],
    memory: TemporalAssociationMemory[str],
    window: int,
    beta: float,
) -> tuple[float, float, float]:
    scores = []
    entropies = []
    top_scores = []
    k = window * 2
    for index, label in enumerate(stream):
        result = memory.recall(vectors[label], beta=beta, top_k=k)
        expected = expected_neighbors(stream, index, window)
        scores.append(temporal_association_score(expected, result.temporal_items, k=k))
        entropies.append(result.content.entropy)
        top_scores.append(result.content.top_score)
    return (
        sum(scores) / len(scores),
        sum(entropies) / len(entropies),
        sum(top_scores) / len(top_scores),
    )


def content_neighbors(substrate: FHRR, query_label: str, vectors: Dict[str, Vector], k: int) -> List[tuple[str, float]]:
    codebook = {label: vector for label, vector in vectors.items() if label != query_label}
    return substrate.top_k(vectors[query_label], codebook, k=k)

