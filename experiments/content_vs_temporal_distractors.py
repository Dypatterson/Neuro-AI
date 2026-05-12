"""Temporal association vs content-similarity distractors.

Run:
    PYTHONPATH=src python3 experiments/content_vs_temporal_distractors.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

from energy_memory.diagnostics import temporal_association_score
from energy_memory.memory import TemporalAssociationMemory
from energy_memory.substrate import FHRR, Vector


STREAM = [
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


def make_vectors(substrate: FHRR, labels: Sequence[str], family_noise: float) -> Dict[str, Vector]:
    vectors = {label: substrate.random_vector() for label in labels}
    family_base = substrate.random_vector()
    for label in MOBILITY_FAMILY:
        vectors[label] = substrate.perturb(family_base, noise=family_noise)
    return vectors


def expected_neighbors(stream: Sequence[str], index: int, window: int) -> set[str]:
    return {
        stream[j]
        for j in range(max(0, index - window), min(len(stream), index + window + 1))
        if j != index
    }


def content_neighbors(substrate: FHRR, query_label: str, vectors: Dict[str, Vector], k: int) -> List[tuple[str, float]]:
    codebook = {label: vector for label, vector in vectors.items() if label != query_label}
    return substrate.top_k(vectors[query_label], codebook, k=k)


def run(seed: int = 11, dim: int = 512, window: int = 2, beta: float = 100.0, family_noise: float = 0.18):
    substrate = FHRR(dim=dim, seed=seed)
    vectors = make_vectors(substrate, STREAM, family_noise=family_noise)
    memory = TemporalAssociationMemory[str](substrate, window=window)
    memory.store_sequence(STREAM, [vectors[label] for label in STREAM])

    query = "stair"
    query_index = STREAM.index(query)
    expected = expected_neighbors(STREAM, query_index, window)
    k = window * 2
    content_top = content_neighbors(substrate, query, vectors, k=k)
    temporal = memory.recall(vectors[query], beta=beta, top_k=k)

    content_score = temporal_association_score(expected, content_top, k=k)
    temporal_score = temporal_association_score(expected, temporal.temporal_items, k=k)
    family_in_content = [label for label, _ in content_top if label in MOBILITY_FAMILY]
    family_in_temporal = [label for label, _ in temporal.temporal_items if label in MOBILITY_FAMILY]

    print(f"dim={dim} window={window} beta={beta} family_noise={family_noise}")
    print(f"query={query}")
    print(f"expected_temporal_neighbors={sorted(expected)}")
    print("content_nearest=[" + ", ".join(f"{label}:{score:.2f}" for label, score in content_top) + "]")
    print("temporal_recall=[" + ", ".join(f"{label}:{score:.2f}" for label, score in temporal.temporal_items) + "]")
    print(f"content temporal-neighbor Recall@{k}: {content_score:.3f}")
    print(f"temporal temporal-neighbor Recall@{k}: {temporal_score:.3f}")
    print(f"content mobility-family hits: {family_in_content}")
    print(f"temporal mobility-family hits: {family_in_temporal}")
    print(f"temporal advantage: {temporal_score - content_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--family-noise", type=float, default=0.18)
    args = parser.parse_args()
    run(seed=args.seed, dim=args.dim, window=args.window, beta=args.beta, family_noise=args.family_noise)


if __name__ == "__main__":
    main()
