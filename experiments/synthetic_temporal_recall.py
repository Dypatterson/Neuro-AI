"""Synthetic temporal association experiment with shuffle control.

Run:
    PYTHONPATH=src python3 experiments/synthetic_temporal_recall.py
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Sequence

from energy_memory.diagnostics import temporal_association_score
from energy_memory.memory import TemporalAssociationMemory
from energy_memory.substrate import FHRR, Vector


ROOMS = {
    "room_a": ["candle", "table", "book", "ink", "window"],
    "room_b": ["stair", "slip", "doctor", "bandage", "crutch"],
    "room_c": ["rain", "windowpane", "letter", "lamp", "silence"],
    "room_d": ["garden", "gate", "key", "path", "bell"],
}


def build_stream() -> List[str]:
    stream: List[str] = []
    for room_items in ROOMS.values():
        stream.extend(room_items)
    return stream


def expected_neighbors(stream: Sequence[str], index: int, window: int) -> set[str]:
    return {
        stream[j]
        for j in range(max(0, index - window), min(len(stream), index + window + 1))
        if j != index
    }


def run(seed: int = 7, dim: int = 512, window: int = 2, beta: float = 8.0, shuffle: bool = False) -> Dict[str, float]:
    substrate = FHRR(dim=dim, seed=seed)
    stream = build_stream()
    labels = list(stream)
    if shuffle:
        rng = random.Random(seed + 1)
        labels = list(labels)
        rng.shuffle(labels)

    vectors = {label: substrate.random_vector() for label in stream}
    memory = TemporalAssociationMemory[str](substrate, window=window)
    memory.store_sequence(labels, [vectors[label] for label in labels])

    scores = []
    examples = []
    for index, label in enumerate(stream):
        result = memory.recall(vectors[label], beta=beta, top_k=window * 2)
        expected = expected_neighbors(stream, index, window)
        score = temporal_association_score(expected, result.temporal_items, k=window * 2)
        scores.append(score)
        if label in {"stair", "slip", "rain", "letter"}:
            examples.append((label, sorted(expected), result.temporal_items[:5], score))

    mean_score = sum(scores) / len(scores)
    print(f"shuffle={shuffle} dim={dim} window={window} beta={beta}")
    print(f"mean temporal Recall@{window * 2}: {mean_score:.3f}")
    for label, expected, retrieved, score in examples:
        retrieved_text = ", ".join(f"{item}:{sim:.2f}" for item, sim in retrieved)
        print(f"  {label:>6} expected={expected} retrieved=[{retrieved_text}] score={score:.2f}")
    return {"mean_temporal_recall": mean_score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--beta", type=float, default=8.0)
    args = parser.parse_args()

    normal = run(seed=args.seed, dim=args.dim, window=args.window, beta=args.beta, shuffle=False)
    print()
    shuffled = run(seed=args.seed, dim=args.dim, window=args.window, beta=args.beta, shuffle=True)
    print()
    delta = normal["mean_temporal_recall"] - shuffled["mean_temporal_recall"]
    print(f"temporal shuffle delta: {delta:.3f}")


if __name__ == "__main__":
    main()

