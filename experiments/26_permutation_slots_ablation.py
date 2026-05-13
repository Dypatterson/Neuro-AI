"""Temporal-shuffle ablation: bag context vs. permutation-indexed slots.

Brainstorm Idea 6 (2026-05-13): unordered context bags commit two
failure modes (temporal inaccuracy, temporal fragmentation). The
discrete permutation-indexed slot encoding eliminates both by
construction. This script measures the size of that improvement on
random FHRR sequences.

Headline metric: ``offset_accuracy`` -- given a context, recover the
correct neighbor's label at each of {-2, -1, +1, +2}. The bag-based
memory can only return *some* neighbor; the permutation memory must
return the right one for the requested signed offset.

Control: shuffle the sequence positions before encoding the contexts.
If permutation slots are doing real temporal work, accuracy should
collapse to chance under shuffling; if the apparent gain is just
content-similarity, accuracy should be similar to unshuffled.

Run:
    PYTHONPATH=src .venv/bin/python experiments/26_permutation_slots_ablation.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch

from energy_memory.memory.torch_temporal_slots import PermutationSlotTemporalMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class Row:
    encoding: str  # "bag" or "permutation"
    condition: str  # "ordered" or "shuffled"
    offset: int
    accuracy: float
    n_probes: int


def _bag_context(substrate: TorchFHRR, vectors, index: int, window: int):
    lo = max(0, index - window)
    hi = min(vectors.shape[0], index + window + 1)
    neighbors = [vectors[j] for j in range(lo, hi) if j != index]
    if not neighbors:
        return vectors[index]
    return substrate.bundle(neighbors)


def _bag_query(substrate, context, vocab_vectors, vocab_labels, top_k: int = 1):
    sims = substrate.similarity_matrix(context, vocab_vectors)
    values, indices = torch.topk(sims, min(top_k, len(vocab_labels)))
    idx = int(indices[0].detach().cpu())
    return vocab_labels[idx]


def _evaluate(
    substrate,
    vectors,
    labels: List[str],
    *,
    window: int,
    condition: str,
    seed: int,
):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    if condition == "shuffled":
        perm = torch.randperm(len(labels), generator=rng).tolist()
        vectors_seq = vectors[perm]
        labels_seq = [labels[p] for p in perm]
    elif condition == "ordered":
        vectors_seq = vectors
        labels_seq = list(labels)
    else:
        raise ValueError(condition)

    # Bag memory: bundle of raw neighbors.
    bag_contexts = []
    for i in range(len(labels_seq)):
        bag_contexts.append(_bag_context(substrate, vectors_seq, i, window))
    bag_contexts = torch.stack(bag_contexts, dim=0)

    # Permutation memory: bundle of permuted neighbors.
    perm_mem = PermutationSlotTemporalMemory(substrate, window=window)
    perm_mem.store_sequence(labels_seq, [vectors_seq[i] for i in range(len(labels_seq))])

    offsets = list(range(-window, window + 1))
    offsets = [k for k in offsets if k != 0]
    rows: List[Row] = []

    for offset in offsets:
        # For each anchor index that has a valid neighbor at this offset,
        # probe and count whether the recovered label matches.
        bag_hits = 0
        perm_hits = 0
        n_probes = 0
        for i in range(len(labels_seq)):
            j = i + offset
            if j < 0 or j >= len(labels_seq):
                continue
            true_label = labels_seq[j]
            # Bag probe: nearest neighbor of the context against vocab.
            bag_pred = _bag_query(
                substrate, bag_contexts[i], vectors_seq, labels_seq, top_k=1
            )
            if bag_pred == true_label:
                bag_hits += 1
            # Permutation probe: ask for the specific offset.
            perm_pred = perm_mem.query_offset(
                perm_mem.context_for(i), offset=offset, top_k=1
            ).top_label
            if perm_pred == true_label:
                perm_hits += 1
            n_probes += 1

        denom = max(n_probes, 1)
        rows.append(Row("bag", condition, offset, bag_hits / denom, n_probes))
        rows.append(Row("permutation", condition, offset, perm_hits / denom, n_probes))
    return rows


def run(dim: int, length: int, window: int, seed: int):
    substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")
    vectors = substrate.random_vectors(length)
    labels = [f"tok_{i}" for i in range(length)]
    rows = []
    rows.extend(_evaluate(substrate, vectors, labels, window=window, condition="ordered", seed=seed))
    rows.extend(_evaluate(substrate, vectors, labels, window=window, condition="shuffled", seed=seed + 1))
    return rows


def _print(rows):
    header = f"{'encoding':<12} {'cond':<10} {'offset':>6} {'acc':>8} {'n':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.encoding:<12} {r.condition:<10} {r.offset:>6d} {r.accuracy:>8.3f} {r.n_probes:>5d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = run(args.dim, args.length, args.window, args.seed)
    _print(rows)

    # Summaries: aggregate accuracy by (encoding, condition).
    print()
    agg = {}
    for r in rows:
        key = (r.encoding, r.condition)
        agg.setdefault(key, []).append(r.accuracy)
    print(f"{'encoding':<12} {'cond':<10} {'mean_acc':>10}")
    print("-" * 36)
    for (enc, cond), values in agg.items():
        print(f"{enc:<12} {cond:<10} {sum(values) / len(values):>10.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
