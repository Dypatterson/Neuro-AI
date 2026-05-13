"""Coupled-recall benchmark: bag vs. permutation temporal encoding.

Brainstorm Idea 6 follow-through (2026-05-13): report 009 confirmed
that permutation-indexed slots recover the correct neighbor at the
correct signed offset on isolated contexts. This script tests the
broader claim — that the upgrade improves *coupled recall* under
content ambiguity, i.e. when only the temporal channel can resolve
which stored token is the right answer.

Stress design:

  * A *family* of K near-identical content vectors (perturbations of a
    shared base) is placed at K distinct positions in a longer random
    stream. Content similarity between any two family members is
    high (~ family_noise close to 1); content similarity to the rest
    of the vocabulary is near 0.

  * Coupled recall is cued with the shared family base as the content
    cue and the *directed context* of one specific family-member
    position as the temporal cue. The temporal channel is the only
    way to disambiguate which family member is being asked for.

  * The directed temporal cue is constructed from the **stream's true
    neighbors at that position**, encoded the same way the memory
    encodes its contexts (bag or permutation). This is a fair-fight
    measurement: each encoding gets the strongest cue available to it.

Headline metric: ``top1_correct`` — for each family-member position,
does coupled_recall pick that exact position as the top label?

Drill-downs:

  * mean entropy of the joint weight distribution (sharper is better
    for unambiguous targets)
  * mean rank of the true target (lower is better)
  * mean similarity at the chosen top vs. the true target

Control: a ``random_temporal`` cue (random FHRR vector). If either
encoding's gain is real, it must come from the temporal channel
carrying directional information, so randomizing the temporal cue
should collapse both encodings to chance ≈ 1/K.

Run:
    PYTHONPATH=src .venv/bin/python experiments/27_permutation_slots_coupled_recall.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from energy_memory.memory.torch_temporal import TorchTemporalAssociationMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class Row:
    encoding: str
    cue_type: str  # "directed" or "random"
    seed: int
    top1_correct: float
    mean_rank: float
    mean_entropy: float
    n_probes: int


def _build_stream(
    substrate: TorchFHRR,
    family_size: int,
    distractor_count: int,  # unused; kept for CLI compatibility
    family_noise: float,
    window: int,
    seed: int,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
    """Construct a directionality-stressing stream.

    The stream is built from ``family_size`` sub-windows, each of the form

        [n_{k,1}, ..., n_{k,window}, FAM_k, n_{k,window+1}, ..., n_{k,2*window}]

    where each ``n_{k,*}`` is one of a shared pool of ``2*window`` distractor
    atoms reused across every family member, placed in a different
    permutation for each member.

    Consequence:
      * The unordered multiset of FAM_k's neighbors is *identical* for
        every k. Therefore the bag context of every family member is
        identical (up to floating-point noise from the bundle normalize).
      * The signed multiset (atom, offset) is *unique* for every k.
        Therefore the permutation context distinguishes the members.

    This isolates the directionality channel as the only available
    disambiguator. The content cue (family base) is uninformative within
    the family, the bag temporal cue is uninformative across members,
    and only the permutation temporal cue carries enough signal to pick
    the right family member.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    family_base = substrate.random_vectors(1)[0]

    family_members = [
        substrate.perturb(family_base, noise=family_noise)
        for _ in range(family_size)
    ]

    pool_size = 2 * window
    distractor_pool = substrate.random_vectors(pool_size)
    pool_labels = [f"d{i}" for i in range(pool_size)]

    labels: List[str] = []
    vectors_list: List[torch.Tensor] = []
    family_indices: List[int] = []

    for k in range(family_size):
        perm = torch.randperm(pool_size, generator=rng).tolist()
        # Pre-anchor neighbors at offsets -window .. -1
        for off in range(-window, 0):
            slot = perm[off + window]
            labels.append(f"fam{k}_n{off}_{pool_labels[slot]}")
            vectors_list.append(distractor_pool[slot])
        # Anchor (family member)
        family_indices.append(len(labels))
        labels.append(f"fam{k}")
        vectors_list.append(family_members[k])
        # Post-anchor neighbors at offsets 1 .. window
        for off in range(1, window + 1):
            slot = perm[off + window - 1]
            labels.append(f"fam{k}_n{off}_{pool_labels[slot]}")
            vectors_list.append(distractor_pool[slot])

    return labels, torch.stack(vectors_list, dim=0), family_base, family_indices


def _directed_temporal_cue(
    substrate: TorchFHRR,
    vectors: torch.Tensor,
    index: int,
    window: int,
    encoding: str,
) -> torch.Tensor:
    """Build a temporal cue matching the encoding the memory uses."""
    lo = max(0, index - window)
    hi = min(vectors.shape[0], index + window + 1)
    terms = []
    for j in range(lo, hi):
        if j == index:
            continue
        if encoding == "bag":
            terms.append(vectors[j])
        else:
            terms.append(substrate.permute(vectors[j], j - index))
    if not terms:
        return vectors[index]
    return substrate.bundle(terms)


def _evaluate(
    substrate: TorchFHRR,
    labels: List[str],
    vectors: torch.Tensor,
    family_base: torch.Tensor,
    family_indices: List[int],
    *,
    encoding: str,
    cue_type: str,
    window: int,
    content_beta: float,
    temporal_beta: float,
    cue_seed: int,
) -> Row:
    memory = TorchTemporalAssociationMemory(substrate, window=window, encoding=encoding)
    memory.store_sequence(labels, [vectors[i] for i in range(vectors.shape[0])])

    rng = torch.Generator(device="cpu").manual_seed(cue_seed)

    correct = 0
    ranks: List[int] = []
    entropies: List[float] = []
    for idx in family_indices:
        if cue_type == "directed":
            temporal_cue = _directed_temporal_cue(
                substrate, vectors, idx, window, encoding
            )
        elif cue_type == "random":
            # Random unit-magnitude FHRR vector.
            phase = torch.rand(substrate.dim, generator=rng, device="cpu") * (2.0 * 3.141592653589793)
            temporal_cue = torch.polar(torch.ones(substrate.dim, device="cpu"), phase).to(substrate.device)
        else:
            raise ValueError(cue_type)

        result = memory.coupled_recall(
            family_base,
            temporal_cue,
            content_beta=content_beta,
            temporal_beta=temporal_beta,
            feedback=0.75,
            max_iter=8,
            top_k=len(labels),
        )
        # Did the chosen top label correspond to position `idx`?
        if result.top_label == labels[idx]:
            correct += 1

        # Rank of the true target in coupled_recall's temporal_items.
        all_labels = [item[0] for item in result.temporal_items]
        if labels[idx] in all_labels:
            ranks.append(all_labels.index(labels[idx]))
        else:
            ranks.append(len(all_labels))

        # Final-step entropy from the trace.
        entropies.append(result.trace[-1].entropy)

    n = len(family_indices)
    return Row(
        encoding=encoding,
        cue_type=cue_type,
        seed=cue_seed,
        top1_correct=correct / max(n, 1),
        mean_rank=statistics.fmean(ranks) if ranks else float("nan"),
        mean_entropy=statistics.fmean(entropies) if entropies else float("nan"),
        n_probes=n,
    )


def run(
    *,
    dim: int,
    family_size: int,
    distractor_count: int,
    family_noise: float,
    window: int,
    content_beta: float,
    temporal_beta: float,
    seeds: List[int],
) -> List[Row]:
    rows: List[Row] = []
    for seed in seeds:
        substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")
        labels, vectors, family_base, family_indices = _build_stream(
            substrate,
            family_size=family_size,
            distractor_count=distractor_count,
            family_noise=family_noise,
            window=window,
            seed=seed + 100,
        )
        for encoding in ("bag", "permutation"):
            for cue_type in ("directed", "random"):
                rows.append(
                    _evaluate(
                        substrate,
                        labels,
                        vectors,
                        family_base,
                        family_indices,
                        encoding=encoding,
                        cue_type=cue_type,
                        window=window,
                        content_beta=content_beta,
                        temporal_beta=temporal_beta,
                        cue_seed=seed + 200,
                    )
                )
    return rows


def _aggregate(rows: List[Row]):
    """Pool across seeds for each (encoding, cue_type) pair."""
    grouped = {}
    for r in rows:
        key = (r.encoding, r.cue_type)
        grouped.setdefault(key, []).append(r)

    print(f"{'encoding':<12} {'cue':<10} {'top1':>10} {'rank':>8} {'entropy':>9} {'n_seeds':>8}")
    print("-" * 60)
    summary = []
    for (enc, cue), group in grouped.items():
        top1 = statistics.fmean(r.top1_correct for r in group)
        rank = statistics.fmean(r.mean_rank for r in group)
        ent = statistics.fmean(r.mean_entropy for r in group)
        print(f"{enc:<12} {cue:<10} {top1:>10.3f} {rank:>8.2f} {ent:>9.3f} {len(group):>8}")
        summary.append({
            "encoding": enc, "cue_type": cue,
            "top1_mean": top1, "rank_mean": rank, "entropy_mean": ent,
            "n_seeds": len(group),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--family-size", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=25)
    parser.add_argument("--family-noise", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--content-beta", type=float, default=80.0)
    parser.add_argument("--temporal-beta", type=float, default=8.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41, 53, 67])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = run(
        dim=args.dim,
        family_size=args.family_size,
        distractor_count=args.distractors,
        family_noise=args.family_noise,
        window=args.window,
        content_beta=args.content_beta,
        temporal_beta=args.temporal_beta,
        seeds=args.seeds,
    )

    print(f"{'encoding':<12} {'cue':<10} {'seed':>6} {'top1':>8} {'rank':>8} {'entropy':>9}")
    print("-" * 56)
    for r in rows:
        print(f"{r.encoding:<12} {r.cue_type:<10} {r.seed:>6} {r.top1_correct:>8.3f} {r.mean_rank:>8.2f} {r.mean_entropy:>9.3f}")

    print()
    summary = _aggregate(rows)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "rows": [asdict(r) for r in rows],
            "summary": summary,
            "config": vars(args) | {"out": str(args.out)},
        }, indent=2))
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
