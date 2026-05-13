"""Reconstruction characterization (Phase 3, post-018 follow-up).

[Report 018](../reports/018_phase3_synergy_comparison.md) showed
that all three learned codebooks (Hebbian, reconstruction,
error_driven) produce ~125x higher settled synergy at the masked
slot than random, but reconstruction and error_driven have Recall@1
*below* random (0.043 / 0.031 vs. random 0.084). The report
hypothesized that these codebooks merge co-occurring atoms more
aggressively, making the right answer's neighborhood structurally
clearer but pushing the argmax winner toward the neighbors. That
hypothesis was a story, not a measurement.

This script tests the hypothesis directly across four axes:

  1. **Atom-pair geometry.** Pairwise cosine similarity distribution
     across the entire 2050-atom codebook for each artifact.
     Random should sit tightly around 0; learned codebooks should
     show heavier upper tails. Reconstruction/error_driven should
     have heavier tails than Hebbian if the merging hypothesis is
     right.

  2. **Recall@K for K in {1, 5, 10, 20, 50}.** Under the
     over-merging hypothesis, the right answer should rise sharply
     in the top-K for learned codebooks even when top-1 misses.
     Specifically: reconstruction should show large Recall@K gains
     at K >= 5 over random, even though Recall@1 regresses.

  3. **Target-rank distribution.** Beyond binary Recall@K, the
     median (and mean) rank of the true target in the full
     decoded list. Lower rank = closer to the argmax.

  4. **Frequency-bucketed settled synergy.** Does each codebook
     help rare tokens more than frequent ones, or vice versa? The
     hypothesis: reconstruction's neighborhood-merging should help
     rare tokens (they get pulled into their context's cluster)
     and hurt frequent tokens (function words become confusable).

Run:
    PYTHONPATH=src .venv/bin/python experiments/33_reconstruction_characterization.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.diagnostics.synergy import synergy_score
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window,
    mask_positions as mask_positions_for_condition,
    masked_window,
)
from energy_memory.phase2.metrics import build_frequency_buckets, wilson_interval
from energy_memory.phase2.persistence import load_codebook, load_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


CODEBOOKS = [
    ("random",         "reports/phase2_full_matrix/phase2_codebook.pt"),
    ("hebbian",        "reports/phase3c_reconstruction/phase3c_codebook_hebbian.pt"),
    ("reconstruction", "reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt"),
    ("error_driven",   "reports/phase3b_error_driven/phase3b_codebook_error_driven.pt"),
]

K_VALUES = [1, 5, 10, 20, 50]


@dataclass
class WindowRecord:
    label: str
    seed: int
    target_token: int
    target_bucket: str
    target_rank: int  # 0 = top-1 correct; -1 = not in top-K_max
    settled_synergy_at_mask: float


def _atom_similarity_profile(
    substrate: TorchFHRR,
    codebook: "torch.Tensor",
    decode_ids: Sequence[int],
    sample_pairs: int = 200_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Sample pairwise cosine similarities across decodable atoms.

    Full N x N for N=2050 is ~4M pairs at complex64 = 32 MB and a
    single matmul. Tractable but we only need a representative
    sample for distribution stats. Sample without replacement.
    """
    atoms = codebook[torch.tensor(list(decode_ids), device=codebook.device)]
    n = atoms.shape[0]

    rng = torch.Generator(device="cpu").manual_seed(seed)
    # Sample upper-triangular pairs (i, j) with i < j.
    max_pairs = n * (n - 1) // 2
    n_sample = min(sample_pairs, max_pairs)
    sims_collected: List[float] = []
    # Sample pairs in batches.
    target = n_sample
    while len(sims_collected) < target:
        batch = min(target - len(sims_collected), 50_000)
        i = torch.randint(0, n, (batch,), generator=rng)
        j = torch.randint(0, n, (batch,), generator=rng)
        # Reject self-pairs.
        keep = i != j
        i = i[keep]; j = j[keep]
        # Compute sims as the mean of real(conj(a) * b) per pair.
        a = atoms[i].cpu()
        b = atoms[j].cpu()
        # FHRR similarity = mean(real(conj(a) * b))
        s = (a.conj() * b).real.mean(dim=-1)
        sims_collected.extend(s.tolist())

    sims_collected = sims_collected[:target]
    sims_tensor = torch.tensor(sims_collected, dtype=torch.float64)
    return {
        "n_pairs": len(sims_collected),
        "mean": float(sims_tensor.mean().item()),
        "std": float(sims_tensor.std(unbiased=False).item()),
        "p05": float(torch.quantile(sims_tensor, 0.05).item()),
        "p50": float(torch.quantile(sims_tensor, 0.50).item()),
        "p95": float(torch.quantile(sims_tensor, 0.95).item()),
        "p99": float(torch.quantile(sims_tensor, 0.99).item()),
        "p999": float(torch.quantile(sims_tensor, 0.999).item()),
        "max": float(sims_tensor.max().item()),
    }


def _decode_with_rank(
    substrate: TorchFHRR,
    state: "torch.Tensor",
    position: "torch.Tensor",
    codebook: "torch.Tensor",
    decode_ids: Sequence[int],
    target_token: int,
) -> int:
    """Return the rank of target_token in the full decoded list.

    0 = top-1 correct, N-1 = bottom of list, -1 if target not in
    decode_ids (shouldn't happen given our filter).
    """
    slot_query = substrate.unbind(state, position)
    candidate_matrix = codebook[
        torch.tensor(list(decode_ids), device=codebook.device)
    ]
    scores = substrate.similarity_matrix(slot_query, candidate_matrix)
    # Argsort descending: highest score first
    sorted_idx = torch.argsort(scores, descending=True).cpu().tolist()
    decode_list = list(decode_ids)
    try:
        target_position_in_decode = decode_list.index(target_token)
    except ValueError:
        return -1
    # rank = position of target_position_in_decode in sorted_idx
    for rank, idx in enumerate(sorted_idx):
        if idx == target_position_in_decode:
            return rank
    return -1


def _evaluate_codebook(
    *,
    label: str,
    codebook_path: Path,
    seed: int,
    repo_root: Path,
    dim: int,
    device: str,
    window_size: int,
    landscape_size: int,
    beta: float,
    test_samples: int,
    mask_position: str,
    wikitext_name: str,
    frequency_buckets: Optional[Dict[str, str]] = None,
) -> List[WindowRecord]:
    substrate = TorchFHRR(dim=dim, seed=seed, device=device)

    vocab_files = list(codebook_path.parent.glob("*vocab*.json"))
    if not vocab_files:
        return []
    vocab = load_vocabulary(vocab_files[0])

    splits = load_corpus_splits("wikitext", repo_root, wikitext_name=wikitext_name)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    train_windows = make_windows(train_ids, window_size)
    validation_windows = make_windows(validation_ids, window_size)
    if not train_windows or not validation_windows:
        return []

    codebook = load_codebook(codebook_path, device=str(substrate.device))
    if codebook.shape[0] != len(vocab.id_to_token) or codebook.shape[1] != dim:
        return []

    positions = build_position_vectors(substrate, window_size)
    decode_ids = [
        index
        for index, token in enumerate(vocab.id_to_token)
        if token not in {vocab.unk_token, vocab.mask_token}
    ]
    if frequency_buckets is None:
        frequency_buckets = build_frequency_buckets(vocab.counts)

    # Matrix slice alignment (W=8 index=1, L=64 index=0).
    matrix_window_index = 1
    matrix_landscape_index = 0
    slice_seed = seed + 10000 * matrix_window_index + 1000 * matrix_landscape_index
    landscape_windows = sample_windows(train_windows, landscape_size, seed=slice_seed)
    generalization_windows = sample_windows(
        validation_windows,
        min(test_samples, len(validation_windows)),
        seed=seed + 10000 * matrix_window_index + 2,
    )

    memory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(landscape_windows):
        memory.store(
            encode_window(substrate, positions, codebook, window),
            label=f"window_{index}",
        )

    masked_positions = mask_positions_for_condition(
        window_size, mask_count=1, position_kind=mask_position
    )
    mask_idx = masked_positions[0]

    records: List[WindowRecord] = []
    for window in generalization_windows:
        target_token = window[mask_idx]
        if target_token == vocab.unk_id:
            continue
        cue_window = masked_window(window, [mask_idx], vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        rank = _decode_with_rank(
            substrate,
            result.state,
            positions[mask_idx],
            codebook,
            decode_ids,
            target_token,
        )
        settled = synergy_score(
            substrate,
            positions[mask_idx],
            codebook[target_token],
            binding=result.state,
        ).synergy
        bucket = frequency_buckets.get(vocab.decode_token(target_token), "unknown")
        records.append(WindowRecord(
            label=label,
            seed=seed,
            target_token=target_token,
            target_bucket=bucket,
            target_rank=rank,
            settled_synergy_at_mask=settled,
        ))
    return records


def _aggregate(records: List[WindowRecord]) -> Dict[str, Dict]:
    """Aggregate by codebook label."""
    by_label: Dict[str, List[WindowRecord]] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)

    out: Dict[str, Dict] = {}
    for label, group in by_label.items():
        n = len(group)
        # Recall@K for K in K_VALUES
        recall_at_k = {}
        for k in K_VALUES:
            hits = sum(1 for r in group if 0 <= r.target_rank < k)
            lo, hi = wilson_interval(hits, n)
            recall_at_k[k] = {
                "value": hits / n,
                "ci_lower": lo,
                "ci_upper": hi,
            }
        # Rank stats over the records where target was found (rank >= 0)
        ranks = [r.target_rank for r in group if r.target_rank >= 0]
        rank_stats = {
            "n_in_topK": len(ranks),
            "mean": statistics.fmean(ranks) if ranks else float("nan"),
            "median": statistics.median(ranks) if ranks else float("nan"),
            "p25": statistics.quantiles(ranks, n=4)[0] if len(ranks) >= 4 else float("nan"),
            "p75": statistics.quantiles(ranks, n=4)[2] if len(ranks) >= 4 else float("nan"),
        }
        # Settled synergy by frequency bucket
        by_bucket: Dict[str, List[WindowRecord]] = defaultdict(list)
        for r in group:
            by_bucket[r.target_bucket].append(r)
        bucket_summary: Dict[str, Dict[str, float]] = {}
        for bucket, brs in by_bucket.items():
            ss = [r.settled_synergy_at_mask for r in brs]
            hits_top1 = sum(1 for r in brs if 0 <= r.target_rank < 1)
            hits_top5 = sum(1 for r in brs if 0 <= r.target_rank < 5)
            bucket_summary[bucket] = {
                "n": len(brs),
                "settled_synergy_mean": statistics.fmean(ss),
                "recall_at_1": hits_top1 / len(brs),
                "recall_at_5": hits_top5 / len(brs),
            }
        out[label] = {
            "n": n,
            "recall_at_k": recall_at_k,
            "target_rank_stats": rank_stats,
            "by_frequency_bucket": bucket_summary,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/Users/dypatterson/Desktop/Neuro-AI"))
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps"])
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--mask-position", type=str, default="end")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 1, 2, 3, 4, 5])
    parser.add_argument("--wikitext-name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--atom-sample-pairs", type=int, default=200_000)
    parser.add_argument(
        "--out-csv", type=Path,
        default=Path("reports/reconstruction_characterization.csv"),
    )
    args = parser.parse_args()

    # Build atom similarity profiles per codebook once (codebook-only, no seed).
    print("Atom similarity profiles (sample of pairs):")
    print(f"{'label':<18} {'mean':>9} {'std':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'p999':>9} {'max':>9}")
    atom_profiles: Dict[str, Dict] = {}
    for label, rel_path in CODEBOOKS:
        codebook_path = args.repo_root / rel_path
        if not codebook_path.exists():
            print(f"  [skip] {label}: missing")
            continue
        substrate = TorchFHRR(dim=args.dim, seed=99, device=args.device)
        vocab = load_vocabulary(next(codebook_path.parent.glob("*vocab*.json")))
        decode_ids = [
            i for i, t in enumerate(vocab.id_to_token)
            if t not in {vocab.unk_token, vocab.mask_token}
        ]
        codebook = load_codebook(codebook_path, device=str(substrate.device))
        profile = _atom_similarity_profile(
            substrate, codebook, decode_ids,
            sample_pairs=args.atom_sample_pairs, seed=11,
        )
        atom_profiles[label] = profile
        print(
            f"{label:<18} "
            f"{profile['mean']:>9.4f} {profile['std']:>9.4f} "
            f"{profile['p50']:>9.4f} {profile['p95']:>9.4f} "
            f"{profile['p99']:>9.4f} {profile['p999']:>9.4f} "
            f"{profile['max']:>9.4f}"
        )
    print()

    # Build frequency buckets once from the random codebook's vocab (all
    # learned codebooks share the same vocab; verified in report 017).
    random_path = args.repo_root / CODEBOOKS[0][1]
    vocab = load_vocabulary(next(random_path.parent.glob("*vocab*.json")))
    frequency_buckets = build_frequency_buckets(vocab.counts)

    all_records: List[WindowRecord] = []
    for label, rel_path in CODEBOOKS:
        codebook_path = args.repo_root / rel_path
        if not codebook_path.exists():
            continue
        for seed in args.seeds:
            records = _evaluate_codebook(
                label=label,
                codebook_path=codebook_path,
                seed=seed,
                repo_root=args.repo_root,
                dim=args.dim,
                device=args.device,
                window_size=args.window_size,
                landscape_size=args.landscape_size,
                beta=args.beta,
                test_samples=args.test_samples,
                mask_position=args.mask_position,
                wikitext_name=args.wikitext_name,
                frequency_buckets=frequency_buckets,
            )
            if not records:
                continue
            all_records.extend(records)
            top1 = sum(1 for r in records if 0 <= r.target_rank < 1) / len(records)
            top5 = sum(1 for r in records if 0 <= r.target_rank < 5) / len(records)
            top20 = sum(1 for r in records if 0 <= r.target_rank < 20) / len(records)
            print(
                f"{label:<18} seed={seed:>3}  n={len(records):>4}  "
                f"R@1={top1:.3f}  R@5={top5:.3f}  R@20={top20:.3f}"
            )

    aggregates = _aggregate(all_records)

    print()
    print("Per-codebook Recall@K (pooled across seeds):")
    header = f"{'label':<18} " + "".join(f"{'R@'+str(k):>14}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for label, agg in aggregates.items():
        row = f"{label:<18}"
        for k in K_VALUES:
            v = agg["recall_at_k"][k]
            row += f" {v['value']:.3f}[{v['ci_lower']:.3f},{v['ci_upper']:.3f}]"
        print(row)

    print()
    print("Per-codebook target-rank stats:")
    print(f"{'label':<18} {'mean':>10} {'median':>10} {'p25':>10} {'p75':>10}")
    print("-" * 60)
    for label, agg in aggregates.items():
        s = agg["target_rank_stats"]
        print(
            f"{label:<18} {s['mean']:>10.1f} {s['median']:>10.1f} "
            f"{s['p25']:>10.1f} {s['p75']:>10.1f}"
        )

    print()
    print("Settled synergy by frequency bucket (pooled across seeds):")
    bucket_order = ["q1_most_frequent", "q2", "q3", "q4_least_frequent"]
    header = f"{'label':<18} " + "".join(f"{b:>20}" for b in bucket_order)
    print(header)
    print("-" * len(header))
    for label, agg in aggregates.items():
        row = f"{label:<18}"
        for bucket in bucket_order:
            b = agg["by_frequency_bucket"].get(bucket)
            if b is None:
                row += f" {'--':>19}"
            else:
                row += f"  syn={b['settled_synergy_mean']:.3f} n={b['n']:>3d}"
        print(row)

    print()
    print("Recall@1 by frequency bucket (pooled across seeds):")
    print(header)
    print("-" * len(header))
    for label, agg in aggregates.items():
        row = f"{label:<18}"
        for bucket in bucket_order:
            b = agg["by_frequency_bucket"].get(bucket)
            if b is None:
                row += f" {'--':>19}"
            else:
                row += f"  R@1={b['recall_at_1']:.3f} n={b['n']:>3d}"
        print(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "seed", "target_token", "target_bucket",
            "target_rank", "settled_synergy_at_mask",
        ])
        for r in all_records:
            w.writerow([
                r.label, r.seed, r.target_token, r.target_bucket,
                r.target_rank, f"{r.settled_synergy_at_mask:.6f}",
            ])

    with args.out_csv.with_suffix(".json").open("w") as f:
        json.dump({
            "config": vars(args) | {"repo_root": str(args.repo_root), "out_csv": str(args.out_csv)},
            "atom_profiles": atom_profiles,
            "aggregates": aggregates,
        }, f, indent=2, default=str)
    print(f"\nWrote {args.out_csv} and {args.out_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
