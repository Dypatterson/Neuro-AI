"""Native-W codebook evaluation: do per-scale codebooks actually retrieve?

Report 020 found that per-scale codebooks (W=2,3,4) have orthogonal
atom-pair geometry, while W=8 single-scale codebooks all collapse to
mean similarity ~0.41. The open question is whether per-scale
orthogonality reflects (A) a structurally better objective or
(B) under-training (53-71% quality-threshold failure rate during
training).

This experiment evaluates each candidate codebook at the native W
of the per-scale codebooks (W in {2,3,4}). For each W, we compare:

  * random_at_W       — fresh random vectors, dim=4096
  * hebbian_w8        — the W=8 Hebbian codebook (collapsed geometry)
  * reconstruction_w8 — the W=8 reconstruction codebook (collapsed)
  * per_scale_wK      — the per-scale codebook trained for this W

Metrics (matching report 019's framework):

  * Recall@K for K in {1, 5, 10, 20, 50}
  * Median / p25 / p75 target rank
  * Mean settled synergy at the masked slot
  * (atom-pair geometry already in report 020, not re-run here)

If per-scale codebooks beat random at their native W on R@K, they're
useful retrievers despite the low pairwise similarity — interpretation
(A). If they sit at random, interpretation (B) — orthogonal because
under-trained.

Run:
    PYTHONPATH=src .venv/bin/python experiments/35_native_w_codebook_eval.py
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
from energy_memory.phase2.metrics import wilson_interval
from energy_memory.phase2.persistence import load_codebook, load_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


K_VALUES = [1, 5, 10, 20, 50]


@dataclass
class Record:
    label: str           # e.g. "per_scale_w2"
    window_size: int     # 2, 3, or 4
    seed: int
    target_token: int
    target_rank: int     # -1 if not in top-rank_k
    settled_synergy: float


def _decode_with_rank(
    substrate: TorchFHRR,
    state: torch.Tensor,
    position: torch.Tensor,
    codebook: torch.Tensor,
    decode_ids: Sequence[int],
    target_token: int,
) -> int:
    slot_query = substrate.unbind(state, position)
    candidate_matrix = codebook[
        torch.tensor(list(decode_ids), device=codebook.device)
    ]
    scores = substrate.similarity_matrix(slot_query, candidate_matrix)
    sorted_idx = torch.argsort(scores, descending=True).cpu().tolist()
    decode_list = list(decode_ids)
    try:
        target_pos = decode_list.index(target_token)
    except ValueError:
        return -1
    for rank, idx in enumerate(sorted_idx):
        if idx == target_pos:
            return rank
    return -1


def _evaluate(
    *,
    label: str,
    codebook: torch.Tensor,
    vocab,
    substrate: TorchFHRR,
    window_size: int,
    seed: int,
    repo_root: Path,
    landscape_size: int,
    beta: float,
    test_samples: int,
    wikitext_name: str,
) -> List[Record]:
    splits = load_corpus_splits("wikitext", repo_root, wikitext_name=wikitext_name)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    train_windows = make_windows(train_ids, window_size)
    validation_windows = make_windows(validation_ids, window_size)
    if not train_windows or not validation_windows:
        return []

    positions = build_position_vectors(substrate, window_size)
    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    landscape = sample_windows(train_windows, landscape_size, seed=seed)
    test_pool = sample_windows(
        validation_windows,
        min(test_samples * 10, len(validation_windows)),
        seed=seed + 9000,
    )
    # Mask the last position
    mask_idx = window_size - 1

    test_windows = [
        w for w in test_pool
        if w[mask_idx] != vocab.unk_id
    ][:test_samples]

    memory = TorchHopfieldMemory[str](substrate)
    for i, w in enumerate(landscape):
        memory.store(
            encode_window(substrate, positions, codebook, w),
            label=f"w_{i}",
        )

    records: List[Record] = []
    for w in test_windows:
        target = w[mask_idx]
        cue_window = masked_window(w, [mask_idx], vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        rank = _decode_with_rank(
            substrate, result.state, positions[mask_idx],
            codebook, decode_ids, target,
        )
        settled = synergy_score(
            substrate, positions[mask_idx],
            codebook[target], binding=result.state,
        ).synergy
        records.append(Record(
            label=label, window_size=window_size, seed=seed,
            target_token=target, target_rank=rank,
            settled_synergy=float(settled),
        ))
    return records


def _summarize(records: List[Record]) -> Dict:
    by_key: Dict[Tuple[str, int], List[Record]] = defaultdict(list)
    for r in records:
        by_key[(r.label, r.window_size)].append(r)
    out = {}
    for (label, w), group in by_key.items():
        n = len(group)
        recall = {}
        for k in K_VALUES:
            hits = sum(1 for r in group if 0 <= r.target_rank < k)
            lo, hi = wilson_interval(hits, n)
            recall[k] = {"value": hits / n, "ci_lower": lo, "ci_upper": hi}
        ranks = [r.target_rank for r in group if r.target_rank >= 0]
        rank_stats = {
            "n_in_topK": len(ranks),
            "mean": statistics.fmean(ranks) if ranks else float("nan"),
            "median": statistics.median(ranks) if ranks else float("nan"),
            "p25": statistics.quantiles(ranks, n=4)[0] if len(ranks) >= 4 else float("nan"),
            "p75": statistics.quantiles(ranks, n=4)[2] if len(ranks) >= 4 else float("nan"),
        }
        syns = [r.settled_synergy for r in group]
        out[f"{label}@W{w}"] = {
            "label": label,
            "window_size": w,
            "n": n,
            "recall_at_k": recall,
            "target_rank_stats": rank_stats,
            "mean_settled_synergy": statistics.fmean(syns),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path("/Users/dypatterson/Desktop/Neuro-AI"))
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 1, 2, 3, 4, 5])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--out-csv", type=Path,
                        default=Path("reports/native_w_codebook_eval.csv"))
    args = parser.parse_args()

    # Per-scale codebooks share their own vocab (vocab.json in
    # phase4_per_scale). The W=8 reference codebooks share the
    # phase2/phase3 vocab. We use whichever vocab corresponds to the
    # codebook's own training pipeline.
    vocab_per_scale = load_vocabulary(args.repo_root / "reports/phase4_per_scale/vocab.json")
    vocab_w8_refs   = load_vocabulary(
        args.repo_root / "reports/phase3c_reconstruction/phase3c_vocab.json"
    )

    references = [
        # (label, codebook_path, vocab)
        ("hebbian_w8",        "reports/phase3c_reconstruction/phase3c_codebook_hebbian.pt",       vocab_w8_refs),
        ("reconstruction_w8", "reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt", vocab_w8_refs),
    ]
    per_scale = {
        2: ("per_scale_w2", "reports/phase4_per_scale/codebook_w2.pt", vocab_per_scale),
        3: ("per_scale_w3", "reports/phase4_per_scale/codebook_w3.pt", vocab_per_scale),
        4: ("per_scale_w4", "reports/phase4_per_scale/codebook_w4.pt", vocab_per_scale),
    }

    all_records: List[Record] = []
    for W in args.window_sizes:
        ps_entry = per_scale.get(W)
        print(f"\n=== W={W} ===")
        for seed in args.seeds:
            substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)

            # random_at_W (fresh per seed, same vocab as per-scale so decode_ids match)
            vocab_random = vocab_per_scale
            random_cb = substrate.random_vectors(len(vocab_random.id_to_token))
            records = _evaluate(
                label="random_at_W",
                codebook=random_cb, vocab=vocab_random, substrate=substrate,
                window_size=W, seed=seed, repo_root=args.repo_root,
                landscape_size=args.landscape_size, beta=args.beta,
                test_samples=args.test_samples, wikitext_name=args.wikitext_name,
            )
            all_records.extend(records)
            r1 = sum(1 for r in records if 0 <= r.target_rank < 1) / max(len(records), 1)
            r20 = sum(1 for r in records if 0 <= r.target_rank < 20) / max(len(records), 1)
            print(f"  random_at_W       seed={seed} n={len(records)} R@1={r1:.3f} R@20={r20:.3f}")

            # W=8 references
            for label, rel_path, vocab in references:
                cb = load_codebook(args.repo_root / rel_path, device=str(substrate.device))
                if cb.shape[0] != len(vocab.id_to_token):
                    print(f"  [skip] {label}: vocab mismatch ({cb.shape[0]} vs {len(vocab.id_to_token)})")
                    continue
                records = _evaluate(
                    label=label, codebook=cb, vocab=vocab, substrate=substrate,
                    window_size=W, seed=seed, repo_root=args.repo_root,
                    landscape_size=args.landscape_size, beta=args.beta,
                    test_samples=args.test_samples, wikitext_name=args.wikitext_name,
                )
                all_records.extend(records)
                r1 = sum(1 for r in records if 0 <= r.target_rank < 1) / max(len(records), 1)
                r20 = sum(1 for r in records if 0 <= r.target_rank < 20) / max(len(records), 1)
                print(f"  {label:<18} seed={seed} n={len(records)} R@1={r1:.3f} R@20={r20:.3f}")

            # per-scale codebook at native W
            if ps_entry is not None:
                label, rel_path, vocab = ps_entry
                cb = load_codebook(args.repo_root / rel_path, device=str(substrate.device))
                if cb.shape[0] != len(vocab.id_to_token):
                    print(f"  [skip] {label}: vocab mismatch ({cb.shape[0]} vs {len(vocab.id_to_token)})")
                else:
                    records = _evaluate(
                        label=label, codebook=cb, vocab=vocab, substrate=substrate,
                        window_size=W, seed=seed, repo_root=args.repo_root,
                        landscape_size=args.landscape_size, beta=args.beta,
                        test_samples=args.test_samples, wikitext_name=args.wikitext_name,
                    )
                    all_records.extend(records)
                    r1 = sum(1 for r in records if 0 <= r.target_rank < 1) / max(len(records), 1)
                    r20 = sum(1 for r in records if 0 <= r.target_rank < 20) / max(len(records), 1)
                    print(f"  {label:<18} seed={seed} n={len(records)} R@1={r1:.3f} R@20={r20:.3f}")

    aggregates = _summarize(all_records)

    print("\nPooled results (over seeds):")
    header = f"{'condition':<28} {'n':>6} " + "".join(f"{'R@'+str(k):>14}" for k in K_VALUES) + f"{'median_rk':>11}{'syn':>9}"
    print(header)
    print("-" * len(header))
    for cond_key, agg in aggregates.items():
        row = f"{cond_key:<28} {agg['n']:>6}"
        for k in K_VALUES:
            v = agg["recall_at_k"][k]
            row += f" {v['value']:.3f}[{v['ci_lower']:.3f},{v['ci_upper']:.3f}]"
        row += f" {agg['target_rank_stats']['median']:>10.1f}"
        row += f" {agg['mean_settled_synergy']:>8.3f}"
        print(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "window_size", "seed", "target_token", "target_rank", "settled_synergy"])
        for r in all_records:
            w.writerow([r.label, r.window_size, r.seed, r.target_token,
                        r.target_rank, f"{r.settled_synergy:.6f}"])
    with args.out_csv.with_suffix(".json").open("w") as f:
        json.dump({
            "config": vars(args) | {"repo_root": str(args.repo_root), "out_csv": str(args.out_csv)},
            "aggregates": aggregates,
        }, f, indent=2, default=str)
    print(f"\nWrote {args.out_csv} and {args.out_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
