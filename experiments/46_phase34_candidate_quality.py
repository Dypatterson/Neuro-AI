"""Candidate-quality diagnostic for range-shaped Phase 4 replay.

Reports 070/072 showed that ``range_shaped + rebind + window_preserving``
creates more Phase 4 replay candidates than standard replay, but held-out
metrics do not improve. This script asks whether those extra candidates are
actually novel/useful or whether they collapse into existing basins.

It intentionally does not run Phase 5. The diagnostic is local to Phase 4
candidate quality:

  * candidate encoder-provenance support
  * candidate final-top-index diversity
  * near-duplicate rate against the pre-insertion memory
  * query similarity to existing patterns
  * d_eff before/after candidate insertion
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from energy_memory.phase2.corpus import (
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window_with_provenance,
    mask_positions as compute_mask_positions,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.trajectory import TracedHopfieldMemory, TrajectoryTrace
from energy_memory.substrate.torch_fhrr import TorchFHRR


RoleAtomPair = Tuple[int, int]


@dataclass
class CandidateRecord:
    condition: str
    seed: int
    scale: int
    cycle: int
    final_top_index: int
    final_top_score: float
    query_top_score: float
    encoder_terms: List[RoleAtomPair]
    d_eff_before: float
    d_eff_after: float


@dataclass(frozen=True)
class ScaleSummary:
    condition: str
    seed: int
    scale: int
    candidates: int
    provenance_cells: int
    provenance_rect: float
    atom_entropy: float
    winner_unique: int
    winner_entropy: float
    winner_gini: float
    near_duplicate_rate: float
    query_near_existing_rate: float
    mean_final_top_score: float
    mean_query_top_score: float
    mean_delta_d_eff: float
    d_eff_initial: float
    d_eff_final: float
    memory_size_final: int
    store_final: int


def _condition_config(
    condition: str,
    *,
    replay_batch_size: int,
    store_capacity: int,
    resolve_threshold: float,
    smoothing_alpha: float,
) -> ReplayConfig:
    shared = dict(
        store_threshold=0.0,
        store_capacity=store_capacity,
        resolve_threshold=resolve_threshold,
        replay_every=1,
        replay_batch_size=replay_batch_size,
        max_age=100,
        tag_overlap_threshold=None,
    )
    if condition == "standard":
        return ReplayConfig(replay_sampler="standard", **shared)
    if condition == "range_rebind_window":
        return ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="rebind",
            range_shaped_rebind_mode="window_preserving",
            range_shaped_smoothing_alpha=smoothing_alpha,
            **shared,
        )
    raise ValueError(f"unknown condition: {condition}")


def _sub_window_for_scale(
    cue_window: Sequence[int],
    *,
    eval_window_size: int,
    masked_pos: int,
    scale: int,
    mask_id: int,
) -> Tuple[List[int], int]:
    half = scale // 2
    sub_start = masked_pos - half
    if scale % 2 == 0:
        sub_start = masked_pos - half + 1
    sub_start = max(0, sub_start)
    if sub_start + scale > eval_window_size:
        sub_start = eval_window_size - scale
    sub_window = list(cue_window[sub_start:sub_start + scale])
    local_masked = masked_pos - sub_start
    cue_w = list(sub_window)
    cue_w[local_masked] = mask_id
    return cue_w, local_masked


def _d_eff_or_zero(substrate: TorchFHRR, memory: TracedHopfieldMemory) -> float:
    if memory.stored_count < 2:
        return 0.0
    value = substrate.d_eff(memory._pattern_matrix())
    if torch.isnan(value):
        return 0.0
    return float(value.detach().cpu())


def _query_top_score(substrate: TorchFHRR, memory: TracedHopfieldMemory, query) -> float:
    if memory.stored_count == 0:
        return 0.0
    sims = substrate.similarity_matrix(query, memory._pattern_matrix())
    return float(sims.max().detach().cpu())


def _histogram_rect(
    pairs: Sequence[RoleAtomPair],
    *,
    n_roles: int,
    n_atoms: int,
) -> float:
    hist = torch.zeros(n_roles, n_atoms, dtype=torch.float64)
    for role, atom in pairs:
        if 0 <= role < n_roles and 0 <= atom < n_atoms:
            hist[role, atom] += 1.0
    total = hist.sum()
    if total <= 0:
        return 0.0
    hist = hist / total
    role = hist.sum(dim=1, keepdim=True)
    atom = hist.sum(dim=0, keepdim=True)
    fact = role @ atom
    eps = 1e-9
    p = (hist.flatten() + eps)
    q = (fact.flatten() + eps)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p.log() - q.log())).sum())


def _entropy_from_counts(counts: Iterable[int]) -> float:
    values = [float(v) for v in counts if v > 0]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    probs = [v / total for v in values]
    return -sum(p * math.log(p) for p in probs) / math.log(len(values))


def _gini_from_counts(counts: Iterable[int]) -> float:
    values = sorted(float(v) for v in counts if v > 0)
    total = sum(values)
    if total <= 0.0:
        return 0.0
    n = len(values)
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


def _summarize_records(
    records: Sequence[CandidateRecord],
    *,
    condition: str,
    seed: int,
    scale: int,
    n_atoms: int,
    near_duplicate_threshold: float,
    d_eff_initial: float,
    d_eff_final: float,
    memory_size_final: int,
    store_final: int,
) -> ScaleSummary:
    pairs: List[RoleAtomPair] = []
    winners: Dict[int, int] = {}
    atom_counts = [0 for _ in range(n_atoms)]
    for rec in records:
        pairs.extend(rec.encoder_terms)
        winners[rec.final_top_index] = winners.get(rec.final_top_index, 0) + 1
        for _, atom in rec.encoder_terms:
            if 0 <= atom < n_atoms:
                atom_counts[atom] += 1

    final_scores = [rec.final_top_score for rec in records]
    query_scores = [rec.query_top_score for rec in records]
    deltas = [rec.d_eff_after - rec.d_eff_before for rec in records]
    return ScaleSummary(
        condition=condition,
        seed=seed,
        scale=scale,
        candidates=len(records),
        provenance_cells=len(set(pairs)),
        provenance_rect=_histogram_rect(pairs, n_roles=scale, n_atoms=n_atoms),
        atom_entropy=_entropy_from_counts(atom_counts),
        winner_unique=len(winners),
        winner_entropy=_entropy_from_counts(winners.values()),
        winner_gini=_gini_from_counts(winners.values()),
        near_duplicate_rate=(
            sum(1 for s in final_scores if s >= near_duplicate_threshold)
            / max(len(final_scores), 1)
        ),
        query_near_existing_rate=(
            sum(1 for s in query_scores if s >= near_duplicate_threshold)
            / max(len(query_scores), 1)
        ),
        mean_final_top_score=(
            statistics.fmean(final_scores) if final_scores else 0.0
        ),
        mean_query_top_score=(
            statistics.fmean(query_scores) if query_scores else 0.0
        ),
        mean_delta_d_eff=statistics.fmean(deltas) if deltas else 0.0,
        d_eff_initial=d_eff_initial,
        d_eff_final=d_eff_final,
        memory_size_final=memory_size_final,
        store_final=store_final,
    )


def run_condition(
    *,
    condition: str,
    seed: int,
    args,
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    vocab,
    codebook,
) -> List[ScaleSummary]:
    torch.manual_seed(seed * 1009 + sum(ord(c) for c in condition))
    substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)
    codebook = codebook.to(substrate.device)

    masked_pos = compute_mask_positions(args.eval_window_size, 1, args.mask_position)[0]
    eval_windows_all = make_windows(validation_ids, args.eval_window_size)
    test_windows = sample_windows(
        eval_windows_all,
        min(args.test_samples, len(eval_windows_all)),
        seed=seed + 9000,
    )
    test_set = set(map(tuple, test_windows))
    cue_pool = sample_windows(
        eval_windows_all,
        min(len(eval_windows_all), args.n_cues * 20 + 5000),
        seed=seed + 7000,
    )
    cue_stream = [
        w for w in cue_pool
        if tuple(w) not in test_set and not any(t == vocab.unk_id for t in w)
    ][: args.n_cues]

    units: Dict[int, UnifiedReplayMemory] = {}
    positions_by_scale = {}
    records_by_scale: Dict[int, List[CandidateRecord]] = {}
    d_eff_initial_by_scale: Dict[int, float] = {}

    config = _condition_config(
        condition,
        replay_batch_size=args.replay_batch_size,
        store_capacity=args.store_capacity,
        resolve_threshold=args.resolve_threshold,
        smoothing_alpha=args.smoothing_alpha,
    )
    cons_config = ConsolidationConfig(
        m=4,
        alpha=0.25,
        death_threshold=0.0,
        death_window=10_000,
    )

    for scale in args.scales:
        train_windows = make_windows(train_ids, scale)
        landscape = sample_windows(
            train_windows,
            min(args.landscape_size, len(train_windows)),
            seed=seed + scale * 100,
        )
        positions = build_position_vectors(substrate, scale)
        positions_by_scale[scale] = positions
        memory = TracedHopfieldMemory[int](substrate, snapshot_k=8)
        for idx, window in enumerate(landscape):
            encoded, _terms = encode_window_with_provenance(
                substrate,
                positions,
                codebook,
                window,
            )
            memory.store(encoded, label=idx)
        cons = ConsolidationState(cons_config, device=str(substrate.device))
        unit = UnifiedReplayMemory[int](
            substrate=substrate,
            memory=memory,
            consolidation=cons,
            config=config,
            replay_position_vectors=positions,
            replay_codebook=codebook,
        )
        unit.attach_initial_patterns()
        units[scale] = unit
        records_by_scale[scale] = []
        d_eff_initial_by_scale[scale] = _d_eff_or_zero(substrate, memory)

    for step, cue_window in enumerate(cue_stream, start=1):
        for scale, unit in units.items():
            cue_w, _local_masked = _sub_window_for_scale(
                cue_window,
                eval_window_size=args.eval_window_size,
                masked_pos=masked_pos,
                scale=scale,
                mask_id=vocab.mask_id,
            )
            cue_vec, encoder_terms = encode_window_with_provenance(
                substrate,
                positions_by_scale[scale],
                codebook,
                cue_w,
            )
            unit.retrieve_and_observe(
                cue_vec,
                beta=args.beta,
                max_iter=8,
                encoder_terms=list(encoder_terms),
            )

        if step % args.replay_every != 0:
            continue

        for scale, unit in units.items():
            cycle_no = step // args.replay_every

            def handler(trace: TrajectoryTrace, *, sc=scale, u=unit, cyc=cycle_no):
                before = _d_eff_or_zero(substrate, u.memory)
                query_score = _query_top_score(substrate, u.memory, trace.query)
                records_by_scale[sc].append(
                    CandidateRecord(
                        condition=condition,
                        seed=seed,
                        scale=sc,
                        cycle=cyc,
                        final_top_index=(
                            -1 if trace.final_top_index is None
                            else int(trace.final_top_index)
                        ),
                        final_top_score=float(trace.final_top_score),
                        query_top_score=query_score,
                        encoder_terms=(
                            [] if trace.encoder_terms is None
                            else [(int(r), int(a)) for r, a in trace.encoder_terms]
                        ),
                        d_eff_before=before,
                        d_eff_after=before,
                    )
                )
                new_idx = u.memory.stored_count
                u.memory.store(trace.final_state.detach().clone(), label=new_idx)
                after = _d_eff_or_zero(substrate, u.memory)
                records_by_scale[sc][-1].d_eff_after = after
                return new_idx

            unit.run_replay_cycle(
                beta=args.beta,
                max_iter=8,
                candidate_handler=handler,
            )

    summaries: List[ScaleSummary] = []
    for scale, unit in units.items():
        summaries.append(
            _summarize_records(
                records_by_scale[scale],
                condition=condition,
                seed=seed,
                scale=scale,
                n_atoms=int(codebook.shape[0]),
                near_duplicate_threshold=args.near_duplicate_threshold,
                d_eff_initial=d_eff_initial_by_scale[scale],
                d_eff_final=_d_eff_or_zero(substrate, unit.memory),
                memory_size_final=unit.memory.stored_count,
                store_final=len(unit.store),
            )
        )
    return summaries


def _aggregate(rows: Sequence[ScaleSummary]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[ScaleSummary]] = {}
    for row in rows:
        grouped.setdefault(f"{row.condition}|W{row.scale}", []).append(row)
    metrics = [
        "candidates",
        "provenance_cells",
        "provenance_rect",
        "atom_entropy",
        "winner_unique",
        "winner_entropy",
        "winner_gini",
        "near_duplicate_rate",
        "query_near_existing_rate",
        "mean_final_top_score",
        "mean_query_top_score",
        "mean_delta_d_eff",
        "d_eff_initial",
        "d_eff_final",
        "memory_size_final",
        "store_final",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for key, vals in grouped.items():
        out[key] = {"n_seeds": float(len(vals))}
        for metric in metrics:
            numbers = [float(getattr(v, metric)) for v in vals]
            out[key][f"{metric}_mean"] = statistics.fmean(numbers)
            out[key][f"{metric}_min"] = min(numbers)
            out[key][f"{metric}_max"] = max(numbers)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 11, 23])
    parser.add_argument("--conditions", nargs="+", default=["standard", "range_rebind_window"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--corpus-source", choices=["repo_sample", "wikitext"], default="repo_sample")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument(
        "--codebook-path",
        default="reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--landscape-size", type=int, default=16)
    parser.add_argument("--eval-window-size", type=int, default=4)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--n-cues", type=int, default=60)
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--replay-every", type=int, default=5)
    parser.add_argument("--replay-batch-size", type=int, default=4)
    parser.add_argument("--store-capacity", type=int, default=500)
    parser.add_argument("--resolve-threshold", type=float, default=0.2)
    parser.add_argument("--smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_candidate_quality/results_n3.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    splits = load_corpus_splits(
        args.corpus_source,
        repo_root,
        wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    codebook = load_codebook(Path(args.codebook_path), device=args.device)
    if int(codebook.shape[0]) != len(vocab.id_to_token):
        raise ValueError(
            f"codebook rows ({codebook.shape[0]}) do not match vocab "
            f"({len(vocab.id_to_token)})"
        )

    rows: List[ScaleSummary] = []
    for seed in args.seeds:
        for condition in args.conditions:
            rows.extend(
                run_condition(
                    condition=condition,
                    seed=seed,
                    args=args,
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                    vocab=vocab,
                    codebook=codebook,
                )
            )

    print(
        f"Candidate quality seeds={args.seeds} conditions={args.conditions} "
        f"scales={args.scales} codebook={args.codebook_path}"
    )
    header = (
        f"{'condition':<22} {'seed':>5} {'W':>2} {'cand':>5} {'cells':>5} "
        f"{'rect':>7} {'winU':>5} {'near':>5} {'qNear':>5} "
        f"{'top':>6} {'qtop':>6} {'dEff':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.condition:<22} {row.seed:>5} {row.scale:>2} "
            f"{row.candidates:>5} {row.provenance_cells:>5} "
            f"{row.provenance_rect:>7.4f} {row.winner_unique:>5} "
            f"{row.near_duplicate_rate:>5.2f} "
            f"{row.query_near_existing_rate:>5.2f} "
            f"{row.mean_final_top_score:>6.3f} "
            f"{row.mean_query_top_score:>6.3f} "
            f"{row.d_eff_final:>7.2f}"
        )

    aggregate = _aggregate(rows)
    print("\nAggregates:")
    for key in sorted(aggregate):
        stats = aggregate[key]
        print(
            f"  {key:<26} cand={stats['candidates_mean']:.1f} "
            f"cells={stats['provenance_cells_mean']:.1f} "
            f"rect={stats['provenance_rect_mean']:.4f} "
            f"winU={stats['winner_unique_mean']:.1f} "
            f"near={stats['near_duplicate_rate_mean']:.3f} "
            f"qNear={stats['query_near_existing_rate_mean']:.3f} "
            f"dEff={stats['d_eff_final_mean']:.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "seeds": args.seeds,
                    "conditions": args.conditions,
                    "dim": args.dim,
                    "device": args.device,
                    "corpus_source": args.corpus_source,
                    "max_vocab": args.max_vocab,
                    "codebook_path": args.codebook_path,
                    "scales": args.scales,
                    "landscape_size": args.landscape_size,
                    "eval_window_size": args.eval_window_size,
                    "n_cues": args.n_cues,
                    "test_samples": args.test_samples,
                    "replay_every": args.replay_every,
                    "replay_batch_size": args.replay_batch_size,
                    "resolve_threshold": args.resolve_threshold,
                    "smoothing_alpha": args.smoothing_alpha,
                    "beta": args.beta,
                    "near_duplicate_threshold": args.near_duplicate_threshold,
                },
                "rows": [asdict(row) for row in rows],
                "aggregate": aggregate,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
