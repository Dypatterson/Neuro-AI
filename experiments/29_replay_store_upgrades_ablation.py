"""Ablation: tag_count and inhibition-of-return on the Phase 4 replay store.

Brainstorm Idea 4a/4c (2026-05-13) added two upgrades to ReplayStore:

  4a -- tag_count: overlapping queries collapse onto a single trace
        with an incremented count. Sampling weight scales with re-tag
        frequency. (Joo & Frank 2023.)
  4c -- inhibition of return: a sampled trace has its suppression
        multiplier scaled down; non-sampled traces recover.
        (Biderman SFMA 2023.)

The unit tests in ``tests/test_phase4_replay_store_upgrades.py`` pin
the data-structure invariants, but no experiment yet shows the
upgrades change *which traces survive*, *how the candidate channel
behaves*, or *how the slow consolidation variables fill in* under
real retrieval dynamics. This script runs the 2x2 ablation.

Stream design:

  * ``n_patterns`` stored windows partitioned into ``n_frequent``
    frequent sources and ``n_oneshot`` one-shot sources.
  * Cues are perturbed copies of source windows. Frequent sources are
    sampled with probability ``frequent_fraction``; one-shot with the
    remainder.
  * The same seed produces the same cue stream across all four
    conditions, so any difference in outcomes is attributable to the
    replay-store configuration.

Four conditions (matrix on (tag_overlap, suppression)):

  * A: baseline      -- tag_overlap=None, no IoR
  * B: tag only      -- tag_overlap=0.7, no IoR
  * C: IoR only      -- tag_overlap=None, suppression decay/recovery
  * D: both          -- tag_overlap=0.7, suppression decay/recovery

Metrics:

  * candidate_count: total resolved-replay candidates emitted.
  * candidate_unique_sources: distinct source windows the candidates
    came from (a measure of replay diversity).
  * store_size_final: trace count in the replay store at end.
  * mean_tag_count_final / max_tag_count_final
  * mean_suppression_final
  * sampling_gini: Gini coefficient over how many times each stored
    trace got sampled across the run (0 = perfectly even, 1 = one
    trace dominates). IoR should reduce this; tag_count should
    increase it.
  * u2_max / u3_max: maximum value across the second and third slow
    Benna-Fusi variables at the end of the run. Higher = more
    patterns reached deep consolidation.
  * freq_survival_ratio: of the patterns that hit u3 > 0.5 * u3_max,
    what fraction had frequent source IDs? (Tag_count should bias
    toward frequent survival.)

Run:
    PYTHONPATH=src .venv/bin/python experiments/29_replay_store_upgrades_ablation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class ConditionResult:
    condition: str
    seed: int
    candidate_count: int
    candidate_unique_sources: int
    store_size_final: int
    mean_tag_count_final: float
    max_tag_count_final: int
    mean_suppression_final: float
    sampling_gini: float
    u2_max: float
    u3_max: float
    n_patterns_u3_active: int
    freq_survival_ratio: float
    # Only meaningful when --enable-candidates is set. With candidate
    # addition disabled, the landscape never changes from its initial
    # state of 32 patterns, so these collapse to (32, 0, 0.0).
    memory_size_final: int = 32
    n_added_candidates: int = 0
    added_candidates_freq_fraction: float = 0.0


def _make_conditions(
    *,
    store_capacity: int,
    replay_batch_size: int,
) -> Dict[str, ReplayConfig]:
    shared = dict(
        store_threshold=0.001,
        store_capacity=store_capacity,
        resolve_threshold=globals().get("RESOLVE_THRESHOLD", 0.30),
        replay_every=15,
        replay_batch_size=replay_batch_size,
        max_age=20,
    )
    return {
        "A_baseline": ReplayConfig(
            tag_overlap_threshold=None,
            suppression_decay=1.0,
            suppression_recovery=0.0,
            **shared,
        ),
        "B_tag_only": ReplayConfig(
            tag_overlap_threshold=0.7,
            suppression_decay=1.0,
            suppression_recovery=0.0,
            **shared,
        ),
        "C_ior_only": ReplayConfig(
            tag_overlap_threshold=None,
            suppression_decay=0.4,
            suppression_recovery=0.02,
            **shared,
        ),
        "D_both": ReplayConfig(
            tag_overlap_threshold=0.7,
            suppression_decay=0.4,
            suppression_recovery=0.02,
            **shared,
        ),
    }


def _gini(counts: List[int]) -> float:
    """Gini coefficient over a vector of counts. 0 = uniform, 1 = concentrated."""
    if not counts:
        return 0.0
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    cum = 0.0
    total = sum(sorted_counts)
    if total <= 0:
        return 0.0
    for i, x in enumerate(sorted_counts, start=1):
        cum += i * x
    return (2.0 * cum) / (n * total) - (n + 1) / n


def _build_stream(
    substrate: TorchFHRR,
    n_frequent: int,
    n_oneshot: int,
    window_size: int,
    seed: int,
):
    codebook_size = 64
    codebook = substrate.random_vectors(codebook_size)
    positions = build_position_vectors(substrate, window_size)
    rng = torch.Generator(device="cpu").manual_seed(seed + 17)

    n_patterns = n_frequent + n_oneshot
    windows = [
        tuple(
            int(t.item())
            for t in torch.randint(0, codebook_size, (window_size,), generator=rng)
        )
        for _ in range(n_patterns)
    ]
    bindings = torch.stack(
        [encode_window(substrate, positions, codebook, w) for w in windows],
        dim=0,
    )
    # Frequent IDs are the first n_frequent; one-shot IDs are the rest.
    return windows, bindings, codebook, positions


def _make_cue_stream(
    n_steps: int,
    n_frequent: int,
    n_oneshot: int,
    frequent_fraction: float,
    seed: int,
) -> List[int]:
    """Return a deterministic sequence of source-pattern indices."""
    rng = torch.Generator(device="cpu").manual_seed(seed + 41)
    n_patterns = n_frequent + n_oneshot
    cue_indices: List[int] = []
    for _ in range(n_steps):
        if torch.rand((1,), generator=rng).item() < frequent_fraction:
            # Frequent source
            idx = int(torch.randint(0, n_frequent, (1,), generator=rng).item())
        else:
            idx = n_frequent + int(
                torch.randint(0, n_oneshot, (1,), generator=rng).item()
            )
        cue_indices.append(idx)
    return cue_indices


def _run_condition(
    condition_name: str,
    config: ReplayConfig,
    *,
    seed: int,
    bindings: torch.Tensor,
    cue_indices: List[int],
    n_frequent: int,
    cue_noise: float,
    beta: float,
    substrate: TorchFHRR,
    enable_candidates: bool = False,
) -> ConditionResult:
    n_patterns = bindings.shape[0]
    memory = TracedHopfieldMemory[int](substrate, snapshot_k=8)
    for idx in range(n_patterns):
        memory.store(bindings[idx], label=idx)

    consolidation = ConsolidationState(
        ConsolidationConfig(m=4, alpha=0.25, death_threshold=0.0, death_window=10_000),
        device="cpu",
    )
    unified = UnifiedReplayMemory[int](substrate, memory, consolidation, config=config)
    unified.attach_initial_patterns()

    # Track sampling counts: how often each pattern's settled trace is the
    # closest match to a sampled trace's query. This is a proxy for which
    # source pattern's traces are getting replayed.
    sampling_counts: Dict[int, int] = {i: 0 for i in range(n_patterns)}
    candidate_sources: List[int] = []
    added_candidate_sources: List[int] = []

    def candidate_handler(trace) -> Optional[int]:
        sims = substrate.similarity_matrix(trace.query, bindings)
        src_idx = int(torch.argmax(sims).detach().cpu())
        candidate_sources.append(src_idx)
        sampling_counts[src_idx] += 1
        if not enable_candidates:
            return None  # measurement-only mode
        # Real Phase 4 dynamics: candidate becomes a new pattern.
        new_idx = memory.stored_count
        memory.store(trace.final_state, label=memory.stored_count)
        added_candidate_sources.append(src_idx)
        return new_idx

    cue_rng = torch.Generator(device="cpu").manual_seed(seed + 71)
    for src_idx in cue_indices:
        # Identical perturbation for the same source index given a fixed seed
        # would be too deterministic; vary by step using cue_rng.
        cue = substrate.perturb(bindings[src_idx], noise=cue_noise)
        unified.retrieve_and_observe(cue, beta=beta, max_iter=8)
        if unified.should_replay():
            unified.run_replay_cycle(
                beta=beta, max_iter=8, candidate_handler=candidate_handler
            )

    store_stats = unified.store.stats()
    u = unified.consolidation.u  # [n_patterns, m]
    # u_k corresponds to column k-1 in the tensor; u_2 is column 1, u_3 is column 2.
    u2_max = float(u[:, 1].max().item()) if u.shape[1] > 1 and u.shape[0] else 0.0
    u3_max = float(u[:, 2].max().item()) if u.shape[1] > 2 and u.shape[0] else 0.0
    u3_threshold = 0.5 * u3_max if u3_max > 0 else 0.0
    n_patterns_u3_active = (
        int((u[:, 2] > u3_threshold).sum().item()) if u3_max > 0 else 0
    )
    if u3_max > 0:
        u3_active_mask = u[:, 2] > u3_threshold
        active_indices = torch.nonzero(u3_active_mask).flatten().tolist()
        freq_in_active = sum(1 for i in active_indices if i < n_frequent)
        freq_survival_ratio = freq_in_active / max(len(active_indices), 1)
    else:
        freq_survival_ratio = 0.0

    counts = list(sampling_counts.values())
    gini = _gini(counts)

    n_added = len(added_candidate_sources)
    added_freq_fraction = (
        sum(1 for s in added_candidate_sources if s < n_frequent) / max(n_added, 1)
        if n_added > 0
        else 0.0
    )

    return ConditionResult(
        condition=condition_name,
        seed=seed,
        candidate_count=len(candidate_sources),
        candidate_unique_sources=len(set(candidate_sources)),
        store_size_final=store_stats["size"],
        mean_tag_count_final=store_stats["mean_tag_count"],
        max_tag_count_final=store_stats["max_tag_count"],
        mean_suppression_final=store_stats["mean_suppression"],
        sampling_gini=gini,
        u2_max=u2_max,
        u3_max=u3_max,
        n_patterns_u3_active=n_patterns_u3_active,
        freq_survival_ratio=freq_survival_ratio,
        memory_size_final=memory.stored_count,
        n_added_candidates=n_added,
        added_candidates_freq_fraction=added_freq_fraction,
    )


def _run_seed(
    *,
    seed: int,
    dim: int,
    window_size: int,
    n_frequent: int,
    n_oneshot: int,
    n_steps: int,
    frequent_fraction: float,
    cue_noise: float,
    beta: float,
    conditions: Dict[str, ReplayConfig],
    enable_candidates: bool = False,
) -> List[ConditionResult]:
    substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")
    windows, bindings, _, _ = _build_stream(
        substrate,
        n_frequent=n_frequent,
        n_oneshot=n_oneshot,
        window_size=window_size,
        seed=seed,
    )
    cue_indices = _make_cue_stream(
        n_steps=n_steps,
        n_frequent=n_frequent,
        n_oneshot=n_oneshot,
        frequent_fraction=frequent_fraction,
        seed=seed,
    )

    results: List[ConditionResult] = []
    for name, config in conditions.items():
        results.append(
            _run_condition(
                name,
                config,
                seed=seed,
                bindings=bindings,
                cue_indices=cue_indices,
                n_frequent=n_frequent,
                cue_noise=cue_noise,
                beta=beta,
                substrate=substrate,
                enable_candidates=enable_candidates,
            )
        )
    return results


def _print_rows(results: List[ConditionResult]) -> None:
    header = (
        f"{'cond':<12} {'seed':>5} {'cands':>6} {'uniq':>5} "
        f"{'store':>6} {'mTag':>5} {'xTag':>5} {'mSup':>5} "
        f"{'gini':>5} {'u2max':>7} {'u3max':>7} {'n_u3':>5} {'fSurv':>6} "
        f"{'mem':>5} {'+add':>5} {'addFq':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.condition:<12} {r.seed:>5} {r.candidate_count:>6d} "
            f"{r.candidate_unique_sources:>5d} {r.store_size_final:>6d} "
            f"{r.mean_tag_count_final:>5.2f} {r.max_tag_count_final:>5d} "
            f"{r.mean_suppression_final:>5.2f} {r.sampling_gini:>5.2f} "
            f"{r.u2_max:>7.3f} {r.u3_max:>7.3f} {r.n_patterns_u3_active:>5d} "
            f"{r.freq_survival_ratio:>6.2f} "
            f"{r.memory_size_final:>5d} {r.n_added_candidates:>5d} "
            f"{r.added_candidates_freq_fraction:>6.2f}"
        )


def _aggregate(results: List[ConditionResult]) -> Dict[str, dict]:
    by_cond: Dict[str, List[ConditionResult]] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)
    summary = {}
    for cond, rs in by_cond.items():
        summary[cond] = {
            "n_seeds": len(rs),
            "candidate_count_mean": statistics.fmean(r.candidate_count for r in rs),
            "candidate_unique_sources_mean": statistics.fmean(r.candidate_unique_sources for r in rs),
            "store_size_final_mean": statistics.fmean(r.store_size_final for r in rs),
            "mean_tag_count_final_mean": statistics.fmean(r.mean_tag_count_final for r in rs),
            "max_tag_count_final_mean": statistics.fmean(r.max_tag_count_final for r in rs),
            "mean_suppression_final_mean": statistics.fmean(r.mean_suppression_final for r in rs),
            "sampling_gini_mean": statistics.fmean(r.sampling_gini for r in rs),
            "u2_max_mean": statistics.fmean(r.u2_max for r in rs),
            "u3_max_mean": statistics.fmean(r.u3_max for r in rs),
            "n_patterns_u3_active_mean": statistics.fmean(r.n_patterns_u3_active for r in rs),
            "freq_survival_ratio_mean": statistics.fmean(r.freq_survival_ratio for r in rs),
            "memory_size_final_mean": statistics.fmean(r.memory_size_final for r in rs),
            "n_added_candidates_mean": statistics.fmean(r.n_added_candidates for r in rs),
            "added_candidates_freq_fraction_mean": statistics.fmean(
                r.added_candidates_freq_fraction for r in rs
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--n-frequent", type=int, default=8)
    parser.add_argument("--n-oneshot", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--frequent-fraction", type=float, default=0.8)
    parser.add_argument("--cue-noise", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--store-capacity", type=int, default=200)
    parser.add_argument("--replay-batch-size", type=int, default=10)
    parser.add_argument("--resolve-threshold", type=float, default=0.30)
    parser.add_argument(
        "--enable-candidates",
        action="store_true",
        help="Real Phase 4 dynamics: candidate handler adds settled states "
             "as new patterns, growing the landscape across replay cycles.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41, 53, 67])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    global RESOLVE_THRESHOLD
    RESOLVE_THRESHOLD = args.resolve_threshold
    conditions = _make_conditions(
        store_capacity=args.store_capacity,
        replay_batch_size=args.replay_batch_size,
    )

    all_results: List[ConditionResult] = []
    for seed in args.seeds:
        all_results.extend(
            _run_seed(
                seed=seed,
                dim=args.dim,
                window_size=args.window_size,
                n_frequent=args.n_frequent,
                n_oneshot=args.n_oneshot,
                n_steps=args.n_steps,
                frequent_fraction=args.frequent_fraction,
                cue_noise=args.cue_noise,
                beta=args.beta,
                conditions=conditions,
                enable_candidates=args.enable_candidates,
            )
        )

    _print_rows(all_results)
    print()
    summary = _aggregate(all_results)
    print("Per-condition aggregates (over seeds):")
    for cond, stats in summary.items():
        print(f"\n  {cond}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": vars(args) | {"out": str(args.out)},
            "rows": [asdict(r) for r in all_results],
            "aggregate": summary,
        }, indent=2))
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
