"""Phase 4 range-shaped replay downstream pilot.

This is the first small comparison after moving range-shaped replay into
``UnifiedReplayMemory``. It uses synthetic skewed encoder windows so the run is
cheap enough for local CPU, then drives the real Phase 4 replay cycle:

  replay store -> sampler -> optional rebind -> Hopfield re-settle ->
  candidate insertion -> consolidation update

The result is still a pilot. It is not Phase 5 evidence and it does not measure
Delta E. The only claim it can support is whether static range-shaped replay
changes downstream Phase 4 candidate/provenance geometry relative to standard
whole-trace replay.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window_with_provenance,
)
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.trajectory import (
    TracedHopfieldMemory,
    TrajectorySnapshot,
    TrajectoryTrace,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


RoleAtomPair = Tuple[int, int]


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    seed: int
    sampled_total: int
    candidate_count: int
    candidate_accept_rate: float
    candidate_cells: int
    candidate_rect: float
    candidate_role_entropy: float
    candidate_atom_entropy: float
    candidate_winner_unique: int
    candidate_winner_gini: float
    mean_final_top_score: float
    store_final: int
    memory_size_final: int
    consolidation_n_patterns: int
    u2_max: float
    u3_max: float
    d_eff_initial: float
    d_eff_final: float


def _generate_skewed_windows(
    *,
    n_traces: int,
    n_roles: int,
    n_atoms: int,
    window_size: int,
    skew_concentration: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    rng = random.Random(seed)
    windows: List[Tuple[int, ...]] = []
    for _ in range(n_traces):
        atoms: List[int] = []
        for role in range(window_size):
            base = (role * skew_concentration) % n_atoms
            atoms.append((base + rng.randrange(skew_concentration)) % n_atoms)
        windows.append(tuple(atoms))
    return windows


def _seeded_trace(
    *,
    query: torch.Tensor,
    encoder_terms: List[RoleAtomPair],
    final_top_index: int,
    final_top_score: float,
) -> TrajectoryTrace:
    # Enough trajectory structure for gate diagnostics while keeping the store
    # construction deterministic. The replay cycle will generate real traces.
    snapshots = [
        TrajectorySnapshot(
            step=1,
            top_k_indices=[int(final_top_index)],
            top_k_weights=[float(final_top_score)],
            entropy=0.65,
            energy=-float(final_top_score),
        )
    ]
    return TrajectoryTrace(
        query=query.detach().clone(),
        encoder_terms=list(encoder_terms),
        snapshots=snapshots,
        final_state=query.detach().clone(),
        final_top_score=float(final_top_score),
        final_top_index=int(final_top_index),
        converged=True,
    )


def _condition_configs(
    *,
    replay_batch_size: int,
    resolve_threshold: float,
    store_capacity: int,
    smoothing_alpha: float,
) -> Dict[str, ReplayConfig]:
    shared = dict(
        store_threshold=0.0,
        store_capacity=store_capacity,
        resolve_threshold=resolve_threshold,
        replay_every=1,
        replay_batch_size=replay_batch_size,
        max_age=100,
        tag_overlap_threshold=None,
    )
    return {
        "standard": ReplayConfig(replay_sampler="standard", **shared),
        "range_skip": ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="skip",
            range_shaped_smoothing_alpha=smoothing_alpha,
            **shared,
        ),
        "range_closest": ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="closest",
            range_shaped_smoothing_alpha=smoothing_alpha,
            **shared,
        ),
        "range_rebind_single": ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="rebind",
            range_shaped_rebind_mode="single_binding",
            range_shaped_smoothing_alpha=smoothing_alpha,
            **shared,
        ),
        "range_rebind_window": ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="rebind",
            range_shaped_rebind_mode="window_preserving",
            range_shaped_window_size=None,
            range_shaped_smoothing_alpha=smoothing_alpha,
            **shared,
        ),
    }


def _joint_histogram(
    pairs: Sequence[RoleAtomPair],
    *,
    n_roles: int,
    n_atoms: int,
) -> torch.Tensor:
    hist = torch.zeros(n_roles, n_atoms, dtype=torch.float64)
    for role, atom in pairs:
        if 0 <= role < n_roles and 0 <= atom < n_atoms:
            hist[role, atom] += 1.0
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    p = p.flatten() + eps
    q = q.flatten() + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p.log() - q.log())).sum())


def _rectangularity(
    pairs: Sequence[RoleAtomPair],
    *,
    n_roles: int,
    n_atoms: int,
) -> float:
    hist = _joint_histogram(pairs, n_roles=n_roles, n_atoms=n_atoms)
    if hist.sum() <= 0:
        return 0.0
    role = hist.sum(dim=1, keepdim=True)
    atom = hist.sum(dim=0, keepdim=True)
    return _kl(hist, role @ atom)


def _entropy_from_counts(counts: Iterable[int]) -> float:
    values = [float(c) for c in counts if c > 0]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    probs = [v / total for v in values]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(values))


def _gini_from_counts(counts: Iterable[int]) -> float:
    values = sorted(float(c) for c in counts if c > 0)
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    n = len(values)
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


def _build_run_state(
    *,
    dim: int,
    device: str,
    seed: int,
    n_traces: int,
    n_roles: int,
    n_atoms: int,
    window_size: int,
    skew_concentration: int,
):
    substrate = TorchFHRR(dim=dim, seed=seed, device=device)
    positions = build_position_vectors(substrate, count=n_roles)
    codebook = substrate.random_vectors(n_atoms)
    windows = _generate_skewed_windows(
        n_traces=n_traces,
        n_roles=n_roles,
        n_atoms=n_atoms,
        window_size=window_size,
        skew_concentration=skew_concentration,
        seed=seed,
    )

    memory = TracedHopfieldMemory[int](substrate, snapshot_k=8)
    traces: List[TrajectoryTrace] = []
    gate_rng = random.Random(seed + 10_000)
    gate_signals: List[float] = []

    for idx, window in enumerate(windows):
        encoded, encoder_terms = encode_window_with_provenance(
            substrate,
            positions[:window_size],
            codebook,
            window,
        )
        memory.store(encoded, label=idx)
        traces.append(
            _seeded_trace(
                query=encoded,
                encoder_terms=list(encoder_terms),
                final_top_index=idx,
                final_top_score=0.45,
            )
        )
        gate_signals.append(gate_rng.uniform(0.1, 1.0))

    return substrate, positions, codebook, memory, traces, gate_signals


def run_condition(
    *,
    condition: str,
    config: ReplayConfig,
    seed: int,
    dim: int,
    device: str,
    n_traces: int,
    n_roles: int,
    n_atoms: int,
    window_size: int,
    skew_concentration: int,
    replay_cycles: int,
) -> ConditionResult:
    condition_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(condition))
    torch.manual_seed(seed * 101 + condition_offset % 997)
    (
        substrate,
        positions,
        codebook,
        memory,
        traces,
        gate_signals,
    ) = _build_run_state(
        dim=dim,
        device=device,
        seed=seed,
        n_traces=n_traces,
        n_roles=n_roles,
        n_atoms=n_atoms,
        window_size=window_size,
        skew_concentration=skew_concentration,
    )

    d_eff_initial = float(substrate.d_eff(memory._pattern_matrix()).detach().cpu())
    consolidation = ConsolidationState(
        ConsolidationConfig(m=4, alpha=0.25, death_threshold=0.0, death_window=10_000),
        device=device,
    )
    unified = UnifiedReplayMemory[int](
        substrate=substrate,
        memory=memory,
        consolidation=consolidation,
        config=config,
        replay_position_vectors=positions,
        replay_codebook=codebook,
    )
    unified.attach_initial_patterns()
    for trace, gate in zip(traces, gate_signals):
        unified.store.add(trace, gate_signal=gate)

    candidate_terms: List[RoleAtomPair] = []
    candidate_winners: List[int] = []
    candidate_scores: List[float] = []

    def candidate_handler(trace: TrajectoryTrace) -> Optional[int]:
        if trace.encoder_terms:
            candidate_terms.extend((int(r), int(a)) for r, a in trace.encoder_terms)
        if trace.final_top_index is not None:
            candidate_winners.append(int(trace.final_top_index))
        candidate_scores.append(float(trace.final_top_score))
        new_idx = memory.stored_count
        memory.store(trace.final_state.detach().clone(), label=new_idx)
        return new_idx

    sampled_total = 0
    candidate_count = 0
    for _ in range(replay_cycles):
        cycle = unified.run_replay_cycle(
            beta=5.0,
            max_iter=8,
            candidate_handler=candidate_handler,
        )
        sampled_total += int(cycle.get("sampled", 0))
        candidate_count += int(cycle.get("candidates", 0))

    u = unified.consolidation.u
    u2_max = float(u[:, 1].max().detach().cpu()) if u.shape[1] > 1 and u.shape[0] else 0.0
    u3_max = float(u[:, 2].max().detach().cpu()) if u.shape[1] > 2 and u.shape[0] else 0.0
    d_eff_final = float(substrate.d_eff(memory._pattern_matrix()).detach().cpu())

    role_counts = [0 for _ in range(n_roles)]
    atom_counts = [0 for _ in range(n_atoms)]
    for role, atom in candidate_terms:
        if 0 <= role < n_roles:
            role_counts[role] += 1
        if 0 <= atom < n_atoms:
            atom_counts[atom] += 1
    winner_counts: Dict[int, int] = {}
    for winner in candidate_winners:
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    return ConditionResult(
        condition=condition,
        seed=seed,
        sampled_total=sampled_total,
        candidate_count=candidate_count,
        candidate_accept_rate=candidate_count / max(sampled_total, 1),
        candidate_cells=len(set(candidate_terms)),
        candidate_rect=_rectangularity(
            candidate_terms,
            n_roles=n_roles,
            n_atoms=n_atoms,
        ),
        candidate_role_entropy=_entropy_from_counts(role_counts),
        candidate_atom_entropy=_entropy_from_counts(atom_counts),
        candidate_winner_unique=len(winner_counts),
        candidate_winner_gini=_gini_from_counts(winner_counts.values()),
        mean_final_top_score=(
            statistics.fmean(candidate_scores) if candidate_scores else 0.0
        ),
        store_final=len(unified.store),
        memory_size_final=memory.stored_count,
        consolidation_n_patterns=unified.consolidation.n_patterns,
        u2_max=u2_max,
        u3_max=u3_max,
        d_eff_initial=d_eff_initial,
        d_eff_final=d_eff_final,
    )


def _aggregate(rows: Sequence[ConditionResult]) -> Dict[str, Dict[str, float]]:
    by_condition: Dict[str, List[ConditionResult]] = {}
    for row in rows:
        by_condition.setdefault(row.condition, []).append(row)

    aggregate: Dict[str, Dict[str, float]] = {}
    metric_names = [
        "sampled_total",
        "candidate_count",
        "candidate_accept_rate",
        "candidate_cells",
        "candidate_rect",
        "candidate_role_entropy",
        "candidate_atom_entropy",
        "candidate_winner_unique",
        "candidate_winner_gini",
        "mean_final_top_score",
        "store_final",
        "memory_size_final",
        "consolidation_n_patterns",
        "u2_max",
        "u3_max",
        "d_eff_initial",
        "d_eff_final",
    ]
    for condition, condition_rows in by_condition.items():
        aggregate[condition] = {"n_seeds": float(len(condition_rows))}
        for name in metric_names:
            values = [float(getattr(row, name)) for row in condition_rows]
            aggregate[condition][f"{name}_mean"] = statistics.fmean(values)
            aggregate[condition][f"{name}_min"] = min(values)
            aggregate[condition][f"{name}_max"] = max(values)
    return aggregate


def _print_rows(rows: Sequence[ConditionResult]) -> None:
    header = (
        f"{'condition':<21} {'seed':>5} {'samp':>5} {'cand':>5} "
        f"{'acc':>5} {'cells':>5} {'rect':>7} {'aH':>5} {'winU':>5} "
        f"{'score':>6} {'store':>5} {'mem':>5} {'u3':>7} {'dEff':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.condition:<21} {row.seed:>5} {row.sampled_total:>5} "
            f"{row.candidate_count:>5} {row.candidate_accept_rate:>5.2f} "
            f"{row.candidate_cells:>5} {row.candidate_rect:>7.4f} "
            f"{row.candidate_atom_entropy:>5.2f} "
            f"{row.candidate_winner_unique:>5} "
            f"{row.mean_final_top_score:>6.3f} {row.store_final:>5} "
            f"{row.memory_size_final:>5} {row.u3_max:>7.3f} "
            f"{row.d_eff_final:>7.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 11, 23])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-traces", type=int, default=128)
    parser.add_argument("--n-roles", type=int, default=8)
    parser.add_argument("--n-atoms", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--skew-concentration", type=int, default=4)
    parser.add_argument("--replay-cycles", type=int, default=12)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--resolve-threshold", type=float, default=0.2)
    parser.add_argument("--smoothing-alpha", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "standard",
            "range_skip",
            "range_closest",
            "range_rebind_single",
            "range_rebind_window",
        ],
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_range_shaped_downstream_pilot/results.json",
    )
    args = parser.parse_args()

    configs = _condition_configs(
        replay_batch_size=args.replay_batch_size,
        resolve_threshold=args.resolve_threshold,
        store_capacity=args.n_traces * 2,
        smoothing_alpha=args.smoothing_alpha,
    )
    unknown = sorted(set(args.conditions) - set(configs))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    print(
        "Phase 4 range-shaped downstream pilot "
        f"seeds={args.seeds} dim={args.dim} n_traces={args.n_traces} "
        f"n_roles={args.n_roles} n_atoms={args.n_atoms} "
        f"window_size={args.window_size} cycles={args.replay_cycles}"
    )

    rows: List[ConditionResult] = []
    for seed in args.seeds:
        for condition in args.conditions:
            row = run_condition(
                condition=condition,
                config=configs[condition],
                seed=seed,
                dim=args.dim,
                device=args.device,
                n_traces=args.n_traces,
                n_roles=args.n_roles,
                n_atoms=args.n_atoms,
                window_size=args.window_size,
                skew_concentration=args.skew_concentration,
                replay_cycles=args.replay_cycles,
            )
            rows.append(row)

    _print_rows(rows)
    aggregate = _aggregate(rows)
    print("\nAggregates:")
    for condition in args.conditions:
        stats = aggregate[condition]
        print(
            f"  {condition:<21} cand={stats['candidate_count_mean']:.1f} "
            f"cells={stats['candidate_cells_mean']:.1f} "
            f"rect={stats['candidate_rect_mean']:.4f} "
            f"atomH={stats['candidate_atom_entropy_mean']:.3f} "
            f"winnerU={stats['candidate_winner_unique_mean']:.1f} "
            f"u3={stats['u3_max_mean']:.4f} "
            f"dEff={stats['d_eff_final_mean']:.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "seeds": args.seeds,
                    "dim": args.dim,
                    "device": args.device,
                    "n_traces": args.n_traces,
                    "n_roles": args.n_roles,
                    "n_atoms": args.n_atoms,
                    "window_size": args.window_size,
                    "skew_concentration": args.skew_concentration,
                    "replay_cycles": args.replay_cycles,
                    "replay_batch_size": args.replay_batch_size,
                    "resolve_threshold": args.resolve_threshold,
                    "smoothing_alpha": args.smoothing_alpha,
                    "conditions": args.conditions,
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
