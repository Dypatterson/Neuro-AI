"""Pre-gate 2 — Range-shaped replay sampler diagnostic.

Codex-recommended pre-Phase-5'-commit gate testing whether range-shaped
replay (Dorrell-Whittington ICLR 2025) can produce data-side rectangular
joint support over (role, atom) when running over realistic Phase 4
trace structure. The brainstorm smoke
(brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_a_range_shaped_replay.py)
established that the sampler factorizes a FakeTrace buffer (KL-to-factored
2.06 -> 0.03). This experiment ports the sampler to real `TrajectoryTrace`
+ `ReplayStore` and verifies the rectangularization claim holds on
realistic Phase 4 data.

What this is NOT
----------------
- Not a full Phase 4 consolidation experiment with range-shaped sampling.
  That requires wiring `RangeShapedReplaySampler` into `UnifiedReplayMemory`
  and re-running Phase 5 ΔE at n>=10 — multi-day work named here but
  deferred.
- Not Phase 5 verified evidence. Per phase-5-checklist.md:10-16, verified
  means n_seeds >= 10 against the ΔE headline. This is an n=3 sampler
  diagnostic.

What this IS
------------
A scoped sampler diagnostic: build the `RangeShapedReplaySampler` against
the existing `ReplayStore` + `TrajectoryTrace` API (no new fields needed —
the S1 trace-schema extension is already wired), populate the store with
synthetic-but-realistic traces carrying known (role, atom) co-occurrence
skew, and verify the sampler rectangularizes the joint distribution at
production scale on real Phase 4 dataclasses. Compares baseline
`ReplayStore.sample()` priority-weighted sampling against the range-shaped
variant.

Anti-homunculus check
---------------------
The sampler reads buffer state and emits per-call sampling decisions
governed by marginal distributions. No controller decides which traces
to replay based on metrics; the sampling weights are static functions
of the buffer's encoder_terms distribution. ✅ PASS — sampling discipline,
not arbitration.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window_with_provenance,
)
from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler
from energy_memory.phase4.replay_loop import ReplayStore
from energy_memory.phase4.trajectory import TrajectoryTrace
from energy_memory.substrate.torch_fhrr import TorchFHRR


# -----------------------------------------------------------------------------
# Synthetic Phase 4 trace generator
# -----------------------------------------------------------------------------


def _generate_skewed_traces(
    substrate: TorchFHRR,
    *,
    n_traces: int,
    n_roles: int,
    n_atoms: int,
    window_size: int,
    skew_concentration: int,
    seed: int,
) -> List[TrajectoryTrace]:
    """Generate synthetic Phase 4 traces with known (role, atom) skew.

    Each trace is a window of `window_size` (role, atom) bindings.
    Within a trace, roles are sequential (0, 1, ..., window_size-1).
    Atoms are sampled from a skewed distribution that concentrates each
    role's atoms in a small subset of the atom codebook — mimicking the
    natural co-occurrence structure of language (role-position-dependent
    word distributions).

    Specifically, for role r the atom is sampled uniformly from the
    `skew_concentration` atoms in the range [r * skew_concentration,
    (r+1) * skew_concentration), modulo n_atoms. This produces clear
    role-atom correlation that range-shaping should break.

    Each trace's TrajectoryTrace is built with:
      - query = encode_window_with_provenance(...)
      - encoder_terms = the (role, atom) tuples for that window
      - gate_signal = random in [0.1, 1.0]
    """
    rng = random.Random(seed)
    positions = build_position_vectors(substrate, count=n_roles)
    codebook = substrate.random_vectors(n_atoms)

    traces: List[TrajectoryTrace] = []
    gate_signals: List[float] = []
    for _ in range(n_traces):
        token_ids: List[int] = []
        for r in range(window_size):
            base = (r * skew_concentration) % n_atoms
            token_ids.append((base + rng.randrange(skew_concentration)) % n_atoms)
        bundle, encoder_terms = encode_window_with_provenance(
            substrate, positions[: len(token_ids)], codebook, token_ids
        )
        traces.append(TrajectoryTrace(
            query=bundle.detach().clone(),
            encoder_terms=list(encoder_terms),
        ))
        gate_signals.append(rng.uniform(0.1, 1.0))
    return traces, gate_signals


# -----------------------------------------------------------------------------
# Joint-distribution metrics
# -----------------------------------------------------------------------------


def _joint_histogram_from_pairs(
    pairs: Sequence[Tuple[int, int]],
    n_roles: int,
    n_atoms: int,
) -> torch.Tensor:
    H = torch.zeros(n_roles, n_atoms, dtype=torch.float64)
    for (r, a) in pairs:
        H[r, a] += 1.0
    s = H.sum().item()
    if s > 0:
        H = H / s
    return H


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    p = p.flatten() + eps
    q = q.flatten() + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p.log() - q.log())).sum())


def _rectangularity(H: torch.Tensor, eps: float = 1e-9) -> float:
    """KL distance from H to the outer product of its marginals."""
    pr = H.sum(dim=1, keepdim=True)
    pc = H.sum(dim=0, keepdim=True)
    fact = pr @ pc
    return _kl(H, fact, eps=eps)


# -----------------------------------------------------------------------------
# Diagnostic experiment
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    seed: int
    n_traces: int
    n_roles: int
    n_atoms: int
    window_size: int
    skew_concentration: int
    n_sampled: int
    baseline_n_pairs: int
    rangeshape_n_pairs: int
    buffer_joint_rect: float
    baseline_joint_rect: float
    rangeshape_joint_rect: float
    kl_baseline_to_rangeshape: float
    kl_rangeshape_to_baseline: float
    buffer_cells_used: int
    baseline_cells_reached: int
    rangeshape_cells_reached: int
    total_cells: int
    role_marginal_L1: float
    atom_marginal_L1: float
    fraction_missing_pairs: float


def _all_pairs_from_indices(
    store: ReplayStore, indices: Sequence[int]
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in indices:
        trace = store.traces[i]
        if trace.encoder_terms is None:
            continue
        pairs.extend([(int(r), int(a)) for (r, a) in trace.encoder_terms])
    return pairs


def run_cell(
    *,
    seed: int,
    n_traces: int,
    n_roles: int,
    n_atoms: int,
    window_size: int,
    skew_concentration: int,
    n_sampled: int,
    device: str,
) -> CellResult:
    substrate = TorchFHRR(dim=4096, seed=seed, device=device)
    traces, gate_signals = _generate_skewed_traces(
        substrate,
        n_traces=n_traces,
        n_roles=n_roles,
        n_atoms=n_atoms,
        window_size=window_size,
        skew_concentration=skew_concentration,
        seed=seed,
    )

    store = ReplayStore(capacity=n_traces + 10)
    for trace, g in zip(traces, gate_signals):
        store.add(trace, gate_signal=g)

    # Buffer's full (role, atom) joint, weighted by gate_signal per trace
    # (matching the priority signal each sampler uses).
    buffer_pairs: List[Tuple[int, int]] = []
    for t in traces:
        if t.encoder_terms is None:
            continue
        buffer_pairs.extend([(int(r), int(a)) for (r, a) in t.encoder_terms])
    H_buffer = _joint_histogram_from_pairs(buffer_pairs, n_roles, n_atoms)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 7 + 1)

    # Baseline sample: priority-weighted whole-trace picks WITH REPLACEMENT.
    # We bypass ReplayStore.sample() because it caps n at len(traces) and
    # samples WITHOUT replacement (line 262 of replay_loop.py) — so with
    # 400 traces and n_sampled=1000, store.sample() would return all 400
    # traces once and yield only 400 * window_size pairs. The real Phase 4
    # consolidation does many small replay_batch_size draws across an
    # epoch, which is equivalent to a with-replacement priority-weighted
    # sequence. Match that with a direct multinomial on the priorities.
    # tag_count=1 and suppression=1 in this synthetic setup (no overlap
    # collapse, no suppression decay), so priority ≈ gate_signal.
    priorities = torch.tensor(store.gate_signals, dtype=torch.float32)
    priorities = priorities / priorities.sum()
    baseline_indices = torch.multinomial(
        priorities, n_sampled, replacement=True, generator=generator
    ).tolist()
    baseline_pairs = _all_pairs_from_indices(store, baseline_indices)
    H_baseline = _joint_histogram_from_pairs(baseline_pairs, n_roles, n_atoms)
    baseline_n_pairs = len(baseline_pairs)

    # Range-shaped sample: (role, atom)-factored. Use sample_pairs to
    # test the SAMPLER's algorithm (which (role, atom) pairs it emits),
    # not what's available in the buffer. This matches the brainstorm
    # smoke's approach: emit the pair regardless of whether a stored
    # trace backs it (the consolidation step rebinds on the fly via the
    # substrate for missing pairs).
    rs_sampler = RangeShapedReplaySampler(store)
    generator.manual_seed(seed * 7 + 2)
    # Sample n_sampled * window_size pairs so total pair counts roughly
    # match baseline (baseline emits window_size pairs per trace).
    n_rs_pairs_target = n_sampled * window_size
    rs_pair_triples = rs_sampler.sample_pairs(
        n=n_rs_pairs_target, generator=generator
    )
    rs_pairs = [(r, a) for (r, a, _) in rs_pair_triples]
    H_rangeshape = _joint_histogram_from_pairs(rs_pairs, n_roles, n_atoms)
    # Fraction of sampled pairs that are NOT backed by an existing
    # buffer trace (would need rebind-on-the-fly in a real Phase 4
    # integration).
    fraction_missing = (
        sum(1 for (_, _, tix) in rs_pair_triples if tix is None)
        / max(1, len(rs_pair_triples))
    )

    # Marginal preservation: range-shaped should produce marginals close
    # to baseline's (it samples roles and atoms independently from their
    # buffer-derived weights).
    role_marg_baseline = H_baseline.sum(dim=1)
    role_marg_rangeshape = H_rangeshape.sum(dim=1)
    atom_marg_baseline = H_baseline.sum(dim=0)
    atom_marg_rangeshape = H_rangeshape.sum(dim=0)

    total_cells = n_roles * n_atoms

    return CellResult(
        seed=seed,
        n_traces=n_traces,
        n_roles=n_roles,
        n_atoms=n_atoms,
        window_size=window_size,
        skew_concentration=skew_concentration,
        n_sampled=n_sampled,
        baseline_n_pairs=baseline_n_pairs,
        rangeshape_n_pairs=len(rs_pairs),
        buffer_joint_rect=_rectangularity(H_buffer),
        baseline_joint_rect=_rectangularity(H_baseline),
        rangeshape_joint_rect=_rectangularity(H_rangeshape),
        kl_baseline_to_rangeshape=_kl(H_baseline, H_rangeshape),
        kl_rangeshape_to_baseline=_kl(H_rangeshape, H_baseline),
        buffer_cells_used=int((H_buffer > 0).sum()),
        baseline_cells_reached=int((H_baseline > 0).sum()),
        rangeshape_cells_reached=int((H_rangeshape > 0).sum()),
        total_cells=total_cells,
        role_marginal_L1=float((role_marg_baseline - role_marg_rangeshape).abs().sum()),
        atom_marginal_L1=float((atom_marg_baseline - atom_marg_rangeshape).abs().sum()),
        fraction_missing_pairs=fraction_missing,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 11, 23])
    parser.add_argument("--n_traces", type=int, default=400)
    parser.add_argument("--n_roles", type=int, default=8)
    parser.add_argument("--n_atoms", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--skew_concentration", type=int, default=4,
        help="Each role's atoms drawn from a contiguous block of this size.",
    )
    parser.add_argument("--n_sampled", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="reports/phase5_range_shaped_replay_gate/results.json",
    )
    args = parser.parse_args()

    print(
        f"seeds={args.seeds}  n_traces={args.n_traces}  "
        f"n_roles={args.n_roles}  n_atoms={args.n_atoms}  "
        f"window_size={args.window_size}  "
        f"skew_concentration={args.skew_concentration}  "
        f"n_sampled={args.n_sampled}  device={args.device}"
    )

    results: List[CellResult] = []
    for seed in args.seeds:
        r = run_cell(
            seed=seed,
            n_traces=args.n_traces,
            n_roles=args.n_roles,
            n_atoms=args.n_atoms,
            window_size=args.window_size,
            skew_concentration=args.skew_concentration,
            n_sampled=args.n_sampled,
            device=args.device,
        )
        results.append(r)
        print(
            f"  seed={r.seed}  "
            f"buf_rect={r.buffer_joint_rect:.4f}  "
            f"base_rect={r.baseline_joint_rect:.4f}  "
            f"rs_rect={r.rangeshape_joint_rect:.4f}  "
            f"cells base={r.baseline_cells_reached}/{r.total_cells} "
            f"rs={r.rangeshape_cells_reached}/{r.total_cells}  "
            f"base_pairs={r.baseline_n_pairs}  rs_pairs={r.rangeshape_n_pairs}  "
            f"missing={r.fraction_missing_pairs:.3f}  "
            f"role_L1={r.role_marginal_L1:.3f}  "
            f"atom_L1={r.atom_marginal_L1:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seeds": args.seeds,
            "n_traces": args.n_traces,
            "n_roles": args.n_roles,
            "n_atoms": args.n_atoms,
            "window_size": args.window_size,
            "skew_concentration": args.skew_concentration,
            "n_sampled": args.n_sampled,
            "device": args.device,
        },
        "results": [
            {
                "seed": r.seed,
                "baseline_n_pairs": r.baseline_n_pairs,
                "rangeshape_n_pairs": r.rangeshape_n_pairs,
                "buffer_joint_rect": r.buffer_joint_rect,
                "baseline_joint_rect": r.baseline_joint_rect,
                "rangeshape_joint_rect": r.rangeshape_joint_rect,
                "kl_baseline_to_rangeshape": r.kl_baseline_to_rangeshape,
                "kl_rangeshape_to_baseline": r.kl_rangeshape_to_baseline,
                "buffer_cells_used": r.buffer_cells_used,
                "baseline_cells_reached": r.baseline_cells_reached,
                "rangeshape_cells_reached": r.rangeshape_cells_reached,
                "total_cells": r.total_cells,
                "role_marginal_L1": r.role_marginal_L1,
                "atom_marginal_L1": r.atom_marginal_L1,
                "fraction_missing_pairs": r.fraction_missing_pairs,
            }
            for r in results
        ],
        "aggregates": {
            "buffer_joint_rect_mean": (
                sum(r.buffer_joint_rect for r in results) / len(results)
            ),
            "baseline_joint_rect_mean": (
                sum(r.baseline_joint_rect for r in results) / len(results)
            ),
            "rangeshape_joint_rect_mean": (
                sum(r.rangeshape_joint_rect for r in results) / len(results)
            ),
            "baseline_cells_reached_mean": (
                sum(r.baseline_cells_reached for r in results) / len(results)
            ),
            "rangeshape_cells_reached_mean": (
                sum(r.rangeshape_cells_reached for r in results) / len(results)
            ),
            "fraction_missing_pairs_mean": (
                sum(r.fraction_missing_pairs for r in results) / len(results)
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
