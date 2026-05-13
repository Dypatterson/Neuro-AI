"""Synergy probe across the Phase 4 pipeline.

Brainstorm Idea 8 follow-through (2026-05-13): the GIB-inspired synergy
estimator (``energy_memory.diagnostics.synergy``) is a measurement
primitive. This script applies it to three real artifacts in the Phase
4 pipeline so we can see whether compositional role-filler structure
survives the dynamics:

  1. **Raw bindings.** Stored Hopfield patterns built directly from
     ``encode_window`` — bundle of bind(position, codebook[token]).
     This is the structural ceiling; if synergy is not high here, the
     substrate isn't doing FHRR compositionality.

  2. **Hopfield-settled states.** Each retrieved settled state, the
     output that downstream consumers (replay gate, decoder, candidate
     handler) actually see. Does Hopfield settling preserve the
     role-filler factorization that was bundled into the stored
     pattern?

  3. **Replay-resolved candidates.** Settled states that crossed the
     resolve_threshold during a replay cycle. These are the artifacts
     that get consolidated as new patterns. They are the
     project's actual long-term compositional inventory.

Headline metric: ``synergy_ratio = synergy(settled) / synergy(raw)``,
plus the corresponding ratio for ``synergy(candidate) / synergy(raw)``.
A ratio near 1.0 means dynamics preserve structure; a ratio near 0
means dynamics collapse compositions back to atom-like representations.

Drill-downs: the raw recover / baseline-from-role / baseline-from-binding
components at each checkpoint, plus the candidate count and the fraction
of probes where the dynamics improved or degraded synergy relative to
raw.

This is also the second of the two prerequisite checks named in
[reports/008_lsr_kernel_sweep.md](../reports/008_lsr_kernel_sweep.md)
for re-opening the LSR question: do consolidated bindings actually
show a regime disjoint in their synergy scores?

Run:
    PYTHONPATH=src .venv/bin/python experiments/28_synergy_probe_phase4.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from energy_memory.diagnostics.synergy import mean_synergy
from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class StageSummary:
    stage: str
    n_artifacts: int
    n_probes: int
    recover: float
    baseline_from_role: float
    baseline_from_binding: float
    synergy: float


@dataclass
class SeedSummary:
    seed: int
    raw: StageSummary
    settled: StageSummary
    candidates: Optional[StageSummary]
    n_candidates: int
    settled_synergy_ratio: float
    candidate_synergy_ratio: Optional[float]


def _measure(
    substrate: TorchFHRR,
    artifacts: torch.Tensor,
    artifact_windows: List[tuple],
    positions: torch.Tensor,
    codebook: torch.Tensor,
    stage_name: str,
) -> StageSummary:
    """Compute mean synergy across (artifact, position, atom) triples.

    For each artifact (one stored window or one settled state), and each
    position in its associated window, the synergy of recovering the
    atom at that position from the artifact and the position role.
    """
    roles, fillers, bindings = [], [], []
    for artifact, window in zip(artifacts, artifact_windows):
        for slot, token_id in enumerate(window):
            roles.append(positions[slot])
            fillers.append(codebook[token_id])
            bindings.append(artifact)
    measurement = mean_synergy(substrate, roles, fillers, bindings=bindings)
    return StageSummary(
        stage=stage_name,
        n_artifacts=len(artifacts),
        n_probes=len(roles),
        recover=measurement.recover,
        baseline_from_role=measurement.baseline_from_role,
        baseline_from_binding=measurement.baseline_from_binding,
        synergy=measurement.synergy,
    )


def _run_seed(
    *,
    seed: int,
    dim: int,
    window_size: int,
    codebook_size: int,
    n_windows: int,
    n_retrievals: int,
    beta: float,
    cue_noise: float,
    replay_every: int,
    replay_batch_size: int,
    resolve_threshold: float,
    store_threshold: float,
) -> SeedSummary:
    substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")

    # Codebook of atom vectors; position vectors.
    codebook = substrate.random_vectors(codebook_size)
    positions = build_position_vectors(substrate, window_size)
    positions_tensor = torch.stack(list(positions), dim=0)

    # Random windows of token IDs.
    rng = torch.Generator(device="cpu").manual_seed(seed + 19)
    windows = [
        tuple(
            int(t.item())
            for t in torch.randint(0, codebook_size, (window_size,), generator=rng)
        )
        for _ in range(n_windows)
    ]

    # Build the traced Hopfield memory and store encoded windows.
    memory = TracedHopfieldMemory[str](substrate, snapshot_k=8)
    raw_bindings: List[torch.Tensor] = []
    for window in windows:
        encoded = encode_window(substrate, positions, codebook, window)
        memory.store(encoded, label=str(window))
        raw_bindings.append(encoded)
    raw_tensor = torch.stack(raw_bindings, dim=0)

    raw_summary = _measure(
        substrate, raw_tensor, windows, positions_tensor, codebook, "raw"
    )

    # Wire the unified replay loop with the default replay-store upgrades
    # (tag_overlap = 0.7, suppression on) so the candidate channel sees
    # the same dynamics the project would run.
    consolidation = ConsolidationState(
        ConsolidationConfig(m=4, alpha=0.25, death_threshold=0.01, death_window=10_000),
        device="cpu",
    )
    replay_config = ReplayConfig(
        store_threshold=store_threshold,
        store_capacity=n_retrievals,
        resolve_threshold=resolve_threshold,
        replay_every=replay_every,
        replay_batch_size=replay_batch_size,
        max_age=10,
    )
    unified = UnifiedReplayMemory[str](
        substrate, memory, consolidation, config=replay_config
    )
    unified.attach_initial_patterns()

    # Drive perturbed retrievals; capture settled states and which window
    # they originated from (the window we cued against).
    cue_rng = torch.Generator(device="cpu").manual_seed(seed + 31)
    settled_states: List[torch.Tensor] = []
    settled_windows: List[tuple] = []
    candidate_states: List[torch.Tensor] = []
    candidate_windows: List[tuple] = []

    def on_candidate(trace) -> Optional[int]:
        # Record the candidate without re-storing it (we just measure).
        # Map back to a source window by finding the closest raw binding
        # to the trace's query — that's what the replay store was cued on.
        sims = substrate.similarity_matrix(trace.query, raw_tensor)
        src_idx = int(torch.argmax(sims).detach().cpu())
        candidate_states.append(trace.final_state.detach())
        candidate_windows.append(windows[src_idx])
        return None  # don't add to memory

    for step in range(n_retrievals):
        idx = int(torch.randint(0, n_windows, (1,), generator=cue_rng).item())
        cue = substrate.perturb(raw_tensor[idx], noise=cue_noise)
        _, trace = unified.retrieve_and_observe(cue, beta=beta, max_iter=8)
        settled_states.append(trace.final_state.detach())
        settled_windows.append(windows[idx])
        if unified.should_replay():
            unified.run_replay_cycle(
                beta=beta, max_iter=8, candidate_handler=on_candidate
            )

    settled_tensor = torch.stack(settled_states, dim=0)
    settled_summary = _measure(
        substrate,
        settled_tensor,
        settled_windows,
        positions_tensor,
        codebook,
        "settled",
    )

    candidate_summary: Optional[StageSummary] = None
    candidate_ratio: Optional[float] = None
    if candidate_states:
        candidate_tensor = torch.stack(candidate_states, dim=0)
        candidate_summary = _measure(
            substrate,
            candidate_tensor,
            candidate_windows,
            positions_tensor,
            codebook,
            "candidates",
        )
        candidate_ratio = (
            candidate_summary.synergy / raw_summary.synergy
            if raw_summary.synergy != 0
            else float("nan")
        )

    return SeedSummary(
        seed=seed,
        raw=raw_summary,
        settled=settled_summary,
        candidates=candidate_summary,
        n_candidates=len(candidate_states),
        settled_synergy_ratio=(
            settled_summary.synergy / raw_summary.synergy
            if raw_summary.synergy != 0
            else float("nan")
        ),
        candidate_synergy_ratio=candidate_ratio,
    )


def _print(seed_results: List[SeedSummary]) -> None:
    header = (
        f"{'seed':>6} {'stage':<12} {'recover':>9} {'sim_role':>9} {'sim_bind':>9} "
        f"{'synergy':>9} {'n_art':>6} {'n_probe':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in seed_results:
        for stage in (s.raw, s.settled, s.candidates):
            if stage is None:
                continue
            print(
                f"{s.seed:>6} {stage.stage:<12} {stage.recover:>9.3f} "
                f"{stage.baseline_from_role:>9.3f} {stage.baseline_from_binding:>9.3f} "
                f"{stage.synergy:>9.3f} {stage.n_artifacts:>6d} {stage.n_probes:>8d}"
            )
    print()
    print(
        f"{'seed':>6} {'n_cands':>8} {'settled/raw':>13} {'cand/raw':>11}"
    )
    print("-" * 42)
    for s in seed_results:
        c_str = (
            f"{s.candidate_synergy_ratio:>11.3f}"
            if s.candidate_synergy_ratio is not None
            else f"{'--':>11}"
        )
        print(f"{s.seed:>6} {s.n_candidates:>8d} {s.settled_synergy_ratio:>13.3f} {c_str}")


def _aggregate(seed_results: List[SeedSummary]) -> dict:
    def _mean_attr(stage_name: str, attr: str) -> float:
        values = []
        for s in seed_results:
            stage = getattr(s, stage_name)
            if stage is not None:
                values.append(getattr(stage, attr))
        return statistics.fmean(values) if values else float("nan")

    settled_ratios = [s.settled_synergy_ratio for s in seed_results]
    candidate_ratios = [
        s.candidate_synergy_ratio
        for s in seed_results
        if s.candidate_synergy_ratio is not None
    ]
    return {
        "raw_mean_synergy": _mean_attr("raw", "synergy"),
        "settled_mean_synergy": _mean_attr("settled", "synergy"),
        "candidate_mean_synergy": _mean_attr("candidates", "synergy"),
        "settled_ratio_mean": statistics.fmean(settled_ratios),
        "settled_ratio_stdev": (
            statistics.pstdev(settled_ratios) if len(settled_ratios) > 1 else 0.0
        ),
        "candidate_ratio_mean": (
            statistics.fmean(candidate_ratios) if candidate_ratios else None
        ),
        "n_seeds_with_candidates": sum(1 for s in seed_results if s.candidates is not None),
        "n_seeds": len(seed_results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--n-windows", type=int, default=32)
    parser.add_argument("--n-retrievals", type=int, default=120)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--cue-noise", type=float, default=0.15)
    parser.add_argument("--replay-every", type=int, default=20)
    parser.add_argument("--replay-batch-size", type=int, default=6)
    parser.add_argument("--resolve-threshold", type=float, default=0.5)
    parser.add_argument("--store-threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 41, 53, 67])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    seed_results: List[SeedSummary] = []
    for seed in args.seeds:
        seed_results.append(
            _run_seed(
                seed=seed,
                dim=args.dim,
                window_size=args.window_size,
                codebook_size=args.codebook_size,
                n_windows=args.n_windows,
                n_retrievals=args.n_retrievals,
                beta=args.beta,
                cue_noise=args.cue_noise,
                replay_every=args.replay_every,
                replay_batch_size=args.replay_batch_size,
                resolve_threshold=args.resolve_threshold,
                store_threshold=args.store_threshold,
            )
        )

    _print(seed_results)
    print()
    agg = _aggregate(seed_results)
    print("Aggregate:")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": vars(args) | {"out": str(args.out)},
            "seeds": [
                {
                    "seed": s.seed,
                    "n_candidates": s.n_candidates,
                    "settled_synergy_ratio": s.settled_synergy_ratio,
                    "candidate_synergy_ratio": s.candidate_synergy_ratio,
                    "raw": asdict(s.raw),
                    "settled": asdict(s.settled),
                    "candidates": asdict(s.candidates) if s.candidates else None,
                }
                for s in seed_results
            ],
            "aggregate": agg,
        }, indent=2))
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
