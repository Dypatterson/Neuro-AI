"""Phase-3 sidebar: LSR vs softmax kernel beta-sweep at D=4096.

Tests the central claim from Dense Associative Memory with Epanechnikov
Energy (arXiv:2506.10801): that the log-sum-ReLU separation function
admits an intermediate-beta regime in which the M stored patterns
remain stable fixed points *and* centroid blend attractors coexist as
additional stable points. The softmax kernel exhibits a sharp
generalization/memorization disjoint at the same dimensionality, so a
side-by-side sweep on identical cues is the cleanest first probe.

Headline metric (per phase): the joint score
    J(beta) = clean_recall@1 * ambiguous_blend_fraction
on the LSR kernel. A non-zero J(beta) at any beta means the two
regimes coexist; the maximizing beta locates the emergent zone.
A near-zero J(beta) for all beta on softmax is the control.

Drill-downs: clean top-1 recall, ambiguous-cue entropy, ambiguous-cue
similarity to true centroid vs. to nearest single pattern, settled-energy
trace length. No controller, no per-cue routing -- the same retrieve()
call is made for both cue types.

Run:
    PYTHONPATH=src .venv/bin/python experiments/25_lsr_kernel_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class SweepRow:
    kernel: str
    beta: float
    clean_top1: float
    clean_mean_entropy: float
    ambiguous_blend_fraction: float
    ambiguous_mean_entropy: float
    ambiguous_sim_centroid: float
    ambiguous_sim_nearest: float
    mean_iterations: float

    @property
    def joint_score(self) -> float:
        return self.clean_top1 * self.ambiguous_blend_fraction


def _bundle_centroid(substrate: TorchFHRR, patterns: torch.Tensor) -> torch.Tensor:
    """Mean of a set of stored patterns, projected back to the substrate manifold."""
    return substrate.normalize(patterns.mean(dim=0))


def _make_cues(
    substrate: TorchFHRR,
    patterns: torch.Tensor,
    num_clean: int,
    num_ambiguous: int,
    blend_size: int,
    clean_noise: float,
    seed: int,
):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    n = patterns.shape[0]
    clean_indices = torch.randperm(n, generator=rng)[:num_clean].tolist()
    clean_cues = [(idx, substrate.perturb(patterns[idx], noise=clean_noise)) for idx in clean_indices]

    ambiguous_cues = []
    for _ in range(num_ambiguous):
        members = torch.randperm(n, generator=rng)[:blend_size].tolist()
        centroid = _bundle_centroid(substrate, patterns[members])
        ambiguous_cues.append((tuple(members), centroid))
    return clean_cues, ambiguous_cues


def _evaluate(
    memory: TorchHopfieldMemory[int],
    substrate: TorchFHRR,
    patterns: torch.Tensor,
    clean_cues,
    ambiguous_cues,
    *,
    kernel: str,
    beta: float,
    max_iter: int,
) -> SweepRow:
    clean_hits = 0
    clean_entropy = 0.0
    iteration_counts: List[int] = []

    for true_idx, cue in clean_cues:
        result = memory.retrieve(cue, beta=beta, max_iter=max_iter, kernel=kernel)
        clean_hits += int(result.top_index == true_idx)
        clean_entropy += result.entropy
        iteration_counts.append(result.iterations)

    blend_hits = 0
    ambig_entropy = 0.0
    sim_centroid_sum = 0.0
    sim_nearest_sum = 0.0
    for members, cue in ambiguous_cues:
        result = memory.retrieve(cue, beta=beta, max_iter=max_iter, kernel=kernel)
        settled = result.state
        member_patterns = patterns[list(members)]
        centroid = _bundle_centroid(substrate, member_patterns)
        sim_to_centroid = float(substrate.similarity(settled, centroid))
        # Nearest single stored pattern's similarity to the settled state.
        sims_all = substrate.similarity_matrix(settled, patterns)
        sim_to_nearest = float(sims_all.max().detach().cpu())
        sim_centroid_sum += sim_to_centroid
        sim_nearest_sum += sim_to_nearest
        # Blend success: settled state is closer to the cue's true centroid than
        # to its single nearest stored pattern. This is the emergent attractor
        # condition (a centroid that is itself an attractor, not the same as
        # any single stored pattern).
        if sim_to_centroid > sim_to_nearest:
            blend_hits += 1
        ambig_entropy += result.entropy
        iteration_counts.append(result.iterations)

    n_clean = max(len(clean_cues), 1)
    n_ambig = max(len(ambiguous_cues), 1)
    return SweepRow(
        kernel=kernel,
        beta=beta,
        clean_top1=clean_hits / n_clean,
        clean_mean_entropy=clean_entropy / n_clean,
        ambiguous_blend_fraction=blend_hits / n_ambig,
        ambiguous_mean_entropy=ambig_entropy / n_ambig,
        ambiguous_sim_centroid=sim_centroid_sum / n_ambig,
        ambiguous_sim_nearest=sim_nearest_sum / n_ambig,
        mean_iterations=sum(iteration_counts) / max(len(iteration_counts), 1),
    )


def run(
    *,
    dim: int,
    num_patterns: int,
    num_clean: int,
    num_ambiguous: int,
    blend_size: int,
    clean_noise: float,
    betas: Sequence[float],
    kernels: Sequence[str],
    max_iter: int,
    seed: int,
    device: str,
) -> List[SweepRow]:
    substrate = TorchFHRR(dim=dim, seed=seed, device=device)
    patterns = substrate.random_vectors(num_patterns)
    memory: TorchHopfieldMemory[int] = TorchHopfieldMemory(substrate)
    for index, pattern in enumerate(patterns):
        memory.store(pattern, label=index)

    clean_cues, ambiguous_cues = _make_cues(
        substrate,
        patterns,
        num_clean=num_clean,
        num_ambiguous=num_ambiguous,
        blend_size=blend_size,
        clean_noise=clean_noise,
        seed=seed + 1,
    )

    rows: List[SweepRow] = []
    for kernel in kernels:
        for beta in betas:
            rows.append(
                _evaluate(
                    memory,
                    substrate,
                    patterns,
                    clean_cues,
                    ambiguous_cues,
                    kernel=kernel,
                    beta=beta,
                    max_iter=max_iter,
                )
            )
    return rows


def _print_table(rows: Sequence[SweepRow]) -> None:
    header = (
        f"{'kernel':<8} {'beta':>10} {'clean@1':>9} {'clean_H':>9} "
        f"{'blend':>8} {'amb_H':>8} {'simC':>7} {'simN':>7} {'J':>7} {'iters':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.kernel:<8} {r.beta:>10.4g} {r.clean_top1:>9.3f} {r.clean_mean_entropy:>9.3f} "
            f"{r.ambiguous_blend_fraction:>8.3f} {r.ambiguous_mean_entropy:>8.3f} "
            f"{r.ambiguous_sim_centroid:>7.3f} {r.ambiguous_sim_nearest:>7.3f} "
            f"{r.joint_score:>7.3f} {r.mean_iterations:>6.2f}"
        )


def _default_betas() -> List[float]:
    # Log-spaced sweep covering the project's identified crossover zone
    # (0.001-0.01) and extending into the sharp-memorization regime.
    return [10 ** (x / 4.0) for x in range(-16, 9)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--patterns", type=int, default=64)
    parser.add_argument("--clean", type=int, default=64)
    parser.add_argument("--ambiguous", type=int, default=64)
    parser.add_argument("--blend-size", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        default=["softmax", "lsr"],
        choices=["softmax", "lsr"],
    )
    parser.add_argument("--betas", type=float, nargs="*", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to dump rows as JSON for later analysis.",
    )
    args = parser.parse_args()

    betas = args.betas if args.betas else _default_betas()
    rows = run(
        dim=args.dim,
        num_patterns=args.patterns,
        num_clean=args.clean,
        num_ambiguous=args.ambiguous,
        blend_size=args.blend_size,
        clean_noise=args.noise,
        betas=betas,
        kernels=args.kernels,
        max_iter=args.max_iter,
        seed=args.seed,
        device=args.device,
    )
    _print_table(rows)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
