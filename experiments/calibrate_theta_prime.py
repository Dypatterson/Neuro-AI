"""Empirical theta'(beta) calibration spike (Path C, C.1.4).

Runs the Geometry of Consolidation E1 protocol on the project's FHRR
substrate to back out the empirical retrieval-success boundary d-bar at
which the Hopfield layer's top-1 retrieval crosses 50% for each beta.
That crossing d-bar defines the calibrated theta'(beta), replacing the
theta' ~= 1/beta starting approximation.

Spec: notes/emergent-codebook/consolidation-geometry-diagnostic.md:172
Precommit: notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md (C.1.4)

Usage:
    PYTHONPATH=src .venv/bin/python experiments/calibrate_theta_prime.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _parse_float_list(arg: str) -> List[float]:
    return [float(x) for x in arg.split(",") if x.strip()]


def _achieved_d_bar(members: torch.Tensor) -> float:
    """Mean pairwise FHRR cosine distance among cluster members.

    FHRR cosine similarity: mean(p_i.conj * p_j).real over D components.
    Distance: 1 - similarity. Returns the mean over upper-triangular pairs.
    """
    n, d = members.shape
    if n < 2:
        return float("nan")
    sim_matrix = (members.conj() @ members.T).real / d
    iu = torch.triu_indices(n, n, offset=1)
    sims = sim_matrix[iu[0], iu[1]]
    return float((1.0 - sims).mean().detach().cpu())


def _achieved_d_eff(members: torch.Tensor) -> float:
    """Participation ratio of the centered Gram. Mirrors TorchFHRR.d_eff."""
    n = members.shape[0]
    if n < 2:
        return float("nan")
    centered = members - members.mean(dim=0, keepdim=True)
    gram = centered @ centered.conj().T / n
    tr_g = gram.diagonal().real.sum()
    tr_g_sq = (gram.abs() * gram.abs()).sum()
    if float(tr_g_sq.detach().cpu()) <= 0.0:
        return float("nan")
    return float((tr_g * tr_g / tr_g_sq).detach().cpu())


def _build_cluster(
    substrate: TorchFHRR,
    centroid: torch.Tensor,
    n_members: int,
    sigma: float,
    d_eff_target: float,
) -> torch.Tensor:
    """Generate ``n_members`` FHRR vectors around ``centroid`` with
    controlled spread.

    The covariance is restricted to a ``k = round(d_eff_target)``-dim
    subspace of phase perturbations so achieved d_eff approximates the
    target. Each member's phase offset is sigma * sum_j a_{ij} * direction_j,
    with directions iid normal across phase space and a_{ij} ~ N(0, 1/k).

    For very small sigma the achieved d-bar is ~ sigma^2 / 2 (since FHRR
    similarity ~ cos(noise) ~ 1 - noise^2/2). At large sigma the phase
    wraps and similarity asymptotes to ~0, i.e. d-bar -> 1. The caller
    bisects sigma to hit a target d-bar.
    """
    D = substrate.dim
    device = substrate.device
    k = max(1, int(round(d_eff_target)))
    # k orthogonal-ish random phase directions: just iid normal, large D
    # gives near-orthogonal directions.
    directions = torch.randn(
        (k, D), generator=substrate.generator, device="cpu"
    ).to(device)
    # Per-member coefficients with variance 1/k so total spread ~ sigma.
    coeffs = torch.randn(
        (n_members, k), generator=substrate.generator, device="cpu"
    ).to(device) / math.sqrt(k)
    phase_offsets = sigma * (coeffs @ directions)  # [n_members, D] real
    members = centroid[None, :] * torch.polar(
        torch.ones_like(phase_offsets), phase_offsets
    )
    # Re-normalize to unit magnitude per component (FHRR convention).
    members = substrate.normalize(members)
    return members


def _calibrate_sigma_for_target_d_bar(
    substrate: TorchFHRR,
    target_d_bar: float,
    n_members: int,
    d_eff_target: float,
    n_probe_clusters: int = 4,
) -> float:
    """Bisect noise sigma so achieved d-bar ~ target_d_bar.

    Searches sigma in [1e-3, 8.0]. Bisection on the monotone map
    sigma -> mean d-bar across ``n_probe_clusters`` probe clusters.
    Returns the sigma value to use for real cluster construction.
    """
    lo, hi = 1e-3, 8.0

    def _achieved_at(sigma: float) -> float:
        d_bars = []
        for _ in range(n_probe_clusters):
            centroid = substrate.random_vector()
            members = _build_cluster(
                substrate, centroid, n_members, sigma, d_eff_target
            )
            d_bars.append(_achieved_d_bar(members))
        return sum(d_bars) / len(d_bars)

    # Pin the bracket: if even hi underflows the target (saturates below it),
    # just return hi. If lo overshoots the target, return lo.
    d_bar_hi = _achieved_at(hi)
    if d_bar_hi < target_d_bar:
        return hi
    d_bar_lo = _achieved_at(lo)
    if d_bar_lo > target_d_bar:
        return lo
    for _ in range(16):
        mid = math.sqrt(lo * hi)  # geometric bisection — sigma is multiplicative
        d_bar_mid = _achieved_at(mid)
        if d_bar_mid < target_d_bar:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _run_cell(
    substrate: TorchFHRR,
    beta: float,
    target_d_bar: float,
    n_clusters: int,
    n_members: int,
    d_eff_target: float,
) -> Dict[str, float]:
    """Generate ``n_clusters`` clusters at target d-bar, store centroids
    in a TorchHopfieldMemory at this beta, and probe top-1 retrieval
    from a held-out member of each cluster.

    Returns:
        {
          "target_d_bar": ...,
          "achieved_d_bar_mean": ...,
          "achieved_d_eff_mean": ...,
          "success_rate": fraction of clusters where top-1 == correct
                          centroid index,
          "n_clusters": n_clusters,
        }
    """
    sigma = _calibrate_sigma_for_target_d_bar(
        substrate, target_d_bar, n_members, d_eff_target
    )

    centroids: List[torch.Tensor] = []
    held_out: List[torch.Tensor] = []
    achieved_d_bars: List[float] = []
    achieved_d_effs: List[float] = []

    for _ in range(n_clusters):
        centroid = substrate.random_vector()
        members = _build_cluster(
            substrate, centroid, n_members, sigma, d_eff_target
        )
        achieved_d_bars.append(_achieved_d_bar(members))
        achieved_d_effs.append(_achieved_d_eff(members))
        centroids.append(centroid)
        # Held-out probe: a member of the cluster that's NOT the centroid.
        held_out.append(members[0])

    # Store all centroids in one Hopfield memory at this beta.
    memory = TorchHopfieldMemory[int](substrate)
    for i, c in enumerate(centroids):
        memory.store(c, label=i)

    # Use single-step retrieval (max_iter=1): theta'(beta) is the
    # one-shot MHN recall boundary in the classical Modern Hopfield
    # storage capacity sense. Iterative settling at small beta diffuses
    # the query and is not what theta'(beta) characterizes.
    n_success = 0
    for i, probe in enumerate(held_out):
        result = memory.retrieve(probe, beta=beta, max_iter=1)
        if result.top_index == i:
            n_success += 1

    return {
        "target_d_bar": target_d_bar,
        "achieved_d_bar_mean": sum(achieved_d_bars) / len(achieved_d_bars),
        "achieved_d_eff_mean": sum(achieved_d_effs) / len(achieved_d_effs),
        "success_rate": n_success / n_clusters,
        "n_clusters": n_clusters,
        "sigma_used": sigma,
    }


def _empirical_theta_prime(
    success_curve: List[Tuple[float, float]],
) -> Tuple[float, bool, bool]:
    """Find smallest d-bar at which success rate drops below 50%.

    success_curve: list of (d_bar, success_rate), sorted by d_bar ascending.
    Returns: (theta_prime, boundary_above_grid, boundary_below_grid).
    """
    d_bars = [d for d, _ in success_curve]
    success = [s for _, s in success_curve]
    # If success starts below 50%, the boundary is below the grid.
    if success[0] < 0.5:
        return d_bars[0], False, True
    # Find first crossing.
    for i in range(1, len(d_bars)):
        if success[i] < 0.5:
            return d_bars[i], False, False
    # Never drops below 50% — boundary above grid.
    return d_bars[-1], True, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--betas", type=str, default="0.01,0.1,1.0,3.0,10.0,30.0,100.0"
    )
    parser.add_argument(
        "--d-bar-grid",
        type=str,
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.96,0.99",
    )
    parser.add_argument("--d-eff", type=float, default=8.0)
    parser.add_argument("--n-clusters-per-cell", type=int, default=50)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--n-cluster-members", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="notes/emergent-codebook/theta_prime_calibration.json",
    )
    args = parser.parse_args()

    betas = _parse_float_list(args.betas)
    d_bar_grid = sorted(_parse_float_list(args.d_bar_grid))

    substrate = TorchFHRR(dim=args.D, seed=args.seed)
    print(f"[calibrate_theta_prime] substrate device: {substrate.device}")
    print(f"[calibrate_theta_prime] betas: {betas}")
    print(f"[calibrate_theta_prime] d_bar grid: {d_bar_grid}")
    print(
        f"[calibrate_theta_prime] {args.n_clusters_per_cell} clusters/cell, "
        f"{args.n_cluster_members} members/cluster, D={args.D}, "
        f"d_eff={args.d_eff}"
    )

    t0 = time.time()
    calibration: Dict[str, Dict[str, object]] = {}
    for beta in betas:
        print(f"\n[calibrate_theta_prime] beta = {beta}")
        success_curve: List[Tuple[float, float]] = []
        cells: Dict[str, Dict[str, float]] = {}
        for d_bar in d_bar_grid:
            t_cell = time.time()
            cell = _run_cell(
                substrate,
                beta=beta,
                target_d_bar=d_bar,
                n_clusters=args.n_clusters_per_cell,
                n_members=args.n_cluster_members,
                d_eff_target=args.d_eff,
            )
            success_curve.append((d_bar, cell["success_rate"]))
            cells[f"{d_bar}"] = cell
            elapsed = time.time() - t_cell
            print(
                f"  d_bar={d_bar:.3f} -> success={cell['success_rate']:.3f} "
                f"(achieved d_bar={cell['achieved_d_bar_mean']:.3f}, "
                f"d_eff={cell['achieved_d_eff_mean']:.2f}, "
                f"sigma={cell['sigma_used']:.3f}, {elapsed:.1f}s)"
            )
        theta_prime, above, below = _empirical_theta_prime(success_curve)
        calibration[str(beta)] = {
            "theta_prime": theta_prime,
            "success_curve": {str(d): s for d, s in success_curve},
            "cells": cells,
            "boundary_above_grid": above,
            "boundary_below_grid": below,
        }

    elapsed_total = time.time() - t0

    out: Dict[str, object] = {
        "metadata": {
            "date": "2026-05-26",
            "D": args.D,
            "n_clusters_per_cell": args.n_clusters_per_cell,
            "n_cluster_members": args.n_cluster_members,
            "d_eff_used": args.d_eff,
            "seed": args.seed,
            "betas": betas,
            "d_bar_grid": d_bar_grid,
            "wall_clock_seconds": elapsed_total,
            "device": str(substrate.device),
        },
        "calibration": calibration,
        "approximation_baseline": {str(b): 1.0 / b for b in betas},
        "note": (
            f"Spike-level n={args.n_clusters_per_cell} per cell, "
            f"single d_eff={args.d_eff}. Replace with deeper sweep for "
            f"production use."
        ),
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        # Resolve relative to repo root (parent of experiments/).
        repo_root = Path(__file__).resolve().parent.parent
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n[calibrate_theta_prime] wrote {output_path}")
    print(f"[calibrate_theta_prime] total wall clock: {elapsed_total:.1f}s")
    print("\n=== Calibration table ===")
    print(f"{'beta':>8}  {'theta_prime':>12}  {'1/beta':>12}  {'flags':>20}")
    for b in betas:
        entry = calibration[str(b)]
        flags = []
        if entry["boundary_above_grid"]:
            flags.append("above_grid")
        if entry["boundary_below_grid"]:
            flags.append("below_grid")
        flag_str = ",".join(flags) if flags else "ok"
        print(
            f"{b:>8.4g}  {entry['theta_prime']:>12.4g}  "
            f"{1.0 / b:>12.4g}  {flag_str:>20}"
        )


if __name__ == "__main__":
    main()
