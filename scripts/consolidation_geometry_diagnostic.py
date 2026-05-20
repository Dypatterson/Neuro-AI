"""Consolidation-geometry diagnostic (d̄, d_eff) over a substrate snapshot.

Operationalization of [consolidation-geometry-diagnostic.md](../notes/emergent-codebook/consolidation-geometry-diagnostic.md)
adapted to the substrate-state regime: the original spec computed d̄ and
d_eff over per-atom *context-bag history* during live consolidation. Our
snapshots don't carry that history, so we substitute a substrate-state
operationalization that answers the same question for the path-decision
diagnostic in [report 044].

For each substrate snapshot, compute:

- **Substrate-level d̄, d_eff** — over the full set of stored patterns:
    - d̄ = mean pairwise FHRR cosine *distance* (= 1 − similarity) across
      all pattern pairs. High → atoms spread out in FHRR space.
    - d_eff = participation ratio of the pattern matrix's complex
      Hermitian covariance: (Σ λ)² / Σ λ². High → variation spans many
      directions. Low → atoms cluster in a low-dim subspace.
- **Per-atom d̄_t, d_eff_t** — for each atom t, compute the same two
  numbers over t's k-NN cluster (the k nearest atoms in FHRR space).
  This is the substrate-state analogue of the spec's per-atom
  context-bag d̄/d_eff.
- **Regime classification** at β=10 (the project working value): per the
  spec, θ' ≈ 1/β = 0.1. Atom t is "tight" if d̄_t < θ', "spread"
  otherwise. The substrate is *substrate-tight* if substrate-level
  d̄ < θ', else *substrate-spread*.

Anti-homunculus discipline: this is a *measurement*, not an actuator. No
mechanism reads d̄ / d_eff and decides which atoms to keep. The
diagnostic-actuator dynamic-form session (STATUS blocker #4) is the
right venue for re-expressing whatever response to this diagnostic the
architecture should have, as a continuous local dynamic of which d̄ /
d_eff is a fast-timescale snapshot — not a rule of the form
"if d̄_t > θ' then kill atom t".

Usage:
  python scripts/consolidation_geometry_diagnostic.py \\
    --snapshot reports/phase5_snapshots_local/seed17/phase3_phase4_w4_step1500.pt \\
    --output reports/phase5_geometry_diag/seed17_step1500.json \\
    --k-nn 5 --beta 10.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch


def _pairwise_fhrr_similarity(patterns: torch.Tensor) -> torch.Tensor:
    """Return [N, N] mean-Re(a*·b) similarity matrix among complex patterns.

    For normalized FHRR vectors with |x_i| = 1, diagonal is exactly 1.
    """
    n, d = patterns.shape
    # X X^H  but using conj on the second argument keeps the math right:
    # sim[i,j] = (1/D) Σ_k Re(conj(x_ik) x_jk)
    inner = patterns @ patterns.conj().T  # [N, N] complex
    return inner.real / d


def _pairwise_fhrr_distance(patterns: torch.Tensor) -> torch.Tensor:
    return 1.0 - _pairwise_fhrr_similarity(patterns)


def _participation_ratio(patterns: torch.Tensor) -> float:
    """(Σ λ)² / Σ λ². NaN for empty input.

    Uses the Gram matrix X̃ X̃^H / N (shape [N, N]) instead of the
    feature covariance X^H X / N (shape [D, D]). They share the same
    non-zero eigenvalues, so participation ratio is identical, but
    eigh on [N, N] is O(N³) vs O(D³) — for N=1024, D=4096 this is
    a ~64× speedup. The 4096×4096 complex eigh on CPU is intractable
    when called once per atom × 1024 atoms × 5 snapshots.
    """
    n = patterns.shape[0]
    if n < 2:
        return float("nan")
    centered = patterns - patterns.mean(dim=0, keepdim=True)
    # Gram matrix is [N, N] complex Hermitian.
    gram = centered @ centered.conj().T / n
    eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
    s = eigvals.sum()
    sq = (eigvals * eigvals).sum()
    if float(sq) <= 0:
        return float("nan")
    return float(s * s / sq)


def _per_atom_diagnostics(
    patterns: torch.Tensor,
    k_nn: int,
) -> List[Dict]:
    """For each atom t, compute d̄_t, d_eff_t over its k-NN cluster.

    The atom itself is excluded from its own k-NN. If there are fewer
    than 3 atoms total, per-atom diagnostics are returned as NaN.
    """
    n = patterns.shape[0]
    out: List[Dict] = []
    if n < 2:
        for i in range(n):
            out.append({
                "atom_idx": i,
                "k_used": 0,
                "d_bar": float("nan"),
                "d_eff": float("nan"),
            })
        return out

    sim = _pairwise_fhrr_similarity(patterns)
    # Mask self by setting diagonal to -inf for top-k selection.
    sim_masked = sim.clone()
    sim_masked.fill_diagonal_(float("-inf"))

    # k_used per atom = min(k_nn, n-1).
    k_eff = min(k_nn, n - 1)
    # topk over rows: for each atom, the k_eff most-similar OTHER atoms.
    top_idx = torch.topk(sim_masked, k=k_eff, dim=1).indices  # [N, k_eff]

    for i in range(n):
        neighbors_idx = top_idx[i].tolist()
        cluster = patterns[neighbors_idx]  # [k_eff, D]
        # d̄_t = mean pairwise distance among cluster members.
        if k_eff >= 2:
            dist_mat = 1.0 - _pairwise_fhrr_similarity(cluster)
            # off-diagonal mean
            mask = ~torch.eye(k_eff, dtype=torch.bool, device=dist_mat.device)
            d_bar = float(dist_mat[mask].mean())
        else:
            d_bar = float("nan")
        # d_eff_t = participation ratio of cluster covariance.
        d_eff = _participation_ratio(cluster)
        out.append({
            "atom_idx": i,
            "k_used": k_eff,
            "d_bar": d_bar,
            "d_eff": d_eff,
        })
    return out


def _summary_stats(values: List[float]) -> Dict[str, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"n": 0, "mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "median": float("nan")}
    finite_sorted = sorted(finite)
    n = len(finite_sorted)
    return {
        "n": n,
        "mean": sum(finite) / n,
        "min": finite_sorted[0],
        "max": finite_sorted[-1],
        "median": finite_sorted[n // 2],
    }


def diagnose(snapshot_path: Path, k_nn: int, beta: float) -> Dict:
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    patterns = snap["patterns"]
    metadata = snap.get("metadata", {}) or {}
    label = snap.get("label", "")
    n_atoms, dim = patterns.shape
    theta_prime = 1.0 / beta

    # --- substrate-level ---
    if n_atoms >= 2:
        sub_dist = 1.0 - _pairwise_fhrr_similarity(patterns)
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=sub_dist.device)
        substrate_d_bar = float(sub_dist[mask].mean())
        substrate_d_bar_std = float(sub_dist[mask].std())
        substrate_d_eff = _participation_ratio(patterns)
    else:
        substrate_d_bar = float("nan")
        substrate_d_bar_std = float("nan")
        substrate_d_eff = float("nan")

    substrate_regime = (
        "tight" if (math.isfinite(substrate_d_bar)
                    and substrate_d_bar < theta_prime) else "spread"
        if math.isfinite(substrate_d_bar) else "unknown"
    )

    # --- per-atom ---
    per_atom = _per_atom_diagnostics(patterns, k_nn=k_nn)
    d_bars = [a["d_bar"] for a in per_atom]
    d_effs = [a["d_eff"] for a in per_atom]
    n_tight = sum(
        1 for d in d_bars if math.isfinite(d) and d < theta_prime
    )
    n_spread = sum(
        1 for d in d_bars if math.isfinite(d) and d >= theta_prime
    )

    return {
        "snapshot": str(snapshot_path),
        "label": label,
        "metadata": metadata,
        "n_atoms": n_atoms,
        "dim": dim,
        "beta": beta,
        "theta_prime": theta_prime,
        "k_nn": k_nn,
        # Substrate-level summary
        "substrate": {
            "d_bar_mean": substrate_d_bar,
            "d_bar_std": substrate_d_bar_std,
            "d_eff": substrate_d_eff,
            "d_eff_ratio_to_dim": (
                substrate_d_eff / dim
                if math.isfinite(substrate_d_eff) else float("nan")
            ),
            "regime": substrate_regime,
            "ratio_d_bar_to_theta_prime": (
                substrate_d_bar / theta_prime
                if math.isfinite(substrate_d_bar) else float("nan")
            ),
        },
        # Per-atom panel
        "per_atom": per_atom,
        "per_atom_summary": {
            "d_bar": _summary_stats(d_bars),
            "d_eff": _summary_stats(d_effs),
            "n_tight": n_tight,
            "n_spread": n_spread,
            "fraction_tight": (
                n_tight / (n_tight + n_spread)
                if (n_tight + n_spread) > 0 else float("nan")
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", required=True,
        help="Path to a substrate snapshot .pt (e.g. phase3_phase4_w4_step1800.pt)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the diagnostic JSON",
    )
    parser.add_argument(
        "--k-nn", type=int, default=5,
        help="Number of nearest neighbors per atom for per-atom diagnostics",
    )
    parser.add_argument(
        "--beta", type=float, default=10.0,
        help="Retrieval sharpness β; θ' ≈ 1/β. Default 10 (project working value)",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)

    result = diagnose(snapshot_path, k_nn=args.k_nn, beta=args.beta)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    sub = result["substrate"]
    pas = result["per_atom_summary"]
    print(f"[done] wrote {out_path}")
    print(f"  n_atoms={result['n_atoms']}  dim={result['dim']}  "
          f"β={args.beta}  θ'={result['theta_prime']:.4f}  k_nn={args.k_nn}")
    print(f"  substrate d̄ = {sub['d_bar_mean']:.4f} ± {sub['d_bar_std']:.4f}  "
          f"d_eff = {sub['d_eff']:.2f} "
          f"(d_eff/D = {sub['d_eff_ratio_to_dim']:.4f})  "
          f"regime = {sub['regime']} (d̄/θ' = {sub['ratio_d_bar_to_theta_prime']:.3f})")
    pa_dbar = pas["d_bar"]
    pa_deff = pas["d_eff"]
    print(f"  per-atom d̄: mean={pa_dbar['mean']:.4f}  median={pa_dbar['median']:.4f}  "
          f"min={pa_dbar['min']:.4f}  max={pa_dbar['max']:.4f}")
    print(f"  per-atom d_eff: mean={pa_deff['mean']:.2f}  median={pa_deff['median']:.2f}  "
          f"min={pa_deff['min']:.2f}  max={pa_deff['max']:.2f}")
    print(f"  n_tight={pas['n_tight']}  n_spread={pas['n_spread']}  "
          f"fraction_tight={pas['fraction_tight']:.3f}")


if __name__ == "__main__":
    main()
