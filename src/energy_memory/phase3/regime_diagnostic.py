from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class AtomRegime:
    atom_id: int
    d_bar: float
    d_eff: float
    regime: str
    theta_prime_used: float


@dataclass
class CodebookRegimeDiagnostics:
    per_atom: Dict[int, AtomRegime] = field(default_factory=dict)
    regime_counts: Dict[str, int] = field(default_factory=dict)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


def pairwise_fhrr_similarity(patterns: torch.Tensor) -> torch.Tensor:
    n, d = patterns.shape
    inner = patterns @ patterns.conj().T
    return inner.real / d


def pairwise_fhrr_distance(patterns: torch.Tensor) -> torch.Tensor:
    return 1.0 - pairwise_fhrr_similarity(patterns)


def participation_ratio(patterns: torch.Tensor) -> float:
    # Gram on [N,N] shares non-zero eigvals with feature covariance — O(N^3) vs O(D^3).
    n = patterns.shape[0]
    if n < 2:
        return float("nan")
    centered = patterns - patterns.mean(dim=0, keepdim=True)
    gram = centered @ centered.conj().T / n
    eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
    s = eigvals.sum()
    sq = (eigvals * eigvals).sum()
    if float(sq) <= 0:
        return float("nan")
    return float(s * s / sq)


def summary_stats(values: List[float]) -> Dict[str, float]:
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


def per_atom_regime_diagnostics(
    patterns: torch.Tensor,
    k_nn: int,
    beta: float,
    theta_prime_fn: Optional[Callable[[float], float]] = None,
) -> Dict[int, AtomRegime]:
    # Classification rule preserved from scripts/consolidation_geometry_diagnostic.py lines 92-143.
    if theta_prime_fn is None:
        theta_prime_fn = lambda b: 1.0 / b  # noqa: E731
    theta_prime = float(theta_prime_fn(beta))

    n = patterns.shape[0]
    out: Dict[int, AtomRegime] = {}

    if n < 2:
        for i in range(n):
            out[i] = AtomRegime(
                atom_id=i,
                d_bar=float("nan"),
                d_eff=float("nan"),
                regime="borderline",
                theta_prime_used=theta_prime,
            )
        return out

    sim = pairwise_fhrr_similarity(patterns)
    sim_masked = sim.clone()
    sim_masked.fill_diagonal_(float("-inf"))

    k_eff = min(k_nn, n - 1)
    top_idx = torch.topk(sim_masked, k=k_eff, dim=1).indices

    for i in range(n):
        neighbors_idx = top_idx[i].tolist()
        cluster = patterns[neighbors_idx]
        if k_eff >= 2:
            dist_mat = 1.0 - pairwise_fhrr_similarity(cluster)
            mask = ~torch.eye(k_eff, dtype=torch.bool, device=dist_mat.device)
            d_bar = float(dist_mat[mask].mean())
        else:
            d_bar = float("nan")
        d_eff = participation_ratio(cluster)

        if not math.isfinite(d_bar):
            regime = "borderline"
        elif d_bar < theta_prime:
            regime = "tight"
        else:
            regime = "spread"

        out[i] = AtomRegime(
            atom_id=i,
            d_bar=d_bar,
            d_eff=d_eff,
            regime=regime,
            theta_prime_used=theta_prime,
        )
    return out


def compute_codebook_regime_diagnostics(
    patterns: torch.Tensor,
    *,
    k_nn: int = 8,
    beta: float = 10.0,
    theta_prime_fn: Optional[Callable[[float], float]] = None,
) -> CodebookRegimeDiagnostics:
    per_atom = per_atom_regime_diagnostics(
        patterns, k_nn=k_nn, beta=beta, theta_prime_fn=theta_prime_fn
    )
    regime_counts: Dict[str, int] = {"tight": 0, "spread": 0, "borderline": 0}
    for atom in per_atom.values():
        regime_counts[atom.regime] = regime_counts.get(atom.regime, 0) + 1

    d_bars = [a.d_bar for a in per_atom.values()]
    d_effs = [a.d_eff for a in per_atom.values()]
    theta_primes = [a.theta_prime_used for a in per_atom.values()]

    summary = {
        "d_bar": summary_stats(d_bars),
        "d_eff": summary_stats(d_effs),
        "theta_prime": summary_stats(theta_primes),
    }

    return CodebookRegimeDiagnostics(
        per_atom=per_atom,
        regime_counts=regime_counts,
        summary=summary,
    )
