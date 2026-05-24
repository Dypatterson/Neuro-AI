"""Global pattern-pair separability diagnostic for the Phase 5 A1' substrate.

> **IMPORTANT (Codex audit 2026-05-23):** Despite the historical
> filename, this script does **NOT** compute the Fisher separation
> index defined in Varner 2026 (arXiv:2603.20115). The Varner index
> is `S = (c̄_within − c̄_between) / [½(σ_within + σ_between)]` computed
> in PCA space over a **designated functional subset vs background**
> partition — i.e. per-cue, partition-specific. This script instead
> reports a substrate-wide global pattern-pair statistic
>
>   S_global = (μ_self − μ_cross)² / (σ_self² + σ_cross²)
>
> where μ_self / σ_self are the diagonal of the cosine matrix and
> μ_cross / σ_cross are the strict upper triangle. **The brainstorm
> Finding 2 "S > 0.30 implies Varner log-prior bias works" prediction
> does not apply to this script's output** — it requires the per-cue
> Varner-correct diagnostic, which is not built here.
>
> **What this script *does* tell you faithfully:**
>
>   - μ_self ≈ 1 (sanity check that patterns are unit-norm)
>   - μ_cross / σ_cross — the overall correlation structure of the
>     stored substrate. Random unitary FHRR patterns at D have
>     μ_cross ≈ 0, σ_cross ≈ 1/√D. The A+B+A1' substrate at seed
>     17/11/23 came back with μ_cross ≈ 0.25, σ_cross ≈ 0.12 — the
>     atoms have clustered onto a manifold lower-dimensional than
>     D=4096 but are NOT collapsed to a single ray.
>   - σ_cross / (1/√D) — the multiplicative excess over the
>     theoretical FHRR crosstalk floor.
>
> **What it does NOT predict:** whether Varner log-prior bias on
> K-branch settling will fix the substrate-saturation failure mode.
> That requires either (a) the per-cue Varner-correct diagnostic
> built fresh, or (b) running the Varner spike directly.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/fisher_separation_diagnostic.py \\
        --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt

For a synthetic demonstration (no snapshot needed) run with --synthetic
to validate the script against a known-good random-FHRR baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Used only for the synthetic demonstration.
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


def global_pattern_pair_separability(patterns: torch.Tensor) -> dict:
    """Compute global pattern-pair separability statistics.

    **This is NOT the Varner 2026 Fisher separation index.** Varner's
    index is `(c̄_within − c̄_between) / [½(σ_within + σ_between)]`
    computed in PCA space over a designated functional subset vs
    background — i.e. it requires a per-cue partition. This function
    instead computes a substrate-wide global statistic:

        S_global = (μ_self − μ_cross)² / (σ_self² + σ_cross²)

    over the full pattern-pair cosine-similarity matrix. The Varner
    threshold S > 0.30 does NOT apply to S_global. See module docstring.

    Args:
        patterns: complex FHRR patterns, shape [N, D]. Assumed unit-norm.

    Returns:
        dict with mean/std of self vs cross similarities and S_global.
    """
    N, D = patterns.shape
    # Cosine similarity: for unit-norm complex vectors, use Re<p_i, p_j>/D.
    # (For FHRR, patterns are unit-modulus per component, so ||p||² = D.)
    # Use the real part of the inner product as the similarity score.
    if patterns.is_complex():
        norms = torch.sqrt((patterns.conj() * patterns).real.sum(dim=-1))
        # Normalise (defensive — substrate should produce unit-norm).
        p = patterns / norms.unsqueeze(-1)
        G = (p @ p.conj().T).real
    else:
        norms = patterns.norm(dim=-1, keepdim=True)
        p = patterns / norms
        G = p @ p.T

    # Diagonal (self-similarity, should be ≈ 1).
    self_sim = G.diag()
    # Off-diagonal upper triangular (cross-pattern similarity).
    iu = torch.triu_indices(N, N, offset=1)
    cross_sim = G[iu[0], iu[1]]

    mu_self = float(self_sim.mean())
    sd_self = float(self_sim.std())
    mu_cross = float(cross_sim.mean())
    sd_cross = float(cross_sim.std())

    # Add a small denominator floor so degenerate σ=0 doesn't divide by zero.
    denom = sd_self ** 2 + sd_cross ** 2
    S = (mu_self - mu_cross) ** 2 / max(denom, 1e-12)

    # Also compute distribution-shape moments on the off-diagonal (the
    # interesting part for substrate-collapse detection).
    q05 = float(cross_sim.quantile(0.05))
    q50 = float(cross_sim.quantile(0.50))
    q95 = float(cross_sim.quantile(0.95))

    return {
        "N": int(N),
        "D": int(D),
        "mu_self": mu_self,
        "sd_self": sd_self,
        "mu_cross": mu_cross,
        "sd_cross": sd_cross,
        # Codex audit 2026-05-23: renamed from `fisher_S` to make clear
        # this is global pattern-pair separability, NOT the Varner per-cue
        # functional-subset/background PCA index (which has the same name
        # in the brainstorm prediction but a different formula and
        # partition requirement). See module docstring.
        "S_global": S,
        "cross_q05": q05,
        "cross_q50": q50,
        "cross_q95": q95,
        "theoretical_crosstalk_floor_1_over_sqrt_D": 1.0 / D ** 0.5,
        "sd_cross_div_theoretical_floor": sd_cross / (1.0 / D ** 0.5),
    }


def verdict(S: float) -> str:
    # Codex audit 2026-05-23: this is global S, NOT the Varner per-cue
    # functional-subset/background index. The Varner 0.30 threshold
    # does NOT apply. Report S_global only; leave the strategic
    # interpretation to the user.
    return ("S_global only — Varner threshold not applicable to this "
            "statistic (different formula, no functional-subset "
            "partition, no PCA). See module docstring.")


def load_snapshot(path: Path) -> torch.Tensor:
    snap = torch.load(path, map_location="cpu", weights_only=False)
    if "patterns" not in snap:
        raise KeyError(
            f"Snapshot {path} missing 'patterns' key; got: {list(snap.keys())}"
        )
    return snap["patterns"]


def synthesise_baseline(
    *, D: int, N: int, seed: int = 17, device: str = "cpu"
) -> torch.Tensor:
    """Random FHRR baseline (worst case for the substrate-collapse hypothesis).

    Produces N unit-modulus complex vectors of dimension D — the "perfect"
    case where every pattern is independently random. This is what the
    A1' substrate SHOULD look like if the design intent (preserve d_eff,
    avoid pattern collapse) succeeded.
    """
    substrate = TorchFHRR(dim=D, device=device, seed=seed)
    return substrate.random_vectors(N)


def synthesise_collapsed(
    *, D: int, N: int, n_base: int, seed: int = 17, device: str = "cpu"
) -> torch.Tensor:
    """d_eff-collapsed FHRR substrate simulation.

    Models the post-death substrate where N=1064 surviving atoms span
    only ~n_base effective dimensions (per report 044 d_eff = ~5 of 4096).
    Implementation: pick n_base random FHRR base vectors, then make N
    atoms each by averaging a random small subset of bases plus a tiny
    perturbation (so atoms cluster on a low-d manifold rather than being
    independently random).
    """
    substrate = TorchFHRR(dim=D, device=device, seed=seed)
    bases = substrate.random_vectors(n_base)  # [n_base, D]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    out = torch.zeros(N, D, dtype=torch.complex64, device=device)
    for i in range(N):
        # Random sparse mixture over bases (so atoms cluster in
        # base-subspace) + small phase perturbation.
        weights = torch.zeros(n_base)
        k = max(1, int(torch.randint(1, max(2, n_base // 2 + 1),
                                     (1,), generator=g).item()))
        idx = torch.randperm(n_base, generator=g)[:k]
        weights[idx] = torch.rand(k, generator=g)
        mixture = (bases * weights.to(device).unsqueeze(-1)).sum(dim=0)
        # Re-unitarise to FHRR (per-component unit modulus).
        mag = mixture.abs().clamp_min(1e-12)
        atom = mixture / mag
        # Small per-component phase noise (analogue of fine-grained
        # substrate drift).
        phase_noise = (torch.randn(D, generator=g) * 0.05).to(device)
        atom = atom * torch.polar(torch.ones(D, device=device), phase_noise)
        out[i] = atom
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--snapshot", type=Path,
                     help="Path to a .pt substrate snapshot.")
    grp.add_argument("--synthetic", action="store_true",
                     help="Skip snapshot; run synthetic random-FHRR demo.")
    grp.add_argument("--synthetic-collapsed", type=int, metavar="N_BASE",
                     help="Skip snapshot; run d_eff-collapsed demo where "
                          "atoms span N_BASE effective dims.")
    ap.add_argument("--synthetic-dim", type=int, default=4096,
                    help="Dim for the synthetic demos.")
    ap.add_argument("--synthetic-N", type=int, default=1064,
                    help="Atoms for the synthetic demos.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON output path.")
    args = ap.parse_args()

    if args.synthetic:
        print("Mode: SYNTHETIC random-FHRR baseline")
        print(f"  D={args.synthetic_dim}, N={args.synthetic_N}, seed=17")
        patterns = synthesise_baseline(
            D=args.synthetic_dim, N=args.synthetic_N, seed=17,
        )
        label = (f"synthetic_random_fhrr_D{args.synthetic_dim}"
                 f"_N{args.synthetic_N}")
    elif args.synthetic_collapsed is not None:
        nb = int(args.synthetic_collapsed)
        print(f"Mode: SYNTHETIC d_eff-collapsed (n_base={nb})")
        print(f"  D={args.synthetic_dim}, N={args.synthetic_N}, "
              f"n_base={nb}, seed=17")
        patterns = synthesise_collapsed(
            D=args.synthetic_dim, N=args.synthetic_N, n_base=nb, seed=17,
        )
        label = (f"synthetic_collapsed_D{args.synthetic_dim}"
                 f"_N{args.synthetic_N}_nbase{nb}")
    else:
        print(f"Mode: SNAPSHOT {args.snapshot}")
        patterns = load_snapshot(args.snapshot)
        label = str(args.snapshot)
        print(f"  loaded patterns shape={tuple(patterns.shape)}, "
              f"dtype={patterns.dtype}")

    stats = global_pattern_pair_separability(patterns)
    stats["source"] = label

    print()
    print("Fisher separation diagnostic")
    print("=" * 60)
    print(f"  N atoms        : {stats['N']}")
    print(f"  Dim            : {stats['D']}")
    print(f"  Self-sim mean  : {stats['mu_self']:.6f}")
    print(f"  Self-sim std   : {stats['sd_self']:.6f}")
    print(f"  Cross-sim mean : {stats['mu_cross']:.6f}")
    print(f"  Cross-sim std  : {stats['sd_cross']:.6f}")
    print(f"  Cross-sim q05  : {stats['cross_q05']:.6f}")
    print(f"  Cross-sim q50  : {stats['cross_q50']:.6f}")
    print(f"  Cross-sim q95  : {stats['cross_q95']:.6f}")
    print()
    print(f"  Theoretical 1/sqrt(D)            : "
          f"{stats['theoretical_crosstalk_floor_1_over_sqrt_D']:.6f}")
    print(f"  Observed sd_cross / theoretical  : "
          f"{stats['sd_cross_div_theoretical_floor']:.3f}x")
    print()
    print(f"  >>> Global pattern-pair S_global = {stats['S_global']:.4f}")
    print(f"  >>> (NOT the Varner per-cue functional-subset index;")
    print(f"  >>>  Varner threshold S > 0.30 does NOT apply here.)")
    print()
    print(f"NOTE: {verdict(stats['S_global'])}")
    print("=" * 60)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2))
        print(f"\n[json] {args.output}")


if __name__ == "__main__":
    main()
