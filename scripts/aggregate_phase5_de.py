"""Aggregate Phase 5 headline ΔE outputs across seeds.

Reads per-seed JSON files produced by ``experiments/40_phase5_branching.py
--mode headline`` and emits:

  - Per-seed mean ΔE (E_content - E_role) for each tag (K4, K1, K4_g0)
  - Across-seed mean ± 95% t-CI (matches the methodology that graduated
    Phase 4 in report 038: per-seed summary is the unit, n_seeds is the
    sample size)
  - Sign test on per-seed means: count of seeds positive + Wilson 95% CI
    on the proportion
  - Pooled per-cue distribution: mean + 95% bootstrap CI over the union
    of all per-cue deltas across seeds
  - LOSO sensitivity: drop each seed in turn and report the (n-1) CI;
    flag any LOSO CI that includes zero (matches checklist D2)
  - Seed-23 readout side-by-side with the n=N mean (matches checklist D3)

This script is statistics-only. No mechanism decides anything based on
the output. The graduation rules in phase-5-checklist.md (sections A, D)
are applied by reading this output, not by the script.

Usage:
  python scripts/aggregate_phase5_de.py \
      --inputs reports/phase5_headline_n5/seed{17,11,23,1,2}/phase5_headline_seed*.json \
      --output reports/phase5_headline_n5/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Statistics primitives (stdlib-only; no scipy dependency)
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _stddev(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


# Two-sided 95% t critical values for df = 1..30. Beyond df=30, approach
# 1.96 asymptotically; we use the largest tabulated value for df > 30.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_95:
        return _T_95[df]
    return 1.960


def t_ci_95(xs: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for two-sided 95% t-CI on the sample mean."""
    n = len(xs)
    if n < 2:
        m = _mean(xs)
        return m, float("nan"), float("nan")
    m = _mean(xs)
    s = _stddev(xs, ddof=1)
    se = s / math.sqrt(n)
    tc = t_critical_95(n - 1)
    return m, m - tc * se, m + tc * se


def wilson_ci_95(k: int, n: int) -> Tuple[float, float, float]:
    """Wilson score CI for a binomial proportion at 95% (z = 1.96)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return phat, center - half, center + half


def bootstrap_ci_95(
    xs: Sequence[float], n_boot: int = 10000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap 95% CI on the mean."""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return _mean(xs), lo, hi


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _load_per_seed(input_paths: Sequence[Path]) -> List[Dict]:
    seeds = []
    for path in input_paths:
        payload = json.loads(path.read_text())
        seed = int(payload.get("seed"))
        deltas = payload.get("headline_deltas", {}) or {}
        n_atoms = int(payload.get("n_atoms", -1))
        # Substrate snapshot info — only used for provenance printing.
        snap = payload.get("snapshot_info") or {}
        seeds.append({
            "seed": seed,
            "path": str(path),
            "n_atoms": n_atoms,
            "snapshot_label": snap.get("label"),
            "snapshot_metadata": snap.get("metadata"),
            "deltas": deltas,
        })
    seeds.sort(key=lambda s: s["seed"])
    return seeds


def _summarize_tag(tag: str, seeds: List[Dict]) -> Dict:
    """Across-seed + pooled-per-cue stats for one tag (K4/K1/K4_g0)."""
    per_seed_mean: List[Tuple[int, float, int, float]] = []
    pooled_per_cue: List[float] = []
    for s in seeds:
        d = (s["deltas"] or {}).get(tag)
        if d is None:
            continue
        per_cue = d.get("per_cue_delta") or []
        if not per_cue:
            continue
        mean = float(d.get("mean_delta_e_content_minus_role", _mean(per_cue)))
        frac_pos = float(d.get("fraction_positive", 0.0))
        per_seed_mean.append((s["seed"], mean, len(per_cue), frac_pos))
        pooled_per_cue.extend(float(x) for x in per_cue)

    # --- across-seed: each seed contributes one mean ---
    means_only = [m for (_, m, _, _) in per_seed_mean]
    across_mean, across_lo, across_hi = t_ci_95(means_only)
    n_pos_seeds = sum(1 for m in means_only if m > 0)
    sign_phat, sign_lo, sign_hi = wilson_ci_95(n_pos_seeds, len(means_only))

    # --- pooled-per-cue: bootstrap CI on the pooled mean ---
    pooled_mean, pooled_lo, pooled_hi = bootstrap_ci_95(pooled_per_cue)
    n_pos_cues = sum(1 for x in pooled_per_cue if x > 0)
    pooled_sign_phat, pooled_sign_lo, pooled_sign_hi = wilson_ci_95(
        n_pos_cues, len(pooled_per_cue),
    )

    # --- LOSO: drop each seed; recompute across-seed t-CI ---
    loso = []
    for i, (sd, _, _, _) in enumerate(per_seed_mean):
        loo = [m for j, (_, m, _, _) in enumerate(per_seed_mean) if j != i]
        loo_mean, loo_lo, loo_hi = t_ci_95(loo)
        loso.append({
            "dropped_seed": sd,
            "n_remaining": len(loo),
            "mean": loo_mean,
            "ci_lo": loo_lo,
            "ci_hi": loo_hi,
            "ci_excludes_zero": (
                (loo_lo > 0 and loo_hi > 0)
                or (loo_lo < 0 and loo_hi < 0)
                if not (math.isnan(loo_lo) or math.isnan(loo_hi))
                else None
            ),
        })

    return {
        "n_seeds": len(per_seed_mean),
        "per_seed_mean": [
            {"seed": sd, "mean": m, "n_cues": nc, "fraction_positive": fp}
            for (sd, m, nc, fp) in per_seed_mean
        ],
        "across_seed_mean": across_mean,
        "across_seed_ci_lo": across_lo,
        "across_seed_ci_hi": across_hi,
        "across_seed_ci_excludes_zero": (
            (across_lo > 0 and across_hi > 0)
            or (across_lo < 0 and across_hi < 0)
            if not (math.isnan(across_lo) or math.isnan(across_hi))
            else None
        ),
        "n_seeds_positive": n_pos_seeds,
        "fraction_seeds_positive": sign_phat,
        "fraction_seeds_positive_ci_lo": sign_lo,
        "fraction_seeds_positive_ci_hi": sign_hi,
        "pooled_n_cues": len(pooled_per_cue),
        "pooled_mean": pooled_mean,
        "pooled_bootstrap_ci_lo": pooled_lo,
        "pooled_bootstrap_ci_hi": pooled_hi,
        "pooled_ci_excludes_zero": (
            (pooled_lo > 0 and pooled_hi > 0)
            or (pooled_lo < 0 and pooled_hi < 0)
            if not (math.isnan(pooled_lo) or math.isnan(pooled_hi))
            else None
        ),
        "n_cues_positive": n_pos_cues,
        "fraction_cues_positive": pooled_sign_phat,
        "fraction_cues_positive_ci_lo": pooled_sign_lo,
        "fraction_cues_positive_ci_hi": pooled_sign_hi,
        "loso": loso,
        "all_loso_ci_excludes_zero": (
            all(item["ci_excludes_zero"] is True for item in loso)
            if loso else None
        ),
    }


def aggregate(input_paths: Sequence[Path]) -> Dict:
    seeds = _load_per_seed(input_paths)
    tags = sorted({
        tag
        for s in seeds
        for tag in (s["deltas"] or {}).keys()
    })
    summary = {tag: _summarize_tag(tag, seeds) for tag in tags}

    # Seed-23 readout (D3) — pull it out side-by-side regardless of order.
    seed23 = next((s for s in seeds if s["seed"] == 23), None)

    return {
        "n_seeds_total": len(seeds),
        "seeds_included": [s["seed"] for s in seeds],
        "per_seed_provenance": [
            {
                "seed": s["seed"],
                "n_atoms": s["n_atoms"],
                "snapshot_label": s["snapshot_label"],
                "snapshot_metadata": s["snapshot_metadata"],
                "path": s["path"],
            }
            for s in seeds
        ],
        "by_tag": summary,
        "seed23_readout": (
            {
                "seed": 23,
                "deltas": seed23["deltas"] if seed23 else None,
            }
            if seed23 else None
        ),
        "notes": (
            "Headline tag is K4 (K_main=4 with γ>0). K1 is the no-branching "
            "control (checklist B2). K4_g0 is the no-prior control "
            "(checklist B3). Per-seed mean is the unit of analysis; "
            "across-seed t-CI is the headline statistic (matches Phase 4 "
            "graduation methodology in report 038). LOSO CI excludes "
            "zero on every leave-one-out subset is checklist D2."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_ci(mean: float, lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return f"{mean:+.3e}  CI=NA"
    return f"{mean:+.3e}  CI=[{lo:+.3e}, {hi:+.3e}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Per-seed JSON paths from experiments/40 --mode headline",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Aggregate JSON output path",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    result = aggregate(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out}")

    # Console summary.
    print()
    print(f"n_seeds_total = {result['n_seeds_total']}")
    print(f"seeds = {result['seeds_included']}")
    for tag, s in result["by_tag"].items():
        print()
        print(f"=== tag: {tag} ===")
        print(f"  n_seeds: {s['n_seeds']}")
        print(
            "  across-seed Δ:        "
            + _format_ci(
                s["across_seed_mean"],
                s["across_seed_ci_lo"],
                s["across_seed_ci_hi"],
            )
            + f"   CI excludes 0: {s['across_seed_ci_excludes_zero']}"
        )
        print(
            "  pooled-per-cue Δ:     "
            + _format_ci(
                s["pooled_mean"],
                s["pooled_bootstrap_ci_lo"],
                s["pooled_bootstrap_ci_hi"],
            )
            + f"   CI excludes 0: {s['pooled_ci_excludes_zero']}"
        )
        print(
            f"  seeds positive: {s['n_seeds_positive']}/{s['n_seeds']}  "
            f"Wilson [{s['fraction_seeds_positive_ci_lo']:.3f}, "
            f"{s['fraction_seeds_positive_ci_hi']:.3f}]"
        )
        print(
            f"  cues positive:  {s['n_cues_positive']}/{s['pooled_n_cues']}  "
            f"Wilson [{s['fraction_cues_positive_ci_lo']:.3f}, "
            f"{s['fraction_cues_positive_ci_hi']:.3f}]"
        )
        for item in s["per_seed_mean"]:
            print(
                f"    seed {item['seed']:>3}: mean={item['mean']:+.3e}  "
                f"frac+={item['fraction_positive']:.2f}  n_cues={item['n_cues']}"
            )
        print(
            f"  LOSO all-excludes-zero: {s['all_loso_ci_excludes_zero']}"
        )
        for item in s["loso"]:
            print(
                f"    drop seed {item['dropped_seed']:>3}: "
                f"mean={item['mean']:+.3e}  "
                f"CI=[{item['ci_lo']:+.3e}, {item['ci_hi']:+.3e}]  "
                f"excludes 0: {item['ci_excludes_zero']}"
            )


if __name__ == "__main__":
    main()
