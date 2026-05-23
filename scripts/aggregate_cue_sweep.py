"""Cross-seed aggregator for cue-regime sweep results.

Reads N per-seed JSONs produced by:

    scripts/phase5_frozen_snapshot_audit.py --cue-regime-sweep ...

and produces a single cross-seed aggregate JSON + a markdown summary
table. Mirrors the architecture of the cross-seed β sweep
(scripts/colab_phase5_cross_seed_beta_sweep.ipynb cell 6).

Per cell (binding_noise_std, content_distortion), aggregates across
seeds:
  - mean ΔE_raw with 95% CI (t-approx)
  - seeds-positive count
  - mean fraction of role<content<random orderings across seeds
  - mean fraction of random_lowest across seeds (the pathology metric)
  - per-condition basin hit rate (mean across seeds)
  - per-condition role-target rank (mean across seeds)
  - ΔE / 5.5e-3 floor ratio

Output: one JSON + one markdown table that ranks cells by mean ΔE
and by basin hit rate.

Usage::

    python scripts/aggregate_cue_sweep.py \\
        --sweep-root /content/drive/MyDrive/neuro-ai/results/phase5_cross_seed_cue_sweep \\
        --output reports/phase5_audit/cue_sweep_aggregate.json

The --sweep-root directory should contain one `seed{N}.json` per seed.
Missing seeds are noted but not fatal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _t_critical(df: int) -> float:
    """Approximate t critical value for 95% two-sided CI."""
    # Lookup small df; default to 1.96 for large df.
    table = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
             15: 2.131, 20: 2.086, 30: 2.042}
    if df in table:
        return table[df]
    if df < 1:
        return float("nan")
    if df > 30:
        return 1.96
    keys = sorted(table)
    for k in keys:
        if k > df:
            return table[k]
    return 2.0


def _aggregate(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Build the cross-seed cell table from per-seed sweep dicts."""
    seeds = sorted(per_seed)
    if not seeds:
        raise SystemExit("no per-seed sweeps found")

    # Use the first seed's grid as canonical; assume all seeds have the
    # same cell list (asserted below).
    canonical = per_seed[seeds[0]]["cells"]
    canonical_keys = [(c["binding_noise_std"], c["content_distortion"]) for c in canonical]

    for s in seeds:
        these_keys = [(c["binding_noise_std"], c["content_distortion"])
                      for c in per_seed[s]["cells"]]
        if these_keys != canonical_keys:
            raise SystemExit(
                f"cell grid mismatch between seeds: seed {seeds[0]} has "
                f"{len(canonical_keys)} cells but seed {s} has "
                f"{len(these_keys)} cells (or different ordering)"
            )

    magnitude_floor = 5.5e-3

    cells_out: List[Dict[str, Any]] = []
    for cell_idx, (bns, cd) in enumerate(canonical_keys):
        per_seed_dE = []
        per_seed_dE_step3 = []
        per_seed_fpos = []
        per_seed_role_lt_content_lt_random = []
        per_seed_role_lt_content = []
        per_seed_random_lowest = []
        per_seed_hit_role = []
        per_seed_hit_content = []
        per_seed_hit_random = []
        per_seed_rank_role = []
        per_seed_rank_content = []
        per_seed_rank_random = []
        for s in seeds:
            c = per_seed[s]["cells"][cell_idx]
            per_seed_dE.append(c["mean_delta_e_raw"])
            per_seed_dE_step3.append(c["mean_delta_e_step3"])
            per_seed_fpos.append(c["frac_positive_raw"])
            per_seed_role_lt_content_lt_random.append(
                c["frac_role_lt_content_lt_random"]
            )
            per_seed_role_lt_content.append(c["frac_role_lt_content"])
            per_seed_random_lowest.append(c["frac_random_lowest"])
            per_seed_hit_role.append(c["per_condition_basin_hit_rate"]["role"])
            per_seed_hit_content.append(c["per_condition_basin_hit_rate"]["content"])
            per_seed_hit_random.append(c["per_condition_basin_hit_rate"]["random"])
            per_seed_rank_role.append(c["per_condition_mean_role_target_rank"]["role"])
            per_seed_rank_content.append(c["per_condition_mean_role_target_rank"]["content"])
            per_seed_rank_random.append(c["per_condition_mean_role_target_rank"]["random"])

        n = len(seeds)
        mean_dE = statistics.mean(per_seed_dE)
        std_dE = statistics.stdev(per_seed_dE) if n > 1 else 0.0
        se_dE = std_dE / math.sqrt(n) if n else 0.0
        tcrit = _t_critical(n - 1)
        ci = (mean_dE - tcrit * se_dE, mean_dE + tcrit * se_dE)
        n_pos = sum(1 for x in per_seed_dE if x > 0)

        cells_out.append({
            "binding_noise_std": bns,
            "content_distortion": cd,
            "n_seeds": n,
            "mean_delta_e_raw": mean_dE,
            "std_delta_e_raw_across_seeds": std_dE,
            "ci95_delta_e_raw": ci,
            "seeds_positive_raw": n_pos,
            "mean_delta_e_step3": statistics.mean(per_seed_dE_step3),
            "delta_e_over_floor": mean_dE / magnitude_floor,
            "mean_frac_positive_raw": statistics.mean(per_seed_fpos),
            "mean_frac_role_lt_content_lt_random": statistics.mean(
                per_seed_role_lt_content_lt_random
            ),
            "mean_frac_role_lt_content": statistics.mean(per_seed_role_lt_content),
            "mean_frac_random_lowest": statistics.mean(per_seed_random_lowest),
            "mean_basin_hit_role": statistics.mean(per_seed_hit_role),
            "mean_basin_hit_content": statistics.mean(per_seed_hit_content),
            "mean_basin_hit_random": statistics.mean(per_seed_hit_random),
            "mean_rank_role": statistics.mean(per_seed_rank_role),
            "mean_rank_content": statistics.mean(per_seed_rank_content),
            "mean_rank_random": statistics.mean(per_seed_rank_random),
            "per_seed_dE_raw": dict(zip(seeds, per_seed_dE)),
        })

    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "magnitude_floor": magnitude_floor,
        "grid": {
            "binding_noise_std": sorted({c["binding_noise_std"] for c in canonical}),
            "content_distortion": sorted({c["content_distortion"] for c in canonical}),
        },
        "cells": cells_out,
    }


def _render_markdown(agg: Dict[str, Any]) -> str:
    """Markdown summary: full cell table + drill-down rankings.

    The cue-regime grid is diagnostic-only. Even if a cell clears the
    magnitude floor, this renderer must not invite post-hoc headline reruns
    under those cue settings; that would convert a sensitivity sweep into
    retuning.
    """
    lines: List[str] = []
    lines.append(f"# Cross-Seed Cue-Regime Sweep — n={agg['n_seeds']} seeds")
    lines.append("")
    lines.append(f"Seeds: {agg['seeds']}")
    lines.append(f"Magnitude floor: {agg['magnitude_floor']}")
    lines.append(f"Grid: bns ∈ {agg['grid']['binding_noise_std']}, "
                 f"cd ∈ {agg['grid']['content_distortion']}")
    lines.append("")

    # Full cell table.
    lines.append("## Full cell table (ordered by mean ΔE_raw)")
    lines.append("")
    lines.append("| bns | cd | mean ΔE_raw | 95% CI | seeds⁺ | ΔE/floor | role<c<r | random_lowest | hit_role | rank_role |")
    lines.append("|-----|-----|-------------|--------|--------|----------|----------|---------------|----------|-----------|")
    cells_sorted = sorted(agg["cells"], key=lambda c: -c["mean_delta_e_raw"])
    for c in cells_sorted:
        ci = c["ci95_delta_e_raw"]
        lines.append(
            f"| {c['binding_noise_std']:.2f} | {c['content_distortion']:.2f} | "
            f"{c['mean_delta_e_raw']:+.6f} | "
            f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | "
            f"{c['seeds_positive_raw']}/{c['n_seeds']} | "
            f"{c['delta_e_over_floor']:+.4f} | "
            f"{c['mean_frac_role_lt_content_lt_random']:.2f} | "
            f"{c['mean_frac_random_lowest']:.2f} | "
            f"{c['mean_basin_hit_role']:.2f} | "
            f"{c['mean_rank_role']:.1f} |"
        )
    lines.append("")

    # Drill-down rankings.
    lines.append("## Top-5 cells by basin hit rate (role)")
    lines.append("")
    lines.append("| bns | cd | hit_role | mean ΔE_raw | rank_role |")
    lines.append("|-----|-----|----------|-------------|-----------|")
    by_hit = sorted(agg["cells"], key=lambda c: -c["mean_basin_hit_role"])[:5]
    for c in by_hit:
        lines.append(
            f"| {c['binding_noise_std']:.2f} | {c['content_distortion']:.2f} | "
            f"{c['mean_basin_hit_role']:.2f} | "
            f"{c['mean_delta_e_raw']:+.6f} | "
            f"{c['mean_rank_role']:.1f} |"
        )
    lines.append("")

    lines.append("## Top-5 cells by role-target rank (lower = better)")
    lines.append("")
    lines.append("| bns | cd | rank_role | hit_role | mean ΔE_raw |")
    lines.append("|-----|-----|-----------|----------|-------------|")
    by_rank = sorted(agg["cells"], key=lambda c: c["mean_rank_role"])[:5]
    for c in by_rank:
        lines.append(
            f"| {c['binding_noise_std']:.2f} | {c['content_distortion']:.2f} | "
            f"{c['mean_rank_role']:.1f} | "
            f"{c['mean_basin_hit_role']:.2f} | "
            f"{c['mean_delta_e_raw']:+.6f} |"
        )
    lines.append("")

    # Drill-down readout.
    best_dE = max(agg["cells"], key=lambda c: c["mean_delta_e_raw"])
    best_hit = max(agg["cells"], key=lambda c: c["mean_basin_hit_role"])
    best_rank = min(agg["cells"], key=lambda c: c["mean_rank_role"])

    lines.append("## Drill-down flags")
    lines.append("")
    lines.append(f"- Best ΔE cell: bns={best_dE['binding_noise_std']}, "
                 f"cd={best_dE['content_distortion']} → "
                 f"ΔE_raw = {best_dE['mean_delta_e_raw']:+.6f}, "
                 f"ΔE/floor = {best_dE['delta_e_over_floor']:+.4f}")
    lines.append(f"- Best basin-hit cell: bns={best_hit['binding_noise_std']}, "
                 f"cd={best_hit['content_distortion']} → "
                 f"hit_role = {best_hit['mean_basin_hit_role']:.2f}")
    lines.append(f"- Best (lowest) rank cell: bns={best_rank['binding_noise_std']}, "
                 f"cd={best_rank['content_distortion']} → "
                 f"rank_role = {best_rank['mean_rank_role']:.1f}")
    lines.append("")
    lines.append("How to use this drill-down:")
    lines.append("")
    lines.append(
        "- No cue-regime cell, including one above the magnitude floor, "
        "is a graduation result; a favorable cell does not graduate Phase 5 "
        "by itself or authorize post-hoc parameter selection. This sweep is "
        "evidence about the measurement surface."
    )
    lines.append(
        "- If ΔE/floor or basin hit improves in a region of the grid, record "
        "that as support for a successor pre-committed cue distribution or "
        "headline reformulation, not as a winning cell."
    )
    lines.append(
        "- If all cells remain sub-floor and basin hit is near zero, that "
        "strengthens lower-D redesign and weakens basin-shape priors."
    )
    lines.append(
        "- If basin hit improves while paired ΔE stays sub-floor, that "
        "strengthens a successor metric such as role-target basin membership."
    )
    lines.append(
        "- If a cell looks favorable only because random-prior pathologies "
        "move around, keep option 4 (basin-shape priors) under suspicion; "
        "do not advance with a sub-floor caveat unless explicitly chosen."
    )

    return "\n".join(lines)


def _parse_expected_seeds(raw: str) -> Optional[List[int]]:
    if not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _load_per_seed_sweeps(
    sweep_root: Path,
    *,
    expected_seeds: Optional[List[int]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Load seed*.json files, skipping non-sweep files and reporting gaps."""
    per_seed: Dict[int, Dict[str, Any]] = {}
    for p in sorted(sweep_root.glob("seed*.json")):
        name = p.stem  # "seed17"
        try:
            seed = int(name.replace("seed", ""))
        except ValueError:
            print(f"  skipping unparseable filename: {p.name}")
            continue
        with open(p) as f:
            d = json.load(f)
        if "cue_regime_sweep" not in d:
            print(f"  {p.name}: no 'cue_regime_sweep' key, skipping")
            continue
        per_seed[seed] = d["cue_regime_sweep"]
        print(f"  loaded seed {seed} ({len(per_seed[seed]['cells'])} cells)")

    missing = []
    if expected_seeds is not None:
        missing = [s for s in expected_seeds if s not in per_seed]
        if missing:
            print(f"  missing expected seeds: {missing}")
    return per_seed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root", type=Path, required=True,
        help="Directory containing per-seed JSONs named seed{N}.json.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output aggregate JSON path. Markdown companion at <output>.md.",
    )
    parser.add_argument(
        "--expected-seeds", default="",
        help="Optional comma-separated seed list. Missing seeds are reported "
             "but are not fatal, so partial Colab runs can still aggregate.",
    )
    args = parser.parse_args()

    if not args.sweep_root.is_dir():
        raise SystemExit(f"--sweep-root not a directory: {args.sweep_root}")

    per_seed, _missing = _load_per_seed_sweeps(
        args.sweep_root,
        expected_seeds=_parse_expected_seeds(args.expected_seeds),
    )

    agg = _aggregate(per_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[done] wrote {args.output}")

    md_path = args.output.with_suffix(".md")
    md_path.write_text(_render_markdown(agg))
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
