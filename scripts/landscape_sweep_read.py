#!/usr/bin/env python
"""Pre-committed read for the landscape-size diagnostic (Report 117 / 2026-05-30).

Reads N gate0_summary.json files (one per landscape_size), computes the
consolidated-arm per-seed SD sigma_A(L), the frozen-arm floor sigma_C(L), and the
mean consolidation lift mean(A-C)(L), then applies the pre-committed thresholds
from notes/notes/2026-05-30-landscape-sweep-diagnostic-precommit.md.

NO new estimator: sigma is the sample SD (n-1, matching gate0_frame_a._std) over
the per_seed_recall arrays already emitted by the gate.

Usage:
  python scripts/landscape_sweep_read.py L=64:reports/landscape_2026-05-30/L064/gate0_summary.json \\
                                         L=256:reports/landscape_2026-05-30/L256/gate0_summary.json \\
                                         L=512:reports/landscape_2026-05-30/L512/gate0_summary.json
"""
import json
import statistics as st
import sys


def _sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def _arm(summary, name):
    psr = summary["per_seed_recall"][name]
    # keys are stringified seeds; order by int seed for stable pairing
    return [float(psr[k]) for k in sorted(psr, key=int)]


def read_one(path):
    s = json.load(open(path))
    A, C = _arm(s, "A"), _arm(s, "C")
    L = s["header"]["operating_point"]["landscape_size"]
    lift = [a - c for a, c in zip(A, C)]
    return {
        "landscape_size": L,
        "sigma_A": _sd(A),
        "sigma_C": _sd(C),
        "mean_lift_A_minus_C": st.mean(lift),
        "n_seeds": len(A),
    }


def verdict(rows):
    """Apply the pre-committed thresholds to the L=max row (and a capacity guard
    across all rows)."""
    rows = sorted(rows, key=lambda r: r["landscape_size"])
    top = rows[-1]
    sA, lift = top["sigma_A"], top["mean_lift_A_minus_C"]
    # Capacity guard overrides the sigma read (checked at ALL L).
    cap_floor = 0.015
    capacity_wall = any(r["mean_lift_A_minus_C"] < cap_floor for r in rows)
    if capacity_wall:
        return ("CAPACITY-WALL",
                f"mean(A-C) fell below {cap_floor} at some L -> bigger landscape "
                "killed the consolidation signal (Hopfield capacity at beta=10). "
                "Bigger L is NOT a usable fix even if sigma dropped.")
    if sA <= 0.10 and lift >= 0.020:
        return ("VARIANCE-REDUCIBLE",
                f"sigma_A({top['landscape_size']})={sA:.3f} <= 0.10 and "
                f"mean(A-C)={lift:.3f} >= 0.020 -> bigger landscape works; propose "
                "a powered run at the best L (op-point sign-off). level-DiD n drops "
                "~424 -> ~100-187; slope worth revisiting.")
    if sA >= 0.13:
        return ("VARIANCE-IRREDUCIBLE",
                f"sigma_A({top['landscape_size']})={sA:.3f} >= 0.13 (<20% drop from "
                "0.157) -> landscape is not the knob; effect structurally hard to "
                "detect at this architecture -> Frame B mechanism / op-point rethink.")
    return ("PARTIAL",
            f"sigma_A({top['landscape_size']})={sA:.3f} in (0.10, 0.13) -> read the "
            "full sigma_A(L) curve; consider pushing L to 1024 or combining modest "
            "L-up with modest n-up before deciding.")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rows = []
    for tok in argv:
        # accept "L=64:path" or bare "path"
        path = tok.split(":", 1)[1] if tok[:2] == "L=" and ":" in tok else tok
        rows.append(read_one(path))
    rows.sort(key=lambda r: r["landscape_size"])
    print("Landscape-size diagnostic read (pre-committed "
          "2026-05-30-landscape-sweep-diagnostic-precommit.md):\n")
    print(f"  {'L':>5} | {'sigma_A':>8} | {'sigma_C':>8} | {'mean(A-C)':>10} | n")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+--")
    for r in rows:
        print(f"  {r['landscape_size']:>5} | {r['sigma_A']:>8.3f} | "
              f"{r['sigma_C']:>8.3f} | {r['mean_lift_A_minus_C']:>10.3f} | "
              f"{r['n_seeds']}")
    # Reproducibility anchor check on L=64 if present.
    base = next((r for r in rows if r["landscape_size"] == 64), None)
    if base is not None and not (0.13 <= base["sigma_A"] <= 0.18):
        print(f"\n  !! WARNING: sigma_A(L=64)={base['sigma_A']:.3f} did NOT reproduce "
              "the recovered ~0.157 (expected 0.13-0.18). Environment/repro problem "
              "-> STOP and re-derive before trusting the sweep.")
    label, why = verdict(rows)
    print(f"\n  VERDICT: {label}\n  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
