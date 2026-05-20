"""Post-pilot analyzer for the Phase 5 A+B 1-seed retrain.

Loads the new substrate snapshots produced by
``scripts/run_phase5_ab_pilot_seed17.sh`` (output under
``reports/phase5_ab_pilot_seed17/snapshots/``), computes d_eff at each
{W, step} pair, compares against the pre-A+B baseline snapshots in
``reports/phase5_snapshots_local/seed17/``, and reports pass/fail
against the pre-committed mechanism-validity criteria from
[notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md].

**Pre-committed mechanism-validity criteria (BINDING):**

1. ``d_eff ≥ 25`` at step 1800 on the W=4 substrate.
   (Pre-death baseline is ~40-45; post-death pre-A+B is ~3-6;
   25 is "halfway back to pre-death." Pre-committed BEFORE retrain.)

These are NOT graduation criteria; they are mechanism-validity gates.
If the gate passes, the next move is the n=10 Colab retrain. If the
gate fails, the candidate is wrong-shaped — back to design, NOT
re-tune α / λ / step_size.

Anti-homunculus discipline: this script *measures* the trajectory; it
does not adjust any parameter based on the measurement. Per
[design note H4]:
the validity criterion is a measurement, not a target the parameters
are searched over.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analyze_phase5_ab_pilot.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# scripts/ is not a Python package; add it to sys.path so we can reuse
# the diagnose() implementation from the existing geometry diagnostic.
sys.path.insert(0, str(Path(__file__).parent))
from consolidation_geometry_diagnostic import diagnose  # noqa: E402


SEED = 17
WS = (2, 3, 4)
STEPS = (500, 1500, 1700, 1800)
VALIDITY_DEFF_MIN = 25.0
VALIDITY_STEP = 1800
VALIDITY_W = 4


def _try_diagnose(snapshot_path: Path) -> Optional[Dict]:
    if not snapshot_path.is_file():
        return None
    return diagnose(snapshot_path, k_nn=5, beta=10.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("reports/phase5_ab_pilot_seed17"),
        help="Output directory of run_phase5_ab_pilot_seed17.sh",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("reports/phase5_snapshots_local/seed17"),
        help="Pre-A+B baseline snapshot directory (for direct comparison)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase5_ab_pilot_seed17/analysis.json"),
    )
    args = parser.parse_args()

    snap_root = args.pilot_dir / "snapshots"
    pilot_results: Dict[str, Dict] = {}
    for w in WS:
        for step in STEPS:
            path = snap_root / f"phase3_phase4_w{w}_step{step}.pt"
            r = _try_diagnose(path)
            if r is not None:
                pilot_results[f"w{w}_step{step}"] = {
                    "n_atoms": r["n_atoms"],
                    "d_bar": r["substrate"]["d_bar_mean"],
                    "d_eff": r["substrate"]["d_eff"],
                    "d_eff_ratio_to_dim": r["substrate"]["d_eff_ratio_to_dim"],
                    "regime": r["substrate"]["regime"],
                    "snapshot": str(path),
                }

    baseline_results: Dict[str, Dict] = {}
    for w in WS:
        for step in (1500, 1800):
            path = args.baseline_dir / f"phase3_phase4_w{w}_step{step}.pt"
            r = _try_diagnose(path)
            if r is not None:
                baseline_results[f"w{w}_step{step}"] = {
                    "n_atoms": r["n_atoms"],
                    "d_eff": r["substrate"]["d_eff"],
                    "snapshot": str(path),
                }

    # Validity gate
    gate_key = f"w{VALIDITY_W}_step{VALIDITY_STEP}"
    gate = {
        "criterion": f"d_eff ≥ {VALIDITY_DEFF_MIN} at step {VALIDITY_STEP} on W={VALIDITY_W}",
        "binding": True,
        "snapshot": gate_key,
    }
    if gate_key in pilot_results:
        observed = pilot_results[gate_key]["d_eff"]
        gate["observed_d_eff"] = observed
        if math.isfinite(observed):
            gate["passed"] = bool(observed >= VALIDITY_DEFF_MIN)
        else:
            gate["passed"] = False
            gate["note"] = "d_eff is non-finite"
    else:
        gate["observed_d_eff"] = None
        gate["passed"] = False
        gate["note"] = (
            f"Snapshot missing at {snap_root}/phase3_phase4_w{VALIDITY_W}_step{VALIDITY_STEP}.pt"
        )

    # Comparison table: pilot vs baseline at the same (W, step)
    comparison: List[Dict] = []
    for w in WS:
        for step in (1500, 1800):
            key = f"w{w}_step{step}"
            pilot = pilot_results.get(key)
            base = baseline_results.get(key)
            comparison.append({
                "key": key,
                "pilot_d_eff": pilot["d_eff"] if pilot else None,
                "pilot_n_atoms": pilot["n_atoms"] if pilot else None,
                "baseline_d_eff": base["d_eff"] if base else None,
                "baseline_n_atoms": base["n_atoms"] if base else None,
                "delta_d_eff": (
                    pilot["d_eff"] - base["d_eff"]
                    if (pilot and base
                        and math.isfinite(pilot["d_eff"])
                        and math.isfinite(base["d_eff"]))
                    else None
                ),
            })

    output = {
        "seed": SEED,
        "pilot_dir": str(args.pilot_dir),
        "baseline_dir": str(args.baseline_dir),
        "pilot_snapshots": pilot_results,
        "baseline_snapshots": baseline_results,
        "comparison": comparison,
        "mechanism_validity_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    # Console summary
    print(f"=== Phase 5 A+B 1-seed pilot analysis (seed {SEED}) ===\n")
    print("Pilot d_eff trajectory:")
    for w in WS:
        cells = []
        for step in STEPS:
            key = f"w{w}_step{step}"
            r = pilot_results.get(key)
            if r is None:
                cells.append(f"step{step}: --")
            else:
                cells.append(f"step{step}: d_eff={r['d_eff']:.2f} (n={r['n_atoms']})")
        print(f"  W={w}: " + "  ".join(cells))

    print("\nComparison vs pre-A+B baseline (positive Δ = A+B preserved more d_eff):")
    for c in comparison:
        if c["delta_d_eff"] is not None:
            print(
                f"  {c['key']}: pilot {c['pilot_d_eff']:6.2f} (n={c['pilot_n_atoms']:>4}) "
                f"vs baseline {c['baseline_d_eff']:6.2f} (n={c['baseline_n_atoms']:>4})  "
                f"Δ = {c['delta_d_eff']:+.2f}"
            )
        else:
            print(f"  {c['key']}: incomplete data")

    print(f"\n=== Mechanism-validity gate ===")
    print(f"  Criterion: {gate['criterion']}")
    if gate["observed_d_eff"] is None:
        print(f"  Observed: <missing>  — {gate.get('note', '')}")
    else:
        print(f"  Observed d_eff = {gate['observed_d_eff']:.2f}")
    print(f"  {'PASS' if gate['passed'] else 'FAIL'}")
    if gate["passed"]:
        print("\n  Next step: ship to Colab for n=10 retrain.")
    else:
        print("\n  Next step: report as falsification; back to design.")
        print("  NOT permitted: re-tune α / λ / repulsion_step_size.")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
