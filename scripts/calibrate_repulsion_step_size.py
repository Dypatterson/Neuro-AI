"""One-shot calibration of `replay.config.repulsion_step_size`.

Per the Phase 5 A+B design note
[notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md] and
the STATUS.md next-session entry point: alpha_anti and coverage_lambda
are pre-committed from theory (natural unit / formal Candidate A);
repulsion_step_size needs a one-shot calibration that picks the
*scale* of "one substrate tick" against the actual gradient magnitude
on the operating substrate.

This script:

  1. Loads each available post-death W=4 step-1800 snapshot.
  2. For each, computes the repulsion force at alpha_anti=1.0 and
     reports d_eff before / after a single update step for
     ``step ∈ {0.01, 0.1, 1, 10, 100, 1000}``.
  3. Reports the median Δd_eff per step across seeds.
  4. Picks the step where median Δd_eff is ≈ 0.5 in the collapsed
     regime — the "natural unit" of one substrate tick on this
     geometry.

Anti-homunculus discipline (per design note H4): this is a one-shot
measurement of the gradient's scale in the operating regime, NOT an
iterative search for the step that lands d_eff in the [25, 50] target
range. The picked value is binding for the retrain. If the retrain's
d_eff misses [25, 50], that is a falsification result, not an
invitation to re-calibrate.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/calibrate_repulsion_step_size.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from energy_memory.substrate.torch_fhrr import TorchFHRR


SNAPSHOT_ROOT = Path("reports/phase5_snapshots_local")
SEEDS = (1, 2, 11, 17, 23)
SNAPSHOT_NAME = "phase3_phase4_w4_step1800.pt"
STEP_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
ALPHA_ANTI = 1.0
# "Small but visible" per-tick motion: pre-commit the smallest step whose
# median Δd_eff across seeds is ≥ 0.05 in the collapsed regime. This is
# the conservative end of the calibration band — large enough to drive
# visible flow, small enough that per-tick phase changes (~1–2° at
# step=100, vs ~17° at step=1000) stay well under the substrate's
# settling-iteration scale, so retrieval can track pattern motion.
# The pre-commit binds regardless of where d_eff ends up at step 1800.
TARGET_DELTA_DEFF = 0.05
OUTPUT_PATH = Path("reports/phase5_ab_calibration.json")


def _build_substrate(dim: int) -> TorchFHRR:
    return TorchFHRR(dim=dim, seed=0, device="cpu", alpha_anti=ALPHA_ANTI)


def _calibrate_one_snapshot(path: Path) -> Dict:
    snap = torch.load(path, map_location="cpu", weights_only=False)
    patterns = snap["patterns"]
    n, d = patterns.shape
    substrate = _build_substrate(d)
    d_eff_before = float(substrate.d_eff(patterns))
    force = substrate.repulsion_force(patterns)
    force_mag_mean = float(force.abs().mean())
    force_mag_max = float(force.abs().max())
    per_step = {}
    for step in STEP_CANDIDATES:
        new_p = substrate.normalize(patterns + step * force)
        d_after = float(substrate.d_eff(new_p))
        per_step[f"{step:g}"] = {
            "d_eff_after": d_after,
            "delta_d_eff": d_after - d_eff_before,
        }
    return {
        "snapshot": str(path),
        "seed": snap.get("metadata", {}).get("seed"),
        "n_atoms": n,
        "dim": d,
        "d_eff_before": d_eff_before,
        "force_abs_mean": force_mag_mean,
        "force_abs_max": force_mag_max,
        "per_step": per_step,
    }


def _pick_step(results: List[Dict]) -> Dict:
    """Pick the step where median Δd_eff across seeds is ≥ target.

    Uses the smallest step whose median delta meets or exceeds the
    target — the "least aggressive" step that still moves d_eff by the
    natural-unit amount. This is the one-shot calibration's binding
    choice.
    """
    per_step_deltas: Dict[str, List[float]] = {f"{s:g}": [] for s in STEP_CANDIDATES}
    for r in results:
        for s, info in r["per_step"].items():
            per_step_deltas[s].append(info["delta_d_eff"])

    medians: Dict[str, float] = {}
    for s, vals in per_step_deltas.items():
        vals = sorted(vals)
        medians[s] = vals[len(vals) // 2]

    picked = None
    for s in STEP_CANDIDATES:
        key = f"{s:g}"
        if medians[key] >= TARGET_DELTA_DEFF:
            picked = s
            break
    return {
        "medians": medians,
        "target_delta_d_eff": TARGET_DELTA_DEFF,
        "picked_step_size": picked,
        "rule": "smallest step whose median ΔDeff across seeds ≥ target",
    }


def main():
    results: List[Dict] = []
    for seed in SEEDS:
        path = SNAPSHOT_ROOT / f"seed{seed}" / SNAPSHOT_NAME
        if not path.is_file():
            print(f"[skip] missing snapshot: {path}")
            continue
        r = _calibrate_one_snapshot(path)
        results.append(r)
        print(
            f"seed {r['seed']}: n_atoms={r['n_atoms']}  "
            f"d_eff_before={r['d_eff_before']:.3f}  "
            f"|force| mean={r['force_abs_mean']:.3e}  max={r['force_abs_max']:.3e}"
        )
        for s, info in r["per_step"].items():
            print(f"    step={s:>6}  d_eff_after={info['d_eff_after']:.3f}  Δ={info['delta_d_eff']:+.3f}")

    pick = _pick_step(results)
    print()
    print("=== Pre-commit selection (binding) ===")
    print(f"Target Δd_eff per cycle (collapsed regime): {pick['target_delta_d_eff']}")
    print(f"Median ΔDeff per step across seeds:")
    for s, m in pick["medians"].items():
        print(f"    step={s:>6}  median Δ={m:+.3f}")
    print(f"Picked repulsion_step_size = {pick['picked_step_size']}")
    print(f"Rule: {pick['rule']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "snapshots": results,
        "calibration": pick,
        "alpha_anti": ALPHA_ANTI,
        "target_delta_d_eff": TARGET_DELTA_DEFF,
    }, indent=2))
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
