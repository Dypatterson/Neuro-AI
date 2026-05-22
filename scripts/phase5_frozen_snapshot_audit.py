"""Phase 5 frozen-snapshot audit: substrate measurement harness.

Reusable measurement layer for cross-seed substrate inspection and
operating-point characterization. Loads frozen Phase 4 substrate
snapshots (the .pt files produced by experiments/19_phase34_integrated.py
with --snapshot-steps), emits structured JSON/CSV with per-snapshot:

  - geometry: n_atoms, dim, coverage_lambda, epsilon (retrieval_weight_epsilon),
    tau (retrieval_weight_tau)
  - effective-strength distribution: |E_i| min, median, mean, max, std
  - step-3 retrieval-weight bias distribution: bias min, median, mean,
    max, std, n_below_epsilon, n_bias_ge_1, bias_cv = std / |mean|
  - optional β preflight: for each β in {1, 3, 5, 10, 30} on n_cue_probes
    random cues, capture max softmax weight at iteration 1 and final
    iteration, softmax entropy at iteration 1 and final, and the
    trajectory gap max_t(w_top) − w_top(final).

No training, no graduation claim, no retuning. Pure measurement.

Usage
-----
Audit a single snapshot::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \\
        --output reports/phase5_audit/seed17.json

Audit a directory of snapshots (matches *.pt recursively)::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot-dir reports/phase5_a1prime_pilot_seed17/snapshots \\
        --output reports/phase5_audit/a1prime_all.json

Include the β preflight (adds ~30s per snapshot at D=4096)::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot ... --beta-preflight --n-cue-probes 8 \\
        --output reports/phase5_audit/seed17_beta.json

CSV companion alongside JSON::

    python scripts/phase5_frozen_snapshot_audit.py ... --csv-also

Why
---
Report 054 surfaced that on the seed-17 A1' substrate, every atom has
|E_i| ≈ 0.025 (all below ε=0.05), making the step-3 sigmoidal bias
near-uniform and therefore softmax-shift-invariant — step 3 is
empirically inert. That was one seed. This harness is the cross-seed
generalization check: does the saturation pattern hold across all
available A+B+A1' snapshots, or is seed 17 a corner case?

The β preflight is the cheaper, parallelizable arm of the upstream
diagnostics — it tests whether β=10 + D=4096 is the saturation corner,
without retraining, on whatever snapshots already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse experiments/40's snapshot loader so the substrate setup is
# identical to what the headline run sees. Loaded via importlib because
# the filename starts with a digit.
import importlib.util  # noqa: E402

_EXP40_PATH = REPO_ROOT / "experiments" / "40_phase5_branching.py"
_spec = importlib.util.spec_from_file_location("experiments_40", str(_EXP40_PATH))
_exp40 = importlib.util.module_from_spec(_spec)
sys.modules["experiments_40"] = _exp40
_spec.loader.exec_module(_exp40)


def _tensor_summary(t: torch.Tensor) -> Dict[str, float]:
    """min/median/mean/max/std as Python floats. Empty → all NaN."""
    if t.numel() == 0:
        return {
            "min": float("nan"), "median": float("nan"),
            "mean": float("nan"), "max": float("nan"), "std": float("nan"),
        }
    return {
        "min": float(t.min()),
        "median": float(t.median()),
        "mean": float(t.mean()),
        "max": float(t.max()),
        "std": float(t.std(unbiased=False)),
    }


def _settle_for_preflight(
    *,
    memory,
    cue: torch.Tensor,
    beta: float,
    max_iter: int = 12,
) -> Dict[str, float]:
    """Plain unbiased Hopfield settling that captures per-iter softmax stats.

    Returns max_w at iter 1, max_w final, entropy at iter 1, entropy final,
    trajectory_gap = max over iterations of w_top minus final w_top.

    No score_bias, no prior, no γ — this is the raw landscape probe. It
    isolates the substrate's softmax sharpness as a function of β.
    """
    patterns = memory._pattern_matrix()
    substrate = memory.substrate
    device = substrate.device
    state = cue.to(device)

    max_w_per_iter: List[float] = []
    entropy_per_iter: List[float] = []
    top_idx_history: List[int] = []
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns)
        weights = torch.softmax(beta * scores, dim=0)
        max_w_per_iter.append(float(weights.max()))
        top_idx_history.append(int(weights.argmax()))
        safe = weights.clamp(min=1e-12)
        entropy_per_iter.append(float(-(safe * safe.log()).sum()))
        update = (patterns * weights[:, None]).sum(dim=0)
        state = substrate.normalize(update)

    final_idx = top_idx_history[-1]
    # Trajectory gap (Path-3 c_i style): how much did the winner's weight
    # peak above its final value? Captures "lost-out" dynamics — when this
    # is near zero, the substrate is winner-take-all from iteration 1.
    winning_w_history = []
    state2 = cue.to(device)
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state2, patterns)
        weights = torch.softmax(beta * scores, dim=0)
        winning_w_history.append(float(weights[final_idx]))
        update = (patterns * weights[:, None]).sum(dim=0)
        state2 = substrate.normalize(update)
    traj_gap = max(winning_w_history) - winning_w_history[-1]

    return {
        "max_w_iter1": max_w_per_iter[0],
        "max_w_final": max_w_per_iter[-1],
        "entropy_iter1": entropy_per_iter[0],
        "entropy_final": entropy_per_iter[-1],
        "trajectory_gap": traj_gap,
    }


def _run_beta_preflight(
    *,
    memory,
    betas: List[float],
    n_cue_probes: int,
    seed: int,
    max_iter: int = 12,
) -> Dict[str, Dict[str, float]]:
    """Average per-β preflight stats across n_cue_probes random unit cues.

    Cues are fresh complex random vectors normalized to unit FHRR phase
    so the preflight measures the substrate's response geometry, not a
    cue-substrate alignment. Off-substrate cues are a deliberate choice:
    they probe the substrate's pulling-power across β regimes without
    being biased toward any particular stored pattern.
    """
    substrate = memory.substrate
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: Dict[str, Dict[str, float]] = {}
    for beta in betas:
        accum: Dict[str, List[float]] = {
            k: [] for k in (
                "max_w_iter1", "max_w_final", "entropy_iter1",
                "entropy_final", "trajectory_gap",
            )
        }
        for k in range(n_cue_probes):
            real = torch.randn(substrate.dim, generator=g)
            imag = torch.randn(substrate.dim, generator=g)
            cue = substrate.normalize((real + 1j * imag).to(torch.complex64))
            stats = _settle_for_preflight(
                memory=memory, cue=cue, beta=beta, max_iter=max_iter,
            )
            for kname in accum:
                accum[kname].append(stats[kname])
        out[f"beta_{beta:g}"] = {
            kname + "_mean": sum(vals) / len(vals)
            for kname, vals in accum.items()
        }
        out[f"beta_{beta:g}"]["n_cue_probes"] = float(n_cue_probes)
    return out


def audit_snapshot(
    *,
    snapshot_path: Path,
    device: str = "cpu",
    beta_preflight: bool = False,
    betas: Optional[List[float]] = None,
    n_cue_probes: int = 4,
    preflight_seed: int = 17,
) -> Dict[str, Any]:
    """Audit one snapshot. Returns a flat dict suitable for JSON or one CSV row."""
    mem, cons, patterns, positions, info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    e_abs = cons.effective_strength().abs()
    eps = float(cons.config.retrieval_weight_epsilon)
    tau = float(cons.config.retrieval_weight_tau)
    coverage_lambda = float(cons.config.coverage_lambda)

    record: Dict[str, Any] = {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": coverage_lambda,
        "epsilon": eps,
        "tau": tau,
        "effective_strength": _tensor_summary(e_abs),
        "n_below_epsilon": int((e_abs < eps).sum()),
        "n_above_epsilon": int((e_abs >= eps).sum()),
    }

    if coverage_lambda > 0.0:
        bias = cons.retrieval_weight_bias()
        bias_summary = _tensor_summary(bias)
        # CV: std / |mean|. Near zero → near-uniform bias → softmax-shift-invariant.
        bias_cv = (
            bias_summary["std"] / abs(bias_summary["mean"])
            if abs(bias_summary["mean"]) > 1e-12 else float("nan")
        )
        record["retrieval_weight_bias"] = bias_summary
        record["bias_cv"] = bias_cv
        record["n_bias_ge_1"] = int((bias >= 1.0).sum())
        record["n_bias_ge_0p5"] = int((bias >= 0.5).sum())
        # The shift-invariance test: the step-3 mechanism can only move
        # paired ΔE if bias_cv is materially above zero. We pin the
        # threshold at 0.05 (5% relative spread) as the "materially
        # non-uniform" floor — below this, softmax(x − bias) ≈ softmax(x).
        record["step3_shift_invariant_likely"] = bool(bias_cv < 0.05)
    else:
        record["retrieval_weight_bias"] = None
        record["bias_cv"] = None
        record["n_bias_ge_1"] = None
        record["n_bias_ge_0p5"] = None
        record["step3_shift_invariant_likely"] = None

    if beta_preflight:
        record["beta_preflight"] = _run_beta_preflight(
            memory=mem,
            betas=betas or [1.0, 3.0, 5.0, 10.0, 30.0],
            n_cue_probes=n_cue_probes,
            seed=preflight_seed,
        )

    return record


def _flatten_for_csv(record: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    """One-level flatten of nested dicts: {a: {b: 1}} → {a.b: 1}."""
    out: Dict[str, Any] = {}
    for k, v in record.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_for_csv(v, key))
        elif isinstance(v, list):
            # Skip lists in CSV (only beta_preflight is nested anyway).
            continue
        else:
            out[key] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--snapshot", type=Path,
        help="Path to one snapshot .pt file.",
    )
    src.add_argument(
        "--snapshot-dir", type=Path,
        help="Directory; recursively audits all *.pt files inside.",
    )
    src.add_argument(
        "--snapshot-list", type=Path,
        help="Text file with one snapshot path per line.",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path. CSV companion at "
                        "<output>.csv if --csv-also.")
    parser.add_argument("--csv-also", action="store_true",
                        help="Also write a flattened CSV next to the JSON.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--beta-preflight", action="store_true",
                        help="Run the β-sweep softmax sharpness probe "
                        "(~30s per snapshot at D=4096).")
    parser.add_argument("--betas", type=str, default="1,3,5,10,30",
                        help="Comma-separated β values for preflight.")
    parser.add_argument("--n-cue-probes", type=int, default=4,
                        help="Random cues to average preflight stats over.")
    parser.add_argument("--preflight-seed", type=int, default=17,
                        help="RNG seed for the preflight's random cues.")
    args = parser.parse_args()

    if args.snapshot is not None:
        snapshots = [args.snapshot]
    elif args.snapshot_dir is not None:
        snapshots = sorted(args.snapshot_dir.rglob("*.pt"))
        if not snapshots:
            raise SystemExit(f"no .pt files under {args.snapshot_dir}")
    else:
        with open(args.snapshot_list) as f:
            snapshots = [Path(line.strip()) for line in f if line.strip()]

    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    records: List[Dict[str, Any]] = []
    for path in snapshots:
        print(f"[audit] {path}", flush=True)
        records.append(audit_snapshot(
            snapshot_path=path,
            device=args.device,
            beta_preflight=args.beta_preflight,
            betas=betas,
            n_cue_probes=args.n_cue_probes,
            preflight_seed=args.preflight_seed,
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"snapshots": records, "n": len(records)}, f, indent=2)
    print(f"[done] wrote {args.output}")

    if args.csv_also:
        csv_path = args.output.with_suffix(args.output.suffix + ".csv")
        flat_rows = [_flatten_for_csv(r) for r in records]
        # Union of all keys across rows so the CSV header is stable
        # even when some snapshots have coverage_lambda=0 (missing bias fields).
        all_keys: List[str] = []
        seen: set = set()
        for row in flat_rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in flat_rows:
                w.writerow(row)
        print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
