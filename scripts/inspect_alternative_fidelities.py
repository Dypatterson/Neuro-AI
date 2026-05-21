"""Tier-1 disambiguation: compare three role-fidelity metrics on a substrate snapshot.

Per the 2026-05-20 brainstorm
(brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md),
Tier 1 of the disambiguation runs three alternative per-atom fidelity
measurements on the existing A1' substrate snapshot:

  1. compute_role_fidelity (current, mean pairwise unbind distance)
     — report 050 found it uniform at 0.9858 ± 0.0000 at D=4096
  2. compute_role_fidelity_decode_margin (Ganesan-style, per research C)
     — decode each unbind to nearest pattern, take max − second_max
  3. compute_role_fidelity_settled (post-settling, per research B B2)
     — pairwise distance on settled-state unbinds

Pre-committed pass threshold (binding): any metric with std > 0.02
across atoms is considered to "rescue" β. Smaller std means the
metric is still substrate-encoding-dominated and the rescue path
through β fails.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/inspect_alternative_fidelities.py \\
        --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \\
        --output reports/phase5_a1prime_pilot_seed17/alt_fidelities_w4_step1800.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# scripts/ is not a Python package; add the parent's src/ to path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# These imports must come after sys.path manipulation.
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory  # noqa: E402
from energy_memory.phase5.role_fidelity import (  # noqa: E402
    compute_role_fidelity,
    compute_role_fidelity_decode_margin,
    compute_role_fidelity_settled,
)


STD_PASS_THRESHOLD = 0.02


def _summary(name: str, values: torch.Tensor) -> dict:
    v = values.detach().cpu().to(torch.float64)
    return {
        "metric": name,
        "n": int(v.shape[0]),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(v.median()),
        "q05": float(v.quantile(0.05)),
        "q95": float(v.quantile(0.95)),
        "passes_threshold": float(v.std()) > STD_PASS_THRESHOLD,
    }


def compute_schema_bindings(
    *,
    substrate: TorchFHRR,
    schemas: torch.Tensor,
    positions,
) -> torch.Tensor:
    """Per-schema unbinds at each position. [N, W, D]."""
    N = schemas.shape[0]
    W = len(positions)
    D = schemas.shape[-1]
    out = torch.zeros(N, W, D, dtype=schemas.dtype, device=schemas.device)
    for i in range(N):
        for r in range(W):
            out[i, r] = substrate.unbind(schemas[i], positions[r].to(schemas.device))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--beta", type=float, default=10.0,
                    help="Hopfield settling β for the settled-fidelity metric.")
    ap.add_argument("--max-iter", type=int, default=12,
                    help="Hopfield max iterations for settling.")
    args = ap.parse_args()

    print(f"[load] {args.snapshot}")
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    patterns = snap["patterns"]  # [N, D] complex
    positions = snap["positions"]  # list/tensor of W position vectors
    N, D = patterns.shape
    if isinstance(positions, torch.Tensor):
        W = positions.shape[0]
        positions_list = [positions[r] for r in range(W)]
    else:
        positions_list = list(positions)
        W = len(positions_list)
    print(f"  N={N}, D={D}, W={W}")

    # Rebuild a minimal substrate + memory matching the snapshot's D.
    substrate = TorchFHRR(dim=D, device="cpu")
    memory = TorchHopfieldMemory(substrate)
    for i in range(N):
        memory.store(patterns[i], label=i)

    # Metric 1: existing pairwise-distance fidelity (the f_i that
    # report 050 found uniform).
    print("[1/3] computing pairwise-distance fidelity (current f_i)...")
    bindings = compute_schema_bindings(
        substrate=substrate, schemas=patterns, positions=positions_list,
    )
    f_pairwise = compute_role_fidelity(bindings)

    # Metric 2: Ganesan-style decode-margin (vocabulary = substrate's
    # own patterns — substrate-self-consistent decode).
    print("[2/3] computing decode-margin fidelity (Ganesan-style)...")
    f_margin = compute_role_fidelity_decode_margin(
        schema_bindings=bindings, vocabulary=patterns,
    )

    # Metric 3: settling-based pairwise distance.
    print(
        f"[3/3] computing settling-based fidelity (β={args.beta}, "
        f"max_iter={args.max_iter}, N={N} settles)..."
    )
    f_settled = compute_role_fidelity_settled(
        schemas=patterns, positions=positions_list,
        memory=memory, substrate=substrate,
        beta=args.beta, max_iter=args.max_iter,
    )

    summaries = [
        _summary("pairwise_distance_current", f_pairwise),
        _summary("decode_margin_ganesan", f_margin),
        _summary("settling_based", f_settled),
    ]

    print()
    print(f"{'metric':<30s}{'mean':>10s}{'std':>10s}{'min':>10s}{'max':>10s}  band[0.02]")
    print("-" * 80)
    for s in summaries:
        flag = "PASS" if s["passes_threshold"] else "fail"
        print(
            f"{s['metric']:<30s}{s['mean']:>10.4f}{s['std']:>10.4f}"
            f"{s['min']:>10.4f}{s['max']:>10.4f}  {flag}"
        )

    payload = {
        "snapshot": str(args.snapshot),
        "n_atoms": int(N),
        "dim": int(D),
        "n_positions": int(W),
        "std_pass_threshold": STD_PASS_THRESHOLD,
        "metrics": summaries,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
