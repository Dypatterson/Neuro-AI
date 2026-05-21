"""Inspect schema-store diversity at a substrate snapshot.

Computes A1 mechanism-validity criterion #1 (per
notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md):

> Across 5 seeds, the top-8 atoms by ``effective_strength`` should have
> mean pairwise FHRR similarity within the band ``[0.10, 0.60]``
> (matches the pre-A+B baseline 0.36 ± reasonable tolerance). The
> failure criterion is *cross-seed median top-8 similarity > 0.60 at
> step 1800*.

Per-seed reporting at n=1 is a signal, not the criterion's final test
(which is n=5). The script does not aggregate; the caller aggregates.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/inspect_topk_diversity.py \\
        --snapshot reports/phase5_a1_pilot_seed17/snapshots/seed17/phase3_phase4_w4_step1800.pt \\
        --k 8 \\
        --output reports/phase5_a1_pilot_seed17/topk_diversity_w4_step1800.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_K = 8


def inspect_topk(snapshot_path: Path, k: int) -> dict:
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    patterns = snap["patterns"]  # complex, [N, D]
    cons = snap.get("consolidation")
    if cons is None:
        raise KeyError(f"Snapshot at {snapshot_path} has no consolidation block")
    u = cons["u"]  # [N, m]
    m = u.shape[1]
    # Default strength weights: 2^(1-k) for k in 1..m, matching
    # ConsolidationState._strength_weights.
    ks = torch.arange(1, m + 1, dtype=u.dtype)
    weights = 2.0 ** (-ks + 1)
    eff = (u * weights.unsqueeze(0)).sum(dim=1)
    n_atoms, dim = patterns.shape
    k = min(k, n_atoms)
    topk_idx = torch.topk(eff, k=k).indices.tolist()
    topk_patterns = patterns[topk_idx]
    # Pairwise FHRR similarity: |G_ij| where G = topk · topk* / D
    gram = (topk_patterns @ topk_patterns.conj().T) / dim
    sims = gram.abs()
    mask = ~torch.eye(k, dtype=torch.bool)
    off_diag = sims[mask].tolist()
    mean_off_diag = sum(off_diag) / len(off_diag) if off_diag else 0.0
    return {
        "snapshot": str(snapshot_path),
        "n_atoms": int(n_atoms),
        "dim": int(dim),
        "k": k,
        "topk_indices": topk_idx,
        "topk_effective_strength": [float(eff[i]) for i in topk_idx],
        "pairwise_similarity_mean_off_diag": float(mean_off_diag),
        "pairwise_similarity_min": float(min(off_diag)) if off_diag else 0.0,
        "pairwise_similarity_max": float(max(off_diag)) if off_diag else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = inspect_topk(args.snapshot, args.k)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))

    print(f"=== Top-{args.k} schema diversity ({args.snapshot.name}) ===")
    print(f"  n_atoms = {result['n_atoms']}, dim = {result['dim']}")
    print(f"  top-{args.k} indices: {result['topk_indices']}")
    print(
        f"  pairwise |G_ij| (off-diag): "
        f"mean={result['pairwise_similarity_mean_off_diag']:.4f}, "
        f"min={result['pairwise_similarity_min']:.4f}, "
        f"max={result['pairwise_similarity_max']:.4f}"
    )
    # Per the A1 design note criterion 1, band is [0.10, 0.60]; report
    # only — pass/fail is computed at the n=5 aggregation step.
    band_low, band_high = 0.10, 0.60
    inside = band_low <= result['pairwise_similarity_mean_off_diag'] <= band_high
    print(f"  In A1 design-note band [{band_low:.2f}, {band_high:.2f}]: {inside}")


if __name__ == "__main__":
    main()
