"""Cue-degradation sweep on the Torch hot path (Tier C item 2).

Torch port of ``experiments/cue_degradation_sweep.py``. Same five
cue modes × nine temporal betas × DISTRACTOR_STREAM matrix, run
against ``TorchTemporalAssociationMemory.coupled_recall`` instead of
the pure-Python reference path. Adds two knobs the reference can't
offer:

* ``--device {cpu, mps}``: run the sweep on either backend.
* ``--encoding {bag, permutation}``: toggle between the original
  bag encoding and the directed-slot encoding from
  [reports/010_permutation_slots_coupled_recall.md](../reports/010_permutation_slots_coupled_recall.md).
  Default is ``bag`` to match the reference for verification.

Per [reports/014_mps_benchmark_d4096.md](../reports/014_mps_benchmark_d4096.md):
the DISTRACTOR_STREAM is only 20 atoms long, so this experiment sits
below the L=1024 MPS-speedup threshold. ``--device mps`` is supported
for consistency with the torch hot path, but CPU will be faster in
practice. The real win of the port is being able to run at
``--dim 4096`` (which the reference path technically can but is slow
at) and pick up the permutation-slot encoding.

Run:
    PYTHONPATH=src .venv/bin/python experiments/cue_degradation_sweep_torch.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from energy_memory.diagnostics import temporal_association_score
from energy_memory.experiments.synthetic_worlds import (
    DISTRACTOR_STREAM,
    MOBILITY_FAMILY,
    expected_neighbors,
)
from energy_memory.memory.torch_temporal import TorchTemporalAssociationMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR

CUE_MODES = ["exact_pair", "single_neighbor", "neighbor_plus_noise", "wrong_pair", "noise_only"]
TEMPORAL_BETAS = [0.5, 1, 2, 4, 8, 16, 32, 64, 100]


def make_vectors(
    substrate: TorchFHRR,
    labels: List[str],
    family_noise: float,
) -> Dict[str, "torch.Tensor"]:
    """Reference-equivalent: random vectors per label, with a shared family base."""
    vectors = {label: substrate.random_vector() for label in labels}
    family_base = substrate.random_vector()
    for label in MOBILITY_FAMILY:
        vectors[label] = substrate.perturb(family_base, noise=family_noise)
    return vectors


def make_temporal_query(
    substrate: TorchFHRR,
    vectors: Dict[str, "torch.Tensor"],
    mode: str,
) -> "torch.Tensor":
    if mode == "exact_pair":
        return substrate.bundle([vectors["slip"], vectors["doctor"]])
    if mode == "single_neighbor":
        return vectors["slip"]
    if mode == "neighbor_plus_noise":
        return substrate.bundle([vectors["slip"], substrate.random_vector()])
    if mode == "wrong_pair":
        return substrate.bundle([vectors["ladder"], vectors["paint"]])
    if mode == "noise_only":
        return substrate.random_vector()
    raise ValueError(f"unknown cue mode: {mode}")


def classify(top_label: str, recall: float, entropy: float) -> str:
    if top_label != "stair":
        return "flipped"
    if recall >= 0.75 and entropy < 0.20:
        return "committed"
    if recall >= 0.75:
        return "ambiguous_correct"
    if entropy >= 0.45:
        return "ambiguous_low_recall"
    return "wrong_or_sparse"


def sweep(
    *,
    seed: int,
    dim: int,
    device: str,
    encoding: str,
    family_noise: float,
    content_beta: float,
) -> List[Dict]:
    rows: List[Dict] = []
    window = 2
    expected_k = window * 2
    for mode in CUE_MODES:
        for temporal_beta in TEMPORAL_BETAS:
            substrate = TorchFHRR(dim=dim, seed=seed, device=device)
            vectors = make_vectors(substrate, list(DISTRACTOR_STREAM), family_noise)
            memory = TorchTemporalAssociationMemory(
                substrate, window=window, encoding=encoding
            )
            memory.store_sequence(
                list(DISTRACTOR_STREAM),
                [vectors[label] for label in DISTRACTOR_STREAM],
            )
            expected = expected_neighbors(
                DISTRACTOR_STREAM, DISTRACTOR_STREAM.index("stair"), window
            )
            temporal_query = make_temporal_query(substrate, vectors, mode)
            result = memory.coupled_recall(
                vectors["stair"],
                temporal_query,
                content_beta=content_beta,
                temporal_beta=temporal_beta,
                feedback=0.75,
                max_iter=12,
                top_k=expected_k,
            )
            recall = temporal_association_score(
                expected, result.temporal_items, k=expected_k
            )
            final = result.trace[-1]
            rows.append({
                "cue_mode": mode,
                "temporal_beta": temporal_beta,
                "top_label": result.top_label,
                "recall_at_4": recall,
                "final_top_weight": final.top_weight,
                "final_entropy": final.entropy,
                "iterations": len(result.trace),
                "converged": str(result.converged),
                "regime": classify(result.top_label, recall, final.entropy),
            })
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict], header_note: str) -> str:
    lines = ["# Cue-Degradation Sweep (Torch hot path)", ""]
    if header_note:
        lines.append(header_note)
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Cue mode | First committed beta | First correct beta | Failure behavior |")
    lines.append("|---|---:|---:|---|")
    for mode in CUE_MODES:
        subset = [row for row in rows if row["cue_mode"] == mode]
        committed = [row for row in subset if row["regime"] == "committed"]
        correct = [
            row for row in subset
            if float(row["recall_at_4"]) >= 0.75 and row["top_label"] == "stair"
        ]
        first_committed = min(
            (float(row["temporal_beta"]) for row in committed), default=None
        )
        first_correct = min(
            (float(row["temporal_beta"]) for row in correct), default=None
        )
        failures = sorted({
            str(row["regime"]) for row in subset if float(row["recall_at_4"]) < 0.75
        })
        lines.append(
            f"| {mode} | {first_committed if first_committed is not None else 'none'} | "
            f"{first_correct if first_correct is not None else 'none'} | {', '.join(failures) or 'none'} |"
        )
    lines.append("")
    lines.append("## Full Table")
    lines.append("")
    lines.append(
        "| Cue mode | Temporal beta | Top label | Recall@4 | Top weight | Entropy | Iterations | Regime |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {cue_mode} | {temporal_beta} | {top_label} | "
            "{recall_at_4:.3f} | {final_top_weight:.3f} | "
            "{final_entropy:.3f} | {iterations} | {regime} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--encoding", type=str, default="bag", choices=["bag", "permutation"])
    parser.add_argument("--family-noise", type=float, default=0.10)
    parser.add_argument("--content-beta", type=float, default=80.0)
    parser.add_argument("--csv", default="reports/005_cue_degradation_sweep_torch.csv")
    parser.add_argument("--report", default="reports/005_cue_degradation_sweep_torch.md")
    args = parser.parse_args()

    rows = sweep(
        seed=args.seed,
        dim=args.dim,
        device=args.device,
        encoding=args.encoding,
        family_noise=args.family_noise,
        content_beta=args.content_beta,
    )

    header = (
        f"- device: `{args.device}`\n"
        f"- D: `{args.dim}`\n"
        f"- encoding: `{args.encoding}`\n"
        f"- seed: `{args.seed}`"
    )

    write_csv(rows, Path(args.csv))
    report = summarize(rows, header)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote {args.csv}")
    print(f"wrote {args.report}")
    # Echo just the summary section.
    for line in report.split("## Full Table")[0].splitlines():
        print(line)


if __name__ == "__main__":
    main()
