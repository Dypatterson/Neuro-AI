"""Cue-degradation sweep for coupled settling.

Run:
    PYTHONPATH=src:. python3 experiments/cue_degradation_sweep.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from energy_memory.diagnostics import temporal_association_score
from energy_memory.experiments.synthetic_worlds import (
    DISTRACTOR_STREAM,
    build_memory,
    distractor_vectors,
    expected_neighbors,
)
from energy_memory.substrate import FHRR, Vector

CUE_MODES = ["exact_pair", "single_neighbor", "neighbor_plus_noise", "wrong_pair", "noise_only"]
TEMPORAL_BETAS = [0.5, 1, 2, 4, 8, 16, 32, 64, 100]


def make_temporal_query(substrate: FHRR, vectors: Dict[str, Vector], mode: str) -> Vector:
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


def sweep(seed: int = 31, dim: int = 512, family_noise: float = 0.10, content_beta: float = 80.0):
    rows: List[Dict[str, float | int | str]] = []
    window = 2
    expected_k = window * 2
    for mode in CUE_MODES:
        for temporal_beta in TEMPORAL_BETAS:
            substrate = FHRR(dim=dim, seed=seed)
            vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=family_noise)
            memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=window)
            expected = expected_neighbors(DISTRACTOR_STREAM, DISTRACTOR_STREAM.index("stair"), window)
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
            recall = temporal_association_score(expected, result.temporal_items, k=expected_k)
            final = result.trace[-1]
            rows.append(
                {
                    "cue_mode": mode,
                    "temporal_beta": temporal_beta,
                    "top_label": result.top_label,
                    "recall_at_4": recall,
                    "final_top_weight": final.top_weight,
                    "final_entropy": final.entropy,
                    "iterations": len(result.trace),
                    "converged": str(result.converged),
                    "regime": classify(result.top_label, recall, final.entropy),
                }
            )
    return rows


def write_csv(rows: List[Dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, float | int | str]]) -> str:
    lines = ["# MVP 0.5 Cue-Degradation Sweep", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Cue mode | First committed beta | First correct beta | Failure behavior |")
    lines.append("|---|---:|---:|---|")
    for mode in CUE_MODES:
        subset = [row for row in rows if row["cue_mode"] == mode]
        committed = [row for row in subset if row["regime"] == "committed"]
        correct = [row for row in subset if float(row["recall_at_4"]) >= 0.75 and row["top_label"] == "stair"]
        first_committed = min((float(row["temporal_beta"]) for row in committed), default=None)
        first_correct = min((float(row["temporal_beta"]) for row in correct), default=None)
        failures = sorted({str(row["regime"]) for row in subset if float(row["recall_at_4"]) < 0.75})
        lines.append(
            f"| {mode} | {first_committed if first_committed is not None else 'none'} | "
            f"{first_correct if first_correct is not None else 'none'} | {', '.join(failures) or 'none'} |"
        )

    lines.append("")
    lines.append("## Full Table")
    lines.append("")
    lines.append("| Cue mode | Temporal beta | Top label | Recall@4 | Top weight | Entropy | Iterations | Regime |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {cue_mode} | {temporal_beta} | {top_label} | {recall_at_4:.3f} | {final_top_weight:.3f} | {final_entropy:.3f} | {iterations} | {regime} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--family-noise", type=float, default=0.10)
    parser.add_argument("--content-beta", type=float, default=80.0)
    parser.add_argument("--csv", default="reports/005_cue_degradation_sweep.csv")
    parser.add_argument("--report", default="reports/005_cue_degradation_sweep.md")
    args = parser.parse_args()

    rows = sweep(
        seed=args.seed,
        dim=args.dim,
        family_noise=args.family_noise,
        content_beta=args.content_beta,
    )
    write_csv(rows, Path(args.csv))
    report = summarize(rows)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote {args.csv}")
    print(f"wrote {args.report}")
    print(report)


if __name__ == "__main__":
    main()

