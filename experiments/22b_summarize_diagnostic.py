"""Summarize experiment 22 codebook health diagnostic.

Reads reports/phase34_diagnostic/codebook_health_results.json and emits a
markdown table and a CSV with per-condition, per-checkpoint means and
standard deviations across seeds.

Usage:
    PYTHONPATH=src .venv/bin/python experiments/22b_summarize_diagnostic.py \
        --input reports/phase34_diagnostic/codebook_health_results.json \
        --output reports/phase34_diagnostic/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


METRICS = [
    "top1", "topk", "cap_t_05",
    "drift_from_initial",
    "pair_sim_abs_mean", "pair_sim_p95",
    "failure_rate", "consolidations",
    "mean_retrieval_top_score",
]


def fmt(v: float, decimals: int = 3) -> str:
    if isinstance(v, (int,)):
        return f"{v:d}"
    if v != v:  # NaN
        return "—"
    return f"{v:.{decimals}f}"


def aggregate(
    per_seed: Dict[int, List[Dict]],
) -> List[Dict[str, float]]:
    seeds = sorted(per_seed.keys())
    n = min(len(per_seed[s]) for s in seeds)
    out: List[Dict[str, float]] = []
    for i in range(n):
        row: Dict[str, float] = {}
        row["cues_seen"] = per_seed[seeds[0]][i]["cues_seen"]
        for m in METRICS:
            vals = [per_seed[s][i].get(m, float("nan")) for s in seeds]
            valid = [v for v in vals if isinstance(v, (int, float)) and v == v]
            row[f"{m}_mean"] = mean(valid) if valid else float("nan")
            row[f"{m}_std"] = pstdev(valid) if len(valid) > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/phase34_diagnostic/codebook_health_results.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase34_diagnostic/summary.md",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inp = repo_root / args.input
    out = repo_root / args.output
    csv_path = out.with_suffix(".csv")

    data = json.loads(inp.read_text())
    config = data["config"]
    results = data["results"]
    conditions = list(results.keys())

    aggregated: Dict[str, List[Dict[str, float]]] = {
        cond: aggregate(results[cond]) for cond in conditions
    }

    # Write CSV
    with csv_path.open("w") as f:
        writer = csv.writer(f)
        header = ["condition", "cues_seen"]
        for m in METRICS:
            header += [f"{m}_mean", f"{m}_std"]
        writer.writerow(header)
        for cond in conditions:
            for row in aggregated[cond]:
                writer.writerow(
                    [cond, int(row["cues_seen"])]
                    + [row[f"{m}_mean"] for m in METRICS for _ in [0]]
                    + []  # placeholder
                )
        # The above is convoluted; rewrite cleanly.

    with csv_path.open("w") as f:
        writer = csv.writer(f)
        header = ["condition", "cues_seen"]
        for m in METRICS:
            header += [f"{m}_mean", f"{m}_std"]
        writer.writerow(header)
        for cond in conditions:
            for row in aggregated[cond]:
                line = [cond, int(row["cues_seen"])]
                for m in METRICS:
                    line.append(row[f"{m}_mean"])
                    line.append(row[f"{m}_std"])
                writer.writerow(line)

    # Write markdown
    md: List[str] = []
    md.append("# Phase 3+4 codebook health diagnostic — summary\n")
    md.append("Config:\n")
    md.append("```\n")
    md.append(json.dumps(config, indent=2))
    md.append("\n```\n\n")

    md.append("## Headline: top-1 over time (mean ± std across seeds)\n\n")
    n_checks = min(len(aggregated[c]) for c in conditions)
    step_header = "  ".join(
        f"{int(aggregated[conditions[0]][i]['cues_seen']):>5}"
        for i in range(n_checks)
    )
    md.append(f"| condition | {' | '.join(str(int(aggregated[conditions[0]][i]['cues_seen'])) for i in range(n_checks))} |\n")
    md.append("|" + "|".join(["---"] * (n_checks + 1)) + "|\n")
    for cond in conditions:
        cells = [cond]
        for i in range(n_checks):
            m = aggregated[cond][i]["top1_mean"]
            s = aggregated[cond][i]["top1_std"]
            cells.append(f"{fmt(m)}±{fmt(s)}")
        md.append("| " + " | ".join(cells) + " |\n")

    md.append("\n## Codebook drift (mean cosine distance from initial)\n\n")
    md.append(f"| condition | {' | '.join(str(int(aggregated[conditions[0]][i]['cues_seen'])) for i in range(n_checks))} |\n")
    md.append("|" + "|".join(["---"] * (n_checks + 1)) + "|\n")
    for cond in conditions:
        cells = [cond]
        for i in range(n_checks):
            m = aggregated[cond][i]["drift_from_initial_mean"]
            cells.append(fmt(m, 4))
        md.append("| " + " | ".join(cells) + " |\n")

    md.append("\n## Pairwise codebook |cos| mean (collapse → grows toward 1)\n\n")
    md.append(f"| condition | {' | '.join(str(int(aggregated[conditions[0]][i]['cues_seen'])) for i in range(n_checks))} |\n")
    md.append("|" + "|".join(["---"] * (n_checks + 1)) + "|\n")
    for cond in conditions:
        cells = [cond]
        for i in range(n_checks):
            m = aggregated[cond][i]["pair_sim_abs_mean_mean"]
            cells.append(fmt(m, 4))
        md.append("| " + " | ".join(cells) + " |\n")

    md.append("\n## Top-k over time (mean across seeds)\n\n")
    md.append(f"| condition | {' | '.join(str(int(aggregated[conditions[0]][i]['cues_seen'])) for i in range(n_checks))} |\n")
    md.append("|" + "|".join(["---"] * (n_checks + 1)) + "|\n")
    for cond in conditions:
        cells = [cond]
        for i in range(n_checks):
            m = aggregated[cond][i]["topk_mean"]
            cells.append(fmt(m))
        md.append("| " + " | ".join(cells) + " |\n")

    md.append("\n## Mean retrieval top_score on cue stream\n\n")
    md.append(f"| condition | {' | '.join(str(int(aggregated[conditions[0]][i]['cues_seen'])) for i in range(n_checks))} |\n")
    md.append("|" + "|".join(["---"] * (n_checks + 1)) + "|\n")
    for cond in conditions:
        cells = [cond]
        for i in range(n_checks):
            m = aggregated[cond][i]["mean_retrieval_top_score_mean"]
            cells.append(fmt(m))
        md.append("| " + " | ".join(cells) + " |\n")

    out.write_text("".join(md))
    print(f"summary: {out}")
    print(f"csv:     {csv_path}")


if __name__ == "__main__":
    main()
