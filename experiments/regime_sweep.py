"""MVP 0.2 regime sweep over beta, window, and content-distractor tightness.

Run:
    PYTHONPATH=src python3 experiments/regime_sweep.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from energy_memory.diagnostics import temporal_association_score
from energy_memory.experiments.synthetic_worlds import (
    DISTRACTOR_STREAM,
    TEMPORAL_STREAM,
    build_memory,
    content_neighbors,
    distractor_vectors,
    expected_neighbors,
    mean_temporal_recall,
    random_vectors,
)
from energy_memory.substrate import FHRR

BETAS = [1, 2, 4, 8, 16, 32, 64, 100, 200]
WINDOWS = [1, 2, 3, 4]
FAMILY_NOISES = [0.10, 0.18, 0.35, 0.60]


def sweep(seed: int = 17, dim: int = 512) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for window in WINDOWS:
        substrate = FHRR(dim=dim, seed=seed + window)
        vectors = random_vectors(substrate, TEMPORAL_STREAM)
        ordered_memory = build_memory(substrate, TEMPORAL_STREAM, vectors, window=window)
        shuffled_memory = build_memory(
            substrate,
            TEMPORAL_STREAM,
            vectors,
            window=window,
            shuffle=True,
            seed=seed + 100 + window,
        )
        for beta in BETAS:
            ordered_recall, ordered_entropy, ordered_top = mean_temporal_recall(
                substrate, TEMPORAL_STREAM, vectors, ordered_memory, window=window, beta=beta
            )
            shuffle_recall, shuffle_entropy, shuffle_top = mean_temporal_recall(
                substrate, TEMPORAL_STREAM, vectors, shuffled_memory, window=window, beta=beta
            )
            rows.append(
                {
                    "experiment": "temporal_shuffle",
                    "window": window,
                    "beta": beta,
                    "family_noise": "",
                    "ordered_recall": ordered_recall,
                    "shuffle_recall": shuffle_recall,
                    "shuffle_delta": ordered_recall - shuffle_recall,
                    "content_recall": "",
                    "temporal_recall": "",
                    "temporal_advantage": "",
                    "entropy": ordered_entropy,
                    "top_score": ordered_top,
                    "regime": classify_shuffle(ordered_recall, shuffle_recall, ordered_entropy),
                }
            )

    for family_noise in FAMILY_NOISES:
        for window in WINDOWS:
            substrate = FHRR(dim=dim, seed=seed + int(family_noise * 1000) + window)
            vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=family_noise)
            memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=window)
            query = "stair"
            expected = expected_neighbors(DISTRACTOR_STREAM, DISTRACTOR_STREAM.index(query), window)
            k = window * 2
            content_top = content_neighbors(substrate, query, vectors, k=k)
            content_recall = temporal_association_score(expected, content_top, k=k)
            for beta in BETAS:
                temporal = memory.recall(vectors[query], beta=beta, top_k=k)
                temporal_recall = temporal_association_score(expected, temporal.temporal_items, k=k)
                rows.append(
                    {
                        "experiment": "content_distractor",
                        "window": window,
                        "beta": beta,
                        "family_noise": family_noise,
                        "ordered_recall": "",
                        "shuffle_recall": "",
                        "shuffle_delta": "",
                        "content_recall": content_recall,
                        "temporal_recall": temporal_recall,
                        "temporal_advantage": temporal_recall - content_recall,
                        "entropy": temporal.content.entropy,
                        "top_score": temporal.content.top_score,
                        "regime": classify_distractor(temporal_recall, content_recall, temporal.content.entropy),
                    }
                )
    return rows


def classify_shuffle(ordered_recall: float, shuffle_recall: float, entropy: float) -> str:
    if ordered_recall >= 0.85 and ordered_recall - shuffle_recall >= 0.35:
        return "robust_temporal"
    if entropy >= 0.75:
        return "blend"
    if ordered_recall < 0.5:
        return "broken"
    return "weak_temporal"


def classify_distractor(temporal_recall: float, content_recall: float, entropy: float) -> str:
    if temporal_recall >= 0.85 and temporal_recall - content_recall >= 0.5:
        return "robust_temporal"
    if content_recall >= temporal_recall:
        return "interference"
    if entropy >= 0.75:
        return "blend"
    return "weak_temporal"


def write_csv(rows: Iterable[Dict[str, float | int | str]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, float | int | str]]) -> str:
    lines = ["# MVP 0.2 Regime Sweep", ""]
    lines.append("## Regime Counts")
    counts: Dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["experiment"]), str(row["regime"]))
        counts[key] = counts.get(key, 0) + 1
    lines.append("")
    lines.append("| Experiment | Regime | Count |")
    lines.append("|---|---|---:|")
    for (experiment, regime), count in sorted(counts.items()):
        lines.append(f"| {experiment} | {regime} | {count} |")

    lines.append("")
    lines.append("## Best Robust Distractor Conditions")
    robust = [
        row
        for row in rows
        if row["experiment"] == "content_distractor" and row["regime"] == "robust_temporal"
    ]
    robust.sort(key=lambda row: (float(row["temporal_advantage"]), float(row["temporal_recall"])), reverse=True)
    lines.append("")
    lines.append("| Window | Beta | Family noise | Content recall | Temporal recall | Advantage | Entropy |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in robust[:12]:
        lines.append(
            "| {window} | {beta} | {family_noise} | {content_recall:.3f} | {temporal_recall:.3f} | {temporal_advantage:.3f} | {entropy:.3f} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Shuffle-Control Conditions")
    lines.append("")
    lines.append("| Window | Beta | Ordered recall | Shuffle recall | Delta | Entropy | Regime |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row["experiment"] == "temporal_shuffle":
            lines.append(
                "| {window} | {beta} | {ordered_recall:.3f} | {shuffle_recall:.3f} | {shuffle_delta:.3f} | {entropy:.3f} | {regime} |".format(
                    **row
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--csv", default="reports/002_regime_sweep.csv")
    parser.add_argument("--report", default="reports/002_regime_sweep.md")
    args = parser.parse_args()

    rows = sweep(seed=args.seed, dim=args.dim)
    csv_path = Path(args.csv)
    report_path = Path(args.report)
    write_csv(rows, csv_path)
    report = summarize(rows)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print(report)


if __name__ == "__main__":
    main()

