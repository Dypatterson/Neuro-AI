"""Small Phase 2 retrieval-baseline validation run.

This is the Session 3 bridge: persistence, landscape population, and a small
metrics run over both contextual completion and next-token retrieval. It is not
the full 540-condition sweep yet.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    NgramBaseline,
    build_ngram_baseline,
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    load_repo_sample_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    decode_position,
    encode_window,
    mask_positions as mask_positions_for_condition,
    masked_window,
)
from energy_memory.phase2.metrics import RetrievalAggregate, build_frequency_buckets, summarize_binary_outcomes
from energy_memory.phase2.persistence import save_codebook, save_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


def run(
    repo_root: Path,
    dim: int,
    device: str | None,
    max_vocab: int,
    window_sizes: Sequence[int],
    landscape_sizes: Sequence[int],
    test_samples: int,
    betas: Sequence[float],
    mask_counts: Sequence[int],
    mask_positions: Sequence[str],
    seed: int,
    output_dir: Path,
    corpus_source: str,
    wikitext_name: str,
) -> dict:
    splits = load_corpus_splits(corpus_source, repo_root, wikitext_name=wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)

    substrate = TorchFHRR(dim=dim, seed=seed, device=device)
    codebook = substrate.random_vectors(len(vocab.id_to_token))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocab, output_dir / "phase2_vocab.json")
    save_codebook(codebook, output_dir / "phase2_codebook.pt")

    baseline = build_ngram_baseline(train_ids, unk_id=vocab.unk_id)
    frequency_buckets = build_frequency_buckets(vocab.counts)
    decode_ids = [index for index, token in enumerate(vocab.id_to_token) if token not in {vocab.unk_token, vocab.mask_token}]
    report = {
        "dataset_summary": {
            "source": corpus_source,
            "train_windows_by_size": {},
            "validation_windows_by_size": {},
            "vocab_size": len(vocab.id_to_token),
            "device": str(substrate.device),
            "mps": substrate.is_mps,
            "wikitext_name": wikitext_name if corpus_source == "wikitext" else "",
            "window_sizes": list(window_sizes),
            "mask_counts": list(mask_counts),
            "mask_positions": list(mask_positions),
            "test_samples": test_samples,
            "landscape_sizes": list(landscape_sizes),
            "betas": list(betas),
        },
        "rows": [],
    }

    for window_index, window_size in enumerate(window_sizes):
        train_windows = make_windows(train_ids, window_size)
        validation_windows = make_windows(validation_ids, window_size)
        report["dataset_summary"]["train_windows_by_size"][str(window_size)] = len(train_windows)
        report["dataset_summary"]["validation_windows_by_size"][str(window_size)] = len(validation_windows)
        if not train_windows or not validation_windows:
            continue

        positions = build_position_vectors(substrate, window_size)
        generalization_windows = sample_windows(
            validation_windows,
            min(test_samples, len(validation_windows)),
            seed=seed + (10000 * window_index) + 2,
        )

        for landscape_index, landscape_size in enumerate(landscape_sizes):
            slice_seed = seed + (10000 * window_index) + (1000 * landscape_index)
            landscape_windows = sample_windows(train_windows, landscape_size, seed=slice_seed)
            memorization_windows = sample_windows(
                landscape_windows,
                min(test_samples, len(landscape_windows)),
                seed=slice_seed + 1,
            )

            memory = TorchHopfieldMemory[str](substrate)
            for index, window in enumerate(landscape_windows):
                memory.store(encode_window(substrate, positions, codebook, window), label=f"window_{index}")

            next_token_conditions = [
                ("next_token", "memorization", memorization_windows, [], None, None),
                ("next_token", "generalization", generalization_windows, [], None, None),
            ]
            masked_token_conditions = []
            for mask_count in mask_counts:
                for mask_position in mask_positions:
                    current_masked_positions = mask_positions_for_condition(window_size, mask_count, mask_position)
                    masked_token_conditions.extend(
                        [
                            ("masked_token", "memorization", memorization_windows, current_masked_positions, mask_count, mask_position),
                            ("masked_token", "generalization", generalization_windows, current_masked_positions, mask_count, mask_position),
                        ]
                    )

            for beta in betas:
                for objective, retrieval_condition, windows, current_masked_positions, mask_count, mask_position in [
                    *next_token_conditions,
                    *masked_token_conditions,
                ]:
                    if not windows:
                        continue
                    condition_result = evaluate_condition(
                        objective=objective,
                        windows=windows,
                        masked_positions=current_masked_positions,
                        vocab=vocab,
                        baseline=baseline,
                        decode_ids=decode_ids,
                        codebook=codebook,
                        positions=positions,
                        memory=memory,
                        substrate=substrate,
                        beta=beta,
                        frequency_buckets=frequency_buckets,
                    )
                    if condition_result is None:
                        continue
                    summary, baseline_accuracy, by_bucket = condition_result
                    report["rows"].append(
                        {
                            "objective": objective,
                            "retrieval_condition": retrieval_condition,
                            "window_size": window_size,
                            "mask_count": mask_count,
                            "mask_position": mask_position,
                            "landscape_size": landscape_size,
                            "beta": beta,
                            "summary": _aggregate_to_dict(summary),
                            "bigram_accuracy": baseline_accuracy,
                            "frequency_buckets": by_bucket,
                        }
                    )
    return report


def evaluate_condition(
    objective: str,
    windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    vocab,
    baseline: NgramBaseline,
    decode_ids: Sequence[int],
    codebook,
    positions,
    memory: TorchHopfieldMemory[str],
    substrate: TorchFHRR,
    beta: float,
    frequency_buckets: Dict[str, str],
) -> tuple[RetrievalAggregate, float, Dict[str, float]] | None:
    outcomes: List[int] = []
    baseline_outcomes: List[int] = []
    gaps: List[float] = []
    entropies: List[float] = []
    energies: List[float] = []
    top_scores: List[float] = []
    bucket_outcomes: Dict[str, List[int]] = defaultdict(list)

    for window in windows:
        if objective == "masked_token":
            targets = [window[index] for index in masked_positions]
            if any(target == vocab.unk_id for target in targets):
                continue
            cue_window = masked_window(window, masked_positions, vocab.mask_id)
            cue = encode_window(substrate, positions, codebook, cue_window)
            result = memory.retrieve(cue, beta=beta, max_iter=12)
            decoded = [decode_position(substrate, result.state, positions[index], codebook, decode_ids, top_k=2) for index in masked_positions]
            predictions = [items[0][0] for items in decoded]
            gaps.append(sum(items[0][1] - items[1][1] if len(items) > 1 else items[0][1] for items in decoded) / len(decoded))
            correct = int(all(prediction == target for prediction, target in zip(predictions, targets)))
            baseline_predictions = baseline.predict_masked(window, masked_positions, vocab.unk_id)
            baseline_outcomes.append(int(all(prediction == target for prediction, target in zip(baseline_predictions, targets))))
            bucket_token = vocab.decode_token(targets[0])
        else:
            target = window[-1]
            if target == vocab.unk_id:
                continue
            cue = encode_window(substrate, positions[:-1], codebook, window[:-1])
            result = memory.retrieve(cue, beta=beta, max_iter=12)
            decoded = decode_position(substrate, result.state, positions[-1], codebook, decode_ids, top_k=2)
            prediction = decoded[0][0]
            gaps.append(decoded[0][1] - decoded[1][1] if len(decoded) > 1 else decoded[0][1])
            correct = int(prediction == target)
            baseline_outcomes.append(int(baseline.predict_next(window[-2]) == target))
            bucket_token = vocab.decode_token(target)

        outcomes.append(correct)
        entropies.append(result.entropy)
        energies.append(memory.energy(result.state, beta=beta))
        top_scores.append(result.top_score)
        bucket_outcomes[frequency_buckets.get(bucket_token, "unknown")].append(correct)

    if not outcomes:
        return None
    summary = summarize_binary_outcomes(outcomes, gaps, entropies, energies, top_scores)
    baseline_accuracy = sum(baseline_outcomes) / len(baseline_outcomes) if baseline_outcomes else 0.0
    by_bucket = {bucket: sum(values) / len(values) for bucket, values in bucket_outcomes.items() if values}
    return summary, baseline_accuracy, by_bucket


def write_report(report: dict, path: Path) -> None:
    lines = ["# Phase 2 Retrieval Baseline", ""]
    dataset = report["dataset_summary"]
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- source: `{dataset['source']}`")
    lines.append(f"- vocab size: `{dataset['vocab_size']}`")
    lines.append(
        "- train windows: `"
        + ", ".join(f"W{size}={count}" for size, count in dataset["train_windows_by_size"].items())
        + "`"
    )
    lines.append(
        "- validation windows: `"
        + ", ".join(f"W{size}={count}" for size, count in dataset["validation_windows_by_size"].items())
        + "`"
    )
    lines.append(f"- window sizes: `{', '.join(str(value) for value in dataset['window_sizes'])}`")
    lines.append(f"- mask counts: `{', '.join(str(value) for value in dataset['mask_counts'])}`")
    lines.append(f"- mask positions: `{', '.join(dataset['mask_positions'])}`")
    lines.append(f"- test samples: `{dataset['test_samples']}`")
    lines.append(f"- landscape sizes: `{', '.join(str(value) for value in dataset['landscape_sizes'])}`")
    lines.append(f"- betas: `{', '.join(f'{value:g}' for value in dataset['betas'])}`")
    lines.append(f"- device: `{dataset['device']}`")
    lines.append(f"- mps: `{dataset['mps']}`")
    if dataset.get("wikitext_name"):
        lines.append(f"- dataset config: `{dataset['wikitext_name']}`")
    lines.append("")
    lines.append("## Condition Summaries")
    lines.append("")
    lines.append("| Objective | Retrieval | W | Mask | Position | Landscape | Beta | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for payload in _sorted_rows(report["rows"]):
        summary = payload["summary"]
        lines.append(
            "| {objective} | {retrieval_condition} | {window_size} | {mask_count} | {mask_position} | {landscape_size} | {beta:g} | {accuracy:.3f} | [{lower_ci:.3f}, {upper_ci:.3f}] | {bigram:.3f} | {cap:.3f} | {meta:.3f} | {gap:.3f} | {energy:.3f} |".format(
                objective=payload["objective"],
                retrieval_condition=payload["retrieval_condition"],
                window_size=payload["window_size"],
                mask_count=payload["mask_count"] if payload["mask_count"] is not None else "-",
                mask_position=payload["mask_position"] or "-",
                landscape_size=payload["landscape_size"],
                beta=payload["beta"],
                accuracy=summary["accuracy"],
                lower_ci=summary["lower_ci"],
                upper_ci=summary["upper_ci"],
                bigram=payload["bigram_accuracy"],
                cap=summary["cap_coverage_error"]["0.5"],
                meta=summary["metastable_rate"],
                gap=summary["mean_gap"],
                energy=summary["mean_energy"],
            )
        )
    lines.append("")
    lines.append("## Best Generalization Rows")
    lines.append("")
    lines.append("| Objective | W | Mask | Position | Landscape | Beta | Accuracy | Bigram | Gap |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for objective in ("masked_token", "next_token"):
        best = _best_generalization_row(report["rows"], objective)
        if best is None:
            continue
        lines.append(
            "| {objective} | {window_size} | {mask_count} | {mask_position} | {landscape_size} | {beta:g} | {accuracy:.3f} | {bigram:.3f} | {gap:.3f} |".format(
                objective=objective,
                window_size=best["window_size"],
                mask_count=best["mask_count"] if best["mask_count"] is not None else "-",
                mask_position=best["mask_position"] or "-",
                landscape_size=best["landscape_size"],
                beta=best["beta"],
                accuracy=best["summary"]["accuracy"],
                bigram=best["bigram_accuracy"],
                gap=best["summary"]["mean_gap"],
            )
        )
    lines.append("")
    lines.append("## Frequency Buckets")
    lines.append("")
    for payload in _sorted_rows(report["rows"]):
        label = (
            f"{payload['objective']}:{payload['retrieval_condition']}:"
            f"W{payload['window_size']}:"
            f"M{payload['mask_count'] if payload['mask_count'] is not None else '-'}:"
            f"P{payload['mask_position'] or '-'}:"
            f"L{payload['landscape_size']}:beta{payload['beta']:g}"
        )
        lines.append(f"### {label}")
        for bucket, accuracy in sorted(payload["frequency_buckets"].items()):
            lines.append(f"- `{bucket}`: {accuracy:.3f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report: dict, path: Path) -> None:
    rows = []
    for payload in report["rows"]:
        summary = payload["summary"]
        row = {
            "objective": payload["objective"],
            "retrieval_condition": payload["retrieval_condition"],
            "window_size": payload["window_size"],
            "mask_count": payload["mask_count"] if payload["mask_count"] is not None else "",
            "mask_position": payload["mask_position"] or "",
            "landscape_size": payload["landscape_size"],
            "beta": payload["beta"],
            "accuracy": summary["accuracy"],
            "lower_ci": summary["lower_ci"],
            "upper_ci": summary["upper_ci"],
            "effective_n": summary["effective_n"],
            "bigram_accuracy": payload["bigram_accuracy"],
            "mean_gap": summary["mean_gap"],
            "mean_entropy": summary["mean_entropy"],
            "mean_energy": summary["mean_energy"],
            "mean_top_score": summary["mean_top_score"],
            "cap_error_0_3": summary["cap_coverage_error"]["0.3"],
            "cap_error_0_5": summary["cap_coverage_error"]["0.5"],
            "cap_error_0_7": summary["cap_coverage_error"]["0.7"],
            "metastable_rate": summary["metastable_rate"],
        }
        rows.append(row)
    fieldnames = [
        "objective",
        "retrieval_condition",
        "window_size",
        "mask_count",
        "mask_position",
        "landscape_size",
        "beta",
        "accuracy",
        "lower_ci",
        "upper_ci",
        "effective_n",
        "bigram_accuracy",
        "mean_gap",
        "mean_entropy",
        "mean_energy",
        "mean_top_score",
        "cap_error_0_3",
        "cap_error_0_5",
        "cap_error_0_7",
        "metastable_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_to_dict(summary: RetrievalAggregate) -> dict:
    return {
        "accuracy": summary.accuracy,
        "stdev": summary.stdev,
        "lower_ci": summary.lower_ci,
        "upper_ci": summary.upper_ci,
        "effective_n": summary.effective_n,
        "mean_gap": summary.mean_gap,
        "mean_entropy": summary.mean_entropy,
        "mean_energy": summary.mean_energy,
        "mean_top_score": summary.mean_top_score,
        "cap_coverage_error": {str(key): value for key, value in summary.cap_coverage_error.items()},
        "metastable_rate": summary.metastable_rate,
    }


def _sorted_rows(rows: Sequence[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row["objective"],
            row["retrieval_condition"],
            row["window_size"],
            row["mask_count"] if row["mask_count"] is not None else -1,
            row["mask_position"] or "",
            row["landscape_size"],
            row["beta"],
        ),
    )


def _best_generalization_row(rows: Sequence[dict], objective: str) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["objective"] == objective and row["retrieval_condition"] == "generalization"
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["summary"]["accuracy"],
            row["summary"]["mean_gap"],
            -row["bigram_accuracy"],
        ),
    )


def _parse_int_list(raw: str) -> List[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_float_list(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _parse_str_list(raw: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one value")
    allowed = {"center", "edge", "end"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown mask position value(s): {', '.join(unknown)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-sizes", default=None)
    parser.add_argument("--landscape-sizes", default="64,256,1024")
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--betas", default="1,10,100")
    parser.add_argument("--mask-count", type=int, default=1)
    parser.add_argument("--mask-counts", default=None)
    parser.add_argument("--mask-position", choices=["center", "edge", "end"], default="center")
    parser.add_argument("--mask-positions", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="reports/phase2_validation")
    parser.add_argument("--corpus-source", choices=["repo_sample", "wikitext"], default="repo_sample")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    window_sizes = _parse_int_list(args.window_sizes) if args.window_sizes else [args.window_size]
    landscape_sizes = _parse_int_list(args.landscape_sizes)
    betas = _parse_float_list(args.betas)
    mask_counts = _parse_int_list(args.mask_counts) if args.mask_counts else [args.mask_count]
    mask_position_values = _parse_str_list(args.mask_positions) if args.mask_positions else [args.mask_position]
    report = run(
        repo_root=repo_root,
        dim=args.dim,
        device=args.device,
        max_vocab=args.max_vocab,
        window_sizes=window_sizes,
        landscape_sizes=landscape_sizes,
        test_samples=args.test_samples,
        betas=betas,
        mask_counts=mask_counts,
        mask_positions=mask_position_values,
        seed=args.seed,
        output_dir=output_dir,
        corpus_source=args.corpus_source,
        wikitext_name=args.wikitext_name,
    )
    report_path = output_dir / "02_phase2_retrieval_baseline.md"
    csv_path = output_dir / "02_phase2_retrieval_baseline.csv"
    write_report(report, report_path)
    write_csv(report, csv_path)
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")
    for payload in _sorted_rows(report["rows"]):
        label = (
            f"{payload['objective']}:{payload['retrieval_condition']}:"
            f"W{payload['window_size']}:"
            f"M{payload['mask_count'] if payload['mask_count'] is not None else '-'}:"
            f"P{payload['mask_position'] or '-'}:"
            f"L{payload['landscape_size']}:beta{payload['beta']:g}"
        )
        print(
            f"{label}: accuracy={payload['summary']['accuracy']:.3f} "
            f"bigram={payload['bigram_accuracy']:.3f} "
            f"meta_stable={payload['summary']['metastable_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
