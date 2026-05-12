"""Phase 3a: Hebbian codebook learning experiment.

Trains codebook vectors via Hebbian co-occurrence bundling on the
training corpus, then evaluates against the Phase 2 retrieval baseline
using the same masked-token and next-token conditions.  Produces a
side-by-side comparison report: random codebook vs learned codebook.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.codebook_learner import CodebookLearner
from energy_memory.phase2.corpus import (
    NgramBaseline,
    build_ngram_baseline,
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    decode_position,
    encode_window,
    mask_positions as compute_mask_positions,
    masked_window,
)
from energy_memory.phase2.metrics import (
    RetrievalAggregate,
    build_frequency_buckets,
    summarize_binary_outcomes,
)
from energy_memory.phase2.persistence import save_codebook, save_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


# ------------------------------------------------------------------
# Evaluation (reuses Phase 2 logic with a supplied codebook)
# ------------------------------------------------------------------


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
            decoded = [
                decode_position(
                    substrate, result.state, positions[index], codebook,
                    decode_ids, top_k=2,
                )
                for index in masked_positions
            ]
            predictions = [items[0][0] for items in decoded]
            gaps.append(
                sum(
                    items[0][1] - items[1][1] if len(items) > 1 else items[0][1]
                    for items in decoded
                )
                / len(decoded)
            )
            correct = int(
                all(p == t for p, t in zip(predictions, targets))
            )
            baseline_predictions = baseline.predict_masked(
                window, masked_positions, vocab.unk_id,
            )
            baseline_outcomes.append(
                int(all(p == t for p, t in zip(baseline_predictions, targets)))
            )
            bucket_token = vocab.decode_token(targets[0])
        else:
            target = window[-1]
            if target == vocab.unk_id:
                continue
            cue = encode_window(substrate, positions[:-1], codebook, window[:-1])
            result = memory.retrieve(cue, beta=beta, max_iter=12)
            decoded = decode_position(
                substrate, result.state, positions[-1], codebook,
                decode_ids, top_k=2,
            )
            prediction = decoded[0][0]
            gaps.append(
                decoded[0][1] - decoded[1][1]
                if len(decoded) > 1
                else decoded[0][1]
            )
            correct = int(prediction == target)
            baseline_outcomes.append(
                int(baseline.predict_next(window[-2]) == target)
            )
            bucket_token = vocab.decode_token(target)

        outcomes.append(correct)
        entropies.append(result.entropy)
        energies.append(memory.energy(result.state, beta=beta))
        top_scores.append(result.top_score)
        bucket_outcomes[frequency_buckets.get(bucket_token, "unknown")].append(
            correct
        )

    if not outcomes:
        return None
    summary = summarize_binary_outcomes(
        outcomes, gaps, entropies, energies, top_scores,
    )
    baseline_accuracy = (
        sum(baseline_outcomes) / len(baseline_outcomes)
        if baseline_outcomes
        else 0.0
    )
    by_bucket = {
        bucket: sum(values) / len(values)
        for bucket, values in bucket_outcomes.items()
        if values
    }
    return summary, baseline_accuracy, by_bucket


def evaluate_codebook(
    label: str,
    substrate: TorchFHRR,
    codebook,
    vocab,
    train_ids,
    validation_ids,
    baseline: NgramBaseline,
    frequency_buckets: Dict[str, str],
    decode_ids: Sequence[int],
    window_sizes: Sequence[int],
    landscape_sizes: Sequence[int],
    betas: Sequence[float],
    mask_counts: Sequence[int],
    mask_positions: Sequence[str],
    test_samples: int,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []

    for window_index, window_size in enumerate(window_sizes):
        train_windows = make_windows(train_ids, window_size)
        validation_windows = make_windows(validation_ids, window_size)
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
            landscape_windows = sample_windows(
                train_windows, landscape_size, seed=slice_seed,
            )
            memorization_windows = sample_windows(
                landscape_windows,
                min(test_samples, len(landscape_windows)),
                seed=slice_seed + 1,
            )

            memory = TorchHopfieldMemory[str](substrate)
            for index, window in enumerate(landscape_windows):
                memory.store(
                    encode_window(substrate, positions, codebook, window),
                    label=f"window_{index}",
                )

            conditions = []
            for mc in mask_counts:
                for mp in mask_positions:
                    mps = compute_mask_positions(window_size, mc, mp)
                    conditions.append(("masked_token", "memorization", memorization_windows, mps, mc, mp))
                    conditions.append(("masked_token", "generalization", generalization_windows, mps, mc, mp))
            conditions.append(("next_token", "memorization", memorization_windows, [], None, None))
            conditions.append(("next_token", "generalization", generalization_windows, [], None, None))

            for beta in betas:
                for objective, retrieval_cond, windows, mps, mc, mp in conditions:
                    if not windows:
                        continue
                    result = evaluate_condition(
                        objective=objective,
                        windows=windows,
                        masked_positions=mps,
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
                    if result is None:
                        continue
                    summary, bigram_acc, by_bucket = result
                    rows.append({
                        "codebook": label,
                        "objective": objective,
                        "retrieval_condition": retrieval_cond,
                        "window_size": window_size,
                        "mask_count": mc,
                        "mask_position": mp,
                        "landscape_size": landscape_size,
                        "beta": beta,
                        "accuracy": summary.accuracy,
                        "lower_ci": summary.lower_ci,
                        "upper_ci": summary.upper_ci,
                        "bigram_accuracy": bigram_acc,
                        "mean_gap": summary.mean_gap,
                        "mean_entropy": summary.mean_entropy,
                        "mean_energy": summary.mean_energy,
                        "metastable_rate": summary.metastable_rate,
                    })
    return rows


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def write_comparison_report(
    random_rows: List[dict],
    learned_rows: List[dict],
    training_log: List[dict],
    config: dict,
    path: Path,
) -> None:
    lines = [
        "# Phase 3a: Hebbian Codebook Learning Results",
        "",
        "## Training Config",
        "",
    ]
    for key, value in sorted(config.items()):
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Training Log")
    lines.append("")
    lines.append("| Epoch | LR | Mean Drift | Max Sim | Repulsion |")
    lines.append("|---:|---:|---:|---:|---:|")
    for entry in training_log:
        lines.append(
            f"| {entry['epoch']} | {entry['lr']:.5f} | {entry['mean_drift']:.6f} "
            f"| {entry['max_sim']:.4f} | {entry['repulsion_count']} |"
        )
    lines.append("")

    lines.append("## Side-by-Side Comparison")
    lines.append("")
    lines.append(
        "| Objective | Retrieval | W | Mask | Pos | L | Beta "
        "| Random Acc | Learned Acc | Delta | Bigram |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")

    random_index = {_row_key(r): r for r in random_rows}
    for lr_row in sorted(learned_rows, key=_sort_key):
        key = _row_key(lr_row)
        rr = random_index.get(key)
        r_acc = rr["accuracy"] if rr else float("nan")
        l_acc = lr_row["accuracy"]
        delta = l_acc - r_acc
        lines.append(
            "| {obj} | {ret} | {w} | {mc} | {mp} | {ls} | {beta:g} "
            "| {r:.3f} | {l:.3f} | {d:+.3f} | {bg:.3f} |".format(
                obj=lr_row["objective"],
                ret=lr_row["retrieval_condition"],
                w=lr_row["window_size"],
                mc=lr_row["mask_count"] if lr_row["mask_count"] is not None else "-",
                mp=lr_row["mask_position"] or "-",
                ls=lr_row["landscape_size"],
                beta=lr_row["beta"],
                r=r_acc,
                l=l_acc,
                d=delta,
                bg=lr_row["bigram_accuracy"],
            )
        )

    lines.append("")
    lines.append("## Generalization Summary")
    lines.append("")

    for objective in ("masked_token", "next_token"):
        r_gen = [
            r for r in random_rows
            if r["retrieval_condition"] == "generalization" and r["objective"] == objective
        ]
        l_gen = [
            r for r in learned_rows
            if r["retrieval_condition"] == "generalization" and r["objective"] == objective
        ]
        if not r_gen or not l_gen:
            continue
        r_mean = sum(r["accuracy"] for r in r_gen) / len(r_gen)
        l_mean = sum(r["accuracy"] for r in l_gen) / len(l_gen)
        bg_mean = sum(r["bigram_accuracy"] for r in l_gen) / len(l_gen)
        lines.append(
            f"- **{objective}**: random={r_mean:.4f}  learned={l_mean:.4f}  "
            f"delta={l_mean - r_mean:+.4f}  bigram={bg_mean:.4f}"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_csv(
    random_rows: List[dict], learned_rows: List[dict], path: Path,
) -> None:
    all_rows = sorted(random_rows + learned_rows, key=_sort_key)
    if not all_rows:
        return
    fieldnames = [
        "codebook", "objective", "retrieval_condition", "window_size",
        "mask_count", "mask_position", "landscape_size", "beta",
        "accuracy", "lower_ci", "upper_ci", "bigram_accuracy",
        "mean_gap", "mean_entropy", "mean_energy", "metastable_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def _row_key(row: dict) -> tuple:
    return (
        row["objective"],
        row["retrieval_condition"],
        row["window_size"],
        row["mask_count"],
        row["mask_position"],
        row["landscape_size"],
        row["beta"],
    )


def _sort_key(row: dict) -> tuple:
    return (
        row.get("codebook", ""),
        row["objective"],
        row["retrieval_condition"],
        row["window_size"],
        row["mask_count"] if row["mask_count"] is not None else -1,
        row["mask_position"] or "",
        row["landscape_size"],
        row["beta"],
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3a: Hebbian codebook learning experiment",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", choices=["repo_sample", "wikitext"], default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--train-window-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.85)
    parser.add_argument("--repulsion-threshold", type=float, default=0.7)
    parser.add_argument("--repulsion-strength", type=float, default=0.05)

    parser.add_argument("--window-sizes", default="4,8")
    parser.add_argument("--landscape-sizes", default="64,256")
    parser.add_argument("--betas", default="10,30")
    parser.add_argument("--mask-counts", default="1")
    parser.add_argument("--mask-positions", default="center,end")
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--output-dir", default="reports/phase3a_hebbian")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    landscape_sizes = [int(x) for x in args.landscape_sizes.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    mask_counts = [int(x) for x in args.mask_counts.split(",")]
    mask_positions = [x.strip() for x in args.mask_positions.split(",")]

    # ---- corpus & vocab ----
    print("loading corpus...", flush=True)
    splits = load_corpus_splits(args.corpus_source, repo_root, wikitext_name=args.wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)
    print(f"  train: {len(train_ids)} token ids", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    random_codebook = substrate.random_vectors(len(vocab.id_to_token))

    save_vocabulary(vocab, output_dir / "phase3a_vocab.json")
    save_codebook(random_codebook, output_dir / "phase3a_codebook_random.pt")

    baseline = build_ngram_baseline(train_ids, unk_id=vocab.unk_id)
    frequency_buckets = build_frequency_buckets(vocab.counts)
    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    # ---- train codebook ----
    train_windows = make_windows(train_ids, args.train_window_size)
    print(f"training codebook: {len(train_windows)} windows, {args.epochs} epochs", flush=True)

    learner = CodebookLearner(
        substrate, random_codebook, vocab,
        lr=args.lr,
        lr_decay=args.lr_decay,
        repulsion_threshold=args.repulsion_threshold,
        repulsion_strength=args.repulsion_strength,
    )
    training_log: List[dict] = []
    for diagnostics in learner.train(train_windows, epochs=args.epochs):
        training_log.append(diagnostics)
        print(
            f"  epoch {diagnostics['epoch']:2d}: "
            f"lr={diagnostics['lr']:.5f}  "
            f"drift={diagnostics['mean_drift']:.6f}  "
            f"max_sim={diagnostics['max_sim']:.4f}  "
            f"repulsion={diagnostics['repulsion_count']}",
            flush=True,
        )

    learned_codebook = learner.codebook
    save_codebook(learned_codebook, output_dir / "phase3a_codebook_learned.pt")

    # ---- evaluate both codebooks ----
    eval_kwargs = dict(
        substrate=substrate,
        vocab=vocab,
        train_ids=train_ids,
        validation_ids=validation_ids,
        baseline=baseline,
        frequency_buckets=frequency_buckets,
        decode_ids=decode_ids,
        window_sizes=window_sizes,
        landscape_sizes=landscape_sizes,
        betas=betas,
        mask_counts=mask_counts,
        mask_positions=mask_positions,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    print("evaluating random codebook...", flush=True)
    random_rows = evaluate_codebook("random", codebook=random_codebook, **eval_kwargs)
    print("evaluating learned codebook...", flush=True)
    learned_rows = evaluate_codebook("learned", codebook=learned_codebook, **eval_kwargs)

    # ---- write reports ----
    config = {
        "corpus": args.corpus_source,
        "dim": args.dim,
        "device": str(substrate.device),
        "mps": substrate.is_mps,
        "vocab_size": len(vocab.id_to_token),
        "train_window_size": args.train_window_size,
        "train_windows": len(train_windows),
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "repulsion_threshold": args.repulsion_threshold,
        "repulsion_strength": args.repulsion_strength,
        "eval_window_sizes": window_sizes,
        "eval_landscape_sizes": landscape_sizes,
        "eval_betas": betas,
        "eval_mask_counts": mask_counts,
        "eval_mask_positions": mask_positions,
        "test_samples": args.test_samples,
        "seed": args.seed,
    }

    report_path = output_dir / "03_phase3a_hebbian.md"
    csv_path = output_dir / "03_phase3a_hebbian.csv"
    write_comparison_report(random_rows, learned_rows, training_log, config, report_path)
    write_comparison_csv(random_rows, learned_rows, csv_path)
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")

    # ---- print summary ----
    print("\n--- generalization summary ---")
    for objective in ("masked_token", "next_token"):
        r_gen = [
            r for r in random_rows
            if r["retrieval_condition"] == "generalization" and r["objective"] == objective
        ]
        l_gen = [
            r for r in learned_rows
            if r["retrieval_condition"] == "generalization" and r["objective"] == objective
        ]
        if not r_gen or not l_gen:
            continue
        r_mean = sum(r["accuracy"] for r in r_gen) / len(r_gen)
        l_mean = sum(r["accuracy"] for r in l_gen) / len(l_gen)
        bg_mean = sum(r["bigram_accuracy"] for r in l_gen) / len(l_gen)
        print(
            f"  {objective}: random={r_mean:.4f}  learned={l_mean:.4f}  "
            f"delta={l_mean - r_mean:+.4f}  bigram={bg_mean:.4f}"
        )


if __name__ == "__main__":
    main()
