"""Experiment 13: Train per-scale reconstruction codebooks.

The existing reconstruction codebook was trained at W=8 only. Since
the multi-scale architecture relies most heavily on W=2 (which has
the highest individual accuracy), training scale-specific codebooks
should improve each scale's contribution.

Trains separate codebooks for W=2, W=3, W=4 with reconstruction loss,
using fresh landscapes per consolidation. Saves each codebook to
reports/phase4_per_scale/ for use by experiment 14.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from energy_memory.phase2.corpus import (
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.persistence import save_codebook, save_vocabulary
from energy_memory.phase2.reconstruction_learner import ReconstructionLearner
from energy_memory.substrate.torch_fhrr import TorchFHRR


def train_scale_codebook(
    substrate: TorchFHRR,
    train_ids: Sequence[int],
    vocab,
    window_size: int,
    epochs: int,
    landscape_size: int,
    probe_size: int,
    consolidation_k: int,
    quality_threshold: float,
    lr_pull: float,
    lr_push: float,
    train_beta: float,
    seed: int,
) -> tuple:
    train_windows = make_windows(train_ids, window_size)
    if not train_windows:
        raise ValueError(f"no windows at W={window_size}")

    codebook = substrate.random_vectors(len(vocab.id_to_token))
    learner = ReconstructionLearner(
        substrate=substrate,
        codebook=codebook,
        vocab=vocab,
        lr_pull=lr_pull,
        lr_push=lr_push,
        consolidation_k=consolidation_k,
        quality_threshold=quality_threshold,
    )

    log_rows = []
    for epoch in range(epochs):
        landscape = sample_windows(
            train_windows,
            min(landscape_size, len(train_windows)),
            seed=seed + 500 + epoch,
        )
        probes = sample_windows(
            train_windows,
            min(probe_size, len(train_windows)),
            seed=seed + 600 + epoch,
        )
        for diag in learner.train(
            landscape, probes, window_size, beta=train_beta,
        ):
            print(
                f"  W={window_size} ep={epoch + 1}/{epochs} "
                f"cons={diag['consolidation']} "
                f"buf={diag['buffer_size']} "
                f"pull={diag['pulled']} push={diag['pushed']} "
                f"q={diag['mean_quality']:.3f} "
                f"fail={diag['failure_rate']:.3f}",
                flush=True,
            )
            log_rows.append(diag)

    return learner.codebook, log_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-scale codebook training")
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scales", default="2,3,4")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--landscape-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=2000)
    parser.add_argument("--consolidation-k", type=int, default=100)
    parser.add_argument("--quality-threshold", type=float, default=0.15)
    parser.add_argument("--lr-pull", type=float, default=0.1)
    parser.add_argument("--lr-push", type=float, default=0.05)
    parser.add_argument("--train-beta", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        default="reports/phase4_per_scale",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scales = [int(x) for x in args.scales.split(",")]

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    save_vocabulary(vocab, output_dir / "vocab.json")

    summary: dict = {
        "config": vars(args),
        "scales": {},
    }

    for scale in scales:
        print(f"\n=== Training codebook for W={scale} ===", flush=True)
        codebook, log = train_scale_codebook(
            substrate=substrate,
            train_ids=train_ids,
            vocab=vocab,
            window_size=scale,
            epochs=args.epochs,
            landscape_size=args.landscape_size,
            probe_size=args.probe_size,
            consolidation_k=args.consolidation_k,
            quality_threshold=args.quality_threshold,
            lr_pull=args.lr_pull,
            lr_push=args.lr_push,
            train_beta=args.train_beta,
            seed=args.seed,
        )
        codebook_path = output_dir / f"codebook_w{scale}.pt"
        save_codebook(codebook, codebook_path)
        print(f"  saved codebook to {codebook_path}", flush=True)

        summary["scales"][str(scale)] = {
            "codebook_path": str(codebook_path),
            "consolidations": len(log),
            "final_failure_rate": log[-1]["failure_rate"] if log else None,
            "final_mean_quality": log[-1]["mean_quality"] if log else None,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Summary written to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
