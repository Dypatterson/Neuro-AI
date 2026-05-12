"""Experiment 10: W=2 coverage sweep.

Tests whether increasing L at the bigram scale (W=2) closes the gap
to the bigram baseline. The bigram baseline uses ALL training bigrams;
this experiment tests L={1024, 4096, 16384, 65536} to see how accuracy
scales with coverage.

Also compares multi-scale combined with the best single W=2 result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
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
    masked_window,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W=2 coverage sweep",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codebook-path", default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt")
    parser.add_argument("--landscape-sizes", default="1024,4096,16384,65536")
    parser.add_argument("--betas", default="10,30")
    parser.add_argument("--test-samples", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    landscape_sizes = [int(x) for x in args.landscape_sizes.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(args.corpus_source, repo_root, wikitext_name=args.wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)

    codebook_path = Path(args.codebook_path)
    if not codebook_path.exists():
        print(f"ERROR: codebook not found at {codebook_path}", flush=True)
        return
    codebook = load_codebook(codebook_path, device=str(substrate.device))
    print(f"  loaded codebook: {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    window_size = 2
    train_windows = make_windows(train_ids, window_size)
    print(f"  total W=2 training windows: {len(train_windows)}", flush=True)

    positions = build_position_vectors(substrate, window_size)

    test_windows_full = make_windows(validation_ids, window_size)
    test_sample = sample_windows(
        test_windows_full,
        min(args.test_samples, len(test_windows_full)),
        seed=args.seed + 999,
    )

    # Also compute bigram baseline for this exact test set
    from collections import Counter
    bigram_counts: Dict[int, Counter] = defaultdict(Counter)
    for w in train_windows:
        bigram_counts[w[0]][w[1]] += 1

    bigram_top1 = 0
    bigram_total = 0
    for window in test_sample:
        target = window[1]
        if target == vocab.unk_id:
            continue
        bigram_total += 1
        context = window[0]
        if context in bigram_counts:
            predicted = bigram_counts[context].most_common(1)[0][0]
            if predicted == target:
                bigram_top1 += 1
    bigram_acc = bigram_top1 / bigram_total if bigram_total > 0 else 0
    print(f"\n  Bigram baseline on this test set: {bigram_acc:.3f} (N={bigram_total})")

    print(flush=True)
    print(
        f"{'L':>7} {'Beta':>5} "
        f"{'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'N':>5}",
        flush=True,
    )
    print("-" * 50, flush=True)

    for landscape_size in landscape_sizes:
        actual_l = min(landscape_size, len(train_windows))
        landscape = sample_windows(train_windows, actual_l, seed=args.seed + landscape_size)

        print(f"  storing {actual_l} patterns...", flush=True, end="")
        memory = TorchHopfieldMemory[str](substrate)
        for idx, w in enumerate(landscape):
            memory.store(
                encode_window(substrate, positions, codebook, w),
                label=f"w_{idx}",
            )
        print(" done", flush=True)

        for beta in betas:
            top1 = top5 = top10 = total = 0
            for window in test_sample:
                target = window[1]
                if target == vocab.unk_id:
                    continue
                total += 1

                cue_window = [window[0], vocab.mask_id]
                cue = encode_window(substrate, positions, codebook, cue_window)
                result = memory.retrieve(cue, beta=beta, max_iter=12)

                decoded = decode_position(
                    substrate, result.state, positions[1], codebook,
                    decode_ids, top_k=10,
                )
                ranked_ids = [item[0] for item in decoded]
                if target in ranked_ids[:1]:
                    top1 += 1
                if target in ranked_ids[:5]:
                    top5 += 1
                if target in ranked_ids[:10]:
                    top10 += 1

            if total > 0:
                print(
                    f"{actual_l:7d} {beta:5.0f} "
                    f"{top1/total:7.3f} {top5/total:7.3f} "
                    f"{top10/total:7.3f} {total:5d}",
                    flush=True,
                )

    print(f"\n  Bigram baseline: {bigram_acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
