"""Experiment 11: Multi-scale retrieval vs bigram, apples-to-apples.

Uses the same evaluation framework as Phase 3c: center-mask prediction
on W=4 and W=8 validation windows. For each test window, queries
sub-window memories at multiple scales and combines decoded scores.

Compares:
  - Per-scale FHRR Hopfield retrieval
  - Combined multi-scale
  - Bigram baseline (computed on the same test set)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def compute_bigram_baseline(
    train_ids: List[int],
    test_windows: Sequence[tuple[int, ...]],
    masked_pos: int,
    unk_id: int,
) -> float:
    bigram_left: Dict[int, Counter] = defaultdict(Counter)
    bigram_right: Dict[int, Counter] = defaultdict(Counter)
    for i in range(len(train_ids) - 1):
        bigram_left[train_ids[i]][train_ids[i + 1]] += 1
        bigram_right[train_ids[i + 1]][train_ids[i]] += 1

    correct = total = 0
    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1

        votes: Counter = Counter()
        if masked_pos > 0:
            ctx = window[masked_pos - 1]
            votes.update(bigram_left[ctx])
        if masked_pos < len(window) - 1:
            ctx = window[masked_pos + 1]
            votes.update(bigram_right[ctx])
        if votes and votes.most_common(1)[0][0] == target:
            correct += 1

    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-scale vs bigram comparison",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codebook-path", default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt")
    parser.add_argument("--eval-window-sizes", default="4,8")
    parser.add_argument("--scale-landscape-sizes", default="2:4096,3:2048,4:1024,8:512")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--decode-k", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_ws = [int(x) for x in args.eval_window_sizes.split(",")]
    scale_ls = {}
    for item in args.scale_landscape_sizes.split(","):
        s, l = item.split(":")
        scale_ls[int(s)] = int(l)
    scales = sorted(scale_ls.keys())

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

    # Pre-build per-scale memories and positions
    scale_data: Dict[int, Tuple] = {}
    for s in scales:
        train_windows_s = make_windows(train_ids, s)
        actual_l = min(scale_ls[s], len(train_windows_s))
        landscape = sample_windows(train_windows_s, actual_l, seed=args.seed + s * 100)
        positions = build_position_vectors(substrate, s)

        memory = TorchHopfieldMemory[str](substrate)
        for idx, w in enumerate(landscape):
            memory.store(
                encode_window(substrate, positions, codebook, w),
                label=f"w_{idx}",
            )
        scale_data[s] = (memory, positions, landscape)
        print(f"  scale W={s}: L={actual_l} stored", flush=True)

    for eval_ws_size in eval_ws:
        print(f"\n{'='*60}")
        print(f"Evaluating: W={eval_ws_size}, center mask, β={args.beta}")
        print(f"{'='*60}")

        eval_windows = make_windows(validation_ids, eval_ws_size)
        masked_pos = (eval_ws_size + 1) // 2 - 1  # center

        test_sample = sample_windows(
            eval_windows,
            min(args.test_samples, len(eval_windows)),
            seed=args.seed + eval_ws_size * 1000 + 2,
        )

        bigram_acc = compute_bigram_baseline(
            train_ids, test_sample, masked_pos, vocab.unk_id,
        )
        print(f"  Bigram baseline: {bigram_acc:.3f}")

        per_scale_top1 = {s: 0 for s in scales}
        combined_top1 = 0
        total = 0

        for window in test_sample:
            target = window[masked_pos]
            if target == vocab.unk_id:
                continue
            total += 1

            all_decoded: Dict[int, List[Tuple[int, float]]] = {}

            for s in scales:
                if s > eval_ws_size:
                    continue
                if s not in scale_data:
                    continue

                memory, positions, _ = scale_data[s]

                # Extract the sub-window of size s centered on masked_pos
                sub_start = max(0, masked_pos - s // 2)
                if sub_start + s > eval_ws_size:
                    sub_start = eval_ws_size - s
                sub_end = sub_start + s
                sub_window = window[sub_start:sub_end]
                local_masked_pos = masked_pos - sub_start

                cue_window = list(sub_window)
                cue_window[local_masked_pos] = vocab.mask_id
                cue = encode_window(substrate, positions, codebook, cue_window)
                result = memory.retrieve(cue, beta=args.beta, max_iter=12)

                decoded = decode_position(
                    substrate, result.state, positions[local_masked_pos],
                    codebook, decode_ids, top_k=args.decode_k,
                )
                all_decoded[s] = decoded

                ranked = [item[0] for item in decoded]
                if ranked and ranked[0] == target:
                    per_scale_top1[s] += 1

            # Combine across scales
            combined: Dict[int, float] = defaultdict(float)
            for s, decoded in all_decoded.items():
                for tok_id, score in decoded:
                    combined[tok_id] += score
            if combined:
                best = max(combined, key=combined.get)
                if best == target:
                    combined_top1 += 1

        print(f"\n  Results (N={total}):")
        print(f"  {'Scale':>10} {'Top-1':>7}")
        print(f"  {'-'*20}")
        for s in scales:
            if s <= eval_ws_size and total > 0:
                acc = per_scale_top1[s] / total
                vs_bigram = acc / bigram_acc if bigram_acc > 0 else float("inf")
                print(f"  {'W='+str(s):>10} {acc:7.3f}  ({vs_bigram:.1f}x bigram)")
        if total > 0:
            comb_acc = combined_top1 / total
            vs_bigram = comb_acc / bigram_acc if bigram_acc > 0 else float("inf")
            print(f"  {'combined':>10} {comb_acc:7.3f}  ({vs_bigram:.1f}x bigram)")
            print(f"  {'bigram':>10} {bigram_acc:7.3f}")


if __name__ == "__main__":
    main()
