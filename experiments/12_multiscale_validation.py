"""Experiment 12: Multi-scale validation.

Uses the SAME evaluation approach as Phase 3c (mask_positions function,
same seed offsets) to produce directly comparable numbers. Tests both
random and reconstruction codebooks. Also computes bigram baseline
matching Phase 3c's approach.
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
    mask_positions as compute_mask_positions,
    masked_window,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


def compute_bigram_baseline_phase3c(
    train_ids: List[int],
    test_windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    unk_id: int,
) -> float:
    """Phase 3c bigram: for each masked position, predict from adjacent tokens."""
    bigram_forward: Dict[int, Counter] = defaultdict(Counter)
    bigram_backward: Dict[int, Counter] = defaultdict(Counter)
    for i in range(len(train_ids) - 1):
        bigram_forward[train_ids[i]][train_ids[i + 1]] += 1
        bigram_backward[train_ids[i + 1]][train_ids[i]] += 1

    correct = total = 0
    for window in test_windows:
        for pos in masked_positions:
            target = window[pos]
            if target == unk_id:
                continue
            total += 1
            votes: Counter = Counter()
            if pos > 0:
                votes.update(bigram_forward[window[pos - 1]])
            if pos < len(window) - 1:
                votes.update(bigram_backward[window[pos + 1]])
            if votes and votes.most_common(1)[0][0] == target:
                correct += 1
    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codebook-path", default=None)
    parser.add_argument("--eval-window-sizes", default="4,8")
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--decode-k", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_ws = [int(x) for x in args.eval_window_sizes.split(",")]
    scale_ls = {}
    for item in args.scale_landscape.split(","):
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

    if args.codebook_path:
        codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))
        codebook_label = "reconstruction"
    else:
        codebook = substrate.random_vectors(len(vocab.id_to_token))
        codebook_label = "random"
    print(f"  codebook: {codebook_label} {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    # Build per-scale memories
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
        scale_data[s] = (memory, positions)
        print(f"  scale W={s}: L={actual_l}", flush=True)

    for eval_ws_size in eval_ws:
        masked_positions = compute_mask_positions(eval_ws_size, 1, args.mask_position)
        masked_pos = masked_positions[0]

        eval_windows = make_windows(validation_ids, eval_ws_size)
        test_sample = sample_windows(
            eval_windows,
            min(args.test_samples, len(eval_windows)),
            seed=args.seed + eval_ws_size * 1000 + 2,
        )

        bigram_acc = compute_bigram_baseline_phase3c(
            train_ids, test_sample, masked_positions, vocab.unk_id,
        )

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

                memory, positions = scale_data[s]

                # Center the sub-window on the masked position
                half = s // 2
                sub_start = masked_pos - half
                if s % 2 == 0:
                    sub_start = masked_pos - half + 1
                sub_start = max(0, sub_start)
                if sub_start + s > eval_ws_size:
                    sub_start = eval_ws_size - s
                sub_window = window[sub_start:sub_start + s]
                local_masked = masked_pos - sub_start

                cue_window = list(sub_window)
                cue_window[local_masked] = vocab.mask_id
                cue = encode_window(substrate, positions, codebook, cue_window)
                result = memory.retrieve(cue, beta=args.beta, max_iter=12)

                decoded = decode_position(
                    substrate, result.state, positions[local_masked],
                    codebook, decode_ids, top_k=args.decode_k,
                )
                all_decoded[s] = decoded

                ranked = [item[0] for item in decoded]
                if ranked and ranked[0] == target:
                    per_scale_top1[s] += 1

            combined: Dict[int, float] = defaultdict(float)
            for s, decoded in all_decoded.items():
                for tok_id, score in decoded:
                    combined[tok_id] += score
            if combined:
                best = max(combined, key=combined.get)
                if best == target:
                    combined_top1 += 1

        print(f"\n=== W={eval_ws_size}, mask={args.mask_position}, "
              f"codebook={codebook_label}, β={args.beta}, N={total} ===")
        for s in scales:
            if s <= eval_ws_size and total > 0:
                acc = per_scale_top1[s] / total
                print(f"  W={s:d} scale:  {acc:.3f}")
        if total > 0:
            comb = combined_top1 / total
            print(f"  combined:   {comb:.3f}")
            print(f"  bigram:     {bigram_acc:.3f}")
            print(f"  lift vs bigram: {comb / bigram_acc:.1f}x" if bigram_acc > 0 else "")


if __name__ == "__main__":
    main()
