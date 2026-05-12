"""Experiment 09: Multi-scale retrieval.

The full-window matching at W=4 or W=8 requires all context positions to
match, giving poor coverage. At W=2 (bigram scale), only one context
token must match, giving much higher coverage.

This experiment:
  1. Runs per-scale retrieval at W={2, 3, 4, 8} independently
  2. Combines decoded scores across scales via weighted aggregation
  3. Tests whether W=2 matches the bigram baseline
  4. Tests whether multi-scale exceeds any single scale

The target token is always the LAST token of the test window — this way
each scale predicts the same token from increasing context.
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


def build_scale_memory(
    substrate: TorchFHRR,
    codebook,
    train_ids: List[int],
    window_size: int,
    landscape_size: int,
    seed: int,
) -> Tuple[TorchHopfieldMemory, list, list]:
    """Build a Hopfield memory for a single scale."""
    windows = make_windows(train_ids, window_size)
    if not windows:
        return None, [], []
    landscape = sample_windows(windows, min(landscape_size, len(windows)), seed=seed)
    positions = build_position_vectors(substrate, window_size)

    memory = TorchHopfieldMemory[str](substrate)
    for idx, w in enumerate(landscape):
        memory.store(
            encode_window(substrate, positions, codebook, w),
            label=f"w_{idx}",
        )
    return memory, positions, landscape


def decode_at_scale(
    substrate: TorchFHRR,
    codebook,
    memory: TorchHopfieldMemory,
    positions,
    context_window: Sequence[int],
    mask_id: int,
    decode_ids: Sequence[int],
    beta: float,
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    """Decode the last position of a window at a given scale."""
    window_size = len(context_window)
    masked_pos = window_size - 1
    cue_window = list(context_window)
    cue_window[masked_pos] = mask_id
    cue = encode_window(substrate, positions, codebook, cue_window)
    result = memory.retrieve(cue, beta=beta, max_iter=12)
    decoded = decode_position(
        substrate, result.state, positions[masked_pos], codebook,
        decode_ids, top_k=top_k,
    )
    return decoded


def aggregate_scores(
    per_scale_decoded: Dict[int, List[Tuple[int, float]]],
    scale_weights: Dict[int, float],
) -> List[Tuple[int, float]]:
    """Combine decoded scores across scales via weighted sum."""
    combined: Dict[int, float] = defaultdict(float)
    for scale, decoded in per_scale_decoded.items():
        w = scale_weights.get(scale, 1.0)
        for token_id, score in decoded:
            combined[token_id] += w * score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-scale retrieval experiment",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codebook-path", default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt")
    parser.add_argument("--codebook-label", default="reconstruction")
    parser.add_argument("--scales", default="2,3,4,8")
    parser.add_argument("--landscape-size", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--decode-k", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scales = [int(x) for x in args.scales.split(",")]
    max_scale = max(scales)

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
    print(f"  loaded {args.codebook_label} codebook: {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    print(f"\nBuilding memories for scales {scales}...", flush=True)
    scale_memories = {}
    scale_positions = {}
    for s in scales:
        mem, pos, _ = build_scale_memory(
            substrate, codebook, train_ids, s, args.landscape_size,
            seed=args.seed + s * 100,
        )
        if mem is not None:
            scale_memories[s] = mem
            scale_positions[s] = pos
            print(f"  W={s}: {mem.stored_count} patterns stored", flush=True)

    test_windows = make_windows(validation_ids, max_scale)
    if not test_windows:
        print("ERROR: no test windows", flush=True)
        return
    test_sample = sample_windows(
        test_windows,
        min(args.test_samples, len(test_windows)),
        seed=args.seed + 999,
    )

    per_scale_top1 = {s: 0 for s in scales}
    per_scale_top5 = {s: 0 for s in scales}
    per_scale_top10 = {s: 0 for s in scales}
    combined_top1 = 0
    combined_top5 = 0
    combined_top10 = 0
    total = 0

    scale_weights = {s: 1.0 for s in scales}

    print(f"\nEvaluating {len(test_sample)} test windows...", flush=True)
    for window in test_sample:
        target_id = window[-1]
        if target_id == vocab.unk_id:
            continue
        total += 1

        per_scale_decoded = {}
        for s in scales:
            if s not in scale_memories:
                continue
            context = window[max_scale - s:]
            decoded = decode_at_scale(
                substrate, codebook,
                scale_memories[s], scale_positions[s],
                context, vocab.mask_id, decode_ids,
                beta=args.beta, top_k=args.decode_k,
            )
            per_scale_decoded[s] = decoded

            ranked_ids = [item[0] for item in decoded]
            if target_id in ranked_ids[:1]:
                per_scale_top1[s] += 1
            if target_id in ranked_ids[:5]:
                per_scale_top5[s] += 1
            if target_id in ranked_ids[:10]:
                per_scale_top10[s] += 1

        combined = aggregate_scores(per_scale_decoded, scale_weights)
        combined_ids = [item[0] for item in combined]
        if target_id in combined_ids[:1]:
            combined_top1 += 1
        if target_id in combined_ids[:5]:
            combined_top5 += 1
        if target_id in combined_ids[:10]:
            combined_top10 += 1

    print(f"\n=== Results (N={total}, L={args.landscape_size}, β={args.beta}) ===\n")
    print(f"{'Scale':>8} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7}")
    print("-" * 35)
    for s in scales:
        if total > 0:
            t1 = per_scale_top1[s] / total
            t5 = per_scale_top5[s] / total
            t10 = per_scale_top10[s] / total
            print(f"  W={s:<4d} {t1:7.3f} {t5:7.3f} {t10:7.3f}")
    if total > 0:
        ct1 = combined_top1 / total
        ct5 = combined_top5 / total
        ct10 = combined_top10 / total
        print(f"{'combined':>8} {ct1:7.3f} {ct5:7.3f} {ct10:7.3f}")

    print(flush=True)
    best_scale = max(scales, key=lambda s: per_scale_top1[s])
    best_acc = per_scale_top1[best_scale] / total if total > 0 else 0
    comb_acc = combined_top1 / total if total > 0 else 0
    print(
        f"Best single scale: W={best_scale} at {best_acc:.3f}  "
        f"Combined: {comb_acc:.3f}  "
        f"Lift: {comb_acc / best_acc:.1f}x" if best_acc > 0 else "N/A",
        flush=True,
    )


if __name__ == "__main__":
    main()
