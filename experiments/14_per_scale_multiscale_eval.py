"""Experiment 14: Multi-scale evaluation with per-scale codebooks +
confidence-weighted aggregation.

Combines two improvements over experiment 12:
  (a) Each scale uses its own reconstruction-trained codebook
      (from experiment 13) instead of a shared W=8-trained codebook.
  (b) Per-scale scores are weighted by retrieval confidence — using
      either the decode gap (top1 − top2 similarity) or the Hopfield
      top_score (cosine of state to nearest pattern).

Tests four configurations:
  1. Shared codebook + uniform aggregation (baseline, exp 12)
  2. Shared codebook + confidence-weighted aggregation
  3. Per-scale codebooks + uniform aggregation
  4. Per-scale codebooks + confidence-weighted aggregation

Reports each against the bigram baseline computed on the same test set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


def compute_bigram_baseline(
    train_ids: List[int],
    test_windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    unk_id: int,
) -> float:
    forward: Dict[int, Counter] = defaultdict(Counter)
    backward: Dict[int, Counter] = defaultdict(Counter)
    for i in range(len(train_ids) - 1):
        forward[train_ids[i]][train_ids[i + 1]] += 1
        backward[train_ids[i + 1]][train_ids[i]] += 1

    correct = total = 0
    for window in test_windows:
        for pos in masked_positions:
            target = window[pos]
            if target == unk_id:
                continue
            total += 1
            votes: Counter = Counter()
            if pos > 0:
                votes.update(forward[window[pos - 1]])
            if pos < len(window) - 1:
                votes.update(backward[window[pos + 1]])
            if votes and votes.most_common(1)[0][0] == target:
                correct += 1
    return correct / total if total > 0 else 0.0


class ScaleSlot:
    """Holds one scale's codebook + memory + positions."""

    def __init__(
        self,
        substrate: TorchFHRR,
        train_ids: List[int],
        window_size: int,
        landscape_size: int,
        codebook: torch.Tensor,
        seed: int,
    ):
        self.substrate = substrate
        self.window_size = window_size
        self.codebook = codebook
        self.positions = build_position_vectors(substrate, window_size)

        train_windows = make_windows(train_ids, window_size)
        actual_l = min(landscape_size, len(train_windows))
        landscape = sample_windows(train_windows, actual_l, seed=seed)

        self.memory: TorchHopfieldMemory[str] = TorchHopfieldMemory(substrate)
        for idx, w in enumerate(landscape):
            self.memory.store(
                encode_window(substrate, self.positions, codebook, w),
                label=f"w_{idx}",
            )
        self.landscape_size = actual_l


def eval_window(
    window: Sequence[int],
    masked_pos: int,
    eval_ws_size: int,
    scale_slots: Dict[int, ScaleSlot],
    decode_ids: List[int],
    mask_id: int,
    beta: float,
    decode_k: int,
    aggregation: str,
) -> Tuple[int, Dict[int, int], Dict[int, List[Tuple[int, float, float]]]]:
    """Run all scales on one window. Returns (combined_top1_token,
    per_scale_top1_tokens, per_scale_decoded_with_confidence)."""

    per_scale_decoded: Dict[int, List[Tuple[int, float]]] = {}
    per_scale_confidence: Dict[int, float] = {}
    per_scale_top1: Dict[int, int] = {}

    for scale, slot in scale_slots.items():
        if scale > eval_ws_size:
            continue

        half = scale // 2
        sub_start = masked_pos - half
        if scale % 2 == 0:
            sub_start = masked_pos - half + 1
        sub_start = max(0, sub_start)
        if sub_start + scale > eval_ws_size:
            sub_start = eval_ws_size - scale
        sub_window = list(window[sub_start:sub_start + scale])
        local_masked = masked_pos - sub_start

        cue_window = list(sub_window)
        cue_window[local_masked] = mask_id
        cue = encode_window(slot.substrate, slot.positions, slot.codebook, cue_window)
        result = slot.memory.retrieve(cue, beta=beta, max_iter=12)

        decoded = decode_position(
            slot.substrate, result.state, slot.positions[local_masked],
            slot.codebook, decode_ids, top_k=max(decode_k, 2),
        )

        per_scale_decoded[scale] = decoded
        per_scale_top1[scale] = decoded[0][0] if decoded else -1

        if aggregation == "uniform":
            per_scale_confidence[scale] = 1.0
        elif aggregation == "decode_gap":
            if len(decoded) >= 2:
                gap = decoded[0][1] - decoded[1][1]
                per_scale_confidence[scale] = max(gap, 0.0)
            else:
                per_scale_confidence[scale] = 0.0
        elif aggregation == "top_score":
            per_scale_confidence[scale] = max(result.top_score, 0.0)
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")

    combined: Dict[int, float] = defaultdict(float)
    for scale, decoded in per_scale_decoded.items():
        weight = per_scale_confidence[scale]
        for tok_id, score in decoded:
            combined[tok_id] += weight * score

    combined_top1 = max(combined, key=combined.get) if combined else -1
    return combined_top1, per_scale_top1, per_scale_decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--shared-codebook",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument(
        "--per-scale-dir",
        default="reports/phase4_per_scale",
    )
    parser.add_argument("--eval-window-sizes", default="4,8")
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--test-samples", type=int, default=400)
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
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)

    shared_path = Path(args.shared_codebook)
    if not shared_path.exists():
        print(f"ERROR: shared codebook not found: {shared_path}", flush=True)
        return
    shared_codebook = load_codebook(shared_path, device=str(substrate.device))
    print(f"  loaded shared codebook: {shared_codebook.shape}", flush=True)

    per_scale_dir = Path(args.per_scale_dir)
    per_scale_codebooks: Dict[int, torch.Tensor] = {}
    for s in scales:
        path = per_scale_dir / f"codebook_w{s}.pt"
        if not path.exists():
            print(f"  WARNING: per-scale codebook missing for W={s}: {path}",
                  flush=True)
            continue
        per_scale_codebooks[s] = load_codebook(path, device=str(substrate.device))
        print(f"  loaded per-scale codebook W={s}: "
              f"{per_scale_codebooks[s].shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    # Build scale slots for each configuration
    def build_slots(codebook_map: Dict[int, torch.Tensor]) -> Dict[int, ScaleSlot]:
        slots: Dict[int, ScaleSlot] = {}
        for s in scales:
            cb = codebook_map.get(s)
            if cb is None:
                continue
            slots[s] = ScaleSlot(
                substrate=substrate,
                train_ids=train_ids,
                window_size=s,
                landscape_size=scale_ls[s],
                codebook=cb,
                seed=args.seed + s * 100,
            )
            print(f"  slot W={s}: L={slots[s].landscape_size}", flush=True)
        return slots

    print("\nBuilding shared-codebook scale slots:", flush=True)
    shared_slots = build_slots({s: shared_codebook for s in scales})

    print("\nBuilding per-scale-codebook slots:", flush=True)
    per_scale_slots = build_slots(per_scale_codebooks) if per_scale_codebooks else None

    configurations = [
        ("shared_uniform", shared_slots, "uniform"),
        ("shared_decode_gap", shared_slots, "decode_gap"),
        ("shared_top_score", shared_slots, "top_score"),
    ]
    if per_scale_slots:
        configurations.extend([
            ("perscale_uniform", per_scale_slots, "uniform"),
            ("perscale_decode_gap", per_scale_slots, "decode_gap"),
            ("perscale_top_score", per_scale_slots, "top_score"),
        ])

    for eval_ws_size in eval_ws:
        masked_positions = compute_mask_positions(eval_ws_size, 1, args.mask_position)
        masked_pos = masked_positions[0]

        eval_windows = make_windows(validation_ids, eval_ws_size)
        test_sample = sample_windows(
            eval_windows,
            min(args.test_samples, len(eval_windows)),
            seed=args.seed + eval_ws_size * 1000 + 2,
        )

        bigram_acc = compute_bigram_baseline(
            train_ids, test_sample, masked_positions, vocab.unk_id,
        )

        print(f"\n{'=' * 70}")
        print(f"W={eval_ws_size}, mask={args.mask_position}, β={args.beta}")
        print(f"  Bigram baseline: {bigram_acc:.3f} (N up to {len(test_sample)})")
        print(f"{'=' * 70}")

        for config_name, slots, aggregation in configurations:
            if slots is None:
                continue

            per_scale_correct: Dict[int, int] = defaultdict(int)
            combined_correct = 0
            total = 0

            for window in test_sample:
                target = window[masked_pos]
                if target == vocab.unk_id:
                    continue
                total += 1

                combined_top1, per_scale_top1, _ = eval_window(
                    window=window,
                    masked_pos=masked_pos,
                    eval_ws_size=eval_ws_size,
                    scale_slots=slots,
                    decode_ids=decode_ids,
                    mask_id=vocab.mask_id,
                    beta=args.beta,
                    decode_k=args.decode_k,
                    aggregation=aggregation,
                )
                if combined_top1 == target:
                    combined_correct += 1
                for s, pred in per_scale_top1.items():
                    if pred == target:
                        per_scale_correct[s] += 1

            if total == 0:
                continue
            print(f"\n  [{config_name}] N={total}")
            for s in scales:
                if s in slots:
                    acc = per_scale_correct[s] / total
                    print(f"    W={s} scale: {acc:.3f}")
            comb = combined_correct / total
            lift = comb / bigram_acc if bigram_acc > 0 else float("inf")
            print(f"    combined:  {comb:.3f}  ({lift:.2f}x bigram)")


if __name__ == "__main__":
    main()
