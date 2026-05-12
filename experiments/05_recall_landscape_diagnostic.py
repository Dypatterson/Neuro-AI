"""Diagnostic: Recall@K × landscape-size sweep.

Tests whether the generalization bottleneck is:
  (a) atom sharpness (right answer nearby but not top-1 → high Recall@5)
  (b) Hopfield interference (small landscapes beat large ones)

Evaluates the reconstruction codebook on masked-token generalization
across L={16,32,64,128,256} × β={10,30,50} × W={4,8}, reporting
top-1, top-5, and top-10 accuracy for each condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

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


def evaluate_recall(
    windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    vocab,
    decode_ids: Sequence[int],
    codebook,
    positions,
    memory: TorchHopfieldMemory[str],
    substrate: TorchFHRR,
    beta: float,
) -> dict:
    top1 = 0
    top5 = 0
    top10 = 0
    total = 0

    for window in windows:
        targets = [window[idx] for idx in masked_positions]
        if any(t == vocab.unk_id for t in targets):
            continue

        cue_window = masked_window(window, masked_positions, vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)

        for pos_idx, target_id in zip(masked_positions, targets):
            decoded = decode_position(
                substrate, result.state, positions[pos_idx], codebook,
                decode_ids, top_k=10,
            )
            ranked_ids = [item[0] for item in decoded]
            total += 1

            if target_id in ranked_ids[:1]:
                top1 += 1
            if target_id in ranked_ids[:5]:
                top5 += 1
            if target_id in ranked_ids[:10]:
                top10 += 1

    if total == 0:
        return {"top1": 0.0, "top5": 0.0, "top10": 0.0, "n": 0}
    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "top10": top10 / total,
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@K × landscape-size diagnostic",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codebook-path", default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt")
    parser.add_argument("--codebook-label", default="reconstruction")
    parser.add_argument("--window-sizes", default="4,8")
    parser.add_argument("--landscape-sizes", default="16,32,64,128,256")
    parser.add_argument("--betas", default="10,30,50")
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--test-samples", type=int, default=128)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    window_sizes = [int(x) for x in args.window_sizes.split(",")]
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
    print(f"  loaded {args.codebook_label} codebook: {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    print(flush=True)
    print(
        f"{'W':>3} {'L':>5} {'Beta':>5} "
        f"{'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'N':>5} "
        f"{'5/1 ratio':>9}",
        flush=True,
    )
    print("-" * 60, flush=True)

    rows: List[dict] = []

    for ws_idx, window_size in enumerate(window_sizes):
        train_windows = make_windows(train_ids, window_size)
        validation_windows = make_windows(validation_ids, window_size)
        if not train_windows or not validation_windows:
            continue

        positions = build_position_vectors(substrate, window_size)
        masked_positions = compute_mask_positions(window_size, 1, args.mask_position)

        gen_windows = sample_windows(
            validation_windows,
            min(args.test_samples, len(validation_windows)),
            seed=args.seed + (10000 * ws_idx) + 2,
        )

        for ls_idx, landscape_size in enumerate(landscape_sizes):
            slice_seed = args.seed + (10000 * ws_idx) + (1000 * ls_idx)
            landscape_windows = sample_windows(
                train_windows, landscape_size, seed=slice_seed,
            )

            memory = TorchHopfieldMemory[str](substrate)
            for idx, window in enumerate(landscape_windows):
                memory.store(
                    encode_window(substrate, positions, codebook, window),
                    label=f"w_{idx}",
                )

            for beta in betas:
                result = evaluate_recall(
                    windows=gen_windows,
                    masked_positions=masked_positions,
                    vocab=vocab,
                    decode_ids=decode_ids,
                    codebook=codebook,
                    positions=positions,
                    memory=memory,
                    substrate=substrate,
                    beta=beta,
                )
                ratio = (
                    result["top5"] / result["top1"]
                    if result["top1"] > 0
                    else float("inf")
                )
                print(
                    f"{window_size:3d} {landscape_size:5d} {beta:5.0f} "
                    f"{result['top1']:7.3f} {result['top5']:7.3f} "
                    f"{result['top10']:7.3f} {result['n']:5d} "
                    f"{ratio:9.1f}x",
                    flush=True,
                )
                rows.append({
                    "window_size": window_size,
                    "landscape_size": landscape_size,
                    "beta": beta,
                    **result,
                })

    print(flush=True)
    print("=== Summary ===", flush=True)

    for ws in window_sizes:
        ws_rows = [r for r in rows if r["window_size"] == ws]
        if not ws_rows:
            continue
        avg_t1 = sum(r["top1"] for r in ws_rows) / len(ws_rows)
        avg_t5 = sum(r["top5"] for r in ws_rows) / len(ws_rows)
        avg_t10 = sum(r["top10"] for r in ws_rows) / len(ws_rows)
        best = max(ws_rows, key=lambda r: r["top1"])
        print(
            f"  W={ws}: avg top-1={avg_t1:.3f} top-5={avg_t5:.3f} "
            f"top-10={avg_t10:.3f}  |  best top-1={best['top1']:.3f} "
            f"at L={best['landscape_size']} β={best['beta']}",
            flush=True,
        )

    small_rows = [r for r in rows if r["landscape_size"] <= 32]
    large_rows = [r for r in rows if r["landscape_size"] >= 128]
    if small_rows and large_rows:
        small_avg = sum(r["top1"] for r in small_rows) / len(small_rows)
        large_avg = sum(r["top1"] for r in large_rows) / len(large_rows)
        print(flush=True)
        print(
            f"  Interference test: L≤32 avg={small_avg:.3f}  "
            f"L≥128 avg={large_avg:.3f}  "
            f"delta={small_avg - large_avg:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
