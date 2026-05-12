"""Experiment 07: Consistency reranking to close the top-1 / top-K gap.

The Recall@K diagnostic showed the correct answer is often in the top-10
decoded candidates but not top-1. This experiment tests whether
re-encoding each candidate into the full window and measuring similarity
to the retrieved state can discriminate the correct answer.

For each masked position:
  1. Decode top-K candidates from blended Hopfield state (standard)
  2. For each candidate, re-encode the full window with that candidate
     placed at the masked position
  3. Score: similarity(re-encoded window, retrieved state)
  4. Pick the candidate with highest consistency score

This is purely geometric — no supervisor, no learned reranker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

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


def evaluate_reranking(
    gen_windows: Sequence[tuple[int, ...]],
    landscape_windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    vocab,
    decode_ids: Sequence[int],
    codebook,
    positions,
    memory: TorchHopfieldMemory[str],
    substrate: TorchFHRR,
    beta: float,
    rerank_k: int,
) -> dict:
    standard_top1 = 0
    standard_topk = 0
    reranked_top1 = 0
    total = 0

    for window in gen_windows:
        targets = [window[idx] for idx in masked_positions]
        if any(t == vocab.unk_id for t in targets):
            continue

        cue_window = masked_window(window, masked_positions, vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)

        for pos_idx, target_id in zip(masked_positions, targets):
            total += 1

            decoded = decode_position(
                substrate, result.state, positions[pos_idx], codebook,
                decode_ids, top_k=rerank_k,
            )
            ranked_ids = [item[0] for item in decoded]

            if ranked_ids and ranked_ids[0] == target_id:
                standard_top1 += 1
            if target_id in ranked_ids:
                standard_topk += 1

            # Consistency reranking: for each candidate, place it in the
            # masked position, re-encode the full window, and measure
            # similarity to the retrieved state.
            best_score = -2.0
            best_candidate = ranked_ids[0] if ranked_ids else -1
            for cand_id, _ in decoded:
                trial_window = list(cue_window)
                trial_window[pos_idx] = cand_id
                trial_encoded = encode_window(
                    substrate, positions, codebook, trial_window,
                )
                score = substrate.similarity(trial_encoded, result.state)
                if score > best_score:
                    best_score = score
                    best_candidate = cand_id

            if best_candidate == target_id:
                reranked_top1 += 1

    if total == 0:
        return {k: 0.0 for k in ["standard_top1", "standard_topk", "reranked_top1", "n"]}
    return {
        "standard_top1": standard_top1 / total,
        "standard_topk": standard_topk / total,
        "reranked_top1": reranked_top1 / total,
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consistency reranking experiment",
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
    parser.add_argument("--landscape-sizes", default="32,64,128,256")
    parser.add_argument("--betas", default="10,30")
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--rerank-k", type=int, default=10)
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
        f"{'W':>3} {'L':>5} {'Beta':>5} {'K':>3} "
        f"{'Std Top1':>9} {'Std TopK':>9} {'Reranked':>9} "
        f"{'Lift':>7} {'N':>5}",
        flush=True,
    )
    print("-" * 70, flush=True)

    all_rows: List[dict] = []

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
                result = evaluate_reranking(
                    gen_windows=gen_windows,
                    landscape_windows=landscape_windows,
                    masked_positions=masked_positions,
                    vocab=vocab,
                    decode_ids=decode_ids,
                    codebook=codebook,
                    positions=positions,
                    memory=memory,
                    substrate=substrate,
                    beta=beta,
                    rerank_k=args.rerank_k,
                )
                lift = (
                    result["reranked_top1"] / result["standard_top1"]
                    if result["standard_top1"] > 0
                    else float("inf") if result["reranked_top1"] > 0 else 1.0
                )
                print(
                    f"{window_size:3d} {landscape_size:5d} {beta:5.0f} {args.rerank_k:3d} "
                    f"{result['standard_top1']:9.3f} {result['standard_topk']:9.3f} "
                    f"{result['reranked_top1']:9.3f} "
                    f"{lift:6.1f}x {result['n']:5d}",
                    flush=True,
                )
                all_rows.append({
                    "window_size": window_size,
                    "landscape_size": landscape_size,
                    "beta": beta,
                    **result,
                })

    print(flush=True)
    print("=== Summary (averages across conditions) ===", flush=True)
    for ws in window_sizes:
        ws_rows = [r for r in all_rows if r["window_size"] == ws]
        if not ws_rows:
            continue
        for key in ["standard_top1", "standard_topk", "reranked_top1"]:
            avg = sum(r[key] for r in ws_rows) / len(ws_rows)
            best_row = max(ws_rows, key=lambda r: r[key])
            print(
                f"  W={ws} {key:>14}: avg={avg:.3f}  "
                f"best={best_row[key]:.3f} at L={best_row['landscape_size']} β={best_row['beta']}",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
