"""Diagnostic: compare decode strategies to isolate the discrimination bottleneck.

Tests three decode approaches on the same retrievals:
  1. Standard: decode from blended Hopfield state (current approach)
  2. Top-pattern: decode from the single best-matching stored pattern
     (avoids blend noise)
  3. Label-lookup: look up the token directly from the top stored
     window (bypasses decode entirely — tests if retrieval finds
     windows sharing the target token)

If label-lookup >> standard, the bottleneck is decode noise.
If label-lookup ≈ standard, the bottleneck is retrieval quality.
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


def run_diagnostic(
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
) -> dict:
    standard_correct = 0
    top_pattern_correct = 0
    label_lookup_correct = 0
    top3_vote_correct = 0
    total = 0

    patterns = memory._patterns

    for window in gen_windows:
        targets = [window[idx] for idx in masked_positions]
        if any(t == vocab.unk_id for t in targets):
            continue

        cue_window = masked_window(window, masked_positions, vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)

        for pos_idx, target_id in zip(masked_positions, targets):
            total += 1

            # Strategy 1: Standard (decode from blended state)
            decoded_std = decode_position(
                substrate, result.state, positions[pos_idx], codebook,
                decode_ids, top_k=1,
            )
            if decoded_std[0][0] == target_id:
                standard_correct += 1

            # Strategy 2: Top-pattern (decode from best single stored pattern)
            top_pattern = patterns[result.top_index]
            decoded_top = decode_position(
                substrate, top_pattern, positions[pos_idx], codebook,
                decode_ids, top_k=1,
            )
            if decoded_top[0][0] == target_id:
                top_pattern_correct += 1

            # Strategy 3: Label-lookup (bypass decode, read from stored window)
            top_window = landscape_windows[result.top_index]
            if pos_idx < len(top_window) and top_window[pos_idx] == target_id:
                label_lookup_correct += 1

            # Strategy 4: Top-3 vote via label-lookup
            weights = torch.tensor(result.weights)
            top3_indices = torch.argsort(weights, descending=True)[:3].tolist()
            votes: dict[int, float] = {}
            for idx in top3_indices:
                w = landscape_windows[idx]
                if pos_idx < len(w):
                    tok = w[pos_idx]
                    votes[tok] = votes.get(tok, 0.0) + result.weights[idx]
            if votes:
                predicted = max(votes, key=votes.get)
                if predicted == target_id:
                    top3_vote_correct += 1

    if total == 0:
        return {k: 0.0 for k in ["standard", "top_pattern", "label_lookup", "top3_vote", "n"]}
    return {
        "standard": standard_correct / total,
        "top_pattern": top_pattern_correct / total,
        "label_lookup": label_lookup_correct / total,
        "top3_vote": top3_vote_correct / total,
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode strategy diagnostic",
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
        f"{'Standard':>9} {'TopPat':>9} {'Label':>9} {'Top3Vote':>9} "
        f"{'N':>5}",
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
                result = run_diagnostic(
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
                )
                print(
                    f"{window_size:3d} {landscape_size:5d} {beta:5.0f} "
                    f"{result['standard']:9.3f} {result['top_pattern']:9.3f} "
                    f"{result['label_lookup']:9.3f} {result['top3_vote']:9.3f} "
                    f"{result['n']:5d}",
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
        for key in ["standard", "top_pattern", "label_lookup", "top3_vote"]:
            avg = sum(r[key] for r in ws_rows) / len(ws_rows)
            best_row = max(ws_rows, key=lambda r: r[key])
            print(
                f"  W={ws} {key:>12}: avg={avg:.3f}  "
                f"best={best_row[key]:.3f} at L={best_row['landscape_size']} β={best_row['beta']}",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
