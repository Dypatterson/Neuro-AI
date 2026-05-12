"""Experiment 08: Large landscape + mask-free cue sweep.

Two hypotheses:
  1. Larger L = more patterns = higher chance retrieval finds one sharing
     the target token. Interference test showed no degradation at L=256,
     so pushing to 512-2048 should be safe.
  2. Mask-free cue: drop the mask-token binding from the cue bundle,
     giving a cleaner (W-1)-term context signal.

Reports top-1, top-5, top-10 for each condition plus mask-free vs masked cue.
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


def encode_window_maskfree(
    substrate: TorchFHRR, positions, codebook, window, masked_positions,
):
    """Encode a window excluding the masked positions entirely."""
    terms = []
    for i, tok_id in enumerate(window):
        if i not in masked_positions:
            terms.append(substrate.bind(positions[i], codebook[tok_id]))
    if not terms:
        return substrate.random_vector()
    return substrate.bundle(terms)


def evaluate(
    gen_windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    vocab,
    decode_ids: Sequence[int],
    codebook,
    positions,
    memory: TorchHopfieldMemory[str],
    substrate: TorchFHRR,
    beta: float,
    use_maskfree_cue: bool,
) -> dict:
    top1 = top5 = top10 = total = 0

    for window in gen_windows:
        targets = [window[idx] for idx in masked_positions]
        if any(t == vocab.unk_id for t in targets):
            continue

        if use_maskfree_cue:
            cue = encode_window_maskfree(
                substrate, positions, codebook, window, set(masked_positions),
            )
        else:
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
        description="Large landscape + mask-free cue sweep",
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
    parser.add_argument("--landscape-sizes", default="256,512,1024,2048")
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
        f"{'W':>3} {'L':>5} {'Beta':>5} {'Cue':>8} "
        f"{'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'N':>5}",
        flush=True,
    )
    print("-" * 65, flush=True)

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
            if landscape_size > len(train_windows):
                print(
                    f"  skipping L={landscape_size} (only {len(train_windows)} windows available)",
                    flush=True,
                )
                continue

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
                for maskfree in [False, True]:
                    result = evaluate(
                        gen_windows=gen_windows,
                        masked_positions=masked_positions,
                        vocab=vocab,
                        decode_ids=decode_ids,
                        codebook=codebook,
                        positions=positions,
                        memory=memory,
                        substrate=substrate,
                        beta=beta,
                        use_maskfree_cue=maskfree,
                    )
                    cue_label = "maskfree" if maskfree else "masked"
                    print(
                        f"{window_size:3d} {landscape_size:5d} {beta:5.0f} {cue_label:>8} "
                        f"{result['top1']:7.3f} {result['top5']:7.3f} "
                        f"{result['top10']:7.3f} {result['n']:5d}",
                        flush=True,
                    )
                    all_rows.append({
                        "window_size": window_size,
                        "landscape_size": landscape_size,
                        "beta": beta,
                        "cue": cue_label,
                        **result,
                    })

    print(flush=True)
    print("=== Summary ===", flush=True)
    for ws in window_sizes:
        for cue_type in ["masked", "maskfree"]:
            ws_rows = [r for r in all_rows if r["window_size"] == ws and r["cue"] == cue_type]
            if not ws_rows:
                continue
            avg_t1 = sum(r["top1"] for r in ws_rows) / len(ws_rows)
            avg_t10 = sum(r["top10"] for r in ws_rows) / len(ws_rows)
            best = max(ws_rows, key=lambda r: r["top1"])
            print(
                f"  W={ws} cue={cue_type:>8}: avg top-1={avg_t1:.3f} "
                f"top-10={avg_t10:.3f}  |  best top-1={best['top1']:.3f} "
                f"at L={best['landscape_size']} β={best['beta']}",
                flush=True,
            )
    print(flush=True)

    for cue_type in ["masked", "maskfree"]:
        small = [r for r in all_rows if r["landscape_size"] <= 512 and r["cue"] == cue_type]
        large = [r for r in all_rows if r["landscape_size"] >= 1024 and r["cue"] == cue_type]
        if small and large:
            s_avg = sum(r["top1"] for r in small) / len(small)
            l_avg = sum(r["top1"] for r in large) / len(large)
            print(
                f"  Scale test ({cue_type}): L≤512 avg={s_avg:.3f}  "
                f"L≥1024 avg={l_avg:.3f}  delta={l_avg - s_avg:+.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
