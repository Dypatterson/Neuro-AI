"""Experiment 15: Robust validation of multi-scale retrieval.

Validates the headline claim (2x bigram lift) with:
  - N=1000+ test windows per seed
  - 5 random seeds for landscape and test sampling
  - Bootstrap confidence intervals on top-1 accuracy
  - Same-test-set bigram baseline per seed
  - Random-codebook ablation per seed

Reports mean ± 95% CI across seeds for:
  - Per-scale top-1
  - Combined multi-scale top-1
  - Bigram baseline
  - Lift ratio (combined / bigram)
  - Codebook ablation: reconstruction vs random codebook lift

Writes CSV with all per-seed numbers and a markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
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
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


def bootstrap_ci(
    outcomes: List[int],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(outcomes) / n
    samples = []
    for _ in range(n_resamples):
        resample = [outcomes[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(resample) / n)
    samples.sort()
    lo = samples[int(n_resamples * alpha / 2)]
    hi = samples[int(n_resamples * (1 - alpha / 2))]
    return mean, lo, hi


def compute_bigram_baseline(
    train_ids: List[int],
    test_windows: Sequence[tuple[int, ...]],
    masked_positions: Sequence[int],
    unk_id: int,
) -> List[int]:
    forward: Dict[int, Counter] = defaultdict(Counter)
    backward: Dict[int, Counter] = defaultdict(Counter)
    for i in range(len(train_ids) - 1):
        forward[train_ids[i]][train_ids[i + 1]] += 1
        backward[train_ids[i + 1]][train_ids[i]] += 1

    outcomes: List[int] = []
    for window in test_windows:
        for pos in masked_positions:
            target = window[pos]
            if target == unk_id:
                continue
            votes: Counter = Counter()
            if pos > 0:
                votes.update(forward[window[pos - 1]])
            if pos < len(window) - 1:
                votes.update(backward[window[pos + 1]])
            if votes:
                outcomes.append(int(votes.most_common(1)[0][0] == target))
            else:
                outcomes.append(0)
    return outcomes


class ScaleSlot:
    def __init__(
        self,
        substrate: TorchFHRR,
        train_windows: Sequence[tuple[int, ...]],
        window_size: int,
        landscape_size: int,
        codebook: torch.Tensor,
        seed: int,
    ):
        self.substrate = substrate
        self.window_size = window_size
        self.codebook = codebook
        self.positions = build_position_vectors(substrate, window_size)

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
) -> Tuple[int, Dict[int, int]]:
    per_scale_top1: Dict[int, int] = {}
    combined: Dict[int, float] = defaultdict(float)

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
            slot.codebook, decode_ids, top_k=decode_k,
        )
        per_scale_top1[scale] = decoded[0][0] if decoded else -1
        for tok_id, score in decoded:
            combined[tok_id] += score

    combined_top1 = max(combined, key=combined.get) if combined else -1
    return combined_top1, per_scale_top1


def run_eval(
    substrate: TorchFHRR,
    train_ids: List[int],
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    codebook: torch.Tensor,
    scales: List[int],
    scale_landscape: Dict[int, int],
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
    landscape_seed: int,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Returns (per_scale_outcomes, combined_outcomes)."""
    slots: Dict[int, ScaleSlot] = {}
    for s in scales:
        if s > eval_ws_size:
            continue
        train_windows_s = make_windows(train_ids, s)
        slots[s] = ScaleSlot(
            substrate=substrate,
            train_windows=train_windows_s,
            window_size=s,
            landscape_size=scale_landscape[s],
            codebook=codebook,
            seed=landscape_seed + s * 100,
        )

    per_scale_outcomes: Dict[int, List[int]] = {s: [] for s in scales if s <= eval_ws_size}
    combined_outcomes: List[int] = []

    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        combined_top1, per_scale_top1 = eval_window(
            window=window,
            masked_pos=masked_pos,
            eval_ws_size=eval_ws_size,
            scale_slots=slots,
            decode_ids=decode_ids,
            mask_id=mask_id,
            beta=beta,
            decode_k=decode_k,
        )
        combined_outcomes.append(int(combined_top1 == target))
        for s in per_scale_outcomes:
            per_scale_outcomes[s].append(int(per_scale_top1.get(s, -1) == target))

    return per_scale_outcomes, combined_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument(
        "--codebook-path",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--eval-window-sizes", default="4,8")
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--decode-k", type=int, default=20)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seeds", default="11,17,23,31,42")
    parser.add_argument("--output-dir", default="reports/phase4_validation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_ws = [int(x) for x in args.eval_window_sizes.split(",")]
    scale_ls = {}
    for item in args.scale_landscape.split(","):
        s, l = item.split(":")
        scale_ls[int(s)] = int(l)
    scales = sorted(scale_ls.keys())
    seeds = [int(x) for x in args.seeds.split(",")]

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=seeds[0], device=args.device)

    codebook_path = Path(args.codebook_path)
    if not codebook_path.exists():
        print(f"ERROR: codebook not found: {codebook_path}", flush=True)
        return
    recon_codebook = load_codebook(codebook_path, device=str(substrate.device))
    print(f"  loaded reconstruction codebook: {recon_codebook.shape}", flush=True)

    random_codebook = substrate.random_vectors(len(vocab.id_to_token))
    print(f"  built random codebook: {random_codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    rows: List[dict] = []

    for eval_ws_size in eval_ws:
        masked_positions = compute_mask_positions(eval_ws_size, 1, args.mask_position)
        masked_pos = masked_positions[0]
        eval_windows_all = make_windows(validation_ids, eval_ws_size)

        print(f"\n{'=' * 70}")
        print(f"W={eval_ws_size}, mask={args.mask_position}, β={args.beta}")
        print(f"  total validation windows: {len(eval_windows_all)}")
        print(f"  scales: {scales}  landscapes: {scale_ls}")
        print(f"{'=' * 70}")

        for seed in seeds:
            test_windows = sample_windows(
                eval_windows_all,
                min(args.test_samples, len(eval_windows_all)),
                seed=seed * 7 + eval_ws_size * 1000,
            )

            bigram_out = compute_bigram_baseline(
                train_ids, test_windows, masked_positions, vocab.unk_id,
            )
            bigram_mean, bigram_lo, bigram_hi = bootstrap_ci(bigram_out, seed=seed)

            recon_per_scale, recon_combined = run_eval(
                substrate=substrate,
                train_ids=train_ids,
                test_windows=test_windows,
                eval_ws_size=eval_ws_size,
                masked_pos=masked_pos,
                codebook=recon_codebook,
                scales=scales,
                scale_landscape=scale_ls,
                decode_ids=decode_ids,
                mask_id=vocab.mask_id,
                unk_id=vocab.unk_id,
                beta=args.beta,
                decode_k=args.decode_k,
                landscape_seed=seed,
            )
            recon_mean, recon_lo, recon_hi = bootstrap_ci(recon_combined, seed=seed)

            random_per_scale, random_combined = run_eval(
                substrate=substrate,
                train_ids=train_ids,
                test_windows=test_windows,
                eval_ws_size=eval_ws_size,
                masked_pos=masked_pos,
                codebook=random_codebook,
                scales=scales,
                scale_landscape=scale_ls,
                decode_ids=decode_ids,
                mask_id=vocab.mask_id,
                unk_id=vocab.unk_id,
                beta=args.beta,
                decode_k=args.decode_k,
                landscape_seed=seed,
            )
            random_mean, random_lo, random_hi = bootstrap_ci(random_combined, seed=seed)

            lift = recon_mean / bigram_mean if bigram_mean > 0 else float("inf")

            print(
                f"  seed={seed:3d} N={len(recon_combined):4d}  "
                f"bigram={bigram_mean:.3f} [{bigram_lo:.3f},{bigram_hi:.3f}]  "
                f"recon={recon_mean:.3f} [{recon_lo:.3f},{recon_hi:.3f}]  "
                f"random={random_mean:.3f} [{random_lo:.3f},{random_hi:.3f}]  "
                f"lift={lift:.2f}x",
                flush=True,
            )

            row = {
                "eval_w": eval_ws_size,
                "seed": seed,
                "n": len(recon_combined),
                "bigram": bigram_mean,
                "bigram_lo": bigram_lo,
                "bigram_hi": bigram_hi,
                "recon_combined": recon_mean,
                "recon_combined_lo": recon_lo,
                "recon_combined_hi": recon_hi,
                "random_combined": random_mean,
                "random_combined_lo": random_lo,
                "random_combined_hi": random_hi,
                "lift": lift,
            }
            for s in scales:
                if s in recon_per_scale:
                    out = recon_per_scale[s]
                    m, _, _ = bootstrap_ci(out, seed=seed + s)
                    row[f"recon_w{s}"] = m
                if s in random_per_scale:
                    out = random_per_scale[s]
                    m, _, _ = bootstrap_ci(out, seed=seed + s)
                    row[f"random_w{s}"] = m
            rows.append(row)

    # Aggregate across seeds
    print(f"\n{'=' * 70}\nAggregated across seeds (mean ± std)\n{'=' * 70}")
    for eval_ws_size in eval_ws:
        ws_rows = [r for r in rows if r["eval_w"] == eval_ws_size]
        if not ws_rows:
            continue
        def stats(key: str) -> Tuple[float, float]:
            vals = [r[key] for r in ws_rows]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            return mean, math.sqrt(var)

        bg_m, bg_s = stats("bigram")
        rc_m, rc_s = stats("recon_combined")
        rd_m, rd_s = stats("random_combined")
        lift_m, lift_s = stats("lift")

        print(f"\n  W={eval_ws_size}, mask={args.mask_position}")
        print(f"    bigram          : {bg_m:.3f} ± {bg_s:.3f}")
        print(f"    random combined : {rd_m:.3f} ± {rd_s:.3f}  ({rd_m/bg_m:.2f}x bigram)" if bg_m > 0 else "")
        print(f"    recon  combined : {rc_m:.3f} ± {rc_s:.3f}  ({rc_m/bg_m:.2f}x bigram)" if bg_m > 0 else "")
        print(f"    lift ratio      : {lift_m:.2f}x ± {lift_s:.2f}x")
        for s in scales:
            key = f"recon_w{s}"
            if any(key in r for r in ws_rows):
                m, sd = stats(key)
                print(f"    recon W={s} scale : {m:.3f} ± {sd:.3f}")

    # Save CSV
    csv_path = output_dir / "validation_per_seed.csv"
    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n  CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
