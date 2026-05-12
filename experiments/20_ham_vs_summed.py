"""Experiment 20: HAM-aggregated multi-scale vs summed-scores baseline.

Tests whether HAM-style coupled multi-scale settling (bidirectional
energy convergence between scales) improves on the validated summed-
scores aggregation. Both use the frozen Phase 3c reconstruction
codebook.

Conditions:
  A. summed_scores: existing validated approach (sum decoded scores
     across W={2,3,4}, argmax over token ids)
  B. ham_geometric: HAM with geometric-mean consensus
  C. ham_arithmetic: HAM with arithmetic-mean consensus

Run at β=30 (our validated prototype-mode regime) and β=10 (feature
mode where Phase 4 was supposed to engage). For each β, multiple seeds
+ both eval window sizes (W=4 center, W=8 center) to validate against
the same protocol as experiment 15.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
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
from energy_memory.phase5.ham_aggregator import (
    HAMAggregator,
    HAMConfig,
    HAMScaleInput,
    predict_top_k,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


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
    for w in test_windows:
        for pos in masked_positions:
            target = w[pos]
            if target == unk_id:
                continue
            votes: Counter = Counter()
            if pos > 0:
                votes.update(forward[w[pos - 1]])
            if pos < len(w) - 1:
                votes.update(backward[w[pos + 1]])
            outcomes.append(int(votes and votes.most_common(1)[0][0] == target))
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
        self.positions = build_position_vectors(substrate, window_size)
        actual_l = min(landscape_size, len(train_windows))
        landscape = sample_windows(train_windows, actual_l, seed=seed)
        self.memory = TorchHopfieldMemory(substrate)
        for idx, w in enumerate(landscape):
            self.memory.store(
                encode_window(substrate, self.positions, codebook, w),
                label=f"w_{idx}",
            )


def get_sub_window(
    eval_window: tuple[int, ...],
    eval_ws_size: int,
    masked_pos: int,
    scale: int,
) -> Tuple[List[int], int]:
    half = scale // 2
    sub_start = masked_pos - half
    if scale % 2 == 0:
        sub_start = masked_pos - half + 1
    sub_start = max(0, sub_start)
    if sub_start + scale > eval_ws_size:
        sub_start = eval_ws_size - scale
    sub_window = list(eval_window[sub_start:sub_start + scale])
    local_masked_pos = masked_pos - sub_start
    return sub_window, local_masked_pos


def eval_summed_scores(
    slots: Dict[int, ScaleSlot],
    codebook: torch.Tensor,
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
) -> Dict:
    correct_top1 = correct_topk = correct_cap_t05 = 0
    total = 0
    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1
        combined: Dict[int, float] = defaultdict(float)
        for scale, slot in slots.items():
            if scale > eval_ws_size:
                continue
            sub_window, local_masked = get_sub_window(
                window, eval_ws_size, masked_pos, scale,
            )
            cue_w = list(sub_window)
            cue_w[local_masked] = mask_id
            cue = encode_window(
                slot.substrate, slot.positions, codebook, cue_w,
            )
            result = slot.memory.retrieve(cue, beta=beta, max_iter=12)
            decoded = decode_position(
                slot.substrate, result.state,
                slot.positions[local_masked], codebook,
                decode_ids, top_k=decode_k,
            )
            for tok_id, score in decoded:
                combined[tok_id] += score

        if not combined:
            continue
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        if ranked[0][0] == target:
            correct_top1 += 1
        in_topk = False
        tgt_score = -1.0
        for tok_id, score in ranked[:decode_k]:
            if tok_id == target:
                tgt_score = score
                in_topk = True
                break
        if in_topk:
            correct_topk += 1
            if tgt_score >= 0.5:
                correct_cap_t05 += 1

    if total == 0:
        return {"n": 0, "top1": 0.0, "topk": 0.0, "cap_t_05": 0.0}
    return {
        "n": total,
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_05": correct_cap_t05 / total,
    }


def eval_ham(
    slots: Dict[int, ScaleSlot],
    codebook: torch.Tensor,
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    alpha: float,
    consensus_mode: str,
    max_iter: int,
    decode_k: int,
    substrate: TorchFHRR,
) -> Dict:
    aggregator = HAMAggregator(
        substrate=substrate,
        config=HAMConfig(
            beta=beta, max_iter=max_iter, alpha=alpha,
            consensus_mode=consensus_mode,
        ),
    )

    correct_top1 = correct_topk = correct_cap_t05 = 0
    total = 0
    iterations_total = 0

    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1

        scale_inputs: Dict[int, HAMScaleInput] = {}
        for scale, slot in slots.items():
            if scale > eval_ws_size:
                continue
            sub_window, local_masked = get_sub_window(
                window, eval_ws_size, masked_pos, scale,
            )
            scale_inputs[scale] = HAMScaleInput(
                memory=slot.memory,
                positions=slot.positions,
                sub_window=sub_window,
                local_masked_pos=local_masked,
            )

        if not scale_inputs:
            continue
        result = aggregator.retrieve(
            scale_inputs=scale_inputs, codebook=codebook,
            mask_id=mask_id, decode_ids=decode_ids,
        )
        iterations_total += result.iterations

        top_k = predict_top_k(result.consensus, decode_ids, k=decode_k)
        ranked_ids = [t[0] for t in top_k]
        if ranked_ids and ranked_ids[0] == target:
            correct_top1 += 1
        in_topk = False
        tgt_score = -1.0
        for tok_id, score in top_k:
            if tok_id == target:
                tgt_score = score
                in_topk = True
                break
        if in_topk:
            correct_topk += 1
            if tgt_score >= 0.5:
                correct_cap_t05 += 1

    if total == 0:
        return {"n": 0}
    return {
        "n": total,
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_05": correct_cap_t05 / total,
        "mean_iterations": iterations_total / total,
    }


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
    parser.add_argument("--betas", default="10,30")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--output-dir", default="reports/phase5_ham_vs_summed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_ls = {}
    for item in args.scale_landscape.split(","):
        s, l = item.split(":")
        scale_ls[int(s)] = int(l)
    scales = sorted(scale_ls.keys())
    eval_ws = [int(x) for x in args.eval_window_sizes.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
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
    codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))
    print(f"  codebook: {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    rows: List[Dict] = []

    for eval_ws_size in eval_ws:
        masked_positions = compute_mask_positions(eval_ws_size, 1, args.mask_position)
        masked_pos = masked_positions[0]
        eval_windows_all = make_windows(validation_ids, eval_ws_size)

        for seed in seeds:
            test_windows = sample_windows(
                eval_windows_all,
                min(args.test_samples, len(eval_windows_all)),
                seed=seed * 7 + eval_ws_size * 1000,
            )

            bigram_outcomes = compute_bigram_baseline(
                train_ids, test_windows, masked_positions, vocab.unk_id,
            )
            bigram_acc = (
                sum(bigram_outcomes) / len(bigram_outcomes)
                if bigram_outcomes else 0.0
            )

            # Build slots once per seed (deterministic landscape)
            slots = {}
            for s in scales:
                train_windows_s = make_windows(train_ids, s)
                slots[s] = ScaleSlot(
                    substrate=substrate,
                    train_windows=train_windows_s,
                    window_size=s,
                    landscape_size=scale_ls[s],
                    codebook=codebook,
                    seed=seed + s * 100,
                )

            for beta in betas:
                summed = eval_summed_scores(
                    slots=slots, codebook=codebook,
                    test_windows=test_windows,
                    eval_ws_size=eval_ws_size, masked_pos=masked_pos,
                    decode_ids=decode_ids,
                    mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                    beta=beta, decode_k=args.decode_k,
                )

                ham_geom = eval_ham(
                    slots=slots, codebook=codebook,
                    test_windows=test_windows,
                    eval_ws_size=eval_ws_size, masked_pos=masked_pos,
                    decode_ids=decode_ids,
                    mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                    beta=beta, alpha=args.alpha,
                    consensus_mode="geometric_mean",
                    max_iter=args.max_iter, decode_k=args.decode_k,
                    substrate=substrate,
                )

                ham_arith = eval_ham(
                    slots=slots, codebook=codebook,
                    test_windows=test_windows,
                    eval_ws_size=eval_ws_size, masked_pos=masked_pos,
                    decode_ids=decode_ids,
                    mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                    beta=beta, alpha=args.alpha,
                    consensus_mode="arithmetic_mean",
                    max_iter=args.max_iter, decode_k=args.decode_k,
                    substrate=substrate,
                )

                row = {
                    "eval_w": eval_ws_size, "seed": seed, "beta": beta,
                    "bigram": bigram_acc,
                    "summed_top1": summed["top1"], "summed_topk": summed["topk"],
                    "summed_cap_t05": summed["cap_t_05"],
                    "ham_geom_top1": ham_geom["top1"], "ham_geom_topk": ham_geom["topk"],
                    "ham_geom_cap_t05": ham_geom["cap_t_05"],
                    "ham_geom_iter": ham_geom.get("mean_iterations", 0.0),
                    "ham_arith_top1": ham_arith["top1"], "ham_arith_topk": ham_arith["topk"],
                    "ham_arith_cap_t05": ham_arith["cap_t_05"],
                    "ham_arith_iter": ham_arith.get("mean_iterations", 0.0),
                    "n": summed["n"],
                }
                rows.append(row)

                print(
                    f"  W={eval_ws_size} seed={seed} β={beta:>4.0f} N={summed['n']:>4d}  "
                    f"bg={bigram_acc:.3f}  "
                    f"sum1={summed['top1']:.3f} sumK={summed['topk']:.3f}  "
                    f"hgm1={ham_geom['top1']:.3f} hgmK={ham_geom['topk']:.3f}  "
                    f"har1={ham_arith['top1']:.3f} harK={ham_arith['topk']:.3f}",
                    flush=True,
                )

    # Aggregate
    print(f"\n{'=' * 70}\n  Aggregated across seeds (mean ± std)\n{'=' * 70}")
    def agg(filtered: List[Dict], key: str) -> Tuple[float, float]:
        vals = [r[key] for r in filtered]
        if not vals:
            return 0.0, 0.0
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        return m, s

    for eval_ws_size in eval_ws:
        for beta in betas:
            filtered = [
                r for r in rows
                if r["eval_w"] == eval_ws_size and r["beta"] == beta
            ]
            if not filtered:
                continue
            bg_m, _ = agg(filtered, "bigram")
            s_t1_m, s_t1_s = agg(filtered, "summed_top1")
            g_t1_m, g_t1_s = agg(filtered, "ham_geom_top1")
            a_t1_m, a_t1_s = agg(filtered, "ham_arith_top1")
            s_tk_m, _ = agg(filtered, "summed_topk")
            g_tk_m, _ = agg(filtered, "ham_geom_topk")
            a_tk_m, _ = agg(filtered, "ham_arith_topk")
            print(
                f"\n  W={eval_ws_size} β={beta:>4.0f}  bigram={bg_m:.3f}"
            )
            print(
                f"    top1:  summed={s_t1_m:.3f}±{s_t1_s:.3f}  "
                f"ham_geom={g_t1_m:.3f}±{g_t1_s:.3f}  "
                f"ham_arith={a_t1_m:.3f}±{a_t1_s:.3f}"
            )
            print(
                f"    topk:  summed={s_tk_m:.3f}  "
                f"ham_geom={g_tk_m:.3f}  ham_arith={a_tk_m:.3f}"
            )

    json_path = output_dir / "ham_vs_summed_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
