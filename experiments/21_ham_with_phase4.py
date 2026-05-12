"""Experiment 21: HAM + Phase 4 discovery channel (Step 2 option A).

Tests whether populating layer-2 attractors via Phase 4's discovery
channel improves on HAM-arithmetic-only retrieval.

Setup: frozen Phase 3c codebook (the validated one). Stream cues
through HAMWithLayer2. For each cue capture trajectory, compute
engagement × low-resolution gate. Above-threshold traces enter the
replay store. Periodically run replay: re-settle traces through
current layer-1 + current layer-2; resolved consensuses become new
layer-2 attractors.

Two conditions, identical seeds and cue stream:
  A. ham_baseline: HAM-arith with empty layer-2 throughout (Step 1
     winner replicated as control)
  B. ham_with_layer2: same HAM, but Phase 4 discovery populates
     layer-2 from streamed cues

Headline metric: top-1 and cap_t05 on held-out test set, evaluated at
checkpoints. Hypothesis: B > A as layer-2 grows, especially at W=8
β=10 where the Step 1 HAM win was largest.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
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
from energy_memory.phase5.ham_aggregator import (
    HAMConfig, HAMScaleInput, predict_top_k,
)
from energy_memory.phase5.ham_with_layer2 import (
    HAML2Result, HAMWithLayer2, Layer2Config,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


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
    return list(eval_window[sub_start:sub_start + scale]), masked_pos - sub_start


def evaluate_ham(
    aggregator: HAMWithLayer2,
    slots: Dict[int, ScaleSlot],
    codebook: torch.Tensor,
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    decode_k: int,
) -> Dict:
    correct_top1 = correct_topk = correct_cap_t05 = 0
    total = 0
    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1
        scale_inputs = {}
        for scale, slot in slots.items():
            if scale > eval_ws_size:
                continue
            sub_window, local_masked = get_sub_window(
                window, eval_ws_size, masked_pos, scale,
            )
            scale_inputs[scale] = HAMScaleInput(
                memory=slot.memory, positions=slot.positions,
                sub_window=sub_window, local_masked_pos=local_masked,
            )
        if not scale_inputs:
            continue
        result = aggregator.retrieve(
            scale_inputs=scale_inputs, codebook=codebook,
            mask_id=mask_id, decode_ids=decode_ids,
        )
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
        return {"n": 0, "top1": 0.0, "topk": 0.0, "cap_t_05": 0.0}
    return {
        "n": total,
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_05": correct_cap_t05 / total,
    }


def stream_with_discovery(
    aggregator: HAMWithLayer2,
    slots: Dict[int, ScaleSlot],
    codebook: torch.Tensor,
    cue_stream: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    decode_k: int,
    store_threshold: float,
    resolve_threshold: float,
    replay_every: int,
    replay_batch_size: int,
    max_age: int,
    enable_discovery: bool,
    rng: random.Random,
) -> dict:
    """Stream cues, build replay store from high-gate traces, periodically
    replay and emit candidates as layer-2 attractors.

    Returns the stats accumulated during streaming.
    """
    store: List[Tuple[Dict[int, HAMScaleInput], float, int]] = []
    cues_seen = 0
    candidates_added = 0
    traces_stored = 0
    decayed = 0

    for cue_window in cue_stream:
        if any(t == unk_id for t in cue_window):
            continue

        scale_inputs = {}
        for scale, slot in slots.items():
            if scale > eval_ws_size:
                continue
            sub_window, local_masked = get_sub_window(
                cue_window, eval_ws_size, masked_pos, scale,
            )
            scale_inputs[scale] = HAMScaleInput(
                memory=slot.memory, positions=slot.positions,
                sub_window=sub_window, local_masked_pos=local_masked,
            )
        if not scale_inputs:
            continue

        result = aggregator.retrieve(
            scale_inputs=scale_inputs, codebook=codebook,
            mask_id=mask_id, decode_ids=decode_ids,
        )
        cues_seen += 1

        gate = result.engagement * (1.0 - result.resolution)
        if enable_discovery and gate > store_threshold:
            store.append((scale_inputs, gate, 0))
            traces_stored += 1

        # Replay cycle
        if enable_discovery and cues_seen % replay_every == 0 and store:
            weights = torch.tensor([g * (1.0 + a) for _, g, a in store])
            weights = weights.clamp(min=1e-9)
            n_sample = min(replay_batch_size, len(store))
            probs = weights / weights.sum()
            sampled_idx = torch.multinomial(probs, n_sample, replacement=False).tolist()
            sampled_idx.sort(reverse=True)

            for idx in sampled_idx:
                trace_inputs, _, age = store[idx]
                replay_result = aggregator.retrieve(
                    scale_inputs=trace_inputs, codebook=codebook,
                    mask_id=mask_id, decode_ids=decode_ids,
                )
                # Discovery criterion: the replay's dynamics converged OR
                # the consensus is sharper than the resolve_threshold.
                # At V=2048, max(consensus) is bounded near 1/V even when
                # the dynamics are converging, so we accept convergence
                # as a sufficient signal for "the system resolved this."
                resolved = (
                    replay_result.converged
                    or replay_result.resolution >= resolve_threshold
                )
                if resolved:
                    aggregator.add_discovery(replay_result.consensus.detach().clone())
                    candidates_added += 1
                    store.pop(idx)
                else:
                    new_gate = replay_result.engagement * (1.0 - replay_result.resolution)
                    if age + 1 > max_age:
                        store.pop(idx)
                        decayed += 1
                    else:
                        store[idx] = (trace_inputs, new_gate, age + 1)
            aggregator.prune_dead()

    return {
        "cues_seen": cues_seen,
        "candidates_added": candidates_added,
        "traces_stored": traces_stored,
        "decayed": decayed,
        "store_size_end": len(store),
        "layer2_size_end": len(aggregator.layer2),
        "replay_cycles_run": (cues_seen // replay_every) if enable_discovery else 0,
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
    parser.add_argument("--eval-window-size", type=int, default=8)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--n-cues", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--seeds", default="11,17,23")
    # Phase 4 settings
    parser.add_argument("--store-threshold", type=float, default=0.5)
    parser.add_argument("--resolve-threshold", type=float, default=0.15)
    parser.add_argument("--replay-every", type=int, default=50)
    parser.add_argument("--replay-batch-size", type=int, default=10)
    parser.add_argument("--max-age", type=int, default=5)
    # Layer-2 settings
    parser.add_argument("--lambda-l2", type=float, default=0.3)
    parser.add_argument("--beta-l2", type=float, default=10.0)
    parser.add_argument("--l2-capacity", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/phase5_layer2")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))
    print(f"  codebook: {codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    masked_positions = compute_mask_positions(
        args.eval_window_size, 1, args.mask_position,
    )
    masked_pos = masked_positions[0]
    eval_windows_all = make_windows(validation_ids, args.eval_window_size)

    rows: List[Dict] = []

    for seed in seeds:
        test_windows = sample_windows(
            eval_windows_all,
            min(args.test_samples, len(eval_windows_all)),
            seed=seed + 9000,
        )
        test_set = set(map(tuple, test_windows))

        cue_pool_size = min(len(eval_windows_all), args.n_cues * 20 + 5000)
        cue_pool = sample_windows(
            eval_windows_all, cue_pool_size, seed=seed + 7000,
        )
        cue_stream = [
            w for w in cue_pool
            if w not in test_set and not any(t == vocab.unk_id for t in w)
        ][: args.n_cues]

        print(f"\n{'=' * 70}")
        print(f"  seed={seed}  cue stream: {len(cue_stream)} cues")
        print(f"{'=' * 70}")

        initial_gen_state = substrate.generator.get_state().clone()

        for condition in ["ham_baseline", "ham_with_layer2"]:
            substrate.generator.set_state(initial_gen_state)
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

            ham_cfg = HAMConfig(
                beta=args.beta, max_iter=args.max_iter, alpha=args.alpha,
                consensus_mode="arithmetic_mean",
            )
            l2_cfg = Layer2Config(
                lambda_l2=args.lambda_l2, beta_l2=args.beta_l2,
                capacity=args.l2_capacity,
            )
            aggregator = HAMWithLayer2(
                substrate=substrate, ham_config=ham_cfg, layer2_config=l2_cfg,
            )

            initial_eval = evaluate_ham(
                aggregator=aggregator, slots=slots, codebook=codebook,
                test_windows=test_windows,
                eval_ws_size=args.eval_window_size,
                masked_pos=masked_pos, decode_ids=decode_ids,
                mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                decode_k=args.decode_k,
            )
            rows.append({
                "seed": seed, "condition": condition, "cues_seen": 0,
                **initial_eval, "layer2_size": 0,
            })
            print(
                f"  [{condition:>16}] step=    0  "
                f"top1={initial_eval['top1']:.3f}  cap_t05={initial_eval['cap_t_05']:.3f}",
                flush=True,
            )

            enable_discovery = (condition == "ham_with_layer2")
            rng = random.Random(seed)

            # Stream in checkpoint chunks so we can evaluate periodically
            chunk_size = args.checkpoint_every
            for chunk_start in range(0, len(cue_stream), chunk_size):
                chunk = cue_stream[chunk_start:chunk_start + chunk_size]
                stats = stream_with_discovery(
                    aggregator=aggregator, slots=slots, codebook=codebook,
                    cue_stream=chunk,
                    eval_ws_size=args.eval_window_size,
                    masked_pos=masked_pos, decode_ids=decode_ids,
                    mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                    decode_k=args.decode_k,
                    store_threshold=args.store_threshold,
                    resolve_threshold=args.resolve_threshold,
                    replay_every=args.replay_every,
                    replay_batch_size=args.replay_batch_size,
                    max_age=args.max_age,
                    enable_discovery=enable_discovery,
                    rng=rng,
                )
                cues_so_far = chunk_start + stats["cues_seen"]
                ev = evaluate_ham(
                    aggregator=aggregator, slots=slots, codebook=codebook,
                    test_windows=test_windows,
                    eval_ws_size=args.eval_window_size,
                    masked_pos=masked_pos, decode_ids=decode_ids,
                    mask_id=vocab.mask_id, unk_id=vocab.unk_id,
                    decode_k=args.decode_k,
                )
                rows.append({
                    "seed": seed, "condition": condition,
                    "cues_seen": cues_so_far,
                    **ev,
                    "layer2_size": stats["layer2_size_end"],
                    "candidates_added_this_chunk": stats["candidates_added"],
                    "traces_stored_this_chunk": stats["traces_stored"],
                })
                print(
                    f"  [{condition:>16}] step={cues_so_far:>5d}  "
                    f"top1={ev['top1']:.3f}  cap_t05={ev['cap_t_05']:.3f}  "
                    f"L2={stats['layer2_size_end']:>3d}  "
                    f"cands={stats['candidates_added']:>3d}  "
                    f"traces_in={stats['traces_stored']:>3d}  "
                    f"replays={stats['replay_cycles_run']}",
                    flush=True,
                )

    print(f"\n{'=' * 70}\n  Aggregated across seeds (final checkpoint)\n{'=' * 70}")

    final_step = max(r["cues_seen"] for r in rows)
    for condition in ["ham_baseline", "ham_with_layer2"]:
        finals = [
            r for r in rows
            if r["condition"] == condition and r["cues_seen"] == final_step
        ]
        if not finals:
            continue
        t1_vals = [r["top1"] for r in finals]
        cp_vals = [r["cap_t_05"] for r in finals]
        l2_vals = [r["layer2_size"] for r in finals]
        print(
            f"  [{condition:>16}]  step={final_step}  "
            f"top1={mean(t1_vals):.3f}±{(pstdev(t1_vals) if len(t1_vals)>1 else 0):.3f}  "
            f"cap_t05={mean(cp_vals):.3f}±{(pstdev(cp_vals) if len(cp_vals)>1 else 0):.3f}  "
            f"L2_size={mean(l2_vals):.1f}"
        )

    json_path = output_dir / "ham_layer2_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
