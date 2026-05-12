"""Experiment 17: Cap-coverage and meta-stable-state diagnostic.

Re-measures the multi-scale Phase 4 results through the metrics the
project's standing-diagnostics section actually specifies:

  - Cap-coverage at θ ∈ {0.3, 0.5, 0.7}: fraction of retrievals where
    max stored-pattern cosine at convergence ≥ θ. Settled-state version
    (per PROJECT_PLAN.md and src/energy_memory/phase2/metrics.py).
  - Cap-coverage (target-aware) at θ ∈ {0.3, 0.5, 0.7}: fraction of
    retrievals where the correct target token appears among decoded
    candidates with decoded similarity ≥ θ. Architecture-aligned version
    per consolidation-geometry-diagnostic.md ("meaning lives in the
    retrieval neighborhood").
  - Meta-stable-state rate (HEN methodology): fraction where top_score
    < 0.95 — the system fell into a meta-stable basin rather than
    converging to a clear stored pattern.
  - Softmax entropy and energy at convergence: per-scale.
  - Identity (Recall@1) and neighborhood (Recall@10) for comparison.

Reports per scale (W=2, 3, 4) and aggregated. Multiple seeds. Both
reconstruction and random codebooks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
from energy_memory.phase2.metrics import cap_coverage, meta_stable_rate
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


THETAS = (0.3, 0.5, 0.7)


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


def eval_scale(
    slot: ScaleSlot,
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
) -> Dict[str, List]:
    top_scores: List[float] = []
    entropies: List[float] = []
    energies: List[float] = []
    top_decoded_scores: List[float] = []
    target_decoded_scores: List[float] = []
    target_in_topk: List[int] = []
    target_top1: List[int] = []

    s = slot.window_size
    half = s // 2
    sub_start = masked_pos - half
    if s % 2 == 0:
        sub_start = masked_pos - half + 1
    sub_start = max(0, sub_start)
    if sub_start + s > eval_ws_size:
        sub_start = eval_ws_size - s

    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue

        sub_window = list(window[sub_start:sub_start + s])
        local_masked = masked_pos - sub_start
        cue_window = list(sub_window)
        cue_window[local_masked] = mask_id
        cue = encode_window(slot.substrate, slot.positions, slot.codebook, cue_window)
        result = slot.memory.retrieve(cue, beta=beta, max_iter=12)

        top_scores.append(float(result.top_score))
        entropies.append(float(result.entropy))
        energies.append(float(slot.memory.energy(result.state, beta=beta)))

        decoded = decode_position(
            slot.substrate, result.state, slot.positions[local_masked],
            slot.codebook, decode_ids, top_k=decode_k,
        )
        if decoded:
            top_decoded_scores.append(float(decoded[0][1]))
            tgt_score = None
            tgt_in_topk = 0
            for tok_id, score in decoded:
                if tok_id == target:
                    tgt_score = float(score)
                    tgt_in_topk = 1
                    break
            target_decoded_scores.append(tgt_score if tgt_score is not None else -1.0)
            target_in_topk.append(tgt_in_topk)
            target_top1.append(int(decoded[0][0] == target))
        else:
            top_decoded_scores.append(0.0)
            target_decoded_scores.append(-1.0)
            target_in_topk.append(0)
            target_top1.append(0)

    return {
        "top_scores": top_scores,
        "entropies": entropies,
        "energies": energies,
        "top_decoded_scores": top_decoded_scores,
        "target_decoded_scores": target_decoded_scores,
        "target_in_topk": target_in_topk,
        "target_top1": target_top1,
    }


def target_aware_cap_coverage(
    target_decoded_scores: Sequence[float],
    target_in_topk: Sequence[int],
    threshold: float,
) -> float:
    """Fraction of queries where target is in top-K decoded AND its
    decoded similarity exceeds threshold."""
    if not target_decoded_scores:
        return 0.0
    correct = sum(
        1 for in_topk, score in zip(target_in_topk, target_decoded_scores)
        if in_topk and score >= threshold
    )
    return correct / len(target_decoded_scores)


def summarize_run(label: str, data: Dict[str, List]) -> Dict:
    ts = data["top_scores"]
    summary = {
        "label": label,
        "n": len(ts),
        "mean_top_score": mean(ts) if ts else 0.0,
        "std_top_score": pstdev(ts) if len(ts) > 1 else 0.0,
        "mean_entropy": mean(data["entropies"]) if data["entropies"] else 0.0,
        "mean_energy": mean(data["energies"]) if data["energies"] else 0.0,
        "metastable_rate": meta_stable_rate(ts, threshold=0.95),
        "recall_at_1": (sum(data["target_top1"]) / len(data["target_top1"])) if data["target_top1"] else 0.0,
        "recall_at_k": (sum(data["target_in_topk"]) / len(data["target_in_topk"])) if data["target_in_topk"] else 0.0,
    }
    for theta in THETAS:
        summary[f"cap_cov_settled_{theta}"] = cap_coverage(ts, theta)
        summary[f"cap_cov_error_settled_{theta}"] = 1.0 - cap_coverage(ts, theta)
        summary[f"cap_cov_target_{theta}"] = target_aware_cap_coverage(
            data["target_decoded_scores"], data["target_in_topk"], theta,
        )
    return summary


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
    parser.add_argument("--betas", default="5,10,30")
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seeds", default="11,17,23,31,42")
    parser.add_argument("--output-dir", default="reports/phase4_cap_coverage")
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
    betas = [float(x) for x in args.betas.split(",")]

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=seeds[0], device=args.device)

    recon_codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))
    random_codebook = substrate.random_vectors(len(vocab.id_to_token))
    print(f"  recon codebook: {recon_codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    all_rows: List[Dict] = []

    for eval_ws_size in eval_ws:
        masked_positions = compute_mask_positions(eval_ws_size, 1, args.mask_position)
        masked_pos = masked_positions[0]
        eval_windows_all = make_windows(validation_ids, eval_ws_size)

        print(f"\n{'=' * 80}")
        print(f"W={eval_ws_size}, mask={args.mask_position}")
        print(f"{'=' * 80}")

        for seed in seeds:
            test_windows = sample_windows(
                eval_windows_all,
                min(args.test_samples, len(eval_windows_all)),
                seed=seed * 7 + eval_ws_size * 1000,
            )

            for codebook_label, codebook in [
                ("recon", recon_codebook),
                ("random", random_codebook),
            ]:
                for scale in scales:
                    train_windows_s = make_windows(train_ids, scale)
                    slot = ScaleSlot(
                        substrate=substrate,
                        train_windows=train_windows_s,
                        window_size=scale,
                        landscape_size=scale_ls[scale],
                        codebook=codebook,
                        seed=seed + scale * 100,
                    )

                    for beta in betas:
                        data = eval_scale(
                            slot=slot,
                            test_windows=test_windows,
                            eval_ws_size=eval_ws_size,
                            masked_pos=masked_pos,
                            decode_ids=decode_ids,
                            mask_id=vocab.mask_id,
                            unk_id=vocab.unk_id,
                            beta=beta,
                            decode_k=args.decode_k,
                        )

                        label = f"W{eval_ws_size}_seed{seed}_{codebook_label}_scale{scale}_b{beta}"
                        summary = summarize_run(label, data)
                        summary["eval_w"] = eval_ws_size
                        summary["seed"] = seed
                        summary["codebook"] = codebook_label
                        summary["scale"] = scale
                        summary["beta"] = beta
                        all_rows.append(summary)

                        print(
                            f"  seed={seed} cb={codebook_label:>6} W={scale} β={beta:>4.0f} "
                            f"top1={summary['recall_at_1']:.3f} "
                            f"top{args.decode_k}={summary['recall_at_k']:.3f} "
                            f"top_score={summary['mean_top_score']:.3f} "
                            f"ent={summary['mean_entropy']:.3f} "
                            f"meta={summary['metastable_rate']:.3f} "
                            f"cap_t@0.3={summary['cap_cov_target_0.3']:.3f}",
                            flush=True,
                        )

    # Aggregate across seeds
    print(f"\n{'=' * 80}")
    print("Aggregated across seeds (mean ± std)")
    print(f"{'=' * 80}")

    def agg(rows: List[Dict], key: str) -> Tuple[float, float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return 0.0, 0.0
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        return m, s

    for eval_ws_size in eval_ws:
        print(f"\n  W={eval_ws_size}")
        for codebook_label in ["recon", "random"]:
            for scale in scales:
                for beta in betas:
                    rows = [
                        r for r in all_rows
                        if r["eval_w"] == eval_ws_size
                        and r["codebook"] == codebook_label
                        and r["scale"] == scale
                        and r["beta"] == beta
                    ]
                    if not rows:
                        continue
                    t1_m, t1_s = agg(rows, "recall_at_1")
                    tk_m, tk_s = agg(rows, "recall_at_k")
                    ts_m, _ = agg(rows, "mean_top_score")
                    ent_m, _ = agg(rows, "mean_entropy")
                    meta_m, _ = agg(rows, "metastable_rate")
                    cap_t03_m, _ = agg(rows, "cap_cov_target_0.3")
                    cap_t05_m, _ = agg(rows, "cap_cov_target_0.5")
                    cap_s05_m, _ = agg(rows, "cap_cov_settled_0.5")
                    print(
                        f"    [{codebook_label:>6} W={scale} β={beta:>4.0f}]  "
                        f"top1={t1_m:.3f}±{t1_s:.3f}  "
                        f"top{args.decode_k}={tk_m:.3f}±{tk_s:.3f}  "
                        f"top_score={ts_m:.3f}  "
                        f"ent={ent_m:.3f}  "
                        f"meta={meta_m:.3f}  "
                        f"cap_s@0.5={cap_s05_m:.3f}  "
                        f"cap_t@0.3={cap_t03_m:.3f}  "
                        f"cap_t@0.5={cap_t05_m:.3f}"
                    )

    # Save JSON
    out = {
        "config": vars(args),
        "rows": all_rows,
    }
    json_path = output_dir / "cap_coverage_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
