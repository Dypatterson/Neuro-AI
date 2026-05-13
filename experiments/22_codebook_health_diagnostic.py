"""Experiment 22: codebook health diagnostic for the Phase 3+4 re-encode
regression.

Phase34_medium showed online codebook learning from random init collapse
between cues=1000 and cues=2000 (top-1 0.018→0.0, topk 0.156→0.049). Dylan's
commit message named the cause as "consolidations on slot-queries from
non-stationary retrievals oscillate the codebook into noisy territory."
This experiment localizes which component (pull, push, re-encode, or
quality_threshold) drives the collapse.

Six conditions, identical seeds and cue streams within a seed:

  A. control_no_update     — no codebook updates; pure drift baseline
  B. pull_only             — pull only (no push, no re-encode)
  C. push_only             — push only (no pull, no re-encode)
  D. pull_push             — both, no re-encode
  E. pull_push_reencode    — full current behavior (matches phase34_medium B)
  F. pull_push_reencode_q5 — full, but quality_threshold=0.5

Per checkpoint, track:
  - top-1, topk, cap_t05 on held-out windows
  - codebook health: mean off-diagonal pairwise cosine, std, max
  - codebook drift from initial
  - failure rate (cumulative)
  - mean retrieval top_score on cue stream

If pull_only collapses but push_only does not → pull is the culprit.
If push_only collapses but pull_only does not → push.
If pull_push collapses but neither alone does → contrastive interaction.
If pull_push_reencode collapses but pull_push does not → re-encode is amplifying.
If pull_push_reencode_q5 stays healthy → quality threshold is load-bearing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.phase34.reencoding import codebook_drift, reencode_patterns
from energy_memory.substrate.torch_fhrr import TorchFHRR


def pairwise_codebook_stats(
    substrate: TorchFHRR,
    codebook: torch.Tensor,
    n_pairs: int = 4000,
    seed: int = 0,
) -> Dict[str, float]:
    """Random-pair pairwise similarity stats on the codebook.

    Returns mean, std, p95, max of pairwise cosine over n_pairs random pairs.
    Collapse manifests as mean → 1 and std → 0.
    """
    n = codebook.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    mask = i != j
    i = i[mask].to(codebook.device)
    j = j[mask].to(codebook.device)
    a = codebook[i]
    b = codebook[j]
    # FHRR cosine: real-mean of conj(a) * b. Vectors are unit-modulus so
    # similarity values live in [-1, 1] with mean ≈ 0 for independent atoms.
    sims = (a.conj() * b).real.mean(dim=-1)
    sims = sims.detach().cpu()
    return {
        "pair_sim_mean": float(sims.mean()),
        "pair_sim_std": float(sims.std()),
        "pair_sim_abs_mean": float(sims.abs().mean()),
        "pair_sim_p95": float(sims.abs().quantile(0.95)),
        "pair_sim_max": float(sims.abs().max()),
        "n_pairs": int(mask.sum()),
    }


def evaluate(
    substrate: TorchFHRR,
    memory: TorchHopfieldMemory,
    positions,
    codebook: torch.Tensor,
    test_windows: Sequence[tuple[int, ...]],
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
) -> Dict[str, float]:
    correct_top1 = correct_topk = correct_cap_t05 = total = 0
    top_scores: List[float] = []
    window_size = len(positions)
    for window in test_windows:
        if len(window) != window_size:
            continue
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1
        cue_w = list(window)
        cue_w[masked_pos] = mask_id
        cue = encode_window(substrate, positions, codebook, cue_w)
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        top_scores.append(float(result.top_score))
        decoded = decode_position(
            substrate, result.state, positions[masked_pos],
            codebook, decode_ids, top_k=decode_k,
        )
        if not decoded:
            continue
        ranked = decoded
        if ranked[0][0] == target:
            correct_top1 += 1
        for tok_id, score in ranked[:decode_k]:
            if tok_id == target:
                correct_topk += 1
                if score >= 0.5:
                    correct_cap_t05 += 1
                break
    if total == 0:
        return {"n": 0}
    return {
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_05": correct_cap_t05 / total,
        "n": total,
        "mean_retrieval_top_score": (
            mean(top_scores) if top_scores else 0.0
        ),
    }


def run_condition(
    *,
    condition: str,
    substrate: TorchFHRR,
    initial_codebook: torch.Tensor,
    train_windows: Sequence[tuple[int, ...]],
    cue_stream: Sequence[tuple[int, ...]],
    test_windows: Sequence[tuple[int, ...]],
    decode_ids: List[int],
    masked_pos: int,
    mask_id: int,
    unk_id: int,
    landscape_size: int,
    window_size: int,
    beta: float,
    decode_k: int,
    checkpoint_every: int,
    apply_pull: bool,
    apply_push: bool,
    apply_reencode: bool,
    reencode_every: int,
    quality_threshold: float,
    lr_pull: float,
    lr_push: float,
    consolidation_k: int,
    seed: int,
) -> List[Dict]:
    codebook = initial_codebook.clone()
    positions = build_position_vectors(substrate, window_size)
    actual_l = min(landscape_size, len(train_windows))
    source_windows: List[Optional[tuple[int, ...]]] = list(
        sample_windows(train_windows, actual_l, seed=seed + 1000)
    )
    memory = TorchHopfieldMemory(substrate)
    for idx, w in enumerate(source_windows):
        memory.store(encode_window(substrate, positions, codebook, w), label=f"w_{idx}")

    updater: Optional[OnlineCodebookUpdater] = None
    if apply_pull or apply_push:
        updater = OnlineCodebookUpdater(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull if apply_pull else 0.0,
            lr_push=lr_push if apply_push else 0.0,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
        )

    results: List[Dict] = []

    def checkpoint(cues_seen: int, stats: Dict[str, float]) -> None:
        ev = evaluate(
            substrate=substrate, memory=memory, positions=positions,
            codebook=codebook, test_windows=test_windows,
            masked_pos=masked_pos, decode_ids=decode_ids,
            mask_id=mask_id, unk_id=unk_id,
            beta=beta, decode_k=decode_k,
        )
        ev["cues_seen"] = cues_seen
        ev["condition"] = condition
        ev["drift_from_initial"] = codebook_drift(initial_codebook, codebook)
        ev["failure_rate"] = stats.get("failure_rate", 0.0)
        ev["total_observations"] = stats.get("total_observations", 0)
        ev["total_failures"] = stats.get("total_failures", 0)
        ev["consolidations"] = stats.get("consolidations", 0)
        ev.update(pairwise_codebook_stats(
            substrate, codebook, seed=seed + 42 + cues_seen,
        ))
        results.append(ev)
        print(
            f"  [{condition:>26}] step={cues_seen:>4}  "
            f"top1={ev['top1']:.3f}  topk={ev['topk']:.3f}  "
            f"cap_t05={ev['cap_t_05']:.3f}  "
            f"pairμ={ev['pair_sim_abs_mean']:.4f}  "
            f"pair95={ev['pair_sim_p95']:.4f}  "
            f"drift={ev['drift_from_initial']:.4f}  "
            f"cons={ev['consolidations']:>3}  "
            f"fail={ev['failure_rate']:.3f}",
            flush=True,
        )

    checkpoint(cues_seen=0, stats={})

    cues_seen = 0
    cycles_since_reencode = 0
    for cue_window in cue_stream:
        target = cue_window[masked_pos]
        if target == unk_id:
            continue
        cue_w = list(cue_window)
        cue_w[masked_pos] = mask_id
        cue_vec = encode_window(substrate, positions, codebook, cue_w)
        result = memory.retrieve(cue_vec, beta=beta, max_iter=12)

        if updater is not None:
            slot_query = substrate.unbind(result.state, positions[masked_pos])
            candidate_matrix = codebook[
                torch.tensor(decode_ids, device=codebook.device)
            ]
            sims = substrate.similarity_matrix(slot_query, candidate_matrix)
            predicted_local = int(sims.argmax().detach().cpu())
            predicted_id = decode_ids[predicted_local]
            ready = updater.observe(
                target_id=target, slot_query=slot_query, predicted_id=predicted_id,
            )
            if ready:
                updater.consolidate_if_ready()

        cues_seen += 1
        cycles_since_reencode += 1

        if apply_reencode and reencode_every > 0 and cycles_since_reencode >= reencode_every:
            reencode_patterns(
                memory=memory, source_windows=source_windows,
                substrate=substrate, positions=positions, codebook=codebook,
            )
            cycles_since_reencode = 0

        if cues_seen % checkpoint_every == 0:
            stats = updater.stats() if updater else {}
            checkpoint(cues_seen=cues_seen, stats=stats)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--landscape-size", type=int, default=4096)
    parser.add_argument("--eval-window-size", type=int, default=2)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--n-cues", type=int, default=2500)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--lr-pull", type=float, default=0.1)
    parser.add_argument("--lr-push", type=float, default=0.05)
    parser.add_argument("--consolidation-k", type=int, default=100)
    parser.add_argument("--quality-threshold-low", type=float, default=0.15)
    parser.add_argument("--quality-threshold-high", type=float, default=0.5)
    parser.add_argument("--reencode-every", type=int, default=100)
    parser.add_argument("--output-dir", default="reports/phase34_diagnostic")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    train_windows = make_windows(train_ids, args.window_size)
    eval_windows_all = make_windows(validation_ids, args.eval_window_size)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)
    print(f"  train_windows W={args.window_size}: {len(train_windows)}", flush=True)
    print(f"  eval_windows W={args.eval_window_size}: {len(eval_windows_all)}", flush=True)

    masked_pos = compute_mask_positions(
        args.eval_window_size, 1, args.mask_position,
    )[0]
    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    seeds = [int(s) for s in args.seeds.split(",")]
    conditions: List[Dict] = [
        {"name": "A_control_no_update", "pull": False, "push": False, "reenc": False, "qt": args.quality_threshold_low},
        {"name": "B_pull_only", "pull": True, "push": False, "reenc": False, "qt": args.quality_threshold_low},
        {"name": "C_push_only", "pull": False, "push": True, "reenc": False, "qt": args.quality_threshold_low},
        {"name": "D_pull_push", "pull": True, "push": True, "reenc": False, "qt": args.quality_threshold_low},
        {"name": "E_pull_push_reencode", "pull": True, "push": True, "reenc": True, "qt": args.quality_threshold_low},
        {"name": "F_pull_push_reencode_q5", "pull": True, "push": True, "reenc": True, "qt": args.quality_threshold_high},
    ]

    all_results: Dict[str, Dict[int, List[Dict]]] = {c["name"]: {} for c in conditions}

    for seed in seeds:
        substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)
        initial_codebook = substrate.random_vectors(len(vocab.id_to_token))
        initial_gen_state = substrate.generator.get_state().clone()
        print(f"\n{'=' * 70}")
        print(f"  SEED {seed}: random codebook {tuple(initial_codebook.shape)}")
        print(f"{'=' * 70}", flush=True)

        test_windows = sample_windows(
            eval_windows_all, args.test_samples, seed=seed + 9000,
        )
        test_set = set(map(tuple, test_windows))
        cue_pool_size = min(len(eval_windows_all), args.n_cues * 20 + 5000)
        cue_pool = sample_windows(
            eval_windows_all, cue_pool_size, seed=seed + 7000,
        )
        cue_stream = [
            w for w in cue_pool
            if w not in test_set and not any(t == vocab.unk_id for t in w)
        ][:args.n_cues]
        print(f"  cue_stream: {len(cue_stream)}  test: {len(test_windows)}", flush=True)

        for cond in conditions:
            substrate.generator.set_state(initial_gen_state)
            print(f"\n  --- {cond['name']} ---", flush=True)
            results = run_condition(
                condition=cond["name"],
                substrate=substrate,
                initial_codebook=initial_codebook,
                train_windows=train_windows,
                cue_stream=cue_stream,
                test_windows=test_windows,
                decode_ids=decode_ids,
                masked_pos=masked_pos,
                mask_id=vocab.mask_id,
                unk_id=vocab.unk_id,
                landscape_size=args.landscape_size,
                window_size=args.window_size,
                beta=args.beta,
                decode_k=args.decode_k,
                checkpoint_every=args.checkpoint_every,
                apply_pull=cond["pull"],
                apply_push=cond["push"],
                apply_reencode=cond["reenc"],
                reencode_every=args.reencode_every,
                quality_threshold=cond["qt"],
                lr_pull=args.lr_pull,
                lr_push=args.lr_push,
                consolidation_k=args.consolidation_k,
                seed=seed,
            )
            all_results[cond["name"]][seed] = results

    json_path = output_dir / "codebook_health_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)

    print(f"\n{'=' * 70}\n  Summary: top-1 over time (mean across seeds)\n{'=' * 70}")
    print(f"  {'condition':>26}  " + "  ".join(
        f"step={s:>4}" for s in range(0, args.n_cues + 1, args.checkpoint_every)
    ))
    for cond in conditions:
        per_seed = all_results[cond["name"]]
        n_checks = min(len(per_seed[s]) for s in seeds)
        means = []
        for i in range(n_checks):
            vals = [per_seed[s][i].get("top1", float("nan")) for s in seeds]
            means.append(mean(v for v in vals if v == v))
        line = "  ".join(f"{m:>8.3f}" for m in means)
        print(f"  {cond['name']:>26}  {line}")

    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
