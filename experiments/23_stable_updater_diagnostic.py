"""Experiment 23: validate the repulsion-augmented online updater.

The shipped `OnlineCodebookUpdater` (no repulsion) collapses top-1 to
chance by ~3000 cues from random init when running with the streaming
phase34 loop. The Phase 3 deep-dive prescribes repulsion at cosine
threshold 0.7 to prevent codebook collapse. This experiment compares the
two updaters head-to-head on a fast single-scale W=2 setup so we can
decide whether repulsion is sufficient before considering further
candidates (drift-coupled lr, EMA pull target, Hebbian pathway).

Three conditions, identical seeds and cue streams:

  A. baseline       — no codebook learning (control, confirms substrate)
  B. unstable       — current `OnlineCodebookUpdater` (no repulsion)
  C. stable         — `StableOnlineCodebookUpdater` (with repulsion)

Per checkpoint we log: top-1, topk, cap_t05, codebook drift, pairwise
codebook |cos| mean and p95, consolidations, repulsion-event count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

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
from energy_memory.phase34.reencoding import codebook_drift
from energy_memory.phase34.stable_online_codebook import (
    StableOnlineCodebookUpdater,
    StableOnlineCodebookUpdaterV2,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


def pairwise_stats(
    codebook: torch.Tensor, n_pairs: int = 4000, seed: int = 0,
) -> Dict[str, float]:
    n = codebook.shape[0]
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    m = i != j
    i = i[m].to(codebook.device); j = j[m].to(codebook.device)
    sims = (codebook[i].conj() * codebook[j]).real.mean(dim=-1).detach().cpu()
    return {
        "pair_sim_abs_mean": float(sims.abs().mean()),
        "pair_sim_p95": float(sims.abs().quantile(0.95)),
        "pair_sim_max": float(sims.abs().max()),
    }


def evaluate(
    substrate, memory, positions, codebook, test_windows, masked_pos,
    decode_ids, mask_id, unk_id, beta, decode_k,
):
    correct1 = correctk = correct05 = total = 0
    top_scores = []
    for w in test_windows:
        target = w[masked_pos]
        if target == unk_id:
            continue
        total += 1
        cue_w = list(w); cue_w[masked_pos] = mask_id
        cue = encode_window(substrate, positions, codebook, cue_w)
        r = memory.retrieve(cue, beta=beta, max_iter=12)
        top_scores.append(float(r.top_score))
        decoded = decode_position(
            substrate, r.state, positions[masked_pos], codebook,
            decode_ids, top_k=decode_k,
        )
        if not decoded:
            continue
        if decoded[0][0] == target:
            correct1 += 1
        for tok, sc in decoded[:decode_k]:
            if tok == target:
                correctk += 1
                if sc >= 0.5:
                    correct05 += 1
                break
    if total == 0:
        return {"n": 0}
    return {
        "top1": correct1 / total,
        "topk": correctk / total,
        "cap_t_05": correct05 / total,
        "n": total,
        "mean_retrieval_top_score": mean(top_scores) if top_scores else 0.0,
    }


def run_condition(
    *, condition, updater_kind, substrate, initial_codebook, train_windows,
    cue_stream, test_windows, decode_ids, masked_pos, mask_id, unk_id,
    landscape_size, window_size, beta, decode_k, checkpoint_every,
    lr_pull, lr_push, consolidation_k, quality_threshold,
    repulsion_threshold, repulsion_strength, seed,
) -> List[Dict]:
    codebook = initial_codebook.clone()
    positions = build_position_vectors(substrate, window_size)
    actual_l = min(landscape_size, len(train_windows))
    source_windows = list(sample_windows(train_windows, actual_l, seed=seed + 1000))
    memory = TorchHopfieldMemory(substrate)
    for idx, w in enumerate(source_windows):
        memory.store(encode_window(substrate, positions, codebook, w), label=f"w_{idx}")

    updater = None
    if updater_kind == "unstable":
        updater = OnlineCodebookUpdater(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
        )
    elif updater_kind == "stable":
        updater = StableOnlineCodebookUpdater(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
            repulsion_threshold=repulsion_threshold,
            repulsion_strength=repulsion_strength,
        )
    elif updater_kind == "stable_v2":
        updater = StableOnlineCodebookUpdaterV2(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
            repulsion_threshold=repulsion_threshold,
            repulsion_strength=repulsion_strength,
        )

    results: List[Dict] = []

    def checkpoint(cues_seen):
        ev = evaluate(
            substrate, memory, positions, codebook, test_windows,
            masked_pos, decode_ids, mask_id, unk_id, beta, decode_k,
        )
        ev["cues_seen"] = cues_seen
        ev["condition"] = condition
        ev["drift_from_initial"] = codebook_drift(initial_codebook, codebook)
        ev.update(pairwise_stats(codebook, seed=seed + 42 + cues_seen))
        if updater is not None:
            s = updater.stats()
            ev["consolidations"] = s["consolidations"]
            ev["failure_rate"] = s["failure_rate"]
            ev["repulsion_events"] = s.get("repulsion_events_total", 0)
        else:
            ev["consolidations"] = 0
            ev["failure_rate"] = 0.0
            ev["repulsion_events"] = 0
        results.append(ev)
        rep = ev["repulsion_events"]
        print(
            f"  [{condition:>12}] step={cues_seen:>4}  "
            f"top1={ev['top1']:.3f}  topk={ev['topk']:.3f}  "
            f"cap_t05={ev['cap_t_05']:.3f}  "
            f"pairμ={ev['pair_sim_abs_mean']:.4f}  "
            f"p95={ev['pair_sim_p95']:.4f}  "
            f"drift={ev['drift_from_initial']:.4f}  "
            f"cons={ev['consolidations']:>3}  rep={rep:>4}",
            flush=True,
        )

    checkpoint(0)
    cues_seen = 0
    for cue in cue_stream:
        target = cue[masked_pos]
        if target == unk_id:
            continue
        cue_w = list(cue); cue_w[masked_pos] = mask_id
        cue_vec = encode_window(substrate, positions, codebook, cue_w)
        result = memory.retrieve(cue_vec, beta=beta, max_iter=12)
        if updater is not None:
            slot_q = substrate.unbind(result.state, positions[masked_pos])
            cand_mat = codebook[
                torch.tensor(decode_ids, device=codebook.device)
            ]
            sims = substrate.similarity_matrix(slot_q, cand_mat)
            pred_local = int(sims.argmax().detach().cpu())
            pred = decode_ids[pred_local]
            if updater.observe(
                target_id=target, slot_query=slot_q, predicted_id=pred,
            ):
                updater.consolidate_if_ready()
        cues_seen += 1
        if cues_seen % checkpoint_every == 0:
            checkpoint(cues_seen)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--landscape-size", type=int, default=4096)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--n-cues", type=int, default=3500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--lr-pull", type=float, default=0.1)
    parser.add_argument("--lr-push", type=float, default=0.05)
    parser.add_argument("--consolidation-k", type=int, default=100)
    parser.add_argument("--quality-threshold", type=float, default=0.15)
    parser.add_argument("--repulsion-threshold", type=float, default=0.7)
    parser.add_argument("--repulsion-strength", type=float, default=0.1)
    parser.add_argument(
        "--conditions", default="unstable,stable_v2",
        help="comma-separated subset of: baseline,unstable,stable,stable_v2",
    )
    parser.add_argument("--output-dir", default="reports/phase34_stable")
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
    eval_windows_all = make_windows(validation_ids, args.window_size)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    masked_pos = compute_mask_positions(
        args.window_size, 1, args.mask_position,
    )[0]
    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    seeds = [int(s) for s in args.seeds.split(",")]
    conditions_arg = [c.strip() for c in args.conditions.split(",") if c.strip()]
    all_conds = {
        "baseline": {"name": "A_baseline", "kind": None},
        "unstable": {"name": "B_unstable", "kind": "unstable"},
        "stable": {"name": "C_stable", "kind": "stable"},
        "stable_v2": {"name": "D_stable_v2", "kind": "stable_v2"},
    }
    conditions = [all_conds[c] for c in conditions_arg]

    all_results: Dict[str, Dict[int, List[Dict]]] = {c["name"]: {} for c in conditions}

    for seed in seeds:
        substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)
        initial_codebook = substrate.random_vectors(len(vocab.id_to_token))
        initial_gen_state = substrate.generator.get_state().clone()
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}", flush=True)

        test_windows = sample_windows(eval_windows_all, args.test_samples, seed=seed + 9000)
        test_set = set(map(tuple, test_windows))
        cue_pool = sample_windows(
            eval_windows_all,
            min(len(eval_windows_all), args.n_cues * 20 + 5000),
            seed=seed + 7000,
        )
        cue_stream = [
            w for w in cue_pool
            if w not in test_set and not any(t == vocab.unk_id for t in w)
        ][:args.n_cues]
        print(f"  cue_stream={len(cue_stream)}  test={len(test_windows)}", flush=True)

        for cond in conditions:
            substrate.generator.set_state(initial_gen_state)
            print(f"\n  --- {cond['name']} ---", flush=True)
            results = run_condition(
                condition=cond["name"], updater_kind=cond["kind"],
                substrate=substrate, initial_codebook=initial_codebook,
                train_windows=train_windows, cue_stream=cue_stream,
                test_windows=test_windows, decode_ids=decode_ids,
                masked_pos=masked_pos, mask_id=vocab.mask_id,
                unk_id=vocab.unk_id, landscape_size=args.landscape_size,
                window_size=args.window_size, beta=args.beta,
                decode_k=args.decode_k, checkpoint_every=args.checkpoint_every,
                lr_pull=args.lr_pull, lr_push=args.lr_push,
                consolidation_k=args.consolidation_k,
                quality_threshold=args.quality_threshold,
                repulsion_threshold=args.repulsion_threshold,
                repulsion_strength=args.repulsion_strength,
                seed=seed,
            )
            all_results[cond["name"]][seed] = results

    json_path = output_dir / "stable_updater_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
