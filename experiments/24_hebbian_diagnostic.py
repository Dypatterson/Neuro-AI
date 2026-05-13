"""Experiment 24: validate the Hebbian online updater atop pretrained
codebook.

The architecturally-correct runtime update per phase-3-deep-dive.md is
Hebbian reinforcement on retrieval success, not online error-driven
contrastive updates. This experiment validates that:

  A. baseline             — pretrained codebook, no online updates
  B. hebbian              — pretrained codebook + Hebbian updater
  C. unstable_error_driven — pretrained codebook + the original
                             OnlineCodebookUpdater (error-driven). This
                             is the misapplied mode at its intended init
                             condition; we want to see whether it
                             *also* destabilizes the pretrained codebook
                             or only collapses from random init.

The pretrained codebook (phase3c_codebook_reconstruction.pt) achieves
top-1 ≈ 0.121 at W=8 with the multi-scale architecture. For this fast
single-scale W=2 test we use a smaller landscape and report top-1 at
checkpoints against a fixed held-out test set.

Success criterion for Hebbian: top-1 stays at or near the pretrained
codebook's level over the run; the update doesn't damage what's
already there. Bonus if top-1 improves slightly.
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
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase34.hebbian_online import (
    HebbianOnlineCodebookUpdater,
)
from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.phase34.reencoding import codebook_drift
from energy_memory.substrate.torch_fhrr import TorchFHRR


def pairwise_stats(codebook, n_pairs=4000, seed=0):
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
    *, condition, mode, substrate, initial_codebook, train_windows,
    cue_stream, test_windows, decode_ids, masked_pos, mask_id, unk_id,
    landscape_size, window_size, beta, decode_k, checkpoint_every,
    lr_pull, lr_push, lr_hebbian, consolidation_k, quality_threshold,
    success_threshold, seed,
):
    codebook = initial_codebook.clone()
    positions = build_position_vectors(substrate, window_size)
    actual_l = min(landscape_size, len(train_windows))
    source_windows = list(sample_windows(train_windows, actual_l, seed=seed + 1000))
    memory = TorchHopfieldMemory(substrate)
    for idx, w in enumerate(source_windows):
        memory.store(encode_window(substrate, positions, codebook, w), label=f"w_{idx}")

    hebbian = None
    error_driven = None
    if mode == "hebbian":
        hebbian = HebbianOnlineCodebookUpdater(
            substrate=substrate, codebook=codebook,
            lr_hebbian=lr_hebbian,
            success_threshold=success_threshold,
            trivial_skip_threshold=0.95,
        )
    elif mode == "error_driven":
        error_driven = OnlineCodebookUpdater(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
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
        if hebbian is not None:
            s = hebbian.stats()
            ev["hebbian_successes"] = s["successes"]
            ev["hebbian_below_threshold"] = s["below_threshold"]
            ev["hebbian_trivial"] = s["trivial_skips"]
            ev["hebbian_success_rate"] = s["success_rate"]
            ev["atoms_updated"] = s["atoms_updated"]
        elif error_driven is not None:
            s = error_driven.stats()
            ev["error_consolidations"] = s["consolidations"]
            ev["error_failure_rate"] = s["failure_rate"]
        results.append(ev)
        extra = ""
        if hebbian is not None:
            extra = (
                f"  succ={ev['hebbian_successes']:>4}  "
                f"sub={ev['hebbian_below_threshold']:>4}  "
                f"triv={ev['hebbian_trivial']:>4}  "
                f"upd={ev['atoms_updated']:>4}"
            )
        elif error_driven is not None:
            extra = (
                f"  cons={ev['error_consolidations']:>3}  "
                f"fail={ev['error_failure_rate']:.3f}"
            )
        print(
            f"  [{condition:>22}] step={cues_seen:>4}  "
            f"top1={ev['top1']:.3f}  topk={ev['topk']:.3f}  "
            f"cap_t05={ev['cap_t_05']:.3f}  "
            f"drift={ev['drift_from_initial']:.4f}  "
            f"pairμ={ev['pair_sim_abs_mean']:.4f}"
            f"{extra}",
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

        if hebbian is not None:
            # Per deep-dive line 99, q is the cosine of the retrieved
            # completion (slot_query at the masked position) to the
            # correct masked token's codebook entry. This is a
            # training-mode signal — at runtime without ground truth,
            # a different proxy is needed (user feedback, replay
            # coherence, etc.).
            slot_q_vec = substrate.unbind(result.state, positions[masked_pos])
            q = float(substrate.similarity(slot_q_vec, codebook[target]))
            hebbian.observe(q=q, cue_token_ids=list(cue))
        elif error_driven is not None:
            slot_q = substrate.unbind(result.state, positions[masked_pos])
            cand_mat = codebook[
                torch.tensor(decode_ids, device=codebook.device)
            ]
            sims = substrate.similarity_matrix(slot_q, cand_mat)
            pred_local = int(sims.argmax().detach().cpu())
            pred = decode_ids[pred_local]
            if error_driven.observe(
                target_id=target, slot_query=slot_q, predicted_id=pred,
            ):
                error_driven.consolidate_if_ready()

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
    parser.add_argument(
        "--codebook-path",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
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
    parser.add_argument("--lr-hebbian", type=float, default=0.01)
    parser.add_argument("--consolidation-k", type=int, default=100)
    parser.add_argument("--quality-threshold", type=float, default=0.15)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument(
        "--conditions", default="baseline,hebbian,error_driven",
        help="comma-separated subset of: baseline,hebbian,error_driven",
    )
    parser.add_argument("--output-dir", default="reports/phase34_hebbian")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading corpus and pretrained codebook...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    train_windows = make_windows(train_ids, args.window_size)
    eval_windows_all = make_windows(validation_ids, args.window_size)

    masked_pos = compute_mask_positions(
        args.window_size, 1, args.mask_position,
    )[0]
    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    seeds = [int(s) for s in args.seeds.split(",")]
    cond_arg = [c.strip() for c in args.conditions.split(",") if c.strip()]
    all_conds = {
        "baseline": {"name": "A_baseline", "mode": None},
        "hebbian": {"name": "B_hebbian", "mode": "hebbian"},
        "error_driven": {"name": "C_error_driven", "mode": "error_driven"},
    }
    conditions = [all_conds[c] for c in cond_arg]

    all_results: Dict[str, Dict[int, List[Dict]]] = {c["name"]: {} for c in conditions}

    for seed in seeds:
        substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)
        codebook_path = repo_root / args.codebook_path
        initial_codebook = load_codebook(codebook_path, device=str(substrate.device))
        print(f"  loaded codebook {tuple(initial_codebook.shape)} from {codebook_path}", flush=True)
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
                condition=cond["name"], mode=cond["mode"],
                substrate=substrate, initial_codebook=initial_codebook,
                train_windows=train_windows, cue_stream=cue_stream,
                test_windows=test_windows, decode_ids=decode_ids,
                masked_pos=masked_pos, mask_id=vocab.mask_id,
                unk_id=vocab.unk_id, landscape_size=args.landscape_size,
                window_size=args.window_size, beta=args.beta,
                decode_k=args.decode_k, checkpoint_every=args.checkpoint_every,
                lr_pull=args.lr_pull, lr_push=args.lr_push,
                lr_hebbian=args.lr_hebbian,
                consolidation_k=args.consolidation_k,
                quality_threshold=args.quality_threshold,
                success_threshold=args.success_threshold,
                seed=seed,
            )
            all_results[cond["name"]][seed] = results

    json_path = output_dir / "hebbian_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
