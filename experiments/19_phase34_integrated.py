"""Experiment 19: Phase 3+4 integration.

Runs codebook learning (Phase 3) concurrently with Phase 4 replay +
consolidation. The hypothesis is that Phase 4's discovery channel
becomes meaningful only when the codebook is genuinely evolving —
without that, replays produce redundant patterns.

Three conditions, identical seeds and cue stream:
  A. baseline_static: frozen codebook, no replay (control)
  B. phase3_only: codebook learns online, no replay
  C. phase3_phase4: codebook learns online + Phase 4 replay +
     periodic re-encoding of stored patterns through current codebook

For each cue we:
  1. Encode the masked window
  2. Multi-scale retrieve (with trajectory capture in condition C)
  3. Decode the masked position
  4. Compute slot_query; if quality is below threshold, buffer for
     codebook update (conditions B and C)
  5. Conditions C: trace enters replay store if gate fires
  6. Periodically: consolidate codebook (B, C); run Phase 4 replay (C);
     re-encode stored patterns through current codebook (C)
  7. At checkpoints: evaluate cap_t05 + top1 on held-out test set

This is the closest test we have to the project's full Phase 3+4
intent.
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
    masked_window,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.phase34.reencoding import codebook_drift, reencode_patterns
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase4.replay_loop import (
    ReplayConfig,
    UnifiedReplayMemory,
)
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


class ScaleSlot:
    """Per-scale memory + positions + source windows."""

    def __init__(
        self,
        substrate: TorchFHRR,
        train_windows: Sequence[tuple[int, ...]],
        window_size: int,
        landscape_size: int,
        codebook: torch.Tensor,
        seed: int,
        traced: bool,
    ):
        self.substrate = substrate
        self.window_size = window_size
        self.positions = build_position_vectors(substrate, window_size)
        actual_l = min(landscape_size, len(train_windows))
        self.source_windows: List[Optional[tuple[int, ...]]] = list(
            sample_windows(train_windows, actual_l, seed=seed)
        )

        if traced:
            self.memory = TracedHopfieldMemory(substrate, snapshot_k=8)
        else:
            self.memory = TorchHopfieldMemory(substrate)
        for idx, w in enumerate(self.source_windows):
            self.memory.store(
                encode_window(substrate, self.positions, codebook, w),
                label=f"w_{idx}",
            )

        self.landscape_size = actual_l


def evaluate_combined(
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
    """Evaluate masked-token contextual completion via multi-scale combined."""
    correct_top1 = correct_topk = 0
    correct_cap_t05 = correct_cap_t03 = 0
    total = 0
    per_scale_top_scores: Dict[int, List[float]] = {s: [] for s in slots}

    for window in test_windows:
        target = window[masked_pos]
        if target == unk_id:
            continue
        total += 1
        combined: Dict[int, float] = defaultdict(float)
        for scale, slot in slots.items():
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
            cue_w = list(sub_window)
            cue_w[local_masked] = mask_id
            cue = encode_window(
                slot.substrate, slot.positions, codebook, cue_w,
            )
            result = slot.memory.retrieve(cue, beta=beta, max_iter=12)
            per_scale_top_scores[scale].append(float(result.top_score))

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
        tgt_score = None
        in_topk = False
        for tok_id, score in ranked[:decode_k]:
            if tok_id == target:
                tgt_score = score
                in_topk = True
                break
        if in_topk:
            correct_topk += 1
            if tgt_score >= 0.5:
                correct_cap_t05 += 1
            if tgt_score >= 0.3:
                correct_cap_t03 += 1

    if total == 0:
        return {"n": 0}
    return {
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_03": correct_cap_t03 / total,
        "cap_t_05": correct_cap_t05 / total,
        "n": total,
        "mean_top_score_w2": (
            mean(per_scale_top_scores[2]) if 2 in per_scale_top_scores and per_scale_top_scores[2] else 0.0
        ),
    }


def stream_phase34(
    condition: str,
    slots: Dict[int, ScaleSlot],
    cue_stream: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
    test_windows: Sequence[tuple[int, ...]],
    checkpoint_every: int,
    substrate: TorchFHRR,
    codebook_box: List[torch.Tensor],
    initial_codebook: torch.Tensor,
    updaters: Optional[Dict[int, OnlineCodebookUpdater]] = None,
    phase4_units: Optional[Dict[int, UnifiedReplayMemory]] = None,
    reencode_every: int = 0,
) -> List[Dict]:
    """Run one condition's streaming loop. Codebook lives in codebook_box[0]
    so consolidation events can update it in-place across scales."""
    results: List[Dict] = []

    initial_eval = evaluate_combined(
        slots=slots, codebook=codebook_box[0],
        test_windows=test_windows, eval_ws_size=eval_ws_size,
        masked_pos=masked_pos, decode_ids=decode_ids,
        mask_id=mask_id, unk_id=unk_id,
        beta=beta, decode_k=decode_k,
    )
    initial_eval.update({"cues_seen": 0, "condition": condition})
    results.append(initial_eval)
    print(
        f"  step=    0  top1={initial_eval['top1']:.3f}  "
        f"topk={initial_eval['topk']:.3f}  cap_t05={initial_eval['cap_t_05']:.3f}",
        flush=True,
    )

    cues_seen = 0
    cycles_since_reencode = 0
    consolidations = 0
    candidates_total = 0

    for cue_window in cue_stream:
        target = cue_window[masked_pos]
        if target == unk_id:
            continue

        for scale, slot in slots.items():
            if scale > eval_ws_size:
                continue
            half = scale // 2
            sub_start = masked_pos - half
            if scale % 2 == 0:
                sub_start = masked_pos - half + 1
            sub_start = max(0, sub_start)
            if sub_start + scale > eval_ws_size:
                sub_start = eval_ws_size - scale
            sub_window = list(cue_window[sub_start:sub_start + scale])
            local_masked = masked_pos - sub_start
            cue_w = list(sub_window)
            cue_w[local_masked] = mask_id
            cue_vec = encode_window(
                substrate, slot.positions, codebook_box[0], cue_w,
            )

            if phase4_units and scale in phase4_units:
                unit = phase4_units[scale]
                result, trace = unit.memory.retrieve_with_trace(
                    cue_vec, beta=beta, max_iter=12,
                )
                gate = trace.gate_signal()
                if gate > unit.config.store_threshold:
                    unit.store.add(trace, gate_signal=gate)
                if (
                    trace.final_top_index is not None
                    and trace.final_top_index < unit.consolidation.n_patterns
                ):
                    unit.consolidation.reinforce(
                        trace.final_top_index,
                        magnitude=unit.config.retrieval_gain,
                    )
                unit._retrieval_count += 1
            else:
                result = slot.memory.retrieve(cue_vec, beta=beta, max_iter=12)

            # Phase 3 online update (conditions B and C)
            if updaters and scale in updaters:
                slot_query = substrate.unbind(
                    result.state, slot.positions[local_masked],
                )
                # Determine predicted_id: argmax over codebook
                candidate_matrix = codebook_box[0][
                    torch.tensor(decode_ids, device=codebook_box[0].device)
                ]
                sims = substrate.similarity_matrix(slot_query, candidate_matrix)
                predicted_local = int(sims.argmax().detach().cpu())
                predicted_id = decode_ids[predicted_local]
                ready = updaters[scale].observe(
                    target_id=target,
                    slot_query=slot_query,
                    predicted_id=predicted_id,
                )
                if ready:
                    diag = updaters[scale].consolidate_if_ready()
                    if diag is not None:
                        consolidations += 1

        cues_seen += 1
        cycles_since_reencode += 1

        # Phase 4 replay cycles (condition C)
        if phase4_units and cues_seen % phase4_units[next(iter(phase4_units))].config.replay_every == 0:
            for scale, unit in phase4_units.items():
                slot = slots[scale]

                def make_handler(sc, sl):
                    def handler(trace):
                        new_idx = sl.memory.stored_count
                        sl.memory.store(
                            trace.final_state.clone(),
                            label=f"discovered_w{sc}_{new_idx}",
                        )
                        sl.source_windows.append(None)
                        return new_idx
                    return handler

                cycle = unit.run_replay_cycle(
                    beta=beta, max_iter=12,
                    candidate_handler=make_handler(scale, slot),
                )
                candidates_total += cycle.get("candidates", 0)
                unit.garbage_collect()

        # Periodic re-encoding (condition C)
        if reencode_every > 0 and cycles_since_reencode >= reencode_every:
            for scale, slot in slots.items():
                reencode_patterns(
                    memory=slot.memory,
                    source_windows=slot.source_windows,
                    substrate=substrate,
                    positions=slot.positions,
                    codebook=codebook_box[0],
                )
            cycles_since_reencode = 0

        if cues_seen % checkpoint_every == 0:
            eval_result = evaluate_combined(
                slots=slots, codebook=codebook_box[0],
                test_windows=test_windows, eval_ws_size=eval_ws_size,
                masked_pos=masked_pos, decode_ids=decode_ids,
                mask_id=mask_id, unk_id=unk_id,
                beta=beta, decode_k=decode_k,
            )
            eval_result.update({
                "cues_seen": cues_seen,
                "condition": condition,
                "consolidations": consolidations,
                "candidates_total": candidates_total,
                "codebook_drift_from_initial": codebook_drift(
                    initial_codebook, codebook_box[0],
                ),
            })
            results.append(eval_result)

            extra = ""
            if updaters and 2 in updaters:
                s = updaters[2].stats()
                extra += f"  fail={s['failure_rate']:.3f}"
            if phase4_units and 2 in phase4_units:
                store_size = len(phase4_units[2].store)
                u_mean = phase4_units[2].consolidation.stats()["mean_strength"]
                extra += f"  store={store_size}  meanU={u_mean:.3f}  cands={candidates_total}"

            print(
                f"  step={cues_seen:5d}  top1={eval_result['top1']:.3f}  "
                f"topk={eval_result['topk']:.3f}  "
                f"cap_t05={eval_result['cap_t_05']:.3f}  "
                f"drift={eval_result['codebook_drift_from_initial']:.4f}{extra}",
                flush=True,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--codebook-path",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument(
        "--random-codebook", action="store_true",
        help="Initialize codebook from random (test learning from scratch)",
    )
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--eval-window-size", type=int, default=8)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument("--n-cues", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=300)
    parser.add_argument("--test-samples", type=int, default=300)
    # Phase 3 settings
    parser.add_argument("--lr-pull", type=float, default=0.1)
    parser.add_argument("--lr-push", type=float, default=0.05)
    parser.add_argument("--consolidation-k", type=int, default=50)
    parser.add_argument("--quality-threshold", type=float, default=0.15)
    # Phase 4 settings
    parser.add_argument("--replay-every", type=int, default=50)
    parser.add_argument("--replay-batch-size", type=int, default=10)
    parser.add_argument("--store-threshold", type=float, default=0.05)
    parser.add_argument("--resolve-threshold", type=float, default=0.7)
    parser.add_argument("--store-capacity", type=int, default=500)
    parser.add_argument("--consolidation-m", type=int, default=6)
    parser.add_argument("--consolidation-alpha", type=float, default=0.25)
    parser.add_argument("--novelty-strength", type=float, default=1.0)
    parser.add_argument("--retrieval-gain", type=float, default=0.1)
    parser.add_argument("--reencode-every", type=int, default=100)
    parser.add_argument("--output-dir", default="reports/phase34_integrated")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_ls = {}
    for item in args.scale_landscape.split(","):
        s, l = item.split(":")
        scale_ls[int(s)] = int(l)
    scales = sorted(scale_ls.keys())

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    if args.random_codebook:
        initial_codebook = substrate.random_vectors(len(vocab.id_to_token))
        codebook_label = "random"
    else:
        initial_codebook = load_codebook(
            Path(args.codebook_path), device=str(substrate.device),
        )
        codebook_label = "phase3c_pretrained"
    print(f"  codebook ({codebook_label}): {initial_codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    masked_positions = compute_mask_positions(
        args.eval_window_size, 1, args.mask_position,
    )
    masked_pos = masked_positions[0]
    eval_windows_all = make_windows(validation_ids, args.eval_window_size)
    test_windows = sample_windows(
        eval_windows_all, args.test_samples,
        seed=args.seed + 9000,
    )
    test_set = set(map(tuple, test_windows))

    cue_pool_size = min(len(eval_windows_all), args.n_cues * 20 + 5000)
    cue_pool = sample_windows(
        eval_windows_all, cue_pool_size,
        seed=args.seed + 7000,
    )
    cue_stream = [
        w for w in cue_pool
        if w not in test_set and not any(t == vocab.unk_id for t in w)
    ][: args.n_cues]
    print(
        f"  cue stream: {len(cue_stream)} valid cues; test set: {len(test_windows)}",
        flush=True,
    )

    initial_gen_state = substrate.generator.get_state().clone()

    all_results = {}

    def build_slots(codebook, traced):
        slots = {}
        for s in scales:
            train_windows_s = make_windows(train_ids, s)
            slots[s] = ScaleSlot(
                substrate=substrate,
                train_windows=train_windows_s,
                window_size=s,
                landscape_size=scale_ls[s],
                codebook=codebook,
                seed=args.seed + s * 100,
                traced=traced,
            )
        return slots

    # ──────────── Condition A: baseline_static ────────────
    print(f"\n{'=' * 70}")
    print("  Condition A: baseline_static (frozen codebook, no replay)")
    print(f"{'=' * 70}", flush=True)
    substrate.generator.set_state(initial_gen_state)
    cb_a = [initial_codebook.clone()]
    slots_a = build_slots(cb_a[0], traced=False)
    all_results["baseline_static"] = stream_phase34(
        condition="baseline_static",
        slots=slots_a, cue_stream=cue_stream,
        eval_ws_size=args.eval_window_size, masked_pos=masked_pos,
        decode_ids=decode_ids,
        mask_id=vocab.mask_id, unk_id=vocab.unk_id,
        beta=args.beta, decode_k=args.decode_k,
        test_windows=test_windows,
        checkpoint_every=args.checkpoint_every,
        substrate=substrate,
        codebook_box=cb_a,
        initial_codebook=initial_codebook,
        updaters=None,
        phase4_units=None,
        reencode_every=0,
    )

    # ──────────── Condition B: phase3_reencode ────────────
    # Codebook learns + stored patterns re-encoded through current codebook.
    # Re-encoding is part of Phase 4 per PROJECT_PLAN.md ("replay-and-re-encode
    # for stale patterns"); we include it here so the discovery channel — the
    # part unique to the 2026-05-02 spec — can be isolated by comparing C - B.
    print(f"\n{'=' * 70}")
    print("  Condition B: phase3_reencode (codebook learns + re-encode)")
    print(f"{'=' * 70}", flush=True)
    substrate.generator.set_state(initial_gen_state)
    cb_b = [initial_codebook.clone()]
    slots_b = build_slots(cb_b[0], traced=False)
    updaters_b = {
        s: OnlineCodebookUpdater(
            substrate=substrate, codebook=cb_b[0],
            lr_pull=args.lr_pull, lr_push=args.lr_push,
            consolidation_k=args.consolidation_k,
            quality_threshold=args.quality_threshold,
        )
        for s in scales
    }
    all_results["phase3_reencode"] = stream_phase34(
        condition="phase3_reencode",
        slots=slots_b, cue_stream=cue_stream,
        eval_ws_size=args.eval_window_size, masked_pos=masked_pos,
        decode_ids=decode_ids,
        mask_id=vocab.mask_id, unk_id=vocab.unk_id,
        beta=args.beta, decode_k=args.decode_k,
        test_windows=test_windows,
        checkpoint_every=args.checkpoint_every,
        substrate=substrate,
        codebook_box=cb_b,
        initial_codebook=initial_codebook,
        updaters=updaters_b,
        phase4_units=None,
        reencode_every=args.reencode_every,
    )

    # ──────────── Condition C: phase3_phase4 ────────────
    print(f"\n{'=' * 70}")
    print("  Condition C: phase3_phase4 (codebook learns + replay + re-encode)")
    print(f"{'=' * 70}", flush=True)
    substrate.generator.set_state(initial_gen_state)
    cb_c = [initial_codebook.clone()]
    slots_c = build_slots(cb_c[0], traced=True)

    updaters_c = {
        s: OnlineCodebookUpdater(
            substrate=substrate, codebook=cb_c[0],
            lr_pull=args.lr_pull, lr_push=args.lr_push,
            consolidation_k=args.consolidation_k,
            quality_threshold=args.quality_threshold,
        )
        for s in scales
    }

    replay_config = ReplayConfig(
        store_threshold=args.store_threshold,
        store_capacity=args.store_capacity,
        resolve_threshold=args.resolve_threshold,
        replay_every=args.replay_every,
        replay_batch_size=args.replay_batch_size,
        max_age=5,
        novelty_strength=args.novelty_strength,
        retrieval_gain=args.retrieval_gain,
    )
    cons_config = ConsolidationConfig(
        m=args.consolidation_m,
        alpha=args.consolidation_alpha,
        novelty_strength=args.novelty_strength,
        retrieval_gain=args.retrieval_gain,
        death_threshold=0.005,
        death_window=10000,
    )
    phase4_units_c = {}
    for s in scales:
        cons_state = ConsolidationState(cons_config, device=str(substrate.device))
        phase4_units_c[s] = UnifiedReplayMemory(
            substrate=substrate,
            memory=slots_c[s].memory,
            consolidation=cons_state,
            config=replay_config,
        )
        phase4_units_c[s].attach_initial_patterns()

    all_results["phase3_phase4"] = stream_phase34(
        condition="phase3_phase4",
        slots=slots_c, cue_stream=cue_stream,
        eval_ws_size=args.eval_window_size, masked_pos=masked_pos,
        decode_ids=decode_ids,
        mask_id=vocab.mask_id, unk_id=vocab.unk_id,
        beta=args.beta, decode_k=args.decode_k,
        test_windows=test_windows,
        checkpoint_every=args.checkpoint_every,
        substrate=substrate,
        codebook_box=cb_c,
        initial_codebook=initial_codebook,
        updaters=updaters_c,
        phase4_units=phase4_units_c,
        reencode_every=args.reencode_every,
    )

    # ──────────── Save and summarize ────────────
    json_path = output_dir / "phase34_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f,
                  indent=2, default=str)

    print(f"\n{'=' * 70}\n  Summary across conditions\n{'=' * 70}")
    print(
        f"  {'step':>6} {'A_top1':>9} {'B_top1':>9} {'C_top1':>9} | "
        f"{'A_capt5':>9} {'B_capt5':>9} {'C_capt5':>9}"
    )
    n_checkpoints = min(len(all_results[k]) for k in all_results)
    for i in range(n_checkpoints):
        a = all_results["baseline_static"][i]
        b = all_results["phase3_reencode"][i]
        c = all_results["phase3_phase4"][i]
        if "top1" not in a:
            continue
        print(
            f"  {a['cues_seen']:>6}  "
            f"{a['top1']:>8.3f}  {b['top1']:>8.3f}  {c['top1']:>8.3f}  | "
            f"{a['cap_t_05']:>8.3f}  {b['cap_t_05']:>8.3f}  {c['cap_t_05']:>8.3f}"
        )

    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
