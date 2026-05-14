"""Experiment 18: Phase 4 unified — trajectory trace + Benna-Fusi consolidation.

Tests whether the unified Phase 4 architecture (trajectory trace +
engagement-resolution gate + replay + Benna-Fusi consolidation +
discovery channel) maintains contextual-completion accuracy under
codebook drift, vs. a no-replay baseline.

Hypothesis: as the codebook drifts between checkpoints (simulating
Phase 3's continuous learning), the baseline degrades because stored
patterns become stale; the Phase 4 condition recovers via the
discovery channel — replayed trajectories that resolve cleanly under
the new codebook emit candidate patterns that get consolidated.

Setup:
  - Multi-scale Hopfield memories (W=2,3,4) per condition, with the
    W=8 reconstruction codebook
  - 2000 cues streamed from validation
  - 4 checkpoints; codebook drift applied between each checkpoint
  - At each checkpoint, evaluate on a held-out test set:
    - top-1 (identity)
    - top-K (Recall@10)
    - cap_t@0.5 (target appears in top-K with score ≥ 0.5)
    - meta-stable rate
  - Phase 4 condition runs the replay cycle every K cues, with the
    discovery channel enabled (candidates get added to memory)

Drill-downs (Phase 4 only):
  - Engagement over time
  - Replay store size
  - Number of discovered patterns
  - u_k variable distributions at end
  - Pattern death count
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.diagnostics.synergy import synergy_score
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
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase34.reencoding import reencode_patterns
from energy_memory.phase4.replay_loop import (
    ReplayConfig,
    UnifiedReplayMemory,
)
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def perturb_codebook(
    codebook: torch.Tensor,
    substrate: TorchFHRR,
    magnitude: float,
) -> torch.Tensor:
    """Apply mild random perturbation, simulating Phase 3 codebook drift."""
    noise = substrate.random_vectors(codebook.shape[0])
    perturbed = codebook + magnitude * noise
    magnitudes = perturbed.abs().clamp(min=1e-8)
    return perturbed / magnitudes


def atom_pair_geometry(
    codebook: torch.Tensor,
    decode_ids: Sequence[int],
    sample_pairs: int = 100_000,
    seed: int = 11,
) -> Dict[str, float]:
    """Sampled pairwise cosine-similarity stats across a codebook.

    Surfaces the atom-collapse pathology from
    [reports/019_reconstruction_characterization.md](../reports/019_reconstruction_characterization.md):
    learned codebooks collapse toward mean pairwise similarity 0.4 while
    random codebooks sit near 0. Cheap, one-shot per codebook, useful
    as a structural-health diagnostic at every Phase 4 checkpoint.
    """
    atoms = codebook[torch.tensor(list(decode_ids), device=codebook.device)]
    n = atoms.shape[0]
    if n < 2:
        return {"n_pairs": 0, "mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    samples: List[float] = []
    target = min(sample_pairs, n * (n - 1) // 2)
    while len(samples) < target:
        batch = min(target - len(samples), 50_000)
        i = torch.randint(0, n, (batch,), generator=rng)
        j = torch.randint(0, n, (batch,), generator=rng)
        keep = i != j
        i = i[keep]; j = j[keep]
        a = atoms[i].cpu()
        b = atoms[j].cpu()
        s = (a.conj() * b).real.mean(dim=-1)
        samples.extend(s.tolist())
    samples = samples[:target]
    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "n_pairs": len(samples),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
        "max": float(t.max().item()),
    }


class ScaleSlot:
    """Holds memory + positions + codebook for one scale.

    Subclass-friendly: works with either TorchHopfieldMemory (baseline)
    or TracedHopfieldMemory (Phase 4).
    """

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
        self.codebook = codebook
        self.positions = build_position_vectors(substrate, window_size)
        actual_l = min(landscape_size, len(train_windows))
        landscape = sample_windows(train_windows, actual_l, seed=seed)

        if traced:
            self.memory = TracedHopfieldMemory(substrate, snapshot_k=8)
        else:
            self.memory = TorchHopfieldMemory(substrate)
        # Track the source window for each stored pattern so the
        # re-encode helper can refresh stale patterns against the
        # current codebook. Discovered patterns (added later via
        # the Phase 4 replay channel) have no source window — we
        # append None for those.
        self.source_windows: List[Optional[tuple[int, ...]]] = []
        for idx, w in enumerate(landscape):
            self.memory.store(
                encode_window(substrate, self.positions, codebook, w),
                label=f"w_{idx}",
            )
            self.source_windows.append(tuple(w))
        self.landscape_size = actual_l


RANK_K_VALUES = (1, 5, 10, 20, 50)


def _ranks_to_recall_at_k(ranks: Sequence[int], total: int) -> Dict[int, float]:
    """Convert a list of target ranks (-1 = miss) into Recall@K for each K."""
    out: Dict[int, float] = {}
    if total == 0:
        return {k: 0.0 for k in RANK_K_VALUES}
    for k in RANK_K_VALUES:
        hits = sum(1 for r in ranks if 0 <= r < k)
        out[k] = hits / total
    return out


def evaluate_combined(
    slots: Dict[int, ScaleSlot],
    test_windows: Sequence[tuple[int, ...]],
    eval_ws_size: int,
    masked_pos: int,
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
    rank_k: int = 50,
    atom_sample_pairs: int = 100_000,
) -> Dict:
    """Evaluate the multi-scale combined accuracy and diagnostics.

    Uses standard retrieve() (no trajectory capture during eval) to keep
    the diagnostic fast and to avoid contaminating the trajectory store
    with test queries.

    Reports the historic top1 / topk / cap_t metrics (unchanged) plus
    three diagnostics surfaced by reports 018–019:

      * per-scale ``recall_at_k`` for K in (1, 5, 10, 20, 50)
      * per-scale ``mean_settled_synergy_at_mask``
      * per-scale ``atom_pair_geometry`` (mean/std/p95/p99/max of pairwise
        atom similarity)
      * combined ``recall_at_k`` from the score-blended ranking
    """
    correct_top1 = 0
    correct_topk = 0
    correct_cap_t05 = 0
    correct_cap_t03 = 0
    total = 0
    per_scale_top_scores: Dict[int, List[float]] = {s: [] for s in slots}
    per_scale_entropies: Dict[int, List[float]] = {s: [] for s in slots}
    per_scale_ranks: Dict[int, List[int]] = {s: [] for s in slots}
    per_scale_settled_syn: Dict[int, List[float]] = {s: [] for s in slots}
    combined_ranks: List[int] = []

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
            cue_window = list(sub_window)
            cue_window[local_masked] = mask_id
            cue = encode_window(
                slot.substrate, slot.positions, slot.codebook, cue_window,
            )
            result = slot.memory.retrieve(cue, beta=beta, max_iter=12)

            per_scale_top_scores[scale].append(float(result.top_score))
            per_scale_entropies[scale].append(float(result.entropy))

            # Use rank_k (>= decode_k) so per-scale rank stats are deep enough
            # for the full RANK_K_VALUES sweep without re-running retrieval.
            decoded = decode_position(
                slot.substrate, result.state,
                slot.positions[local_masked], slot.codebook,
                decode_ids, top_k=max(decode_k, rank_k),
            )
            # Per-scale rank of the target in this scale's decoded list.
            scale_rank = -1
            for r, (tok_id, _) in enumerate(decoded):
                if tok_id == target:
                    scale_rank = r
                    break
            per_scale_ranks[scale].append(scale_rank)

            # Settled synergy at the masked slot for this scale: how well
            # does the unbind of the settled state at the masked position
            # recover the true target's codebook atom (beyond the binding/
            # role baselines)?
            settled_syn = synergy_score(
                slot.substrate,
                slot.positions[local_masked],
                slot.codebook[target],
                binding=result.state,
            ).synergy
            per_scale_settled_syn[scale].append(settled_syn)

            # Combine only the first decode_k entries (preserving existing
            # combined-score semantics for backward compatibility).
            for tok_id, score in decoded[:decode_k]:
                combined[tok_id] += score

        if not combined:
            continue
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        # Combined target rank: position of target in the score-blended ranking.
        combined_rank = -1
        for r, (tok_id, _) in enumerate(ranked):
            if tok_id == target:
                combined_rank = r
                break
        combined_ranks.append(combined_rank)

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

    per_scale_summary = {}
    for s, slot in slots.items():
        ts = per_scale_top_scores[s]
        ents = per_scale_entropies[s]
        ranks = per_scale_ranks[s]
        syns = per_scale_settled_syn[s]
        per_scale_summary[s] = {
            "mean_top_score": mean(ts) if ts else 0.0,
            "mean_entropy": mean(ents) if ents else 0.0,
            "metastable_rate": (
                sum(1 for x in ts if x < 0.95) / len(ts) if ts else 0.0
            ),
            "recall_at_k": _ranks_to_recall_at_k(ranks, len(ranks)),
            "mean_settled_synergy_at_mask": mean(syns) if syns else 0.0,
            "atom_pair_geometry": atom_pair_geometry(
                slot.codebook, decode_ids,
                sample_pairs=atom_sample_pairs,
            ),
        }

    return {
        "top1": correct_top1 / total,
        "topk": correct_topk / total,
        "cap_t_03": correct_cap_t03 / total,
        "cap_t_05": correct_cap_t05 / total,
        "recall_at_k": _ranks_to_recall_at_k(combined_ranks, total),
        "per_scale": per_scale_summary,
        "n": total,
    }


def stream_and_replay(
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
    drift_magnitude: float,
    substrate: TorchFHRR,
    phase4_units: Optional[Dict[int, UnifiedReplayMemory]] = None,
    initial_codebook: Optional[torch.Tensor] = None,
    rank_k: int = 50,
    atom_sample_pairs: int = 100_000,
    reencode_every: int = 0,
) -> Tuple[List[Dict], torch.Tensor]:
    """Stream cues through the slots, checkpoint, apply drift.

    For Phase 4: phase4_units holds the UnifiedReplayMemory per scale;
    replay cycles trigger every replay_every cues; candidates get added
    to the underlying memory and reinforced.
    """
    results: List[Dict] = []
    codebook = initial_codebook.clone() if initial_codebook is not None else None

    initial_eval = evaluate_combined(
        slots=slots, test_windows=test_windows,
        eval_ws_size=eval_ws_size, masked_pos=masked_pos,
        decode_ids=decode_ids, mask_id=mask_id, unk_id=unk_id,
        beta=beta, decode_k=decode_k,
        rank_k=rank_k, atom_sample_pairs=atom_sample_pairs,
    )
    initial_eval["cues_seen"] = 0
    initial_eval["condition"] = condition
    if phase4_units:
        for s, unit in phase4_units.items():
            initial_eval[f"phase4_stats_w{s}"] = unit.stats()
    results.append(initial_eval)
    print(
        f"  step=    0  top1={initial_eval['top1']:.3f}  "
        f"topk={initial_eval['topk']:.3f}  cap_t05={initial_eval['cap_t_05']:.3f}",
        flush=True,
    )

    cues_seen = 0
    for cue_window in cue_stream:
        if any(t == unk_id for t in cue_window):
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
                substrate, slot.positions, slot.codebook, cue_w,
            )

            if phase4_units and scale in phase4_units:
                unit = phase4_units[scale]
                unit.retrieve_and_observe(cue_vec, beta=beta, max_iter=12)
                if unit.should_replay():
                    def make_handler(scale_idx, slot_ref):
                        def handler(trace):
                            new_idx = slot_ref.memory.stored_count
                            slot_ref.memory.store(
                                trace.final_state.clone(),
                                label=f"discovered_w{scale_idx}_{new_idx}",
                            )
                            # Discovered patterns have no source window;
                            # they will be skipped by reencode_patterns.
                            slot_ref.source_windows.append(None)
                            return new_idx
                        return handler

                    unit.run_replay_cycle(
                        beta=beta, max_iter=12,
                        candidate_handler=make_handler(scale, slot),
                    )
                    unit.garbage_collect()
            else:
                slot.memory.retrieve(cue_vec, beta=beta, max_iter=12)

        cues_seen += 1

        # Periodically re-encode stored patterns through the current
        # codebook. The reencode helper skips entries without a source
        # window (e.g. patterns added via the Phase 4 discovery channel).
        if reencode_every > 0 and cues_seen % reencode_every == 0 and codebook is not None:
            for slot in slots.values():
                reencode_patterns(
                    slot.memory, slot.source_windows,
                    substrate, slot.positions, slot.codebook,
                )

        if cues_seen % checkpoint_every == 0:
            if drift_magnitude > 0 and codebook is not None:
                codebook = perturb_codebook(codebook, substrate, drift_magnitude)
                for slot in slots.values():
                    slot.codebook = codebook

            eval_result = evaluate_combined(
                slots=slots, test_windows=test_windows,
                eval_ws_size=eval_ws_size, masked_pos=masked_pos,
                decode_ids=decode_ids, mask_id=mask_id, unk_id=unk_id,
                beta=beta, decode_k=decode_k,
                rank_k=rank_k, atom_sample_pairs=atom_sample_pairs,
            )
            eval_result["cues_seen"] = cues_seen
            eval_result["condition"] = condition
            if phase4_units:
                for s, unit in phase4_units.items():
                    eval_result[f"phase4_stats_w{s}"] = unit.stats()
            results.append(eval_result)

            extra = ""
            if phase4_units and 2 in phase4_units:
                u_stats = phase4_units[2].consolidation.stats()
                store_stats = phase4_units[2].store.stats()
                extra = (
                    f"  store={store_stats['size']} "
                    f"meanU={u_stats['mean_strength']:.3f} "
                    f"n_pat={u_stats['n_patterns']} "
                    f"cands={phase4_units[2]._candidate_count}"
                )

            print(
                f"  step={cues_seen:5d}  top1={eval_result['top1']:.3f}  "
                f"topk={eval_result['topk']:.3f}  "
                f"cap_t05={eval_result['cap_t_05']:.3f}{extra}",
                flush=True,
            )

    return results, (codebook if codebook is not None else torch.empty(0))


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
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--eval-window-size", type=int, default=8)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--decode-k", type=int, default=10)
    parser.add_argument(
        "--rank-k", type=int, default=50,
        help="Top-K depth used to record target rank per scale; enables "
             "Recall@K for K in (1,5,10,20,50). >= decode_k.",
    )
    parser.add_argument(
        "--atom-sample-pairs", type=int, default=100_000,
        help="Pairs sampled when reporting atom-pair geometry per scale codebook.",
    )
    parser.add_argument("--n-cues", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--drift-magnitude", type=float, default=0.15)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--replay-every", type=int, default=50)
    parser.add_argument("--replay-batch-size", type=int, default=10)
    parser.add_argument("--store-threshold", type=float, default=0.05)
    parser.add_argument("--resolve-threshold", type=float, default=0.85)
    parser.add_argument("--store-capacity", type=int, default=500)
    parser.add_argument("--consolidation-m", type=int, default=6)
    parser.add_argument("--consolidation-alpha", type=float, default=0.25)
    parser.add_argument("--novelty-strength", type=float, default=1.0)
    parser.add_argument("--retrieval-gain", type=float, default=0.1)
    parser.add_argument(
        "--death-window", type=int, default=10000,
        help="Patterns whose strength stays below death_threshold for this "
             "many consolidation steps get pruned. Lower => stale patterns "
             "are replaced by replay-discovered ones faster.",
    )
    parser.add_argument(
        "--death-threshold", type=float, default=0.005,
        help="Strength threshold below which death_window counter advances.",
    )
    parser.add_argument(
        "--reencode-every", type=int, default=0,
        help="Re-encode stored patterns through the current codebook every "
             "N cues. 0 disables. Only landscape patterns are re-encoded; "
             "discovered patterns are skipped (no source window).",
    )
    parser.add_argument("--output-dir", default="reports/phase4_unified")
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
    initial_codebook = load_codebook(
        Path(args.codebook_path), device=str(substrate.device),
    )
    print(f"  codebook: {initial_codebook.shape}", flush=True)

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

    # Oversample heavily because most validation windows contain UNK
    # (W=8 on WikiText-2 with max_vocab=2048 has ~88% UNK-rate per window)
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
        f"  cue stream: {len(cue_stream)} valid cues "
        f"(filtered from {len(cue_pool)} candidates); "
        f"test set: {len(test_windows)}",
        flush=True,
    )

    # Save initial generator state so both conditions get identical position
    # vectors and landscape samples
    initial_generator_state = substrate.generator.get_state().clone()

    all_results = {}

    # ───────────── Condition: baseline (no replay, no consolidation) ─────────────
    print(f"\n{'=' * 70}")
    print("  Condition: baseline (no replay, no consolidation)")
    print(f"{'=' * 70}", flush=True)
    substrate.generator.set_state(initial_generator_state)

    baseline_codebook = initial_codebook.clone()
    baseline_slots: Dict[int, ScaleSlot] = {}
    for s in scales:
        train_windows_s = make_windows(train_ids, s)
        baseline_slots[s] = ScaleSlot(
            substrate=substrate,
            train_windows=train_windows_s,
            window_size=s,
            landscape_size=scale_ls[s],
            codebook=baseline_codebook,
            seed=args.seed + s * 100,
            traced=False,
        )

    baseline_results, baseline_final_cb = stream_and_replay(
        condition="baseline",
        slots=baseline_slots,
        cue_stream=cue_stream,
        eval_ws_size=args.eval_window_size,
        masked_pos=masked_pos,
        decode_ids=decode_ids,
        mask_id=vocab.mask_id,
        unk_id=vocab.unk_id,
        beta=args.beta,
        decode_k=args.decode_k,
        test_windows=test_windows,
        checkpoint_every=args.checkpoint_every,
        drift_magnitude=args.drift_magnitude,
        substrate=substrate,
        phase4_units=None,
        initial_codebook=baseline_codebook,
        rank_k=args.rank_k,
        atom_sample_pairs=args.atom_sample_pairs,
        reencode_every=args.reencode_every,
    )
    all_results["baseline"] = baseline_results

    # ───────────── Condition: phase4 (trajectory + consolidation + replay) ─────────
    print(f"\n{'=' * 70}")
    print("  Condition: phase4 (trajectory + consolidation + replay)")
    print(f"{'=' * 70}", flush=True)
    substrate.generator.set_state(initial_generator_state)

    phase4_codebook = initial_codebook.clone()
    phase4_slots: Dict[int, ScaleSlot] = {}
    phase4_units: Dict[int, UnifiedReplayMemory] = {}
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
        death_threshold=args.death_threshold,
        death_window=args.death_window,
    )
    for s in scales:
        train_windows_s = make_windows(train_ids, s)
        phase4_slots[s] = ScaleSlot(
            substrate=substrate,
            train_windows=train_windows_s,
            window_size=s,
            landscape_size=scale_ls[s],
            codebook=phase4_codebook,
            seed=args.seed + s * 100,
            traced=True,
        )
        cons_state = ConsolidationState(cons_config, device=str(substrate.device))
        phase4_units[s] = UnifiedReplayMemory(
            substrate=substrate,
            memory=phase4_slots[s].memory,
            consolidation=cons_state,
            config=replay_config,
        )
        phase4_units[s].attach_initial_patterns()

    phase4_results, phase4_final_cb = stream_and_replay(
        condition="phase4",
        slots=phase4_slots,
        cue_stream=cue_stream,
        eval_ws_size=args.eval_window_size,
        masked_pos=masked_pos,
        decode_ids=decode_ids,
        mask_id=vocab.mask_id,
        unk_id=vocab.unk_id,
        beta=args.beta,
        decode_k=args.decode_k,
        test_windows=test_windows,
        checkpoint_every=args.checkpoint_every,
        drift_magnitude=args.drift_magnitude,
        substrate=substrate,
        phase4_units=phase4_units,
        initial_codebook=phase4_codebook,
        rank_k=args.rank_k,
        atom_sample_pairs=args.atom_sample_pairs,
        reencode_every=args.reencode_every,
    )
    all_results["phase4"] = phase4_results

    # ───────────── Save results ─────────────
    json_path = output_dir / "phase4_unified_results.json"
    out = {
        "config": vars(args),
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ───────────── Summary ─────────────
    print(f"\n{'=' * 70}\n  Summary: deltas at each checkpoint\n{'=' * 70}")
    print(
        f"  {'step':>6} {'cb_top1':>10} {'p4_top1':>10} {'Δtop1':>8} "
        f"{'cb_capt5':>10} {'p4_capt5':>10} {'Δcap_t5':>10}"
    )
    for i, (b, p) in enumerate(zip(baseline_results, phase4_results)):
        if "top1" not in b or "top1" not in p:
            continue
        d_top1 = p["top1"] - b["top1"]
        d_capt5 = p["cap_t_05"] - b["cap_t_05"]
        print(
            f"  {b['cues_seen']:>6}  "
            f"{b['top1']:>9.3f}  {p['top1']:>9.3f}  {d_top1:>+7.3f}  "
            f"{b['cap_t_05']:>9.3f}  {p['cap_t_05']:>9.3f}  {d_capt5:>+9.3f}"
        )

    print(f"\n  results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
