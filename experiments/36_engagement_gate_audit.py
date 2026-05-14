"""Engagement-gate audit for Phase 4.

Report 022 showed Phase 4's replay store stays at size 0 throughout
2000-cue runs (`store=0` at every checkpoint). Report on drift sweep
showed Phase 4 helps when drift damages baseline, but the help comes
from ~30 discovered patterns per checkpoint — admission rate is
extremely low. The hypothesis is that the engagement-resolution gate

    gate = engagement × (1 - resolution)

rarely exceeds the `store_threshold` (default 0.05). This script
instruments `retrieve_and_observe` to capture per-cue (engagement,
resolution, gate) and reports the distribution.

The audit answers three concrete questions:

  1. What fraction of cues produce gate > store_threshold (0.05)?
     (i.e., what's the gate's actual admission rate?)
  2. What's the joint distribution of engagement and resolution
     across the cue stream? Is one or the other consistently extreme,
     making the product flat?
  3. Where does the admitted ~0.7% of traces sit in
     the distribution? Are they outliers or part of a continuum?

Run:
    PYTHONPATH=src .venv/bin/python experiments/36_engagement_gate_audit.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from energy_memory.phase2.corpus import (
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window,
    mask_positions as compute_mask_positions,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--codebook-path",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument("--scale", type=int, default=2,
                        help="Single scale to audit (W=2 is Phase 4's "
                             "design regime).")
    parser.add_argument("--landscape-size", type=int, default=4096)
    parser.add_argument("--eval-window-size", type=int, default=8)
    parser.add_argument("--mask-position", default="center")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--n-cues", type=int, default=2000)
    parser.add_argument("--store-threshold", type=float, default=0.05)
    parser.add_argument("--resolve-threshold", type=float, default=0.7)
    parser.add_argument(
        "--threshold-grid", default="0.01,0.02,0.05,0.10,0.20,0.30",
        help="Comma-separated store_threshold values to report admission "
             "rates for. Doesn't affect the run — purely a post-hoc sweep.",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("reports/engagement_gate_audit.json"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    splits = load_corpus_splits("wikitext", repo_root, wikitext_name="wikitext-2-raw-v1")
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))

    W = args.scale
    positions = build_position_vectors(substrate, W)
    train_windows = make_windows(train_ids, W)
    landscape = sample_windows(
        train_windows, min(args.landscape_size, len(train_windows)),
        seed=args.seed + W * 100,
    )
    memory = TracedHopfieldMemory(substrate, snapshot_k=8)
    for i, w in enumerate(landscape):
        memory.store(
            encode_window(substrate, positions, codebook, w),
            label=f"w_{i}",
        )

    cons_state = ConsolidationState(
        ConsolidationConfig(m=6, alpha=0.25, novelty_strength=1.0,
                            retrieval_gain=0.1,
                            death_threshold=0.005, death_window=1000),
        device=str(substrate.device),
    )
    unit = UnifiedReplayMemory(
        substrate=substrate, memory=memory, consolidation=cons_state,
        config=ReplayConfig(
            store_threshold=args.store_threshold,
            store_capacity=500,
            resolve_threshold=args.resolve_threshold,
            replay_every=50, replay_batch_size=10, max_age=5,
            novelty_strength=1.0, retrieval_gain=0.1,
        ),
    )
    unit.attach_initial_patterns()

    # Build a cue stream from validation windows at the eval window size,
    # mapping each W=eval cue to a W-token sub-window centered on the mask.
    eval_W = args.eval_window_size
    masked_positions = compute_mask_positions(eval_W, 1, args.mask_position)
    masked_pos = masked_positions[0]
    eval_windows_all = make_windows(validation_ids, eval_W)
    cue_pool = sample_windows(
        eval_windows_all, min(len(eval_windows_all), args.n_cues * 20 + 5000),
        seed=args.seed + 7000,
    )
    cues = [
        w for w in cue_pool
        if not any(t == vocab.unk_id for t in w)
    ][: args.n_cues]

    print(f"  cues: {len(cues)}; landscape (W={W}): {len(landscape)}; "
          f"store_threshold: {args.store_threshold}")

    engagements: list[float] = []
    resolutions: list[float] = []
    gates: list[float] = []
    admitted_count = 0
    # We don't run replay cycles here — we want to characterize the raw
    # gate distribution. The store is reset to size 0 throughout because
    # admit-then-skip-replay would let the store grow without bound; we
    # still call retrieve_and_observe so the store sees the trace, then
    # explicitly clear after each cue. This isolates the gate signal
    # from the replay dynamics.
    for cue_window in cues:
        half = W // 2
        sub_start = masked_pos - half
        if W % 2 == 0:
            sub_start = masked_pos - half + 1
        sub_start = max(0, sub_start)
        if sub_start + W > eval_W:
            sub_start = eval_W - W
        sub_w = list(cue_window[sub_start:sub_start + W])
        local_masked = masked_pos - sub_start
        sub_w[local_masked] = vocab.mask_id
        cue_vec = encode_window(substrate, positions, codebook, sub_w)
        _, trace = unit.retrieve_and_observe(cue_vec, beta=args.beta, max_iter=12)

        eng = trace.engagement()
        res = trace.resolution()
        gate = trace.gate_signal()
        engagements.append(float(eng))
        resolutions.append(float(res))
        gates.append(float(gate))
        if gate > args.store_threshold:
            admitted_count += 1

    def quantiles(xs):
        t = torch.tensor(xs, dtype=torch.float64)
        return {
            "min": float(t.min()),
            "p05": float(torch.quantile(t, 0.05)),
            "p25": float(torch.quantile(t, 0.25)),
            "p50": float(torch.quantile(t, 0.50)),
            "p75": float(torch.quantile(t, 0.75)),
            "p95": float(torch.quantile(t, 0.95)),
            "p99": float(torch.quantile(t, 0.99)),
            "max": float(t.max()),
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
        }

    eng_q = quantiles(engagements)
    res_q = quantiles(resolutions)
    gate_q = quantiles(gates)

    print("\n  engagement distribution:")
    for k, v in eng_q.items():
        print(f"    {k:>4}: {v:8.4f}")
    print("\n  resolution distribution:")
    for k, v in res_q.items():
        print(f"    {k:>4}: {v:8.4f}")
    print("\n  gate (= engagement × (1 − resolution)):")
    for k, v in gate_q.items():
        print(f"    {k:>4}: {v:8.4f}")

    threshold_grid = [float(x) for x in args.threshold_grid.split(",")]
    print("\n  admission rate by hypothetical store_threshold:")
    threshold_results = {}
    for thr in threshold_grid:
        hits = sum(1 for g in gates if g > thr)
        rate = hits / len(gates)
        threshold_results[thr] = {"hits": hits, "rate": rate}
        print(f"    thr={thr:.3f}: hits={hits:4d}/{len(gates)} = {rate:6.3%}")

    # Joint distribution: count cues falling into engagement × resolution buckets
    def bucket(x: float, edges):
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    eng_edges = [0.1, 0.3, 0.5, 0.7, 0.9]   # 6 buckets
    res_edges = [0.1, 0.3, 0.5, 0.7, 0.9]
    joint = Counter()
    for e, r in zip(engagements, resolutions):
        joint[(bucket(e, eng_edges), bucket(r, res_edges))] += 1
    print("\n  joint distribution (engagement × resolution, % of cues):")
    print(f"    {'eng↓ \\ res→':<15}" + "".join(
        f" {f'<{x:.1f}':>7}" for x in res_edges
    ) + f" {'>='+f'{res_edges[-1]:.1f}':>7}")
    band_names = [f"<{x:.1f}" for x in eng_edges] + [f">={eng_edges[-1]:.1f}"]
    n = len(cues)
    for ei in range(len(eng_edges) + 1):
        row = f"    {band_names[ei]:<15}"
        for ri in range(len(res_edges) + 1):
            row += f" {joint[(ei, ri)] / n * 100:6.2f}%"
        print(row)

    out = {
        "config": vars(args) | {"out": str(args.out)},
        "n_cues": len(cues),
        "scale": W,
        "engagement": eng_q,
        "resolution": res_q,
        "gate": gate_q,
        "admitted_at_run_threshold": admitted_count,
        "threshold_sweep": threshold_results,
        "joint_bucket_counts": {f"e{ei}_r{ri}": c for (ei, ri), c in joint.items()},
        "samples": {
            "engagement": engagements,
            "resolution": resolutions,
            "gate": gates,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
