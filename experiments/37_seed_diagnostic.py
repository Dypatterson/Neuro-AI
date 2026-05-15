"""Experiment 37: seed-23 diagnostic.

Targeted run to characterize why seed 23 is the recurring negative outlier in
reports 026, 028, 029, 030, and the death-mechanism re-run. For each of seeds
{17, 11, 23, 1, 2}, captures:

1. Codebook atom-pair geometry (cosine similarity stats across a 100k pair
   sample) — same codebook file for all seeds, so this is identical.
2. Substrate-projected codebook geometry after binding into FHRR space
   (depends on substrate seed → may differ per seed).
3. Position-vector geometry per scale (pure substrate-seed dependent).
4. Test set composition: token frequency distribution at the masked position
   and whether the test set tokens land in low-coverage regions.
5. Baseline (condition A) per-cue retrieval trace stats: distribution of
   final_top_score, mean entropy, fraction of cues that land at high
   confidence vs in the meta-stable band.
6. Phase 4 candidate dynamics: gate_signal distribution across discovered
   candidates, by scale.

Output: reports/seed_diagnostic/seed_<N>_diagnostic.json per seed plus a
summary table. The shape we're looking for: does seed 23 have systematically
different geometry / dynamics that explains the Phase 4 underperformance?
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    build_vocabulary, encode_texts, load_corpus_splits, make_windows, sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors, encode_window, mask_positions as compute_mask_positions,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.phase4.trajectory import TracedHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def atom_pair_stats(codebook: torch.Tensor, n_pairs: int = 100000, seed: int = 0):
    """Sample n_pairs random atom pairs, compute cosine similarity stats."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = codebook.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    a = codebook[i].to(torch.complex64)
    b = codebook[j].to(torch.complex64)
    a = a / a.abs().clamp(min=1e-8)
    b = b / b.abs().clamp(min=1e-8)
    cos = (a.conj() * b).sum(dim=-1).real / codebook.shape[-1]
    cos = cos.cpu().float()
    return {
        'mean': float(cos.mean()),
        'std': float(cos.std()),
        'max': float(cos.max()),
        'p95': float(cos.quantile(0.95)),
        'p99': float(cos.quantile(0.99)),
        'n_pairs': int(n_pairs),
    }


def position_pair_stats(positions, n_pairs: int = 1000, seed: int = 0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = len(positions)
    if n < 2:
        return {'mean': 0.0, 'std': 0.0, 'n_pairs': 0}
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    sims = []
    for ii, jj in zip(i.tolist(), j.tolist()):
        if ii == jj: continue
        a = positions[ii].to(torch.complex64)
        b = positions[jj].to(torch.complex64)
        a = a / a.abs().clamp(min=1e-8)
        b = b / b.abs().clamp(min=1e-8)
        sims.append(float((a.conj() * b).sum().real / a.shape[-1]))
    if not sims: return {'mean': 0.0, 'std': 0.0, 'n_pairs': 0}
    return {'mean': mean(sims), 'std': pstdev(sims), 'n_pairs': len(sims)}


def run_seed(seed: int, args, repo_root: Path, output_dir: Path):
    print(f"\n{'=' * 70}\n  seed {seed}\n{'=' * 70}", flush=True)

    splits = load_corpus_splits('wikitext', repo_root, wikitext_name='wikitext-2-raw-v1')
    vocab = build_vocabulary(splits['train'], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits['train'], vocab)
    validation_ids = encode_texts(splits['validation'], vocab)

    substrate = TorchFHRR(dim=args.dim, seed=seed, device=args.device)
    codebook = load_codebook(Path(args.codebook_path), device=str(substrate.device))

    decode_ids = [i for i, t in enumerate(vocab.id_to_token)
                  if t not in {vocab.unk_token, vocab.mask_token}]

    # 1. Atom-pair geometry (codebook is the same across seeds, but this
    #    verifies and gives us baseline)
    atom_stats = atom_pair_stats(codebook, n_pairs=100000, seed=seed)
    print(f"  atom-pair cos: mean={atom_stats['mean']:.4f} std={atom_stats['std']:.4f} "
          f"p95={atom_stats['p95']:.4f}")

    # 2. Test set composition
    masked_positions = compute_mask_positions(args.eval_window_size, 1, 'center')
    masked_pos = masked_positions[0]
    eval_windows_all = make_windows(validation_ids, args.eval_window_size)
    test_windows = sample_windows(eval_windows_all, args.test_samples, seed=seed + 9000)

    target_tokens = [w[masked_pos] for w in test_windows
                     if w[masked_pos] != vocab.unk_id]
    target_freq = Counter(target_tokens)
    unique_targets = len(target_freq)
    top_targets = target_freq.most_common(10)
    target_entropy = -sum((c/len(target_tokens)) * math.log2(c/len(target_tokens))
                          for c in target_freq.values()) if target_tokens else 0.0

    print(f"  test set: n={len(target_tokens)}, unique_targets={unique_targets}, "
          f"target_entropy={target_entropy:.3f} bits")
    print(f"  top 10 targets: {[(vocab.id_to_token[t], c) for t, c in top_targets[:5]]}")

    # 3. Build positions ONCE per scale (build_position_vectors advances
    #    substrate RNG state — calling it multiple times produces different
    #    vectors and breaks cue/pattern alignment). Capture geometry stats
    #    on the same positions used downstream.
    scales = [2, 3, 4]
    scale_ls = {2: 4096, 3: 2048, 4: 1024}
    scale_diagnostics = {}
    print("  building memories...", flush=True)
    slots = {}
    for s in scales:
        positions = build_position_vectors(substrate, s)
        scale_diagnostics[s] = {
            'position_pair_stats': position_pair_stats(
                positions, n_pairs=100, seed=seed,
            ),
        }
        train_windows_s = make_windows(train_ids, s)
        actual_l = min(scale_ls[s], len(train_windows_s))
        windows = sample_windows(train_windows_s, actual_l, seed=seed + s * 100)
        mem = TracedHopfieldMemory(substrate, snapshot_k=8)
        for idx, w in enumerate(windows):
            mem.store(
                encode_window(substrate, positions, codebook, w),
                label=f"w_{idx}",
            )
        slots[s] = {'mem': mem, 'positions': positions}

    # 5. Baseline retrieval trace stats on test set (condition A, W=2 only)
    print("  running condition A trace dump on test set...", flush=True)
    s = 2  # most-discriminative scale for cap_t05
    top_scores_A = []
    entropies_A = []
    gate_signals_A = []
    settled_correct_A = 0
    for window in test_windows:
        target = window[masked_pos]
        if target == vocab.unk_id: continue
        half = s // 2
        sub_start = max(0, masked_pos - half + (1 if s % 2 == 0 else 0))
        if sub_start + s > args.eval_window_size:
            sub_start = args.eval_window_size - s
        cue_w = list(window[sub_start:sub_start + s])
        cue_w[masked_pos - sub_start] = vocab.mask_id
        cue = encode_window(substrate, slots[s]['positions'], codebook, cue_w)
        result, trace = slots[s]['mem'].retrieve_with_trace(cue, beta=args.beta, max_iter=12)
        top_scores_A.append(float(result.top_score))
        entropies_A.append(trace.engagement())
        gate_signals_A.append(trace.gate_signal())

    print(f"  cond-A W=2: top_score mean={mean(top_scores_A):.3f} std={pstdev(top_scores_A):.3f}  "
          f"engagement mean={mean(entropies_A):.3f}  gate_signal mean={mean(gate_signals_A):.4f} "
          f"max={max(gate_signals_A):.4f}")

    # 6. Phase 4 candidate dynamics: short replay simulation
    print("  running Phase 4 simulation (300 cues) to collect candidate stats...",
          flush=True)
    cue_pool = sample_windows(eval_windows_all, 6000, seed=seed + 7000)
    test_set_tuples = set(map(tuple, test_windows))
    cue_stream = [
        w for w in cue_pool
        if w not in test_set_tuples and not any(t == vocab.unk_id for t in w)
    ][:300]

    phase4_units = {}
    for sc in scales:
        cfg = ConsolidationConfig(m=6, alpha=0.25, novelty_strength=1.0,
                                  retrieval_gain=0.1, death_threshold=0.05,
                                  death_window=10)
        cons = ConsolidationState(cfg, device=str(substrate.device))
        rcfg = ReplayConfig(store_threshold=0.05, store_capacity=500,
                            resolve_threshold=0.7, replay_every=50,
                            replay_batch_size=10, max_age=5,
                            novelty_strength=1.0, retrieval_gain=0.1)
        phase4_units[sc] = UnifiedReplayMemory(
            substrate=substrate, memory=slots[sc]['mem'],
            consolidation=cons, config=rcfg,
        )
        phase4_units[sc].attach_initial_patterns()

    gate_signals_C = {s: [] for s in scales}
    store_gate_signals = []
    n_stored = 0
    for cue_window in cue_stream:
        for sc, unit in phase4_units.items():
            half = sc // 2
            sub_start = max(0, masked_pos - half + (1 if sc % 2 == 0 else 0))
            if sub_start + sc > args.eval_window_size:
                sub_start = args.eval_window_size - sc
            cue_w = list(cue_window[sub_start:sub_start + sc])
            cue_w[masked_pos - sub_start] = vocab.mask_id
            cue = encode_window(substrate, slots[sc]['positions'], codebook, cue_w)
            result, trace = unit.memory.retrieve_with_trace(
                cue, beta=args.beta, max_iter=12,
            )
            gate = trace.gate_signal()
            gate_signals_C[sc].append(gate)
            if gate > unit.config.store_threshold:
                unit.store.add(trace, gate_signal=gate)
                store_gate_signals.append(gate)
                n_stored += 1

    print(f"  Phase 4 (300 cues): {n_stored} traces stored across all scales")
    for sc in scales:
        gs = gate_signals_C[sc]
        n_above = sum(1 for g in gs if g > 0.05)
        print(f"    W={sc}: gate_signal mean={mean(gs):.4f} max={max(gs):.4f}  "
              f"n_above_thresh={n_above}/{len(gs)}")

    out = {
        'seed': seed,
        'atom_pair_geometry': atom_stats,
        'test_set': {
            'n_eval': len(target_tokens), 'unique_targets': unique_targets,
            'target_entropy_bits': target_entropy,
            'top_10_targets': [(vocab.id_to_token[t], c) for t, c in top_targets],
        },
        'scale_position_geometry': scale_diagnostics,
        'condA_w2': {
            'top_score_mean': mean(top_scores_A),
            'top_score_std': pstdev(top_scores_A),
            'top_score_p10': float(torch.tensor(top_scores_A).quantile(0.10)),
            'top_score_p50': float(torch.tensor(top_scores_A).quantile(0.50)),
            'top_score_p90': float(torch.tensor(top_scores_A).quantile(0.90)),
            'engagement_mean': mean(entropies_A),
            'gate_signal_mean': mean(gate_signals_A),
            'gate_signal_max': max(gate_signals_A),
            'frac_committed': sum(1 for s_ in top_scores_A if s_ >= 0.95)/len(top_scores_A),
            'frac_metastable': sum(1 for s_ in top_scores_A if 0.5 <= s_ < 0.95)/len(top_scores_A),
        },
        'phase4_300cues': {
            'n_stored': n_stored,
            'per_scale_gate_signal': {
                sc: {
                    'mean': mean(gs), 'max': max(gs),
                    'n_above_0p05': sum(1 for g in gs if g > 0.05),
                    'n_cues': len(gs),
                } for sc, gs in gate_signals_C.items()
            },
        },
    }
    json_path = output_dir / f'seed_{seed}_diagnostic.json'
    json_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {json_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 11, 23, 1, 2])
    p.add_argument('--dim', type=int, default=4096)
    p.add_argument('--device', default=None)
    p.add_argument('--max-vocab', type=int, default=2048)
    p.add_argument('--eval-window-size', type=int, default=8)
    p.add_argument('--test-samples', type=int, default=300)
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument(
        '--codebook-path',
        default='reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt',
    )
    p.add_argument('--output-dir', default='reports/seed_diagnostic')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args, repo_root, output_dir))

    # Summary table
    print(f"\n{'='*100}")
    print('  SEED-WISE SUMMARY')
    print('='*100)
    print(f"{'seed':>5} {'top_score_mean':>15} {'top_score_std':>14} "
          f"{'frac_metastable':>17} {'gate_sig_mean':>14} {'p4_stored':>10}")
    for r in results:
        cA = r['condA_w2']
        print(f"{r['seed']:>5} {cA['top_score_mean']:>15.4f} {cA['top_score_std']:>14.4f} "
              f"{cA['frac_metastable']:>17.3f} {cA['gate_signal_mean']:>14.5f} "
              f"{r['phase4_300cues']['n_stored']:>10d}")

    (output_dir / 'summary.json').write_text(
        json.dumps({'seeds': [r['seed'] for r in results], 'results': results},
                   indent=2, default=str)
    )
    print(f"\nwrote {output_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
