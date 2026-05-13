"""Phase 3 codebook comparison on synergy AND Recall@1 (n=500).

Follow-up to [reports/017_phase3_codebook_comparison.md](../reports/017_phase3_codebook_comparison.md).
The argmax-Recall@1 metric at n_eff~50 could not distinguish three
codebooks (Hebbian, reconstruction, error_driven) that differ by mean
|diff| ~ 0.82 on the unit-magnitude phase manifold. This script
addresses two open questions in a single pass:

  (1) Does Recall@1 at n=500 (tightening Wilson CIs from +/-8pp to
      +/-1pp) resolve the codebook differences?
  (2) Does synergy -- a continuous geometric structural measurement
      from the GIB-inspired estimator in
      [reports/011_synergy_probe_phase4.md](../reports/011_synergy_probe_phase4.md) --
      separate the codebooks where Recall@1 cannot?

Per test window we record:
  * recall_correct: did argmax-unbind at the masked position produce
    the correct token?
  * raw_synergy:    synergy of the *cue*'s encoded binding (one
    measurement averaged over the cue's W slot positions).
  * settled_synergy: synergy of the *settled state* at the masked
    position only, with role=positions[mask_idx] and
    filler=codebook[true_token]. This asks "can we structurally
    recover the right answer from the settled state?" regardless of
    whether the argmax happens to land on it.

Per codebook we aggregate:
  * pooled Recall@1 + Wilson CI (across all seeds, all windows)
  * mean raw_synergy + bootstrap CI
  * mean settled_synergy + bootstrap CI
  * settled_synergy / raw_synergy ratio (structural preservation)

Headline: which metric (Recall@1 or synergy) separates the codebooks
with non-overlapping CIs at the operating envelope.

Run:
    PYTHONPATH=src .venv/bin/python experiments/32_phase3_synergy_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from energy_memory.diagnostics.synergy import synergy_score
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    decode_position,
    encode_window,
    mask_positions as mask_positions_for_condition,
    masked_window,
)
from energy_memory.phase2.metrics import wilson_interval
from energy_memory.phase2.persistence import load_codebook, load_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


CODEBOOKS = [
    ("random",         "reports/phase2_full_matrix/phase2_codebook.pt"),
    ("hebbian",        "reports/phase3c_reconstruction/phase3c_codebook_hebbian.pt"),
    ("reconstruction", "reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt"),
    ("error_driven",   "reports/phase3b_error_driven/phase3b_codebook_error_driven.pt"),
]


@dataclass
class WindowRecord:
    label: str
    seed: int
    recall_correct: int
    raw_synergy: float
    settled_synergy_at_mask: float


def _per_window_eval(
    *,
    substrate: TorchFHRR,
    memory: TorchHopfieldMemory,
    codebook: "torch.Tensor",
    positions,
    decode_ids: List[int],
    mask_idx: int,
    window: tuple,
    vocab,
    beta: float,
) -> Optional[Dict[str, float]]:
    target_token = window[mask_idx]
    if target_token == vocab.unk_id:
        return None
    cue_window = masked_window(window, [mask_idx], vocab.mask_id)
    cue = encode_window(substrate, positions, codebook, cue_window)

    # Raw synergy: cue itself, averaged over its W slot positions.
    raw_pairs = []
    for slot_i, tok_i in enumerate(cue_window):
        if tok_i in (vocab.unk_id, vocab.mask_id):
            continue
        m = synergy_score(
            substrate, positions[slot_i], codebook[tok_i], binding=cue
        )
        raw_pairs.append(m.synergy)
    raw_synergy = statistics.fmean(raw_pairs) if raw_pairs else 0.0

    # Retrieval and downstream metrics.
    result = memory.retrieve(cue, beta=beta, max_iter=12)
    decoded = decode_position(
        substrate, result.state, positions[mask_idx], codebook, decode_ids, top_k=2
    )
    prediction = decoded[0][0]
    recall_correct = int(prediction == target_token)

    # Settled synergy at the masked slot.
    settled = synergy_score(
        substrate,
        positions[mask_idx],
        codebook[target_token],
        binding=result.state,
    ).synergy

    return {
        "recall_correct": recall_correct,
        "raw_synergy": raw_synergy,
        "settled_synergy_at_mask": settled,
    }


def _evaluate(
    *,
    label: str,
    codebook_path: Path,
    seed: int,
    repo_root: Path,
    dim: int,
    device: str,
    window_size: int,
    landscape_size: int,
    beta: float,
    test_samples: int,
    mask_position: str,
    wikitext_name: str,
) -> List[WindowRecord]:
    substrate = TorchFHRR(dim=dim, seed=seed, device=device)

    vocab_files = list(codebook_path.parent.glob("*vocab*.json"))
    if not vocab_files:
        return []
    vocab = load_vocabulary(vocab_files[0])

    splits = load_corpus_splits("wikitext", repo_root, wikitext_name=wikitext_name)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)

    train_windows = make_windows(train_ids, window_size)
    validation_windows = make_windows(validation_ids, window_size)
    if not train_windows or not validation_windows:
        return []

    codebook = load_codebook(codebook_path, device=str(substrate.device))
    if codebook.shape[0] != len(vocab.id_to_token) or codebook.shape[1] != dim:
        return []

    positions = build_position_vectors(substrate, window_size)
    decode_ids = [
        index
        for index, token in enumerate(vocab.id_to_token)
        if token not in {vocab.unk_token, vocab.mask_token}
    ]

    # Phase 2 matrix slicing alignment (verified in report 017): W=8 sits
    # at window_index=1, L=64 at landscape_index=0.
    matrix_window_index = 1
    matrix_landscape_index = 0
    slice_seed = seed + 10000 * matrix_window_index + 1000 * matrix_landscape_index
    landscape_windows = sample_windows(train_windows, landscape_size, seed=slice_seed)
    generalization_windows = sample_windows(
        validation_windows,
        min(test_samples, len(validation_windows)),
        seed=seed + 10000 * matrix_window_index + 2,
    )

    memory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(landscape_windows):
        memory.store(
            encode_window(substrate, positions, codebook, window),
            label=f"window_{index}",
        )

    masked_positions = mask_positions_for_condition(
        window_size, mask_count=1, position_kind=mask_position
    )
    mask_idx = masked_positions[0]

    records: List[WindowRecord] = []
    for window in generalization_windows:
        rec = _per_window_eval(
            substrate=substrate,
            memory=memory,
            codebook=codebook,
            positions=positions,
            decode_ids=decode_ids,
            mask_idx=mask_idx,
            window=window,
            vocab=vocab,
            beta=beta,
        )
        if rec is None:
            continue
        records.append(WindowRecord(
            label=label, seed=seed, **rec,
        ))
    return records


def _bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple:
    """Percentile bootstrap 95% CI on the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = torch.Generator(device="cpu").manual_seed(seed)
    n = len(values)
    means = []
    arr = torch.tensor(values, dtype=torch.float64)
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=rng)
        means.append(float(arr[idx].mean().item()))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/Users/dypatterson/Desktop/Neuro-AI"))
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps"])
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--mask-position", type=str, default="end")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 1, 2, 3, 4, 5])
    parser.add_argument("--wikitext-name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--out-csv", type=Path, default=Path("reports/phase3_synergy_comparison.csv"))
    args = parser.parse_args()

    all_records: List[WindowRecord] = []
    for label, rel_path in CODEBOOKS:
        codebook_path = args.repo_root / rel_path
        if not codebook_path.exists():
            print(f"[skip] {label}: missing {codebook_path}")
            continue
        for seed in args.seeds:
            records = _evaluate(
                label=label,
                codebook_path=codebook_path,
                seed=seed,
                repo_root=args.repo_root,
                dim=args.dim,
                device=args.device,
                window_size=args.window_size,
                landscape_size=args.landscape_size,
                beta=args.beta,
                test_samples=args.test_samples,
                mask_position=args.mask_position,
                wikitext_name=args.wikitext_name,
            )
            if not records:
                print(f"[skip] {label} seed={seed}: no eligible records")
                continue
            all_records.extend(records)
            mean_raw = statistics.fmean(r.raw_synergy for r in records)
            mean_set = statistics.fmean(r.settled_synergy_at_mask for r in records)
            acc = sum(r.recall_correct for r in records) / len(records)
            print(
                f"{label:<18} seed={seed:>3}  n={len(records):>4}  "
                f"acc={acc:.3f}  raw_syn={mean_raw:.3f}  settled_syn={mean_set:.3f}"
            )

    # Per-label aggregates.
    print()
    print(
        f"{'label':<18} {'n':>6} "
        f"{'recall':>8} {'recall_CI':>20} "
        f"{'raw_syn':>9} {'raw_syn_CI':>20} "
        f"{'set_syn':>9} {'set_syn_CI':>20}"
    )
    print("-" * 120)
    aggregates: Dict[str, Dict] = {}
    for label, _ in CODEBOOKS:
        group = [r for r in all_records if r.label == label]
        if not group:
            continue
        n = len(group)
        successes = sum(r.recall_correct for r in group)
        recall = successes / n
        rec_lo, rec_hi = wilson_interval(successes, n)
        raw_vals = [r.raw_synergy for r in group]
        set_vals = [r.settled_synergy_at_mask for r in group]
        raw_mean = statistics.fmean(raw_vals)
        set_mean = statistics.fmean(set_vals)
        raw_lo, raw_hi = _bootstrap_ci(raw_vals)
        set_lo, set_hi = _bootstrap_ci(set_vals)
        aggregates[label] = {
            "n": n,
            "recall": recall,
            "recall_ci": [rec_lo, rec_hi],
            "raw_synergy_mean": raw_mean,
            "raw_synergy_ci": [raw_lo, raw_hi],
            "settled_synergy_mean": set_mean,
            "settled_synergy_ci": [set_lo, set_hi],
        }
        print(
            f"{label:<18} {n:>6d} "
            f"{recall:>8.3f} [{rec_lo:>6.3f},{rec_hi:>6.3f}]  "
            f"{raw_mean:>9.3f} [{raw_lo:>6.3f},{raw_hi:>6.3f}]  "
            f"{set_mean:>9.3f} [{set_lo:>6.3f},{set_hi:>6.3f}]"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "seed", "recall_correct", "raw_synergy", "settled_synergy_at_mask"])
        for r in all_records:
            w.writerow([r.label, r.seed, r.recall_correct, f"{r.raw_synergy:.6f}", f"{r.settled_synergy_at_mask:.6f}"])

    with args.out_csv.with_suffix(".json").open("w") as f:
        json.dump({
            "config": vars(args) | {"repo_root": str(args.repo_root), "out_csv": str(args.out_csv)},
            "aggregates": aggregates,
        }, f, indent=2, default=str)
    print(f"\nWrote {args.out_csv}  and  {args.out_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
