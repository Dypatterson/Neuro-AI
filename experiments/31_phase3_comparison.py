"""Phase 3 codebook-growth comparison at the Phase 2 operating envelope.

PROJECT_PLAN Tier C item 4 follow-through. Per
[reports/016_phase2_audit_and_phase3_objective.md](../reports/016_phase2_audit_and_phase3_objective.md),
the Phase 3 codebook-growth objective is formally pinned to:

    masked_token : generalization : Recall@1
    at:  beta = 3.0
         landscape_size = 64
         window_size = 8
         mask_count = 1
         mask_position = end
         dataset = wikitext-2-raw-v1
         D = 4096

with the random-codebook baseline at 0.205 (from the Phase 2 full
matrix). This script evaluates each existing learned codebook at
that exact operating envelope, with multi-seed Wilson CIs, and
writes a single comparison CSV.

Eligible codebook artifacts (all 2050 x 4096 complex64):

  * phase2_full_matrix/phase2_codebook.pt              -- random baseline
  * phase3c_reconstruction/phase3c_codebook_random.pt  -- random control
  * phase3c_reconstruction/phase3c_codebook_hebbian.pt -- Hebbian
  * phase3c_reconstruction/phase3c_codebook_reconstruction.pt -- reconstruction
  * phase3b_error_driven/phase3b_codebook_error_driven.pt -- error-driven (deprecated)
  * phase3b_error_driven/phase3b_codebook_hebbian.pt   -- Hebbian (3b run)

The codebooks were trained against the same wikitext-2 vocabulary;
they share the same vocab.json next to the .pt. We load vocab from
the directory the codebook lives in.

Headline output: comparison CSV at
``reports/phase3_comparison.csv`` with one row per
(codebook_label, seed) and one summary row per codebook_label.

Run:
    PYTHONPATH=src .venv/bin/python experiments/31_phase3_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
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
    mask_positions as mask_positions_for_condition,
    masked_window,
)
from energy_memory.phase2.metrics import wilson_interval
from energy_memory.phase2.persistence import load_codebook, load_vocabulary
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class Eval:
    label: str
    codebook_path: str
    seed: int
    n_eval: int
    n_correct: int
    accuracy: float
    lower_ci: float
    upper_ci: float


CODEBOOKS = [
    ("random_baseline_matrix", "reports/phase2_full_matrix/phase2_codebook.pt"),
    ("random_phase3c",        "reports/phase3c_reconstruction/phase3c_codebook_random.pt"),
    ("hebbian_phase3c",       "reports/phase3c_reconstruction/phase3c_codebook_hebbian.pt"),
    ("reconstruction",        "reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt"),
    ("hebbian_phase3b",       "reports/phase3b_error_driven/phase3b_codebook_hebbian.pt"),
    ("error_driven",          "reports/phase3b_error_driven/phase3b_codebook_error_driven.pt"),
]


def _tensor_md5(path: Path) -> str:
    """Hash the bytes of the loaded codebook tensor (not the .pt wrapper).

    Two ``.pt`` files saved at different times produce different file md5s even
    when the underlying tensor is identical (torch.save metadata differs). The
    only honest equivalence check is on the tensor content. See report 039 for
    why this matters here.
    """
    tensor = load_codebook(path, device="cpu")
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()


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
) -> Optional[Eval]:
    """Evaluate a codebook on masked_token:generalization Recall@1."""
    substrate = TorchFHRR(dim=dim, seed=seed, device=device)

    # Load the vocab that was saved alongside this codebook. Rebuilding
    # the vocab from the dataset can produce off-by-special-token shape
    # mismatches; the saved snapshot is the ground truth.
    vocab_path = codebook_path.with_name(
        codebook_path.name.replace("codebook", "vocab").replace(".pt", ".json")
    )
    # Each codebook directory contains its sibling vocab. Find it
    # robustly by walking the directory.
    vocab_files = list(codebook_path.parent.glob("*vocab*.json"))
    if not vocab_files:
        return None
    vocab = load_vocabulary(vocab_files[0])

    splits = load_corpus_splits("wikitext", repo_root, wikitext_name=wikitext_name)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)

    train_windows = make_windows(train_ids, window_size)
    validation_windows = make_windows(validation_ids, window_size)
    if not train_windows or not validation_windows:
        return None

    # Load the codebook, project to substrate device. The codebook width
    # must match this run's D; we already filter for 4096-wide artifacts.
    codebook = load_codebook(codebook_path, device=str(substrate.device))
    if codebook.shape[1] != dim:
        return None
    # Codebook may be wider/narrower than vocab. We assume row i maps to
    # vocab id i; if codebook has fewer rows than vocab, evaluation is
    # limited to first row_count tokens. Our artifacts are all 2050x4096.
    if codebook.shape[0] != len(vocab.id_to_token):
        return None

    positions = build_position_vectors(substrate, window_size)
    decode_ids = [
        index
        for index, token in enumerate(vocab.id_to_token)
        if token not in {vocab.unk_token, vocab.mask_token}
    ]

    # Match Phase 2 matrix slicing semantics exactly. The matrix runs
    # across window_sizes=[4,8,16] and landscape_sizes=[64,256,1024],
    # using
    #   slice_seed = seed + 10000*window_index + 1000*landscape_index
    # and the generalization-window sampler uses seed + 10000*window_index + 2.
    # Our operating envelope has W=8 (window_index=1) and L=64
    # (landscape_index=0). Hardcoding those offsets so a given --seed
    # arg produces the same data slice the matrix saw.
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

    n_correct = 0
    n_eval = 0
    for window in generalization_windows:
        targets = [window[index] for index in masked_positions]
        if any(target == vocab.unk_id for target in targets):
            continue
        cue_window = masked_window(window, masked_positions, vocab.mask_id)
        cue = encode_window(substrate, positions, codebook, cue_window)
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        decoded = [
            decode_position(substrate, result.state, positions[index], codebook, decode_ids, top_k=2)
            for index in masked_positions
        ]
        predictions = [items[0][0] for items in decoded]
        if all(prediction == target for prediction, target in zip(predictions, targets)):
            n_correct += 1
        n_eval += 1

    if n_eval == 0:
        return None
    accuracy = n_correct / n_eval
    lower, upper = wilson_interval(n_correct, n_eval)
    return Eval(
        label=label,
        codebook_path=str(codebook_path),
        seed=seed,
        n_eval=n_eval,
        n_correct=n_correct,
        accuracy=accuracy,
        lower_ci=lower,
        upper_ci=upper,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/Users/dypatterson/Desktop/Neuro-AI"))
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps"])
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--mask-position", type=str, default="end")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--wikitext-name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--out-csv", type=Path, default=Path("reports/phase3_comparison.csv"))
    args = parser.parse_args()

    # Pre-pass: hash each codebook tensor (not the .pt file). Phase3b and
    # phase3c both load phase3a's random and learned codebooks and save them
    # back unchanged for per-directory self-containment, so multiple labels
    # can point at the same tensor. Without this dedup pass the comparison
    # silently double-counts those tensors as if they were independent
    # conditions (report 039 documents the historical bug).
    canonical_for_hash: Dict[str, str] = {}
    aliases: Dict[str, List[str]] = {}
    paths_to_eval: List[tuple[str, Path]] = []
    for label, rel_path in CODEBOOKS:
        codebook_path = args.repo_root / rel_path
        if not codebook_path.exists():
            print(f"[skip] {label}: missing {codebook_path}")
            continue
        h = _tensor_md5(codebook_path)
        if h in canonical_for_hash:
            canonical = canonical_for_hash[h]
            aliases.setdefault(canonical, []).append(label)
            print(f"[alias] {label} is the same tensor as {canonical} (md5={h[:8]}…); evaluated once under {canonical}")
            continue
        canonical_for_hash[h] = label
        paths_to_eval.append((label, codebook_path))

    rows: List[Eval] = []
    for label, codebook_path in paths_to_eval:
        for seed in args.seeds:
            res = _evaluate(
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
            if res is None:
                print(f"[skip] {label} seed={seed}: shape mismatch or no eligible windows")
                continue
            rows.append(res)
            print(
                f"{label:<24} seed={seed}  acc={res.accuracy:.3f}  "
                f"CI=[{res.lower_ci:.3f}, {res.upper_ci:.3f}]  n={res.n_eval}"
            )

    # Per-label aggregate over seeds: pooled Wilson CI on the combined
    # numerator/denominator. Aliases (labels deduplicated above) are surfaced
    # alongside their canonical label so the comparison output is honest
    # about how many distinct tensors back the table.
    print()
    if aliases:
        print("Aliases collapsed in this run:")
        for canonical, alias_list in aliases.items():
            print(f"  {canonical} = {', '.join(alias_list)}")
        print()
    print(f"{'label':<24} {'mean_acc':>9} {'pooled_CI':>20} {'seeds':>6}")
    print("-" * 65)
    aggregates = {}
    for label, _ in CODEBOOKS:
        group = [r for r in rows if r.label == label]
        if not group:
            continue
        total_correct = sum(r.n_correct for r in group)
        total_eval = sum(r.n_eval for r in group)
        mean_acc = statistics.fmean(r.accuracy for r in group)
        lo, hi = wilson_interval(total_correct, total_eval)
        aggregates[label] = {
            "mean_acc": mean_acc,
            "pooled_acc": total_correct / total_eval,
            "pooled_lower": lo,
            "pooled_upper": hi,
            "n_seeds": len(group),
            "total_correct": total_correct,
            "total_eval": total_eval,
            "aliases": aliases.get(label, []),
        }
        print(f"{label:<24} {mean_acc:>9.3f} [{lo:>6.3f}, {hi:>6.3f}]      {len(group):>6}")

    # Write CSV: one row per (label, seed) plus aggregate rows.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "codebook_path", "seed", "n_eval", "n_correct",
            "accuracy", "lower_ci", "upper_ci",
        ])
        for r in rows:
            w.writerow([
                r.label, r.codebook_path, r.seed, r.n_eval, r.n_correct,
                f"{r.accuracy:.6f}", f"{r.lower_ci:.6f}", f"{r.upper_ci:.6f}",
            ])
        w.writerow([])
        w.writerow(["AGGREGATE label", "n_seeds", "total_correct", "total_eval",
                    "pooled_acc", "pooled_lower", "pooled_upper"])
        for label, agg in aggregates.items():
            w.writerow([
                label, agg["n_seeds"], agg["total_correct"], agg["total_eval"],
                f"{agg['pooled_acc']:.6f}", f"{agg['pooled_lower']:.6f}", f"{agg['pooled_upper']:.6f}",
            ])

    # Also dump JSON for downstream tools.
    with args.out_csv.with_suffix(".json").open("w") as f:
        json.dump({
            "config": vars(args) | {"repo_root": str(args.repo_root), "out_csv": str(args.out_csv)},
            "rows": [r.__dict__ for r in rows],
            "aggregates": aggregates,
            "aliases": aliases,
            "unique_tensor_count": len(canonical_for_hash),
        }, f, indent=2, default=str)
    print(f"\nWrote {args.out_csv} and {args.out_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
