"""Atom-pair geometry sweep across per-scale codebooks + references.

Report 019 found that all three single-scale Phase 3 codebooks (Hebbian,
reconstruction, error-driven) collapsed to mean pairwise atom similarity
~0.41 vs. random's ~0.00. The per-scale codebooks in
`reports/phase4_per_scale/` predate that report — we don't know whether
they share the pathology. This script answers that question with the
same diagnostic used in experiment 33: sampled pairwise cosine similarity
across the decodable atoms of each codebook.

Run:
    PYTHONPATH=src .venv/bin/python experiments/34_per_scale_atom_geometry.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from energy_memory.phase2.persistence import load_codebook, load_vocabulary


def atom_pair_geometry(
    codebook: torch.Tensor,
    decode_ids: Sequence[int],
    sample_pairs: int = 200_000,
    seed: int = 11,
) -> Dict[str, float]:
    atoms = codebook[torch.tensor(list(decode_ids), device=codebook.device)]
    n = atoms.shape[0]
    if n < 2:
        return {"n_pairs": 0, "mean": 0.0, "std": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    target = min(sample_pairs, n * (n - 1) // 2)
    samples: List[float] = []
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
        "n_atoms": int(n),
        "dim": int(codebook.shape[1]),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
        "p999": float(torch.quantile(t, 0.999).item()),
        "max": float(t.max().item()),
    }


CODEBOOKS = [
    # (label, codebook path, vocab path-or-None-for-sibling-glob)
    ("random_baseline",        "reports/phase2_full_matrix/phase2_codebook.pt", None),
    ("hebbian_w8",             "reports/phase3c_reconstruction/phase3c_codebook_hebbian.pt", None),
    ("reconstruction_w8",      "reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt", None),
    ("error_driven_w8",        "reports/phase3b_error_driven/phase3b_codebook_error_driven.pt", None),
    ("per_scale_w2",           "reports/phase4_per_scale/codebook_w2.pt", "reports/phase4_per_scale/vocab.json"),
    ("per_scale_w3",           "reports/phase4_per_scale/codebook_w3.pt", "reports/phase4_per_scale/vocab.json"),
    ("per_scale_w4",           "reports/phase4_per_scale/codebook_w4.pt", "reports/phase4_per_scale/vocab.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path("/Users/dypatterson/Desktop/Neuro-AI"))
    parser.add_argument("--sample-pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/per_scale_atom_geometry.json"))
    args = parser.parse_args()

    results: Dict[str, Dict] = {}
    header = (
        f"{'label':<22} {'n_atoms':>8} {'dim':>6} "
        f"{'mean':>9} {'std':>8} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, rel_cb, rel_vocab in CODEBOOKS:
        cb_path = args.repo_root / rel_cb
        if not cb_path.exists():
            print(f"{label:<22}  [missing: {rel_cb}]")
            continue
        if rel_vocab:
            vocab_path = args.repo_root / rel_vocab
        else:
            candidates = list(cb_path.parent.glob("*vocab*.json"))
            if not candidates:
                print(f"{label:<22}  [no vocab found in {cb_path.parent}]")
                continue
            vocab_path = candidates[0]
        vocab = load_vocabulary(vocab_path)
        codebook = load_codebook(cb_path, device="cpu")
        if codebook.shape[0] != len(vocab.id_to_token):
            print(f"{label:<22}  [vocab/codebook size mismatch: "
                  f"{codebook.shape[0]} vs {len(vocab.id_to_token)}]")
            continue
        decode_ids = [
            i for i, t in enumerate(vocab.id_to_token)
            if t not in {vocab.unk_token, vocab.mask_token}
        ]
        profile = atom_pair_geometry(
            codebook, decode_ids,
            sample_pairs=args.sample_pairs, seed=args.seed,
        )
        profile["codebook_path"] = str(cb_path)
        profile["vocab_path"] = str(vocab_path)
        results[label] = profile
        print(
            f"{label:<22} {profile['n_atoms']:>8d} {profile['dim']:>6d} "
            f"{profile['mean']:>9.4f} {profile['std']:>8.4f} "
            f"{profile['p50']:>9.4f} {profile['p95']:>9.4f} "
            f"{profile['p99']:>9.4f} {profile['max']:>9.4f}"
        )

    out_path = args.repo_root / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "config": {
                "sample_pairs": args.sample_pairs,
                "seed": args.seed,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
