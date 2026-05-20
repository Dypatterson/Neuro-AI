"""Experiment 41: MHN memory-cliff diagnostic (Phase 5).

Question
--------
Does Modern Hopfield Network recall break catastrophically as the stored
pattern count n_atoms shrinks on this project's substrate (dim=4096, FHRR
complex phasors, beta=10)? Specifically: does the W=4 post-death substrate
(n_atoms in {6, 10, 12}) sit below a fundamental capacity cliff a la
Sharma-Chandra-Fiete 2022 "MESH", such that no death-mechanism redesign
could recover separable retrieval?

This experiment is descriptive, not prescriptive (anti-homunculus). It
bounds how much a death-mechanism redesign could possibly help.

Setup
-----
- substrate: TorchFHRR(dim=4096, device='cpu')
- memory: TorchHopfieldMemory at beta=10
- n_atoms sweep: {2, 4, 6, 12, 24, 48, 96, 256, 1024}
- noise_scale sweep: {0.1, 0.3, 0.5, 0.7, 0.9}
- 20 trials x 3 seeds per cell

A trial: store n_atoms random FHRR patterns, then for each pattern produce
a noisy query (additive complex Gaussian noise then re-normalize to unit
phasors) and retrieve. Recall = fraction of queries whose top-1 index is
the originating pattern's index.

Noise convention
----------------
The task spec says "query = substrate.normalize(pattern + noise * scale)"
with noise drawn as a real Gaussian. FHRR vectors are complex phasors, so
the additive noise must be complex; we sample a complex Gaussian (real and
imag independent N(0,1)) and scale it. substrate.normalize() restores unit
magnitude per dimension. This is the standard FHRR additive corruption.

A scale of 0 keeps the pattern; scale -> infinity destroys phase coherence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


N_ATOMS_SWEEP = [2, 4, 6, 12, 24, 48, 96, 256, 1024]
# Task-spec noise levels (0.1..0.9). On unit-magnitude complex phasors at
# D=4096, these are extremely mild: a complex N(0,1)*0.9 perturbation followed
# by renormalization shifts phases by ~atan(0.9) ~ 42 deg per dim, and the
# central limit on D=4096 dims means the post-normalization inner product
# stays well above the separation threshold for any reasonable n_atoms.
# We also probe an extended range to actually locate the cliff.
NOISE_SWEEP_SPEC = [0.1, 0.3, 0.5, 0.7, 0.9]
NOISE_SWEEP_EXT = [1.5, 3.0, 6.0, 12.0, 24.0]
NOISE_SWEEP = NOISE_SWEEP_SPEC + NOISE_SWEEP_EXT
SEEDS = [1, 11, 23]
TRIALS_PER_SEED = 20
BETA = 10.0
DIM = 4096
DEVICE = "cpu"


def add_complex_noise(
    pattern: torch.Tensor,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add complex Gaussian noise to a complex phasor vector.

    pattern: complex64 tensor of shape [D] with |pattern_d| == 1.
    scale:   noise strength (real); 0 returns the pattern unchanged.
    """
    d = pattern.shape[-1]
    real = torch.randn(d, generator=generator)
    imag = torch.randn(d, generator=generator)
    noise = torch.complex(real, imag).to(pattern.dtype).to(pattern.device)
    return pattern + scale * noise


def run_one_cell(
    n_atoms: int,
    noise_scale: float,
    seed: int,
    trials: int,
) -> Dict:
    """Return {hits, total, recall} for one (n_atoms, noise, seed) cell."""
    substrate = TorchFHRR(dim=DIM, seed=seed, device=DEVICE)
    memory: TorchHopfieldMemory[int] = TorchHopfieldMemory(substrate)

    # Store n_atoms random patterns, labelled by index.
    patterns = substrate.random_vectors(n_atoms)
    for i in range(n_atoms):
        memory.store(patterns[i], label=i)

    # Independent generator for noise so it doesn't disturb pattern RNG state.
    noise_gen = torch.Generator(device="cpu")
    noise_gen.manual_seed(seed * 1009 + 7)

    hits = 0
    total = 0
    # Trials: pick a random pattern, corrupt, retrieve.
    pick_gen = torch.Generator(device="cpu")
    pick_gen.manual_seed(seed * 31 + 3)
    for _ in range(trials):
        # Choose an index uniformly. For very small n_atoms we still want
        # every pattern probed multiple times in expectation.
        idx = int(torch.randint(0, n_atoms, (1,), generator=pick_gen).item())
        clean = patterns[idx]
        noisy = substrate.normalize(add_complex_noise(clean, noise_scale, noise_gen))
        result = memory.retrieve(noisy, beta=BETA, max_iter=10)
        if result.top_index == idx:
            hits += 1
        total += 1
    return {"hits": hits, "total": total, "recall": hits / total if total else 0.0}


def main() -> None:
    t0 = time.time()
    results: Dict[str, Dict] = {
        "config": {
            "dim": DIM,
            "beta": BETA,
            "device": DEVICE,
            "n_atoms_sweep": N_ATOMS_SWEEP,
            "noise_sweep": NOISE_SWEEP,
            "seeds": SEEDS,
            "trials_per_seed": TRIALS_PER_SEED,
        },
        "cells": {},   # key "n_atoms|noise" -> {recall_mean, recall_per_seed}
    }

    for n_atoms in N_ATOMS_SWEEP:
        for noise in NOISE_SWEEP:
            per_seed: List[float] = []
            cell_hits = 0
            cell_total = 0
            for seed in SEEDS:
                cell = run_one_cell(n_atoms, noise, seed, TRIALS_PER_SEED)
                per_seed.append(cell["recall"])
                cell_hits += cell["hits"]
                cell_total += cell["total"]
            mean = sum(per_seed) / len(per_seed)
            key = f"{n_atoms}|{noise}"
            results["cells"][key] = {
                "n_atoms": n_atoms,
                "noise_scale": noise,
                "recall_per_seed": per_seed,
                "recall_mean": mean,
                "hits": cell_hits,
                "total": cell_total,
            }
            print(
                f"n_atoms={n_atoms:>4d}  noise={noise:.1f}  "
                f"recall_mean={mean:.3f}  per_seed={['%.3f' % r for r in per_seed]}"
            )

    results["wall_time_sec"] = time.time() - t0

    out_dir = Path(__file__).resolve().parents[1] / "reports" / "phase5_memory_cliff"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}  (wall {results['wall_time_sec']:.1f}s)")


if __name__ == "__main__":
    main()
