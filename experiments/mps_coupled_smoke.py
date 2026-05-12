"""Torch/MPS smoke test for coupled temporal settling.

Run with the local venv:
    PYTHONPATH=src:. .venv/bin/python experiments/mps_coupled_smoke.py
"""

from __future__ import annotations

from energy_memory.experiments.synthetic_worlds import DISTRACTOR_STREAM
from energy_memory.memory.torch_temporal import TorchTemporalAssociationMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def main() -> None:
    substrate = TorchFHRR(dim=512, seed=23)
    labels = list(DISTRACTOR_STREAM)
    vectors = {label: substrate.random_vector() for label in labels}
    family_base = substrate.random_vector()
    for label in ["stair", "ladder", "ramp", "step", "escalator"]:
        vectors[label] = substrate.perturb(family_base, noise=0.10)

    memory = TorchTemporalAssociationMemory(substrate, window=2)
    memory.store_sequence(labels, [vectors[label] for label in labels])
    temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])
    result = memory.coupled_recall(
        vectors["stair"],
        temporal_query,
        content_beta=100.0,
        temporal_beta=100.0,
        top_k=4,
    )
    print(f"device: {substrate.device}")
    print(f"mps: {substrate.is_mps}")
    print(f"top anchor: {result.top_label}")
    print("temporal items: " + ", ".join(f"{label}:{score:.2f}" for label, score in result.temporal_items))
    print("trace:")
    for step in result.trace:
        print(f"  iter={step.iteration} top={step.top_label} weight={step.top_weight:.3f} entropy={step.entropy:.3f}")


if __name__ == "__main__":
    main()

