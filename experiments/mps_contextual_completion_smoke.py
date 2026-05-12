"""Torch/MPS smoke test for Phase 2 style contextual completion.

Run with the local venv:
    PYTHONPATH=src:. .venv/bin/python experiments/mps_contextual_completion_smoke.py
"""

from __future__ import annotations

import torch

from energy_memory.experiments.synthetic_worlds import TEMPORAL_STREAM
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


def build_position_vectors(substrate: TorchFHRR, count: int):
    base = substrate.random_vector()
    step = substrate.random_vector()
    positions = [base]
    for _ in range(1, count):
        positions.append(substrate.bind(positions[-1], step))
    return positions


def encode_window(substrate: TorchFHRR, positions, token_vectors, tokens):
    return substrate.bundle(
        substrate.bind(position, token_vectors[token]) for position, token in zip(positions, tokens)
    )


def decode_slot(substrate: TorchFHRR, state, position, labels, vectors, k: int = 5):
    slot_query = substrate.unbind(state, position)
    return substrate.top_k(slot_query, labels, vectors, k=k)


def main() -> None:
    substrate = TorchFHRR(dim=1024, seed=11)
    window_size = 5
    windows = [TEMPORAL_STREAM[i : i + window_size] for i in range(0, 8)]
    vocab = sorted({token for window in windows for token in window} | {"<MASK>"})
    labels = [token for token in vocab if token != "<MASK>"]
    token_vectors = {token: substrate.random_vector() for token in vocab}
    label_matrix = torch.stack([token_vectors[label] for label in labels], dim=0)
    positions = build_position_vectors(substrate, window_size)

    memory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(windows):
        memory.store(encode_window(substrate, positions, token_vectors, window), label=f"window_{index}")

    target_window = list(windows[3])
    masked_index = 2
    masked_window = list(target_window)
    masked_window[masked_index] = "<MASK>"
    cue = encode_window(substrate, positions, token_vectors, masked_window)
    result = memory.retrieve(cue, beta=12.0, max_iter=12)
    decoded = decode_slot(substrate, result.state, positions[masked_index], labels, label_matrix, k=5)

    print(f"device: {substrate.device}")
    print(f"mps: {substrate.is_mps}")
    print(f"retrieved pattern: {result.top_label}")
    print(f"masked position: {masked_index}")
    print(f"target token: {target_window[masked_index]}")
    print("decoded top-5: " + ", ".join(f"{label}:{score:.3f}" for label, score in decoded))
    print(f"success: {decoded[0][0] == target_window[masked_index]}")


if __name__ == "__main__":
    main()
