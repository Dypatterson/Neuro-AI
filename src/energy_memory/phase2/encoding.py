"""Sequence encoding and decoding helpers for contextual completion."""

from __future__ import annotations

from typing import List, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from energy_memory.substrate.torch_fhrr import TorchFHRR


def build_position_vectors(substrate: "TorchFHRR", count: int):
    if count <= 0:
        raise ValueError("count must be positive")
    base = substrate.random_vector()
    step = substrate.random_vector()
    positions = [base]
    for _ in range(1, count):
        positions.append(substrate.bind(positions[-1], step))
    return positions


def encode_window(substrate: "TorchFHRR", positions, codebook, token_ids: Sequence[int]):
    if len(positions) < len(token_ids):
        raise ValueError("positions must cover the token sequence")
    terms = [substrate.bind(positions[index], codebook[token_id]) for index, token_id in enumerate(token_ids)]
    return substrate.bundle(terms)


def masked_window(window: Sequence[int], masked_positions: Sequence[int], mask_id: int) -> List[int]:
    masked = list(window)
    for position in masked_positions:
        masked[position] = mask_id
    return masked


def mask_positions(window_size: int, mask_count: int, position_kind: str) -> List[int]:
    if not 1 <= mask_count <= window_size:
        raise ValueError("mask_count must be between 1 and window_size")
    if position_kind == "edge":
        start = 0
    elif position_kind == "end":
        start = window_size - mask_count
    elif position_kind == "center":
        start = ((window_size - mask_count) + 1) // 2
    else:
        raise ValueError(f"unknown mask position kind: {position_kind}")
    return list(range(start, start + mask_count))


def decode_position(
    substrate: "TorchFHRR",
    state,
    position,
    codebook,
    candidate_ids: Sequence[int],
    top_k: int = 5,
) -> List[tuple[int, float]]:
    if torch is None:  # pragma: no cover - exercised when torch missing
        raise ModuleNotFoundError("decode_position requires torch")
    if top_k <= 0:
        return []
    slot_query = substrate.unbind(state, position)
    candidate_matrix = codebook[torch.tensor(list(candidate_ids), device=codebook.device)]
    scores = substrate.similarity_matrix(slot_query, candidate_matrix)
    values, indices = torch.topk(scores, min(top_k, len(candidate_ids)))
    cpu_values = values.detach().cpu().tolist()
    cpu_indices = indices.detach().cpu().tolist()
    return [(candidate_ids[index], float(value)) for index, value in zip(cpu_indices, cpu_values)]
