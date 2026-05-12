"""Shared Torch math utilities for memory modules."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def torch_normalized_entropy(weights) -> float:
    if weights.numel() <= 1:
        return 0.0
    safe = weights.clamp_min(1e-12)
    raw = -(safe * safe.log()).sum()
    denom = torch.log(torch.tensor(float(weights.numel()), device=weights.device))
    return float((raw / denom).detach().cpu())
