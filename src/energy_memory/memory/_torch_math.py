"""Shared Torch math utilities for memory modules."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def torch_normalized_entropy(weights) -> float:
    if weights.numel() <= 1:
        return 0.0
    return float(torch_normalized_entropy_tensor(weights).detach().cpu())


def torch_normalized_entropy_tensor(weights):
    """Same as `torch_normalized_entropy` but returns a 0-dim tensor (no sync)."""
    if weights.numel() <= 1:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    safe = weights.clamp_min(1e-12)
    raw = -(safe * safe.log()).sum()
    denom = torch.log(torch.tensor(float(weights.numel()), device=weights.device))
    return raw / denom
