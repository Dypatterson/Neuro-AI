"""Optional Torch/MPS FHRR substrate.

This module is intentionally optional. Importing it raises a clear error if
Torch is not installed, while the pure-Python FHRR backend remains the reference
implementation.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class TorchFHRR:
    """Batched FHRR operations backed by Torch tensors."""

    def __init__(self, dim: int = 4096, seed: Optional[int] = None, device: Optional[str] = None):
        if torch is None:  # pragma: no cover - exercised when torch missing
            raise ModuleNotFoundError("TorchFHRR requires torch to be installed") from _IMPORT_ERROR
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    def random_vector(self):
        phase = torch.rand(self.dim, generator=self.generator, device="cpu") * (2.0 * math.pi)
        return torch.polar(torch.ones(self.dim, device="cpu"), phase).to(self.device)

    def random_vectors(self, count: int):
        phase = torch.rand((count, self.dim), generator=self.generator, device="cpu") * (2.0 * math.pi)
        return torch.polar(torch.ones((count, self.dim), device="cpu"), phase).to(self.device)

    def perturb(self, vector, noise: float = 0.15):
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        eps = torch.randn(self.dim, generator=self.generator, device="cpu") * noise
        eps = eps.to(self.device)
        return vector * torch.polar(torch.ones(self.dim, device=self.device), eps)

    def bind(self, left, right):
        return left * right

    def unbind(self, bound, role):
        return bound * role.conj()

    def normalize(self, vector):
        mag = vector.abs().clamp_min(1e-12)
        return vector / mag

    def permute(self, vector, shift: int):
        """Cyclic shift of vector components by ``shift`` positions.

        Used as the VSA permutation operator for directed slot encoding
        (Plate; arXiv:2512.14709 "Attention as Binding"). Preserves
        unit magnitude, composes additively
        (``permute(permute(v, a), b) == permute(v, a + b)``) and is
        exactly invertible via ``permute(v, -shift)``.

        Accepts either a [D] vector or a [..., D] batch.
        """
        return torch.roll(vector, shifts=int(shift), dims=-1)

    def bundle(self, vectors):
        matrix = torch.stack(list(vectors), dim=0)
        return self.normalize(matrix.sum(dim=0))

    def weighted_bundle(self, vectors, weights):
        matrix = torch.stack(list(vectors), dim=0)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        return self.normalize((matrix * weight_tensor[:, None]).sum(dim=0))

    def similarity(self, left, right) -> float:
        return float((left.conj() * right).real.mean().detach().cpu())

    def similarity_matrix(self, query, patterns):
        """Return similarities between one query and a [N, D] pattern matrix."""
        return (patterns.conj() * query[None, :]).real.mean(dim=1)

    def top_k(self, query, labels, vectors, k: int = 5):
        sims = self.similarity_matrix(query, vectors)
        count = min(k, len(labels))
        values, indices = torch.topk(sims, count)
        cpu_values = values.detach().cpu().tolist()
        cpu_indices = indices.detach().cpu().tolist()
        return [(labels[index], float(value)) for index, value in zip(cpu_indices, cpu_values)]
