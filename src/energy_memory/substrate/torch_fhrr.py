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

    def __init__(
        self,
        dim: int = 4096,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        alpha_anti: float = 0.0,
    ):
        if torch is None:  # pragma: no cover - exercised when torch missing
            raise ModuleNotFoundError("TorchFHRR requires torch to be installed") from _IMPORT_ERROR
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
        # Candidate B (dimensionality-preserving repulsion field).
        # alpha_anti is the strength of the substrate's H_anti = -α·log(d_eff)
        # energy term. At alpha_anti=0 the term vanishes and the substrate
        # energy is identical to the prior FHRR substrate. At alpha_anti>0
        # the substrate's effective dimensionality becomes part of the
        # substrate's energy landscape — d_eff is the integral of a
        # continuous local dynamic, not a metric some controller checks.
        # Per the load-bearing precondition (notes/notes/2026-05-20-...md
        # §"Candidate B"), alpha_anti is set ONCE at substrate construction
        # and is NOT adapted from observed d_eff trajectories during
        # training. If alpha_anti is wrong, the next retrain uses a
        # different alpha_anti; it is not a feedback loop on d_eff.
        if alpha_anti < 0.0:
            raise ValueError("alpha_anti must be non-negative")
        self.alpha_anti = float(alpha_anti)

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

    def d_eff(self, patterns) -> "torch.Tensor":
        """Effective dimensionality (participation ratio) of the centered Gram.

        Matches the operationalization in
        ``scripts/consolidation_geometry_diagnostic.py`` so the d_eff
        reported here is on the same scale as the falsification criteria
        in [report 044] and the
        [2026-05-20 diagnostic-actuator note](../../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md).

        d_eff = (Σ λ_k)² / Σ λ_k², where λ_k are eigenvalues of the
        centered Hermitian Gram. Computed via the algebraic identity
        (tr G)² / tr(G²) instead of eigvalsh — same value, differentiable
        through autograd, O(N²·D) instead of O(N³) over the eigendecomp.
        Returns NaN when fewer than 2 patterns.
        """
        n = patterns.shape[0]
        if n < 2:
            return torch.tensor(float("nan"), device=patterns.device)
        centered = patterns - patterns.mean(dim=0, keepdim=True)
        gram = centered @ centered.conj().T / n
        tr_g = gram.diagonal().real.sum()
        tr_g_sq = (gram.abs() * gram.abs()).sum()
        return (tr_g * tr_g) / tr_g_sq.clamp(min=1e-12)

    def substrate_energy_anti(self, patterns) -> "torch.Tensor":
        """H_anti = -α · log(d_eff).

        The dimensionality-preserving repulsion energy from Candidate B
        of the diagnostic-actuator design note. Returns a 0-dim scalar
        tensor (real, float32) so it can be added to any other substrate
        energy term.

        Per the load-bearing precondition: this energy is part of *the*
        substrate energy at every call site that reads substrate energy
        — settling, replay scoring, branch energy, consolidation. The
        method always returns a value (zero at alpha_anti=0 or n<2) so
        call sites can sum it unconditionally without an ``if`` guard.
        """
        if patterns.shape[0] < 2 or self.alpha_anti == 0.0:
            return torch.zeros((), device=patterns.device, dtype=torch.float32)
        deff = self.d_eff(patterns)
        return (-self.alpha_anti * torch.log(deff.clamp(min=1e-12))).to(torch.float32)

    def repulsion_force(self, patterns) -> "torch.Tensor":
        """Per-atom descent direction ``-∂H_anti/∂p̄_i``.

        Returned tensor has the same shape and complex dtype as
        ``patterns``. Caller integrates this into pattern evolution:
        ``new_p = normalize(p + lr * force)``. At alpha_anti=0 returns
        zeros with no autograd work.
        """
        if patterns.shape[0] < 2 or self.alpha_anti == 0.0:
            return torch.zeros_like(patterns)
        p = patterns.detach().clone().requires_grad_(True)
        energy = self.substrate_energy_anti(p)
        grad, = torch.autograd.grad(energy, p)
        # PyTorch returns ∂f/∂conj(z) for complex z with real f, such
        # that ``z -= lr * grad`` is descent. Force is the descent
        # direction itself.
        return -grad
