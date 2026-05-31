"""Cue-space decorrelator (the production form of the validated lever).

Design: notes/emergent-codebook/phase-4-heteroassociative-write-design.md
Warrant: Report 049 §Result 4 + Report 050 Part 2 — cue-space decorrelation rescues
the correlated-key collapse and transfers to real data (the sparse-cue regime where
store-as-is fails).

Mechanism: a **batch ZCA whitening** transform ``P = (Sigma + eps*I)^{-1/2}`` learned
offline from the closed cue covariance ``Sigma = K^H K / N``, applied uniformly to all
cues. This is an **offline/batch statistic** of the cue distribution — anti-homunculus
exempt, the same class as the project's existing batch eigen-computations (``d_eff`` in
``torch_fhrr.py``; the C.3 kernel-trick ``eigvalsh``). It never reads a per-cue metric to
gate, branch, or select.

Why not the online anti-Hebbian (FEP) rule: the real cue covariance is hugely
ill-conditioned (the shared/frequent-token direction concentrates variance; condition
number ~1e5 on real text), so the single-phase anti-Hebbian iteration of
``arxiv:2505.22749`` converges impractically slowly here. The closed-form batch whitening
is the production form; an accelerated online variant is left as future work (it is the
same fixed point, not a different mechanism).
"""

from __future__ import annotations

from typing import Optional

import torch


class CueDecorrelator:
    """A batch ZCA whitening transform for cue keys, fit offline over a closed cue
    set and applied uniformly. ``fit`` then ``apply``; both are pure measurements of
    the cue distribution (no runtime arbitration)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.P: Optional[torch.Tensor] = None
        self.offdiag_before: Optional[float] = None
        self.offdiag_after: Optional[float] = None

    def fit(self, keys: torch.Tensor, *, ridge: float = 1e-5) -> "CueDecorrelator":
        """Learn ``P = Sigma^{-1/2}`` (over the spanned subspace) from the closed cue
        set ``keys`` ([N, D]). Eigen-directions below ``ridge*lmax`` are the
        rank-deficient null space (N<D leaves D-N exactly-zero eigenvalues) and are
        **projected out** (inv-sqrt set to 0) rather than amplified — the keys have no
        component there, so this is exact whitening of the cue subspace. Records the
        cue off-diagonal correlation (on the pre-renorm transform) before/after so the
        decorrelation is auditable (a measurement, not a hidden actuator)."""
        if keys.dim() != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be [N, {self.dim}], got {tuple(keys.shape)}")
        N, D = keys.shape
        sigma = (keys.conj().transpose(0, 1) @ keys) / N        # [D, D] Hermitian
        evals, evecs = torch.linalg.eigh(sigma)
        floor = ridge * evals.max().clamp_min(1e-12)
        inv_sqrt = torch.where(evals > floor, evals.clamp_min(floor) ** -0.5,
                               torch.zeros_like(evals))         # project out the null space
        self.P = (evecs * inv_sqrt.to(evecs.dtype)) @ evecs.conj().transpose(0, 1)
        self.offdiag_before = self._offdiag(sigma)
        self.offdiag_after = self._offdiag(self._cov(keys @ self.P))
        return self

    def apply(self, keys: torch.Tensor) -> torch.Tensor:
        """Decorrelate cue keys with the learned transform, renormalized by L2
        (vector) magnitude.

        L2-renorm, NOT element-wise unit-magnitude (the usual FHRR convention): the
        whitened key lives in the rank-(<=N) signal subspace, so element-wise renorm
        would fill the (D-rank)-dim null space with unit-magnitude noise — catastrophic
        when N<<D (~93% noise at D=4096), the cause of the Report 053 graduation
        collapse (D=4096 obs=1: element-wise 0.086 ≈ floor vs L2 0.549 ≈ ceiling). L2
        preserves the subspace direction; the downstream heteroassociative matmuls do
        not require per-element unit magnitude."""
        if self.P is None:
            raise RuntimeError("fit() the decorrelator before apply()")
        out = keys @ self.P   # P is Hermitian (Sigma^{-1/2}); cov(keys @ P) = I
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _cov(keys: torch.Tensor) -> torch.Tensor:
        return (keys.conj().transpose(0, 1) @ keys) / keys.shape[0]

    @staticmethod
    def _offdiag(cov: torch.Tensor) -> float:
        d = torch.diag(torch.diagonal(cov))
        return float((cov - d).abs().mean())


__all__ = ["CueDecorrelator"]
