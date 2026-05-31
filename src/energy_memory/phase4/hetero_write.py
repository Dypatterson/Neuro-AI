"""Heteroassociative consolidation write (the surgical Phase-4 mechanism).

Design: notes/emergent-codebook/phase-4-heteroassociative-write-design.md
Warrant: Reports 048/049 (the Stage-1 gates that selected it).

Replaces the store-as-is bundle / atom-codebook pull/push with an error-correcting
(delta-rule) heteroassociative role->target write, optionally with a PRECOMMITTED
swap-negative (the anti-homunculus-clean contrastive term). Read out via the
existing Modern Hopfield ``top_index_hits`` basin-membership.

Anti-homunculus (reviewer: PASS-WITH-CONDITIONS, conditions discharged here):
- The write requires a FROZEN, seed-fixed buffer (``freeze()`` before ``write``);
  it is a single batch-offline pass, never a per-observation streaming update
  (runtime error-driven updates are BANNED).
- The negative is a PRECOMMITTED seed-fixed draw from the value codebook
  (``precommit_swap_negatives``), never ``sims.argmax()`` — the module does NOT
  invoke the ``error_driven_learner`` streaming argmax path (the removed thermostat
  at ``error_driven_learner.py:106``).
- The read terminates in a ``top_index`` basin-membership count, never an energy
  scalar / min-over-branches / ΔE (the Phase-5' fence bright line).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Self-contained batched Modern Hopfield cleanup (returns the basin index).
# Same fixed iterative softmax update as the rest of the substrate; kept local
# so this module does not depend on experiments/.
# --------------------------------------------------------------------------- #
def batched_hopfield_topindex(
    substrate, patterns: torch.Tensor, queries: torch.Tensor,
    *, beta: float, max_iter: int = 12, tol: float = 1e-8,
):
    """Settle ``queries`` against ``patterns`` and return
    ``(top_index, entropy, margin)``. ``top_index`` is the basin (argmax) index —
    the basin-membership readout, NOT a top-1 cosine decode."""
    if patterns.numel() == 0:
        raise ValueError("empty pattern matrix")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    state = substrate.normalize(queries).to(patterns.device)
    d = patterns.shape[1]
    prev_energy: Optional[torch.Tensor] = None
    frozen = torch.zeros(state.shape[0], dtype=torch.bool, device=patterns.device)
    final_state = state
    for _ in range(max_iter):
        scores = (state @ patterns.conj().T).real / d
        energy = -torch.logsumexp(beta * scores, dim=1) / beta
        weights = torch.softmax(beta * scores, dim=1)
        nxt = substrate.normalize(weights.to(patterns.dtype) @ patterns)
        if prev_energy is not None:
            conv = ((energy - prev_energy).abs() < tol) & (~frozen)
            final_state = torch.where(conv[:, None], nxt, final_state)
            frozen = frozen | conv
        prev_energy = energy
        state = nxt
    final_state = torch.where(frozen[:, None], final_state, state)
    fscores = (final_state @ patterns.conj().T).real / d
    top_index = torch.argmax(fscores, dim=1)
    if fscores.shape[1] > 1:
        top2 = torch.topk(fscores, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        w = torch.softmax(beta * fscores, dim=1).clamp_min(1e-12)
        entropy = -(w * w.log()).sum(dim=1) / math.log(fscores.shape[1])
    else:
        margin = torch.zeros(fscores.shape[0], device=patterns.device)
        entropy = torch.zeros(fscores.shape[0], device=patterns.device)
    return top_index, entropy, margin


def precommit_swap_negatives(
    value_idx: torch.Tensor, n_values: int, seed: int
) -> torch.Tensor:
    """A PRECOMMITTED seed-fixed negative value index per observation: a uniform
    draw from the value codebook distinct from the true value. This is a fixed
    offline data property — NOT a runtime ``sims.argmax()`` selection (the removed
    thermostat)."""
    if n_values < 2:
        raise ValueError("need >=2 value atoms to draw a distinct negative")
    g = torch.Generator().manual_seed(seed * 104729 + 11)
    neg = value_idx.clone()
    for i in range(len(value_idx)):
        while True:
            j = int(torch.randint(0, n_values, (1,), generator=g))
            if j != int(value_idx[i]):
                neg[i] = j
                break
    return neg


class HeteroConsolidationBuffer:
    """A closed, seed-fixed buffer of precommitted ``(key, value_idx)`` observations
    for the batch-offline heteroassociative write. ``freeze()`` must be called
    before the write fires (the AH condition: a single offline pass over a frozen
    buffer, not a streaming update)."""

    def __init__(self, dim: int, device):
        self.dim = dim
        self.device = device
        self._keys: list[torch.Tensor] = []
        self._vidx: list[int] = []
        self._frozen = False
        self.keys: Optional[torch.Tensor] = None
        self.vidx: Optional[torch.Tensor] = None

    def add(self, key: torch.Tensor, value_idx: int) -> None:
        if self._frozen:
            raise RuntimeError("buffer is frozen; cannot add after freeze()")
        if key.shape[-1] != self.dim:
            raise ValueError(f"key dim {key.shape[-1]} != {self.dim}")
        self._keys.append(key.detach())
        self._vidx.append(int(value_idx))

    def freeze(self) -> "HeteroConsolidationBuffer":
        if self._frozen:
            return self
        if not self._keys:
            raise ValueError("cannot freeze an empty buffer")
        self.keys = torch.stack(self._keys, dim=0).to(self.device)
        self.vidx = torch.tensor(self._vidx, dtype=torch.long, device=self.device)
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._vidx)


def heteroassociative_write(
    buffer: HeteroConsolidationBuffer,
    value_codebook: torch.Tensor,
    *,
    lr: float = 0.5,
    epochs: int = 20,
    contrastive: bool = False,
    lr_push: float = 0.1,
    neg_seed: int = 0,
) -> torch.Tensor:
    """Error-correcting (delta-rule) heteroassociative write over the FROZEN buffer.

    Returns the weight ``H`` (``[D, D]`` complex) mapping a key to its value region:
    ``H k_i ≈ D · v_i``. With ``contrastive=True`` adds an anti-Hebbian push away
    from a precommitted swap-negative (the anti-homunculus-clean G-B negative).

    Raises if the buffer is not frozen (batch-offline only — AH condition 3).
    """
    if not buffer.frozen:
        raise RuntimeError(
            "freeze() the buffer before the write (batch-offline only; "
            "a streaming per-observation write is BANNED)")
    keys = buffer.keys                      # [N, D]
    N, D = keys.shape
    pair_vals = value_codebook[buffer.vidx]  # [N, D]
    neg_vals = None
    if contrastive:
        neg_idx = precommit_swap_negatives(buffer.vidx, value_codebook.shape[0], neg_seed)
        neg_vals = value_codebook[neg_idx]
    H = torch.zeros(D, D, dtype=keys.dtype, device=keys.device)
    for _ in range(epochs):
        R = (keys @ H.transpose(0, 1)) / D   # [N, D] = H k_i
        resid = pair_vals - R
        H = H + lr * (resid.transpose(0, 1) @ keys.conj()) / N
        if neg_vals is not None:
            H = H - lr_push * (neg_vals.transpose(0, 1) @ keys.conj()) / N
    return H


def recall_top_index(
    substrate, H: torch.Tensor, keys: torch.Tensor, value_codebook: torch.Tensor,
    *, beta: float = 10.0, max_iter: int = 12,
):
    """Recover the value basin for each key: cleanup ``H k`` over the value
    codebook -> ``(top_index, entropy, margin)``. Read terminates in a basin index
    count (Phase-5' fence respected)."""
    D = H.shape[0]
    recalled = (keys @ H.transpose(0, 1)) / D
    return batched_hopfield_topindex(
        substrate, value_codebook, recalled, beta=beta, max_iter=max_iter)


__all__ = [
    "batched_hopfield_topindex",
    "precommit_swap_negatives",
    "HeteroConsolidationBuffer",
    "heteroassociative_write",
    "recall_top_index",
]
