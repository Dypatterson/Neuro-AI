"""Role-fidelity-weighted prior (Candidate β, path 3).

See [notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md] for
the design and anti-homunculus audit. β replaces the categorical
``top_k_by_effective_strength`` schema-store selector (failure mode
identified in [reports/049_phase5_a1prime_pilot_seed17.md]) with a
continuous weighted sum over the full schema store:

    prior(cue) = normalize( Σ_i (cue · s_i)_+^p · f_i^q · s_i )

where:

- ``(cue · s_i)_+`` is the non-negative-clamped FHRR cosine
  (max(0, Re(<cue, s_i>) / (||cue|| ||s_i||))).
- ``f_i ∈ [0, 1]`` is the per-schema role-binding fidelity (this module).
- ``s_i`` is the schema vector (FHRR-encoded).

``p, q ≥ 0``. At ``q = 0`` the prior reduces to a content-only weighted
sum (no role-fidelity influence); at ``q = 1`` schemas with cleanly-
distinguishable role decompositions are continuously up-weighted.

This module is **pure**: no state, no scheduled computation, no
per-population arbitration. The role-fidelity is a substrate-derived
geometric property of each schema; the prior construction is a
weighted superposition.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def compute_role_fidelity(schema_bindings: "torch.Tensor") -> "torch.Tensor":
    """Per-schema role-binding fidelity ``f_i ∈ [0, 1]``.

    For each schema, the mean pairwise FHRR distance ``(1 - |G_jk|)``
    among its W unbound fillers, where ``G_jk = (1/D) * <f_j, f_k>``.

    Schemas whose unbind decomposition produces distinct fillers (clean
    role-binding) have ``f_i ≈ 1`` (pairwise distance large). Schemas
    whose unbind collapses (fillers near-identical across positions)
    have ``f_i ≈ 0``.

    The fidelity is **state-free**: it depends only on the schema vector
    and the position vectors via the substrate's unbind operation; it
    is not a controller-set parameter. The 2026-05-20 cue-regime note
    audited PASS on this shape.

    Parameters
    ----------
    schema_bindings : torch.Tensor, shape [N, W, D] complex
        Per-schema unbound fillers, one per role position. Typically
        produced by
        ``experiments/40_phase5_branching.compute_schema_bindings``.

    Returns
    -------
    torch.Tensor, shape [N] float32
        Role-fidelity per schema, in [0, 1].
    """
    if schema_bindings.dim() != 3:
        raise ValueError(
            f"schema_bindings must be [N, W, D], got shape "
            f"{tuple(schema_bindings.shape)}"
        )
    n, w, d = schema_bindings.shape
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=schema_bindings.device)
    if w < 2:
        # Single position → no pairs → fidelity undefined; return 0.
        return torch.zeros(n, dtype=torch.float32, device=schema_bindings.device)

    # Per-schema Gram of fillers: [N, W, W] complex
    gram = (schema_bindings @ schema_bindings.conj().transpose(-1, -2)) / d
    gram_abs = gram.abs()  # [N, W, W], in [0, 1]

    # Mean of (1 - |G_jk|) over off-diagonal pairs
    mask = ~torch.eye(w, dtype=torch.bool, device=schema_bindings.device)
    n_pairs = float(mask.sum().item())  # W*(W-1)
    if n_pairs == 0:  # defensive; W >= 2 guards above
        return torch.zeros(n, dtype=torch.float32, device=schema_bindings.device)

    dists = (1.0 - gram_abs) * mask.to(gram_abs.dtype).unsqueeze(0)  # [N, W, W]
    f = dists.sum(dim=(1, 2)) / n_pairs  # [N]
    return f.clamp(min=0.0, max=1.0).to(torch.float32)


def fidelity_weighted_prior(
    *,
    cue: "torch.Tensor",
    schemas: "torch.Tensor",
    fidelities: "torch.Tensor",
    p: float = 1.0,
    q: float = 1.0,
    substrate=None,
) -> "torch.Tensor":
    """β prior: ``normalize( Σ_i (cue·s_i)_+^p · f_i^q · s_i )``.

    Continuous weighted superposition over the full schema store, with
    no top-k cutoff and no categorical role-vs-content branch. Replaces
    the categorical selector that report 049 identified as the binding
    failure layer.

    Parameters
    ----------
    cue : torch.Tensor, shape [D] complex
        Query vector (the Phase 5 retrieval input).
    schemas : torch.Tensor, shape [N, D] complex
        The full schema store (no top-k pre-filter).
    fidelities : torch.Tensor, shape [N] float32
        Per-schema role-fidelity in [0, 1]. From
        ``compute_role_fidelity``.
    p : float, default 1.0
        Exponent on cue-similarity weight. ``p ≥ 0`` required.
    q : float, default 1.0
        Exponent on role-fidelity weight. ``q ≥ 0`` required. At
        ``q = 0`` the prior is content-only (the natural baseline);
        ``q = 1`` is the recommended setting per the design note.
    substrate : TorchFHRR, optional
        Used to normalize the output. If None, the output is normalized
        via element-wise unit-magnitude (FHRR convention).

    Returns
    -------
    torch.Tensor, shape [D] complex
        Unit-magnitude FHRR prior vector.

    Anti-homunculus note: the prior is a continuous superposition; no
    decision is made over the schema store. Every schema contributes
    its own direction weighted by two local geometric quantities
    (cue-cosine, role-fidelity). The output is one vector, not a
    selected subset. See
    notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md.
    """
    if p < 0.0 or q < 0.0:
        raise ValueError(f"p and q must be non-negative, got p={p}, q={q}")
    if schemas.dim() != 2:
        raise ValueError(
            f"schemas must be [N, D], got shape {tuple(schemas.shape)}"
        )
    if cue.dim() != 1 or cue.shape[0] != schemas.shape[-1]:
        raise ValueError(
            f"cue must be [D] matching schemas' D={schemas.shape[-1]}; "
            f"got {tuple(cue.shape)}"
        )
    if fidelities.shape != (schemas.shape[0],):
        raise ValueError(
            f"fidelities must be [N={schemas.shape[0]}], got "
            f"{tuple(fidelities.shape)}"
        )

    n, d = schemas.shape
    if n == 0:
        # No schemas — return the cue itself as a degenerate prior.
        return cue.clone()

    # Cue-cosine: Re(<cue, s_i>) / (||cue|| · ||s_i||), clamped to [0, ∞).
    inner = (schemas.conj() * cue).sum(dim=-1).real  # [N], float
    cue_norm = cue.norm().clamp(min=1e-12)
    sch_norm = schemas.norm(dim=-1).clamp(min=1e-12)  # [N]
    cosines = (inner / (sch_norm * cue_norm)).clamp(min=0.0)  # [N]

    cue_weight = cosines ** p
    fid_weight = fidelities.to(cosines.dtype) ** q
    weights = cue_weight * fid_weight  # [N], non-negative

    # Weighted sum of schema vectors (cast weights to schema dtype)
    weighted = (weights.unsqueeze(-1).to(schemas.dtype) * schemas).sum(dim=0)  # [D]

    if substrate is not None:
        return substrate.normalize(weighted)
    # Fallback: element-wise unit-magnitude (FHRR convention)
    mag = weighted.abs().clamp(min=1e-12)
    return weighted / mag.to(weighted.dtype)


__all__ = ["compute_role_fidelity", "fidelity_weighted_prior"]
