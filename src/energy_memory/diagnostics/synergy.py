"""Synergy estimator for FHRR-bound role-filler composites.

Operationalization of the Generalized Information Bottleneck (GIB)
synergy concept (arXiv:2509.26327, 2025) for vector-symbolic
architectures, as recommended in brainstorm Idea 8 (2026-05-13) for a
Phase 5 headline metric.

The intuition: a composite ``c = bind(role, filler)`` is *synergistic*
when knowledge of (role, filler) is recoverable from ``c`` **only via
the joint binding**, not from either component alone. We measure this
geometrically rather than information-theoretically:

  recover(c, role)         = sim( unbind(c, role), filler )
  baseline_from_role(role) = sim( role, filler )
  baseline_from_binding(c) = sim( c, filler )
  synergy(c, role, filler) = recover(c, role)
                             - max( baseline_from_role,
                                    baseline_from_binding )

A high score means the binding carries information about the filler
that neither the role alone nor the binding alone (without
factorization) can disclose. The same definition holds with role and
filler swapped — synergy in this VSA is symmetric.

Why this is the right shape:
  * Random independent role and filler -> baselines ~= 0.
  * Perfect FHRR binding -> recover ~= 1.
  * Therefore synergy ~= 1 for clean bindings, ~= 0 for a degenerate
    binding (e.g. ``c = filler`` directly without role).

The estimator generalizes to bundles of multiple role-filler pairs:
unbinding one role recovers the corresponding filler plus crosstalk
noise from the other bindings.

This module is intentionally small and dependency-light. It is meant
as a measurement primitive that experiments can call, not as a piece
of the dynamics. No update rule reads its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class SynergyMeasurement:
    recover: float            # sim(unbind(c, role), filler)
    baseline_from_role: float  # sim(role, filler)
    baseline_from_binding: float  # sim(binding, filler)
    synergy: float            # recover - max(baselines)


def _sim(substrate: TorchFHRR, left, right) -> float:
    return float(substrate.similarity(left, right))


def synergy_score(
    substrate: TorchFHRR,
    role,
    filler,
    binding=None,
) -> SynergyMeasurement:
    """Synergy of a single role-filler binding.

    If ``binding`` is None, it is computed as ``substrate.bind(role, filler)``.
    Use this when validating that the substrate's own bind is synergistic;
    pass an externally produced ``binding`` when measuring a candidate
    composite from elsewhere in the pipeline (e.g. a Hopfield-settled
    state, a layer-2 binding, etc.).
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("synergy_score requires torch") from _IMPORT_ERROR
    if binding is None:
        binding = substrate.bind(role, filler)
    recovered = substrate.unbind(binding, role)
    recover = _sim(substrate, recovered, filler)
    baseline_role = _sim(substrate, role, filler)
    baseline_binding = _sim(substrate, binding, filler)
    synergy = recover - max(baseline_role, baseline_binding)
    return SynergyMeasurement(
        recover=recover,
        baseline_from_role=baseline_role,
        baseline_from_binding=baseline_binding,
        synergy=synergy,
    )


def mean_synergy(
    substrate: TorchFHRR,
    roles: Sequence,
    fillers: Sequence,
    bindings: Sequence = None,
) -> SynergyMeasurement:
    """Synergy averaged over a batch of (role, filler[, binding]) triples.

    All three sequences must have the same length. ``bindings=None`` causes
    each binding to be computed on the fly from its (role, filler) pair.
    """
    if len(roles) != len(fillers):
        raise ValueError("roles and fillers must have the same length")
    if bindings is not None and len(bindings) != len(roles):
        raise ValueError("bindings must match the role/filler length")

    rs, brs, bbs, ss = [], [], [], []
    for i in range(len(roles)):
        binding = bindings[i] if bindings is not None else None
        m = synergy_score(substrate, roles[i], fillers[i], binding=binding)
        rs.append(m.recover)
        brs.append(m.baseline_from_role)
        bbs.append(m.baseline_from_binding)
        ss.append(m.synergy)
    n = len(rs)
    return SynergyMeasurement(
        recover=sum(rs) / n,
        baseline_from_role=sum(brs) / n,
        baseline_from_binding=sum(bbs) / n,
        synergy=sum(ss) / n,
    )


def atom_alone_synergy(substrate: TorchFHRR, atoms: Sequence) -> float:
    """Reference value: synergy for "binding" that is just the filler itself.

    If a system claims compositional structure but its "bindings" are
    literally the filler atoms (no role), this returns ~ 1 - 1 = 0:
    you can recover the filler from the "binding", but you didn't gain
    anything that you couldn't get from the filler alone. This is the
    null-hypothesis baseline for Phase 5 structural claims.

    Definition: synergy(c = filler, role = arbitrary, filler) = 1 - 1 = 0.
    Returned value is the mean over the supplied atom batch (should be
    ~ 0 for any well-behaved substrate).
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("atom_alone_synergy requires torch") from _IMPORT_ERROR
    if not len(atoms):
        return 0.0
    # Sample one role per atom from the substrate's random distribution.
    roles = substrate.random_vectors(len(atoms))
    bindings = atoms  # degenerate: the "binding" IS the filler
    scores: List[float] = []
    for i in range(len(atoms)):
        scores.append(
            synergy_score(
                substrate, roles[i], atoms[i], binding=bindings[i]
            ).synergy
        )
    return sum(scores) / len(scores)
