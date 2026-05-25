"""Range-shaped replay sampling for Phase 4.

The sampler is a static replay discipline: it reads encoder-time provenance
from stored trajectory traces, builds role and atom marginals, and samples from
their product. It does not inspect downstream metrics or switch behavior based
on observed success.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.phase2.encoding import encode_window_with_provenance
from energy_memory.phase4.trajectory import TrajectoryTrace


RoleAtomPair = Tuple[int, int]
SampledPair = Tuple[int, int, Optional[int]]


@dataclass(frozen=True)
class _RoleAtomIndex:
    """Inverted index over trace encoder terms."""

    by_pair: Dict[RoleAtomPair, List[int]]
    role_weights: Dict[int, float]
    atom_weights: Dict[int, float]
    missing_encoder_terms: int


class RangeShapedReplaySampler:
    """Rectangularized sampler over ``(role, atom)`` encoder terms.

    ``sample_pairs`` emits pairs from ``p(role) * p(atom)``. The optional trace
    index points to a stored trace that already contains the pair; ``None``
    means the caller can skip, fall back to a nearby stored trace, or synthesize
    a replay trace by rebinding on the fly. Those choices are static config,
    not adaptive routing.
    """

    def __init__(
        self,
        store,
        *,
        weight_floor: float = 1e-9,
        atom_count: Optional[int] = None,
        atom_smoothing_alpha: float = 0.0,
    ):
        if weight_floor <= 0.0:
            raise ValueError("weight_floor must be positive")
        if atom_smoothing_alpha < 0.0:
            raise ValueError("atom_smoothing_alpha must be non-negative")
        if atom_count is not None and atom_count < 0:
            raise ValueError("atom_count must be non-negative")
        self.store = store
        self.weight_floor = float(weight_floor)
        self.atom_count = atom_count
        self.atom_smoothing_alpha = float(atom_smoothing_alpha)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        by_pair: Dict[RoleAtomPair, List[int]] = defaultdict(list)
        role_weights: Counter = Counter()
        atom_weights: Counter = Counter()
        missing_encoder_terms = 0

        for trace_idx, trace in enumerate(self.store.traces):
            if trace.encoder_terms is None:
                missing_encoder_terms += 1
                continue
            w = max(float(self.store.gate_signals[trace_idx]), self.weight_floor)
            for role, atom in trace.encoder_terms:
                role_i = int(role)
                atom_i = int(atom)
                by_pair[(role_i, atom_i)].append(trace_idx)
                role_weights[role_i] += w
                atom_weights[atom_i] += w

        if self.atom_count is not None and self.atom_smoothing_alpha > 0.0:
            for atom_i in range(self.atom_count):
                atom_weights[atom_i] += self.atom_smoothing_alpha

        self.index = _RoleAtomIndex(
            by_pair=dict(by_pair),
            role_weights=dict(role_weights),
            atom_weights=dict(atom_weights),
            missing_encoder_terms=missing_encoder_terms,
        )

    def sample_pairs(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> List[SampledPair]:
        """Sample ``n`` ``(role, atom, trace_idx)`` triples.

        Trace selection for backed pairs is priority-weighted by each
        candidate trace's gate signal. Passing a seeded ``torch.Generator``
        gives deterministic output.
        """
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("RangeShapedReplaySampler requires torch") from _IMPORT_ERROR
        if n <= 0:
            return []
        if not self.index.role_weights or not self.index.atom_weights:
            return []

        roles = sorted(self.index.role_weights.keys())
        atoms = sorted(self.index.atom_weights.keys())
        role_weights = _normalized_weights(
            [self.index.role_weights[r] for r in roles]
        )
        atom_weights = _normalized_weights(
            [self.index.atom_weights[a] for a in atoms]
        )

        role_pos = torch.multinomial(
            role_weights, n, replacement=True, generator=generator
        )
        atom_pos = torch.multinomial(
            atom_weights, n, replacement=True, generator=generator
        )

        out: List[SampledPair] = []
        for r_pos, a_pos in zip(role_pos.tolist(), atom_pos.tolist()):
            role = roles[int(r_pos)]
            atom = atoms[int(a_pos)]
            trace_idx = self._pick_backing_trace(role, atom, generator=generator)
            out.append((role, atom, trace_idx))
        return out

    def sample_indices(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        fallback: str = "skip",
    ) -> List[int]:
        """Legacy trace-index sampler for non-rebinding call sites.

        ``fallback``:
        - ``skip``: drop unbacked pairs
        - ``closest``: use a priority-weighted trace sharing the role, then atom
        - ``rebind``: invalid here; synthesize a trace from ``sample_pairs``
        """
        if fallback not in {"skip", "closest"}:
            raise ValueError("sample_indices fallback must be 'skip' or 'closest'")
        indices: List[int] = []
        for role, atom, trace_idx in self.sample_pairs(n, generator=generator):
            if trace_idx is not None:
                indices.append(trace_idx)
            elif fallback == "closest":
                closest = self.closest_trace_index(role, atom, generator=generator)
                if closest is not None:
                    indices.append(closest)
        return indices

    def closest_trace_index(
        self,
        role: int,
        atom: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[int]:
        """Priority-weighted fallback sharing role first, atom second."""
        same_role: List[int] = []
        same_atom: List[int] = []
        for (candidate_role, candidate_atom), trace_indices in self.index.by_pair.items():
            if candidate_role == role:
                same_role.extend(trace_indices)
            elif candidate_atom == atom:
                same_atom.extend(trace_indices)
        if same_role:
            return self._pick_weighted_index(same_role, generator=generator)
        if same_atom:
            return self._pick_weighted_index(same_atom, generator=generator)
        return None

    def sample_window_pairs(
        self,
        window_size: int,
        *,
        generator: Optional[torch.Generator] = None,
        anchor_pair: Optional[RoleAtomPair] = None,
    ) -> List[RoleAtomPair]:
        """Sample a full factored window, optionally anchored on one pair."""
        if window_size <= 0:
            return []
        pairs: List[RoleAtomPair] = []
        if anchor_pair is not None:
            pairs.append((int(anchor_pair[0]), int(anchor_pair[1])))
        needed = window_size - len(pairs)
        if needed > 0:
            pairs.extend(
                (role, atom)
                for role, atom, _ in self.sample_pairs(needed, generator=generator)
            )
        return pairs[:window_size]

    def marginal_diagnostics(self) -> dict:
        """Return buffer support and marginal summary diagnostics."""
        return {
            "n_traces": len(self.store.traces),
            "n_pair_cells": len(self.index.by_pair),
            "n_roles": len(self.index.role_weights),
            "n_atoms": len(self.index.atom_weights),
            "n_missing_encoder_terms": self.index.missing_encoder_terms,
            "atom_smoothing_alpha": self.atom_smoothing_alpha,
            "atom_count": self.atom_count,
        }

    def _pick_backing_trace(
        self,
        role: int,
        atom: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[int]:
        candidates = self.index.by_pair.get((role, atom))
        if not candidates:
            return None
        return self._pick_weighted_index(candidates, generator=generator)

    def _pick_weighted_index(
        self,
        candidates: Sequence[int],
        *,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        weights = _normalized_weights(
            [
                max(float(self.store.gate_signals[c]), self.weight_floor)
                for c in candidates
            ]
        )
        pick_pos = int(torch.multinomial(weights, 1, generator=generator).item())
        return int(candidates[pick_pos])


def synthesize_single_binding_trace(
    substrate,
    position_vectors: Sequence[torch.Tensor],
    codebook: torch.Tensor,
    role: int,
    atom: int,
) -> TrajectoryTrace:
    """Synthesize one replay trace from ``bind(position_vectors[role], codebook[atom])``."""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("range-shaped replay synthesis requires torch") from _IMPORT_ERROR
    _validate_role_atom(position_vectors, codebook, role, atom)
    query = substrate.bind(position_vectors[int(role)], codebook[int(atom)])
    return TrajectoryTrace(
        query=query.detach().clone(),
        encoder_terms=[(int(role), int(atom))],
    )


def synthesize_window_preserving_trace(
    substrate,
    position_vectors: Sequence[torch.Tensor],
    codebook: torch.Tensor,
    pairs: Sequence[RoleAtomPair],
) -> TrajectoryTrace:
    """Synthesize a full window and preserve global encoder terms.

    The call intentionally goes through ``encode_window_with_provenance`` so the
    encoded query follows the same bundling path as normal Phase 2 windows.
    The returned trace records the caller's global role ids, not the local
    positions used inside the helper call.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("range-shaped replay synthesis requires torch") from _IMPORT_ERROR
    if not pairs:
        raise ValueError("pairs must be non-empty")
    roles = [int(role) for role, _ in pairs]
    atoms = [int(atom) for _, atom in pairs]
    for role, atom in zip(roles, atoms):
        _validate_role_atom(position_vectors, codebook, role, atom)
    local_positions = [position_vectors[role] for role in roles]
    query, _local_terms = encode_window_with_provenance(
        substrate, local_positions, codebook, atoms
    )
    return TrajectoryTrace(
        query=query.detach().clone(),
        encoder_terms=list(zip(roles, atoms)),
    )


def _normalized_weights(values: Sequence[float]) -> torch.Tensor:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("RangeShapedReplaySampler requires torch") from _IMPORT_ERROR
    weights = torch.tensor(list(values), dtype=torch.float32)
    weights = weights.clamp(min=1e-12)
    return weights / weights.sum()


def _validate_role_atom(
    position_vectors: Sequence[torch.Tensor],
    codebook: torch.Tensor,
    role: int,
    atom: int,
) -> None:
    if role < 0 or role >= len(position_vectors):
        raise IndexError(f"role {role} out of range for {len(position_vectors)} positions")
    if atom < 0 or atom >= int(codebook.shape[0]):
        raise IndexError(f"atom {atom} out of range for codebook size {codebook.shape[0]}")
