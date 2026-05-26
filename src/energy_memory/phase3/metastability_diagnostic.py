"""Passive per-atom metastability EMA diagnostic (C.1.5).

Reads ``ConsolidationState.metastability_ema`` and exposes it as a clean
readout matching the ``BasinDiagnostics`` pattern in basin_diagnostics.py.
Standalone — does not wire into ``ConsolidationDiagnostics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetastabilityDiagnostics:
    per_atom: Dict[int, float] = field(default_factory=dict)
    mean: float = 0.0
    max: float = 0.0
    n_atoms: int = 0
    obs_rate_active: bool = False


def compute_metastability_diagnostics(state) -> MetastabilityDiagnostics:
    obs_rate_active = state.config.metastability_obs_rate > 0.0
    n_atoms = int(state.metastability_ema.shape[0])
    if n_atoms == 0:
        return MetastabilityDiagnostics(
            per_atom={},
            mean=0.0,
            max=0.0,
            n_atoms=0,
            obs_rate_active=obs_rate_active,
        )
    # Single batched sync — respect CLAUDE.md GPU performance rule.
    values = state.metastability_ema.detach().to("cpu").tolist()
    per_atom: Dict[int, float] = {i: float(v) for i, v in enumerate(values)}
    mean_val = float(sum(values) / n_atoms)
    max_val = float(max(values))
    return MetastabilityDiagnostics(
        per_atom=per_atom,
        mean=mean_val,
        max=max_val,
        n_atoms=n_atoms,
        obs_rate_active=obs_rate_active,
    )
