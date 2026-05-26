from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch


@dataclass
class BasinTrace:
    settled_state: torch.Tensor
    top1_atom: int


class BasinTraceBuffer:
    def __init__(self, maxlen: int = 5):
        self._buf: deque = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def append(self, trace: BasinTrace) -> None:
        self._buf.append(trace)

    def __len__(self) -> int:
        return len(self._buf)

    def traces_by_atom(self) -> Dict[int, List[BasinTrace]]:
        out: Dict[int, List[BasinTrace]] = {}
        for trace in self._buf:
            out.setdefault(int(trace.top1_atom), []).append(trace)
        return out


@dataclass
class BasinDiagnostics:
    nc1_per_atom: Dict[int, float] = field(default_factory=dict)
    nc1_singleton_atoms: Set[int] = field(default_factory=set)
    separability_nc2: Optional[float] = None
    n_basins_observed: int = 0
    total_traces: int = 0


def _stack_states(traces: List[BasinTrace]) -> torch.Tensor:
    return torch.stack([t.settled_state for t in traces], dim=0)


# Complex tensors use Hermitian covariance via algebraic d_eff identity; real tensors fall through the same path since conj() is a no-op on real.
def _d_eff_patterns(patterns: torch.Tensor) -> torch.Tensor:
    n = patterns.shape[0]
    if n < 2:
        return torch.tensor(float("nan"), device=patterns.device)
    centered = patterns - patterns.mean(dim=0, keepdim=True)
    gram = centered @ centered.conj().T / n
    tr_g = gram.diagonal().real.sum()
    tr_g_sq = (gram.abs() * gram.abs()).sum()
    return (tr_g * tr_g) / tr_g_sq.clamp(min=1e-12)


def compute_basin_nc1(buffer: BasinTraceBuffer) -> Tuple[Dict[int, float], Set[int]]:
    grouped = buffer.traces_by_atom()
    atoms_sorted = sorted(grouped.keys())
    singleton_atoms: Set[int] = set()
    deff_tensors: List[torch.Tensor] = []
    atoms_with_deff: List[int] = []
    for atom in atoms_sorted:
        traces = grouped[atom]
        if len(traces) == 1:
            singleton_atoms.add(atom)
            atoms_with_deff.append(atom)
            deff_tensors.append(torch.tensor(1.0))
            continue
        patterns = _stack_states(traces)
        deff = _d_eff_patterns(patterns)
        atoms_with_deff.append(atom)
        deff_tensors.append(deff)
    if not deff_tensors:
        return {}, singleton_atoms
    stacked = torch.stack([d.detach().to(torch.float32).reshape(()) for d in deff_tensors]).cpu().tolist()
    nc1: Dict[int, float] = {atom: float(val) for atom, val in zip(atoms_with_deff, stacked)}
    return nc1, singleton_atoms


def compute_basin_separability_nc2(buffer: BasinTraceBuffer) -> Optional[float]:
    grouped = buffer.traces_by_atom()
    if len(grouped) < 2:
        return None
    atoms_sorted = sorted(grouped.keys())
    centroids = []
    for atom in atoms_sorted:
        patterns = _stack_states(grouped[atom])
        centroid = patterns.mean(dim=0)
        norm = centroid.abs().pow(2).sum().sqrt() if torch.is_complex(centroid) else centroid.norm()
        centroid = centroid / norm.clamp(min=1e-12)
        centroids.append(centroid)
    C = torch.stack(centroids, dim=0)
    K = C.shape[0]
    gram = (C @ C.conj().T).real
    eye = torch.eye(K, device=gram.device, dtype=gram.dtype)
    ones = torch.ones((K, K), device=gram.device, dtype=gram.dtype)
    g_etf = (K / (K - 1)) * (eye - ones / K)
    diff = gram - g_etf
    nc2 = (diff * diff).sum().sqrt()
    return float(nc2.detach().cpu())


def compute_basin_separability_pairwise_mean(buffer: BasinTraceBuffer) -> Optional[float]:
    grouped = buffer.traces_by_atom()
    if len(grouped) < 2:
        return None
    atoms_sorted = sorted(grouped.keys())
    centroids = []
    for atom in atoms_sorted:
        patterns = _stack_states(grouped[atom])
        centroid = patterns.mean(dim=0)
        norm = centroid.abs().pow(2).sum().sqrt() if torch.is_complex(centroid) else centroid.norm()
        centroids.append(centroid / norm.clamp(min=1e-12))
    C = torch.stack(centroids, dim=0)
    K = C.shape[0]
    gram = (C @ C.conj().T).real
    off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=gram.device)
    mean_cos = gram[off_diag_mask].mean()
    return float((1.0 - mean_cos).detach().cpu())


def compute_basin_diagnostics(buffer: BasinTraceBuffer) -> BasinDiagnostics:
    nc1, singletons = compute_basin_nc1(buffer)
    nc2 = compute_basin_separability_nc2(buffer)
    grouped = buffer.traces_by_atom()
    return BasinDiagnostics(
        nc1_per_atom=nc1,
        nc1_singleton_atoms=singletons,
        separability_nc2=nc2,
        n_basins_observed=len(grouped),
        total_traces=len(buffer),
    )
