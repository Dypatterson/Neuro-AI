"""Per-atom bimodality diagnostic (C.1.2).

Standalone passive diagnostic per Path C precommit
(notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md).
Tracks per-atom context_bag rolling history, reduces D-dim bags to a 1D
consecutive-cosine signal, and exposes Hartigan's dip test (primary) and
GMM-BIC (fallback) for multimodality detection.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import torch


@dataclass
class ContextBagHistory:
    bags: Deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=5))
    maxlen: int = 5

    def __post_init__(self) -> None:
        if self.bags.maxlen != self.maxlen:
            self.bags = deque(self.bags, maxlen=self.maxlen)

    def append(self, bag: torch.Tensor) -> None:
        self.bags.append(bag)

    def __len__(self) -> int:
        return len(self.bags)


@dataclass
class PerAtomBimodalitySignal:
    signal: torch.Tensor
    dip_p_value: Optional[float] = None
    dip_rejects: bool = False
    delta_bic: Optional[float] = None
    bic_favors_k2: bool = False


@dataclass
class BimodalityDiagnostics:
    per_atom_signal: Dict[int, PerAtomBimodalitySignal] = field(default_factory=dict)
    persistent_rejections: Dict[int, int] = field(default_factory=dict)
    persistent_bimodality: Dict[int, bool] = field(default_factory=dict)
    n_atoms: int = 0


def context_bag_signal(history: ContextBagHistory) -> torch.Tensor:
    n = len(history)
    if n < 2:
        return torch.empty(0)
    bags = list(history.bags)
    # Hermitian inner product handles FHRR (complex); .real is a no-op on real tensors.
    stacked = torch.stack(bags, dim=0)
    a = stacked[:-1]
    b = stacked[1:]
    if torch.is_complex(a):
        inner = (a * b.conj()).sum(dim=-1).real
        norm_a = (a * a.conj()).sum(dim=-1).real.clamp(min=1e-12).sqrt()
        norm_b = (b * b.conj()).sum(dim=-1).real.clamp(min=1e-12).sqrt()
    else:
        inner = (a * b).sum(dim=-1)
        norm_a = (a * a).sum(dim=-1).clamp(min=1e-12).sqrt()
        norm_b = (b * b).sum(dim=-1).clamp(min=1e-12).sqrt()
    cos = inner / (norm_a * norm_b)
    return cos.detach().to(torch.float32)


def _gcm(x: list, y: list) -> list:
    """Greatest convex minorant: lower convex hull of points (x[i], y[i])."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            x1, y1 = x[hull[-2]], y[hull[-2]]
            x2, y2 = x[hull[-1]], y[hull[-1]]
            # Cross product: keep convex (turning left/up).
            if (x2 - x1) * (y[i] - y1) - (y2 - y1) * (x[i] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def _lcm(x: list, y: list) -> list:
    """Least concave majorant: upper concave hull of points (x[i], y[i])."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            x1, y1 = x[hull[-2]], y[hull[-2]]
            x2, y2 = x[hull[-1]], y[hull[-1]]
            if (x2 - x1) * (y[i] - y1) - (y2 - y1) * (x[i] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def _interp(xs: list, ys: list, idxs: list, x_query: list) -> list:
    """Linearly interpolate (xs[idxs], ys[idxs]) at x_query positions."""
    out = []
    j = 0
    for xq in x_query:
        while j + 1 < len(idxs) and xs[idxs[j + 1]] < xq:
            j += 1
        if j + 1 >= len(idxs):
            out.append(ys[idxs[-1]])
            continue
        i0, i1 = idxs[j], idxs[j + 1]
        x0, x1 = xs[i0], xs[i1]
        y0, y1 = ys[i0], ys[i1]
        if x1 == x0:
            out.append(y0)
        else:
            t = (xq - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out


def _dip_statistic(x_sorted: torch.Tensor) -> float:
    """Hartigan dip = max deviation of ECDF from closest unimodal CDF.

    For each candidate mode position m, fit GCM on [0..m] (convex CDF =
    increasing density) and LCM on [m..n-1] (concave CDF = decreasing
    density); the dip is half the sup-deviation, minimized over m.
    """
    n = x_sorted.shape[0]
    if n < 4:
        return 0.0
    xs = x_sorted.tolist()
    ys = [(i + 1) / n for i in range(n)]
    best = float("inf")
    # Iterate candidate mode indices; interior modes give a meaningful unimodal envelope.
    for m in range(1, n - 1):
        gcm_idx = _gcm(xs[: m + 1], ys[: m + 1])
        lcm_idx_local = _lcm(xs[m:], ys[m:])
        lcm_idx = [m + i for i in lcm_idx_local]
        envelope_idx = gcm_idx[:-1] + lcm_idx
        env_y = _interp(xs, ys, envelope_idx, xs)
        dev = max(abs(ys[i] - env_y[i]) for i in range(n))
        if dev < best:
            best = dev
    return 0.5 * best


def hartigan_dip(
    signal: torch.Tensor,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[float], bool]:
    n = signal.shape[0]
    if n < 4:
        return None, False
    x_sorted, _ = torch.sort(signal.to(torch.float32))
    obs = _dip_statistic(x_sorted)
    # Null is uniform on [0,1]; dip is the standard reference null (Hartigan & Hartigan 1985).
    if generator is None:
        null = torch.rand((n_bootstrap, n))
    else:
        null = torch.rand((n_bootstrap, n), generator=generator)
    null_sorted, _ = torch.sort(null, dim=1)
    null_stats = torch.empty(n_bootstrap)
    for b in range(n_bootstrap):
        null_stats[b] = _dip_statistic(null_sorted[b])
    count_ge = int((null_stats >= obs).sum().item())
    p_value = (count_ge + 1) / (n_bootstrap + 1)
    return p_value, p_value < alpha


def _gmm_em_1d(
    x: torch.Tensor,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """1D 2-component Gaussian mixture EM. Returns (weights, means, vars, loglik)."""
    n = x.shape[0]
    x_sorted, _ = torch.sort(x)
    mu = torch.stack([x_sorted[n // 4], x_sorted[(3 * n) // 4]])
    var = torch.full((2,), float(x.var(unbiased=False).clamp(min=1e-6)))
    w = torch.tensor([0.5, 0.5])
    prev_ll = torch.tensor(-float("inf"))
    ll = prev_ll
    for _ in range(n_iter):
        # E-step.
        log_norm = -0.5 * (torch.log(2 * math.pi * var.clamp(min=1e-12)))
        diff = x.unsqueeze(1) - mu.unsqueeze(0)
        log_pdf = log_norm.unsqueeze(0) - 0.5 * (diff * diff) / var.clamp(min=1e-12).unsqueeze(0)
        log_weighted = log_pdf + torch.log(w.clamp(min=1e-12)).unsqueeze(0)
        log_resp_denom = torch.logsumexp(log_weighted, dim=1, keepdim=True)
        ll = log_resp_denom.sum()
        resp = torch.exp(log_weighted - log_resp_denom)
        # M-step.
        nk = resp.sum(dim=0).clamp(min=1e-12)
        w = nk / n
        mu = (resp * x.unsqueeze(1)).sum(dim=0) / nk
        diff2 = x.unsqueeze(1) - mu.unsqueeze(0)
        var = ((resp * diff2 * diff2).sum(dim=0) / nk).clamp(min=1e-6)
        if torch.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return w, mu, var, ll


def _gaussian_loglik_1d(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    mu = x.mean()
    var = x.var(unbiased=False).clamp(min=1e-6)
    return (-0.5 * n * torch.log(2 * math.pi * var) - 0.5 * ((x - mu) * (x - mu)).sum() / var)


def gmm_bic_1d(
    signal: torch.Tensor,
    delta_threshold: float = 10.0,
) -> Tuple[Optional[float], bool]:
    n = signal.shape[0]
    if n < 4:
        return None, False
    x = signal.to(torch.float32)
    ll1 = _gaussian_loglik_1d(x)
    bic1 = 2.0 * math.log(n) - 2.0 * float(ll1)
    _, _, _, ll2 = _gmm_em_1d(x)
    # K=2 has 5 free params: 2 means, 2 vars, 1 mixture weight.
    bic2 = 5.0 * math.log(n) - 2.0 * float(ll2)
    delta_bic = bic1 - bic2
    return float(delta_bic), bool(delta_bic > delta_threshold)


class PersistentBimodalityTracker:
    def __init__(self, maxlen: int = 5):
        self.maxlen = maxlen
        self._hist: Dict[int, Deque[bool]] = {}

    def append(self, atom: int, rejected: bool) -> None:
        if atom not in self._hist:
            self._hist[atom] = deque(maxlen=self.maxlen)
        self._hist[atom].append(bool(rejected))

    def count(self, atom: int) -> int:
        if atom not in self._hist:
            return 0
        return int(sum(1 for v in self._hist[atom] if v))

    def is_persistent(self, atom: int, threshold: int = 3) -> bool:
        return self.count(atom) >= threshold

    def atoms(self):
        return list(self._hist.keys())


def compute_bimodality_diagnostics(
    histories: Dict[int, ContextBagHistory],
    tracker: PersistentBimodalityTracker,
    *,
    use_gmm: bool = False,
) -> BimodalityDiagnostics:
    per_atom: Dict[int, PerAtomBimodalitySignal] = {}
    persistent_counts: Dict[int, int] = {}
    persistent_flags: Dict[int, bool] = {}
    for atom, hist in histories.items():
        sig = context_bag_signal(hist)
        if use_gmm:
            delta_bic, favors_k2 = gmm_bic_1d(sig)
            dip_p, dip_rej = None, False
            rejected_for_tracker = favors_k2
        else:
            dip_p, dip_rej = hartigan_dip(sig)
            delta_bic, favors_k2 = None, False
            rejected_for_tracker = dip_rej
        per_atom[int(atom)] = PerAtomBimodalitySignal(
            signal=sig,
            dip_p_value=dip_p,
            dip_rejects=dip_rej,
            delta_bic=delta_bic,
            bic_favors_k2=favors_k2,
        )
        tracker.append(int(atom), rejected_for_tracker)
        persistent_counts[int(atom)] = tracker.count(int(atom))
        persistent_flags[int(atom)] = tracker.is_persistent(int(atom))
    return BimodalityDiagnostics(
        per_atom_signal=per_atom,
        persistent_rejections=persistent_counts,
        persistent_bimodality=persistent_flags,
        n_atoms=len(per_atom),
    )
