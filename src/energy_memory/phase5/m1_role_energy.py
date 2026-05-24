"""Path D / M1 role-energy stack primitives.

This module is intentionally separate from ``experiments/40_phase5_branching.py``.
It provides the first implementation surface for the M1 path:

* S1 provenance consumption: encoder-time ``(role_index, atom_id)`` tuples.
* P1 per-role weighted Modern Hopfield energy.
* D3 additive cross-K softmax in the Lyapunov-clean form.
* P3 optional centered role/content saliency as a fixed logit contribution.

The code is substrate-level plumbing and settling dynamics. It does not make
graduation claims; report scripts decide which controls to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.phase4.trajectory import TrajectoryTrace
from energy_memory.substrate.torch_fhrr import TorchFHRR


EncoderTerm = Tuple[int, int]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("M1 role-energy stack requires torch") from _IMPORT_ERROR


@dataclass(frozen=True)
class M1Config:
    beta: float = 10.0
    max_iter: int = 12
    tol: float = 1e-8
    d3_mix: float = 0.5
    p3_saliency_gain: float = 0.0
    laplace_count: float = 1.0
    normalize_weighted_rows: bool = False


@dataclass(frozen=True)
class S2CheckResult:
    passed: bool
    energy_trace: List[float]
    final_magnitude: float
    max_energy_increase: float


@dataclass
class M1BranchTelemetry:
    role_index: int
    state: "torch.Tensor"
    energy: float
    top_index: int
    top_score: float
    energy_trace: List[float] = field(default_factory=list)


@dataclass
class M1Result:
    branches: List[M1BranchTelemetry]
    joint_energy_trace: List[float]


@dataclass
class RoleBindingStats:
    """Counts and normalized weights for atom-to-role attribution."""

    counts: "torch.Tensor"  # [n_atoms, n_roles]

    @classmethod
    def empty(
        cls,
        n_atoms: int,
        n_roles: int,
        *,
        device: Optional[str] = None,
    ) -> "RoleBindingStats":
        _require_torch()
        if n_atoms <= 0:
            raise ValueError("n_atoms must be positive")
        if n_roles <= 0:
            raise ValueError("n_roles must be positive")
        return cls(torch.zeros((n_atoms, n_roles), dtype=torch.float32, device=device or "cpu"))

    @classmethod
    def from_encoder_terms(
        cls,
        term_lists: Iterable[Sequence[EncoderTerm]],
        *,
        n_atoms: int,
        n_roles: int,
        device: Optional[str] = None,
    ) -> "RoleBindingStats":
        stats = cls.empty(n_atoms=n_atoms, n_roles=n_roles, device=device)
        for terms in term_lists:
            stats.add_terms(terms)
        return stats

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[TrajectoryTrace],
        *,
        n_atoms: int,
        n_roles: int,
        device: Optional[str] = None,
        require_complete: bool = True,
    ) -> "RoleBindingStats":
        stats = cls.empty(n_atoms=n_atoms, n_roles=n_roles, device=device)
        missing = 0
        for trace in traces:
            if trace.encoder_terms is None:
                missing += 1
                continue
            stats.add_terms(trace.encoder_terms)
        if require_complete and missing:
            raise ValueError(f"{missing} traces are missing encoder_terms provenance")
        return stats

    @classmethod
    def from_pattern_encoder_terms(
        cls,
        pattern_encoder_terms: Sequence[Optional[Sequence[EncoderTerm]]],
        *,
        n_roles: int,
        device: Optional[str] = None,
        require_complete: bool = True,
    ) -> "RoleBindingStats":
        """Build row-role counts from per-pattern encoder provenance.

        ``encode_window_with_provenance`` records ``(role_index, token_id)``.
        M1 retrieves over pattern rows, not token-codebook rows, so this
        adapter intentionally uses the outer list index as the atom row and
        only uses each term's role index as the count source.
        """
        stats = cls.empty(
            n_atoms=len(pattern_encoder_terms), n_roles=n_roles, device=device,
        )
        missing = 0
        for atom_row, terms in enumerate(pattern_encoder_terms):
            if terms is None:
                missing += 1
                continue
            for role_index, _token_id in terms:
                if not 0 <= int(role_index) < stats.n_roles:
                    raise IndexError(f"role_index {role_index} out of range")
                stats.counts[atom_row, int(role_index)] += 1.0
        if require_complete and missing:
            raise ValueError(f"{missing} pattern rows are missing encoder provenance")
        return stats

    @property
    def n_atoms(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_roles(self) -> int:
        return int(self.counts.shape[1])

    def add_terms(self, terms: Sequence[EncoderTerm]) -> None:
        for role_index, atom_id in terms:
            if not 0 <= int(role_index) < self.n_roles:
                raise IndexError(f"role_index {role_index} out of range")
            if not 0 <= int(atom_id) < self.n_atoms:
                raise IndexError(f"atom_id {atom_id} out of range")
            self.counts[int(atom_id), int(role_index)] += 1.0

    def atom_role_weights(self, laplace: float = 1.0) -> "torch.Tensor":
        """Return ``w_{i,r}``, normalized over roles for every atom."""
        c = self.counts
        if laplace < 0.0:
            raise ValueError("laplace must be non-negative")
        smoothed = c + float(laplace)
        denom = smoothed.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return smoothed / denom

    @staticmethod
    def geometric_row_role_scores(
        substrate: TorchFHRR,
        patterns: "torch.Tensor",
        role_vectors: Sequence["torch.Tensor"],
        *,
        mode: str = "unbind_density",
        neighbor_k: int = 8,
    ) -> "torch.Tensor":
        """Return non-negative row-role scores from substrate geometry.

        Count provenance records how a stored row was encoded. For full
        windows, that record is intentionally complete and therefore uniform
        over roles. This geometric adapter asks a different, row-domain
        question over the exact matrix M1 retrieves from: after unbinding a
        pattern row by a role vector, does the recovered filler live in a
        locally dense part of the full unbound filler population? Same-role
        row geometry alone is still an isometry of the pattern matrix; the
        cross-role filler population is what makes the role-local signal
        inspectable without consulting token ids.
        """
        _require_torch()
        if patterns.ndim != 2:
            raise ValueError("patterns must be a [N, D] tensor")
        if len(role_vectors) == 0:
            raise ValueError("role_vectors must not be empty")
        n_patterns = int(patterns.shape[0])
        if n_patterns == 0:
            raise ValueError("patterns must contain at least one row")
        if neighbor_k <= 0:
            raise ValueError("neighbor_k must be positive")

        device = patterns.device
        roles = [role.to(device) for role in role_vectors]

        if mode == "unbind_norm":
            rows = [
                substrate.unbind(patterns, role).norm(dim=1)
                for role in roles
            ]
            return torch.stack(rows, dim=1).to(torch.float32).clamp(min=0.0)
        if mode != "unbind_density":
            raise ValueError("mode must be 'unbind_density' or 'unbind_norm'")

        if n_patterns == 1:
            return torch.ones(
                (1, len(roles)), dtype=torch.float32, device=device,
            )

        n_roles = len(roles)
        k = min(int(neighbor_k), n_patterns * n_roles - 1)
        fillers_by_role = torch.stack(
            [substrate.unbind(patterns, role) for role in roles],
            dim=1,
        )
        flat_fillers = fillers_by_role.reshape(n_patterns * n_roles, patterns.shape[1])
        flat_norms = flat_fillers.norm(dim=1).clamp(min=1e-12)
        role_scores = []
        row_indices = torch.arange(n_patterns, device=device)
        for role_idx in range(n_roles):
            fillers = fillers_by_role[:, role_idx, :]
            gram = (fillers @ flat_fillers.conj().transpose(0, 1)).real
            norms = fillers.norm(dim=1).clamp(min=1e-12)
            sims = gram / (norms[:, None] * flat_norms[None, :])
            sims = sims.clamp(min=0.0)
            sims = sims.clone()
            sims[row_indices, row_indices * n_roles + role_idx] = 0.0
            topk = torch.topk(sims, k=k, dim=1).values
            role_scores.append(topk.mean(dim=1))
        return torch.stack(role_scores, dim=1).to(torch.float32).clamp(min=0.0)

    @staticmethod
    def geometric_row_role_weights(
        substrate: TorchFHRR,
        patterns: "torch.Tensor",
        role_vectors: Sequence["torch.Tensor"],
        *,
        mode: str = "unbind_density",
        neighbor_k: int = 8,
        laplace: float = 1e-6,
        temperature: Optional[float] = 0.05,
    ) -> "torch.Tensor":
        """Return row-normalized geometric role weights for M1."""
        if laplace < 0.0:
            raise ValueError("laplace must be non-negative")
        if temperature is not None and temperature <= 0.0:
            raise ValueError("temperature must be positive")
        scores = RoleBindingStats.geometric_row_role_scores(
            substrate,
            patterns,
            role_vectors,
            mode=mode,
            neighbor_k=neighbor_k,
        )
        if temperature is not None:
            return torch.softmax(scores / float(temperature), dim=1)
        smoothed = scores + float(laplace)
        denom = smoothed.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return smoothed / denom

    def outer_role_weights(self, laplace: float = 1.0) -> "torch.Tensor":
        """Return global role weights normalized over roles."""
        if laplace < 0.0:
            raise ValueError("laplace must be non-negative")
        totals = self.counts.sum(dim=0) + float(laplace)
        return totals / totals.sum().clamp(min=1e-12)


def weighted_patterns(
    substrate: TorchFHRR,
    patterns: "torch.Tensor",
    weights: "torch.Tensor",
    *,
    normalize_rows: bool = False,
) -> "torch.Tensor":
    """Apply P1's in-LSE per-role row weights to a pattern matrix."""
    _require_torch()
    if weights.shape != (patterns.shape[0],):
        raise ValueError(
            f"weights must have shape ({patterns.shape[0]},), got {tuple(weights.shape)}"
        )
    wp = patterns * weights.to(patterns.device).to(patterns.real.dtype)[:, None]
    if not normalize_rows:
        return wp
    row_mag = wp.abs().mean(dim=1)
    keep = row_mag > 1e-12
    if bool(keep.any().detach().cpu()):
        wp = wp.clone()
        wp[keep] = substrate.normalize(wp[keep])
    return wp


def mhn_energy(
    substrate: TorchFHRR,
    state: "torch.Tensor",
    patterns: "torch.Tensor",
    beta: float,
    *,
    score_bias: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    scores = substrate.similarity_matrix(state, patterns)
    logits = beta * scores
    if score_bias is not None:
        logits = logits + score_bias.to(logits.device)
    return -torch.logsumexp(logits, dim=0) / beta


def settle_weighted_mhn(
    substrate: TorchFHRR,
    init_state: "torch.Tensor",
    patterns: "torch.Tensor",
    role_weights: "torch.Tensor",
    *,
    beta: float = 10.0,
    max_iter: int = 12,
    tol: float = 1e-8,
    score_bias: Optional["torch.Tensor"] = None,
    normalize_weighted_rows: bool = False,
) -> Tuple["torch.Tensor", List[float]]:
    """Settle one branch under a weighted per-role MHN energy."""
    _require_torch()
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    wp = weighted_patterns(
        substrate, patterns, role_weights.to(patterns.device),
        normalize_rows=normalize_weighted_rows,
    )
    state = init_state.to(patterns.device).clone()
    trace: List[float] = []
    prev_energy: Optional[float] = None
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, wp)
        logits = beta * scores
        if score_bias is not None:
            logits = logits + score_bias.to(logits.device)
        weights = torch.softmax(logits, dim=0)
        update = (wp * weights.to(wp.dtype)[:, None]).sum(dim=0)
        state = substrate.normalize(update)
        energy = float(mhn_energy(substrate, state, wp, beta, score_bias=score_bias).detach().cpu())
        trace.append(energy)
        if prev_energy is not None and abs(energy - prev_energy) < tol:
            break
        prev_energy = energy
    return state, trace


def centered_idp_saliency(
    substrate: TorchFHRR,
    cue: "torch.Tensor",
    patterns: "torch.Tensor",
    role_vectors: Sequence["torch.Tensor"],
) -> "torch.Tensor":
    """P3 fixed role/content saliency field, centered over atoms.

    The field is a deterministic energy contribution from the cue and
    substrate. It is not used to select a subsystem or early-exit a branch.
    """
    _require_torch()
    if not role_vectors:
        return torch.zeros(patterns.shape[0], dtype=torch.float32, device=patterns.device)
    role_scores = []
    for role in role_vectors:
        slot_query = substrate.unbind(cue, role.to(patterns.device))
        role_scores.append(substrate.similarity_matrix(slot_query, patterns))
    role_score = torch.stack(role_scores, dim=0).mean(dim=0)
    content_score = substrate.similarity_matrix(cue, patterns)

    def zscore(x):
        return (x - x.mean()) / x.std(unbiased=False).clamp(min=1e-6)

    field = zscore(role_score) - zscore(content_score)
    return field - field.mean()


def d3_additive_cross_k_settle(
    substrate: TorchFHRR,
    init_states: Sequence["torch.Tensor"],
    patterns: "torch.Tensor",
    branch_role_weights: "torch.Tensor",
    *,
    beta: float = 10.0,
    max_iter: int = 12,
    mix: float = 0.5,
    branch_score_bias: Optional["torch.Tensor"] = None,
    normalize_weighted_rows: bool = False,
    return_branch_traces: bool = False,
) -> (
    Tuple[List["torch.Tensor"], List[float]]
    | Tuple[List["torch.Tensor"], List[float], List[List[float]]]
):
    """Settle K branches with D3's additive cross-K softmax update."""
    _require_torch()
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be in [0, 1]")
    if len(init_states) == 0:
        raise ValueError("at least one branch is required")
    k_branches = len(init_states)
    if branch_role_weights.shape != (k_branches, patterns.shape[0]):
        raise ValueError(
            "branch_role_weights must have shape "
            f"({k_branches}, {patterns.shape[0]})"
        )
    states = torch.stack([s.to(patterns.device) for s in init_states], dim=0).clone()
    weighted_by_branch = torch.stack([
        weighted_patterns(
            substrate,
            patterns,
            branch_role_weights[k].to(patterns.device),
            normalize_rows=normalize_weighted_rows,
        )
        for k in range(k_branches)
    ], dim=0)
    joint_energy_trace: List[float] = []
    branch_energy_history: List["torch.Tensor"] = []
    for _ in range(max_iter):
        score_rows = []
        for k in range(k_branches):
            score_rows.append(substrate.similarity_matrix(states[k], weighted_by_branch[k]))
        scores = torch.stack(score_rows, dim=0)
        logits = beta * scores
        if branch_score_bias is not None:
            logits = logits + branch_score_bias.to(logits.device)
        if return_branch_traces:
            branch_energy_history.append(
                (-torch.logsumexp(logits, dim=1) / beta).detach()
            )
        pi = torch.softmax(logits, dim=1)
        alpha = torch.softmax(logits, dim=0)
        combined = (1.0 - mix) * pi + mix * alpha
        updates = []
        for k in range(k_branches):
            update = (
                weighted_by_branch[k]
                * combined[k].to(weighted_by_branch.dtype)[:, None]
            ).sum(dim=0)
            updates.append(substrate.normalize(update))
        states = torch.stack(updates, dim=0)
        joint_energy_trace.append(_joint_d3_energy(logits, beta, mix))
    if return_branch_traces:
        if branch_energy_history:
            branch_matrix = torch.stack(branch_energy_history, dim=1)
            branch_energy_traces = [
                [float(x) for x in row]
                for row in branch_matrix.detach().cpu().tolist()
            ]
        else:
            branch_energy_traces = [[] for _ in range(k_branches)]
        return [states[k] for k in range(k_branches)], joint_energy_trace, branch_energy_traces
    return [states[k] for k in range(k_branches)], joint_energy_trace


def _joint_d3_energy(logits: "torch.Tensor", beta: float, mix: float) -> float:
    per_branch = -torch.logsumexp(logits, dim=1).sum() / beta
    cross_k = -torch.logsumexp(logits, dim=0).sum() / beta
    energy = (1.0 - mix) * per_branch + mix * cross_k
    return float(energy.detach().cpu())


def run_m1_stack(
    substrate: TorchFHRR,
    cue: "torch.Tensor",
    patterns: "torch.Tensor",
    role_vectors: Sequence["torch.Tensor"],
    atom_role_weights: "torch.Tensor",
    *,
    branch_roles: Optional[Sequence[int]] = None,
    config: M1Config = M1Config(),
) -> M1Result:
    """Run the P1 + D3 + P3 M1 stack for one cue."""
    _require_torch()
    if not role_vectors:
        raise ValueError("role_vectors must not be empty")
    n_roles = len(role_vectors)
    if atom_role_weights.shape != (patterns.shape[0], n_roles):
        raise ValueError(
            "atom_role_weights must have shape "
            f"({patterns.shape[0]}, {n_roles})"
        )
    roles = list(range(n_roles)) if branch_roles is None else [int(r) for r in branch_roles]
    for r in roles:
        if not 0 <= r < n_roles:
            raise IndexError(f"branch role {r} out of range")

    init_states = [
        substrate.unbind(cue.to(patterns.device), role_vectors[r].to(patterns.device))
        for r in roles
    ]
    branch_weights = torch.stack([
        atom_role_weights[:, r].to(patterns.device)
        for r in roles
    ], dim=0)

    branch_bias = None
    if config.p3_saliency_gain != 0.0:
        field = centered_idp_saliency(substrate, cue, patterns, role_vectors)
        branch_bias = config.p3_saliency_gain * field[None, :].repeat(len(roles), 1)

    states, joint_trace, branch_energy_traces = d3_additive_cross_k_settle(
        substrate,
        init_states,
        patterns,
        branch_weights,
        beta=config.beta,
        max_iter=config.max_iter,
        mix=config.d3_mix,
        branch_score_bias=branch_bias,
        normalize_weighted_rows=config.normalize_weighted_rows,
        return_branch_traces=True,
    )

    branches: List[M1BranchTelemetry] = []
    for role, state, energy_trace in zip(roles, states, branch_energy_traces):
        wp = weighted_patterns(
            substrate,
            patterns,
            atom_role_weights[:, role].to(patterns.device),
            normalize_rows=config.normalize_weighted_rows,
        )
        scores = substrate.similarity_matrix(state, wp)
        energy = float(mhn_energy(substrate, state, wp, config.beta).detach().cpu())
        top_index = int(torch.argmax(scores).detach().cpu())
        top_score = float(scores[top_index].detach().cpu())
        branches.append(M1BranchTelemetry(
            role_index=role,
            state=state,
            energy=energy,
            top_index=top_index,
            top_score=top_score,
            energy_trace=energy_trace,
        ))
    return M1Result(branches=branches, joint_energy_trace=joint_trace)


def run_s2_weighted_mhn_check(
    *,
    seed: int = 17,
    dim: int = 512,
    n_atoms: int = 48,
    n_roles: int = 4,
    beta: float = 10.0,
    max_iter: int = 12,
) -> S2CheckResult:
    """Synthetic S2 check for weighted MHN energy and basin magnitude."""
    _require_torch()
    substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")
    patterns = substrate.random_vectors(n_atoms)
    counts = torch.ones((n_atoms, n_roles), dtype=torch.float32)
    for i in range(n_atoms):
        counts[i, i % n_roles] += 4.0
    stats = RoleBindingStats(counts)
    weights = stats.atom_role_weights(laplace=0.0)[:, 0]
    query = substrate.perturb(patterns[0], noise=0.03)
    state, trace = settle_weighted_mhn(
        substrate,
        query,
        patterns,
        weights,
        beta=beta,
        max_iter=max_iter,
    )
    increases = [
        trace[i] - trace[i - 1]
        for i in range(1, len(trace))
    ]
    max_inc = max(increases) if increases else 0.0
    final_mag = float(state.abs().mean().detach().cpu())
    passed = bool(trace and max_inc <= 1e-5 and 0.7 <= final_mag <= 1.3)
    return S2CheckResult(
        passed=passed,
        energy_trace=trace,
        final_magnitude=final_mag,
        max_energy_increase=max_inc,
    )
