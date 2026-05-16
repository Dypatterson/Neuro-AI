"""Trajectory trace for Hopfield settling.

Captures the *path* of co-activations across iteration steps during
Hopfield retrieval — not just the endpoint. Per the 2026-05-02 design,
this trajectory is first-class information that supports:

  - engagement detection (was the landscape exerting force across many
    patterns?)
  - re-settling during replay (re-run the same query through the current
    landscape, possibly with updated codebook)
  - co-activation analysis (which patterns kept appearing together?)

The trace is a passive observer of settling — it does not influence the
dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory._torch_math import (
    torch_normalized_entropy,
    torch_normalized_entropy_tensor as _torch_entropy_tensor,
)
from energy_memory.memory.torch_hopfield import (
    TorchHopfieldMemory,
    TorchRetrievalResult,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

T = TypeVar("T")


@dataclass(frozen=True)
class TrajectorySnapshot:
    """One settling step's co-activation snapshot."""

    step: int
    top_k_indices: List[int]
    top_k_weights: List[float]
    entropy: float
    energy: float


@dataclass
class TrajectoryTrace:
    """Full settling trajectory: query, per-step snapshots, final state.

    Sufficient information to:
      - compute engagement (mean entropy across snapshots)
      - re-settle (using the stored query) through an updated landscape
      - inspect co-activation structure (which patterns competed when)
    """

    query: "torch.Tensor"
    snapshots: List[TrajectorySnapshot] = field(default_factory=list)
    final_state: Optional["torch.Tensor"] = None
    final_top_score: float = 0.0
    final_top_index: Optional[int] = None
    converged: bool = False
    age: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.snapshots)

    def engagement(self) -> float:
        """Mean softmax entropy across settling steps.

        High engagement = the landscape exerted force across multiple
        patterns (many active simultaneously). Low engagement = single-
        basin lock-in.
        """
        if not self.snapshots:
            return 0.0
        return sum(s.entropy for s in self.snapshots) / len(self.snapshots)

    def resolution(self) -> float:
        """Max cosine similarity of settled state to any stored pattern.

        High resolution = clean basin lock (state is a stored pattern).
        Low resolution = state is a blended/metastable mixture.
        """
        return self.final_top_score

    def gate_signal(self) -> float:
        """The replay-store gate signal: engagement × (1 - resolution).

        High = unresolved trajectory with significant landscape force.
        Both axes must fire for the gate to trigger.
        """
        return self.engagement() * (1.0 - self.resolution())


class TracedHopfieldMemory(TorchHopfieldMemory, Generic[T]):
    """Hopfield memory that captures trajectory traces during retrieval.

    Subclass of TorchHopfieldMemory. The standard `retrieve()` method is
    unchanged; use `retrieve_with_trace()` to also obtain the trajectory.
    """

    def __init__(
        self,
        substrate: TorchFHRR,
        snapshot_k: int = 8,
        snapshot_min_weight: float = 0.01,
    ):
        super().__init__(substrate)
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("TracedHopfieldMemory requires torch") from _IMPORT_ERROR
        self.snapshot_k = snapshot_k
        self.snapshot_min_weight = snapshot_min_weight

    def retrieve_with_trace(
        self,
        query,
        beta: float = 8.0,
        max_iter: int = 10,
        tol: float = 1e-8,
        score_bias: Optional["torch.Tensor"] = None,
    ) -> tuple[TorchRetrievalResult[T], TrajectoryTrace]:
        """Settle the memory toward an attractor, returning state + trajectory.

        ``score_bias`` (optional): a [n_patterns] tensor that is *subtracted*
        from ``beta * scores`` per pattern before the softmax. Used to wire
        in Saighi-style per-pattern self-inhibition (A_k). When None or
        all-zero, retrieval behaves identically to the no-inhibition baseline.
        """
        if not self._patterns:
            raise ValueError("cannot retrieve from an empty Hopfield memory")
        if beta <= 0.0:
            raise ValueError("beta must be positive")

        patterns = self._pattern_matrix()
        state = query.to(self.substrate.device)
        bias = None
        if score_bias is not None:
            bias = score_bias.to(self.substrate.device)
            if bias.shape != (patterns.shape[0],):
                raise ValueError(
                    f"score_bias must have shape ({patterns.shape[0]},), "
                    f"got {tuple(bias.shape)}"
                )

        # Same deferred-sync pattern as TorchHopfieldMemory.retrieve, with
        # the converged state captured on-device via torch.where (no list of
        # max_iter intermediate states — see CLAUDE.md "GPU performance rule
        # of thumb"). Snapshot tensors per iter are tiny (k=8 floats/ints) so
        # we keep the simple list of those.
        n_patterns = patterns.shape[0]
        k = min(self.snapshot_k, n_patterns)
        device = self.substrate.device

        energy_tensors: List["torch.Tensor"] = []
        snapshot_topk_indices: List["torch.Tensor"] = []
        snapshot_topk_values: List["torch.Tensor"] = []
        snapshot_entropies: List["torch.Tensor"] = []
        frozen = torch.zeros((), dtype=torch.bool, device=device)
        final_state = state
        prev_energy: Optional["torch.Tensor"] = None

        for _ in range(max_iter):
            scores = self._scores(state, patterns)
            energy_t = self._energy_from_scores(scores, beta=beta, kernel="softmax")
            logits = beta * scores
            if bias is not None:
                logits = logits - bias
            weights = torch.softmax(logits, dim=0)
            next_state = self.substrate.normalize(
                (patterns * weights[:, None]).sum(dim=0),
            )
            top_values, top_indices = torch.topk(weights, k)

            energy_tensors.append(energy_t)
            snapshot_topk_indices.append(top_indices)
            snapshot_topk_values.append(top_values)
            snapshot_entropies.append(_torch_entropy_tensor(weights))
            if prev_energy is not None:
                converged_now = ((energy_t - prev_energy).abs() < tol) & (~frozen)
                final_state = torch.where(converged_now, next_state, final_state)
                frozen = frozen | converged_now
            prev_energy = energy_t
            state = next_state
        final_state = torch.where(frozen, final_state, state)

        # Single batched sync.
        energies = torch.stack(energy_tensors).detach().cpu().tolist()
        entropies = torch.stack(snapshot_entropies).detach().cpu().tolist()
        topk_idx_cpu = torch.stack(snapshot_topk_indices).detach().cpu().tolist()
        topk_val_cpu = torch.stack(snapshot_topk_values).detach().cpu().tolist()

        converged = False
        converged_iter = max_iter
        for kk in range(1, max_iter):
            if abs(energies[kk] - energies[kk - 1]) < tol:
                converged = True
                converged_iter = kk + 1
                break
        energy_trace = energies[:converged_iter]
        state = final_state

        snapshots: List[TrajectorySnapshot] = []
        for step_i in range(converged_iter):
            values = topk_val_cpu[step_i]
            indices = topk_idx_cpu[step_i]
            kept_indices = [idx for idx, v in zip(indices, values)
                            if v >= self.snapshot_min_weight]
            kept_values = [v for v in values if v >= self.snapshot_min_weight]
            snapshots.append(TrajectorySnapshot(
                step=step_i + 1,
                top_k_indices=kept_indices,
                top_k_weights=kept_values,
                entropy=float(entropies[step_i]),
                energy=float(energies[step_i]),
            ))

        final_scores = self._scores(state, patterns)
        final_weights = torch.softmax(beta * final_scores, dim=0)
        top_index = int(torch.argmax(final_scores).detach().cpu())
        top_label = self.labels[top_index] if self.labels else None
        top_score = float(final_scores[top_index].detach().cpu())

        result = TorchRetrievalResult(
            state=state,
            weights=final_weights.detach().cpu().tolist(),
            scores=final_scores.detach().cpu().tolist(),
            top_index=top_index,
            top_label=top_label,
            top_score=top_score,
            entropy=torch_normalized_entropy(final_weights),
            energy_trace=energy_trace,
            iterations=len(energy_trace),
            converged=converged,
        )

        trace = TrajectoryTrace(
            query=query.detach().clone(),
            snapshots=snapshots,
            final_state=state.detach().clone(),
            final_top_score=top_score,
            final_top_index=top_index,
            converged=converged,
        )

        return result, trace

    def _build_snapshot(
        self,
        step: int,
        weights: "torch.Tensor",
        energy: float,
    ) -> TrajectorySnapshot:
        n_patterns = weights.shape[0]
        k = min(self.snapshot_k, n_patterns)
        top_values, top_indices = torch.topk(weights, k)
        keep = top_values >= self.snapshot_min_weight
        kept_indices = top_indices[keep].detach().cpu().tolist()
        kept_weights = top_values[keep].detach().cpu().tolist()
        entropy = torch_normalized_entropy(weights)
        return TrajectorySnapshot(
            step=step,
            top_k_indices=kept_indices,
            top_k_weights=kept_weights,
            entropy=float(entropy),
            energy=float(energy),
        )
