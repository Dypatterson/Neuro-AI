"""Optional Torch-backed Hopfield retrieval.

This mirrors the pure-Python Hopfield memory so Phase 2 style retrieval can run
on CPU or MPS without changing the retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory._torch_math import torch_normalized_entropy
from energy_memory.substrate.torch_fhrr import TorchFHRR

T = TypeVar("T")


@dataclass(frozen=True)
class TorchRetrievalResult(Generic[T]):
    state: "torch.Tensor"
    weights: List[float]
    scores: List[float]
    top_index: int
    top_label: Optional[T]
    top_score: float
    entropy: float
    energy_trace: List[float]
    iterations: int
    converged: bool


class TorchHopfieldMemory(Generic[T]):
    """Stores Torch patterns and retrieves with iterative softmax settling."""

    def __init__(self, substrate: TorchFHRR):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("TorchHopfieldMemory requires torch") from _IMPORT_ERROR
        self.substrate = substrate
        self._patterns: List["torch.Tensor"] = []
        self.labels: List[T] = []
        self._cached_matrix: Optional["torch.Tensor"] = None

    @property
    def stored_count(self) -> int:
        return len(self._patterns)

    def store(self, pattern, label: Optional[T] = None) -> None:
        if pattern.shape[-1] != self.substrate.dim:
            raise ValueError(f"expected dimension {self.substrate.dim}, got {pattern.shape[-1]}")
        self._patterns.append(pattern.to(self.substrate.device))
        self.labels.append(label)  # type: ignore[arg-type]
        self._cached_matrix = None

    def invalidate_cache(self) -> None:
        """Drop the cached pattern matrix; force a rebuild on next retrieval.

        Call after any direct mutation of `_patterns` or `labels` that
        bypasses `store()` / `remove_pattern()`.
        """
        self._cached_matrix = None

    def remove_pattern(self, idx: int) -> None:
        """Remove the pattern at `idx` and invalidate the cache.

        Keeps `_patterns`/`labels` in sync. Callers must also keep any
        external per-pattern bookkeeping (consolidation state, source
        windows) aligned.
        """
        self._patterns.pop(idx)
        if idx < len(self.labels):
            self.labels.pop(idx)
        self._cached_matrix = None

    def retrieve(
        self,
        query,
        beta: float = 8.0,
        max_iter: int = 10,
        tol: float = 1e-8,
        kernel: str = "softmax",
    ) -> TorchRetrievalResult[T]:
        if not self._patterns:
            raise ValueError("cannot retrieve from an empty Hopfield memory")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if kernel not in ("softmax", "lsr"):
            raise ValueError(f"unknown kernel {kernel!r}; expected 'softmax' or 'lsr'")
        patterns = self._pattern_matrix()
        state = query.to(self.substrate.device)

        # Run all max_iter iterations on-device without syncing, then replay
        # the convergence check from a single batched CPU sync at the end.
        # The converged state is captured on-device via torch.where against a
        # frozen mask, so we don't hold max_iter intermediate states in memory
        # (that ballooned the async-queue residency on MPS, see CLAUDE.md
        # "GPU performance rule of thumb").
        energy_tensors: List["torch.Tensor"] = []
        device = self.substrate.device
        frozen = torch.zeros((), dtype=torch.bool, device=device)
        final_state = state
        prev_energy: Optional["torch.Tensor"] = None
        for _ in range(max_iter):
            scores = self._scores(state, patterns)
            energy = self._energy_from_scores(scores, beta=beta, kernel=kernel)
            energy_tensors.append(energy)
            weights = self._weights(scores, beta=beta, kernel=kernel)
            next_state = self.substrate.normalize((patterns * weights[:, None]).sum(dim=0))
            if prev_energy is not None:
                converged_now = ((energy - prev_energy).abs() < tol) & (~frozen)
                final_state = torch.where(converged_now, next_state, final_state)
                frozen = frozen | converged_now
            prev_energy = energy
            state = next_state
        # If never frozen (never converged), fall through to the last next_state.
        final_state = torch.where(frozen, final_state, state)

        energies = torch.stack(energy_tensors).detach().cpu().tolist()
        converged = False
        converged_iter = max_iter
        for k in range(1, max_iter):
            if abs(energies[k] - energies[k - 1]) < tol:
                converged = True
                converged_iter = k + 1
                break
        energy_trace = energies[:converged_iter]
        state = final_state

        final_scores = self._scores(state, patterns)
        final_weights = self._weights(final_scores, beta=beta, kernel=kernel)
        top_index = int(torch.argmax(final_scores).detach().cpu())
        top_label = self.labels[top_index] if self.labels else None
        return TorchRetrievalResult(
            state=state,
            weights=final_weights.detach().cpu().tolist(),
            scores=final_scores.detach().cpu().tolist(),
            top_index=top_index,
            top_label=top_label,
            top_score=float(final_scores[top_index].detach().cpu()),
            entropy=torch_normalized_entropy(final_weights),
            energy_trace=energy_trace,
            iterations=len(energy_trace),
            converged=converged,
        )

    def energy(self, state, beta: float = 8.0, patterns=None, kernel: str = "softmax") -> float:
        pattern_matrix = self._pattern_matrix() if patterns is None else patterns
        score_tensor = self._scores(state, pattern_matrix)
        return float(self._energy_from_scores(score_tensor, beta=beta, kernel=kernel).detach().cpu())

    @staticmethod
    def _energy_from_scores(score_tensor, beta: float, kernel: str):
        """Compute energy directly from a pre-computed score tensor.

        Returns a 0-dim tensor (no CPU sync), so the retrieve loop can
        accumulate energies and sync them in a single batch.
        """
        if kernel == "softmax":
            return -torch.logsumexp(beta * score_tensor, dim=0) / beta
        if kernel == "lsr":
            # Lagrangian conjugate of the truncated-quadratic separation function:
            #   F(s) = (beta/2) * sum_i max(0, s_i)^2  -> E = -F.
            relu_scores = torch.clamp(score_tensor, min=0.0)
            return -0.5 * beta * (relu_scores * relu_scores).sum()
        raise ValueError(f"unknown kernel {kernel!r}")

    def _pattern_matrix(self):
        if self._cached_matrix is None:
            self._cached_matrix = torch.stack(self._patterns, dim=0)
        return self._cached_matrix

    def _scores(self, state, patterns):
        return self.substrate.similarity_matrix(state, patterns)

    @staticmethod
    def _weights(scores, beta: float, kernel: str):
        if kernel == "softmax":
            return torch.softmax(beta * scores, dim=0)
        if kernel == "lsr":
            relu_scores = torch.clamp(beta * scores, min=0.0)
            total = relu_scores.sum()
            if float(total.detach().cpu()) <= 0.0:
                # No pattern is in the support; fall back to a uniform mixture so
                # the iteration is well-defined (energy is 0 here, which is the
                # LSR ceiling — settling has no gradient to follow).
                return torch.full_like(scores, 1.0 / scores.numel())
            return relu_scores / total
        raise ValueError(f"unknown kernel {kernel!r}")
