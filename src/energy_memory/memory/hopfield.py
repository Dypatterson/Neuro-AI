"""Modern Hopfield-style associative retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from energy_memory.memory._math import logsumexp, normalized_entropy, softmax
from energy_memory.substrate import FHRR, Vector

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalResult(Generic[T]):
    state: Vector
    weights: List[float]
    scores: List[float]
    top_index: int
    top_label: Optional[T]
    top_score: float
    entropy: float
    energy_trace: List[float]
    iterations: int
    converged: bool


class HopfieldMemory(Generic[T]):
    """Stores vectors as attractor patterns and retrieves by iterative settling."""

    def __init__(self, substrate: FHRR):
        self.substrate = substrate
        self.patterns: List[Vector] = []
        self.labels: List[T] = []

    @property
    def stored_count(self) -> int:
        return len(self.patterns)

    def store(self, pattern: Vector, label: Optional[T] = None) -> None:
        self.substrate._check_dim(pattern)
        self.patterns.append(pattern)
        self.labels.append(label)  # type: ignore[arg-type]

    def retrieve(
        self,
        query: Vector,
        beta: float = 8.0,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> RetrievalResult[T]:
        if not self.patterns:
            raise ValueError("cannot retrieve from an empty Hopfield memory")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        state = query
        energy_trace: List[float] = []
        converged = False

        for iteration in range(1, max_iter + 1):
            scores = self._scores(state)
            weights = softmax([beta * score for score in scores])
            next_state = self.substrate.weighted_bundle(self.patterns, weights)
            energy = self.energy(state, beta=beta)
            energy_trace.append(energy)
            if iteration > 1 and abs(energy_trace[-1] - energy_trace[-2]) < tol:
                converged = True
                state = next_state
                break
            state = next_state

        final_scores = self._scores(state)
        final_weights = softmax([beta * score for score in final_scores])
        top_index = max(range(len(final_scores)), key=lambda i: final_scores[i])
        entropy = normalized_entropy(final_weights)
        top_label = self.labels[top_index] if self.labels else None
        return RetrievalResult(
            state=state,
            weights=final_weights,
            scores=final_scores,
            top_index=top_index,
            top_label=top_label,
            top_score=final_scores[top_index],
            entropy=entropy,
            energy_trace=energy_trace,
            iterations=len(energy_trace),
            converged=converged,
        )

    def energy(self, state: Vector, beta: float = 8.0) -> float:
        scores = self._scores(state)
        scaled = [beta * score for score in scores]
        return -logsumexp(scaled) / beta

    def _scores(self, state: Vector) -> List[float]:
        return [self.substrate.similarity(pattern, state) for pattern in self.patterns]

