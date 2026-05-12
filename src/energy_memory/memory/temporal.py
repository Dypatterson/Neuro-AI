"""Temporal association channel layered over content-addressable memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, TypeVar

from energy_memory.memory._math import normalized_entropy, softmax
from energy_memory.memory.hopfield import HopfieldMemory, RetrievalResult
from energy_memory.substrate import FHRR, Vector

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalRecallResult(Generic[T]):
    content: RetrievalResult[T]
    temporal_state: Vector
    temporal_items: List[tuple[T, float]]


@dataclass(frozen=True)
class JointTemporalRecallResult(Generic[T]):
    state: Vector
    weights: List[float]
    content_scores: List[float]
    temporal_scores: List[float]
    joint_scores: List[float]
    top_index: int
    top_label: T
    temporal_state: Vector
    temporal_items: List[tuple[T, float]]
    entropy: float


@dataclass(frozen=True)
class CoupledTemporalStep(Generic[T]):
    iteration: int
    top_label: T
    top_weight: float
    entropy: float
    top_joint_score: float


@dataclass(frozen=True)
class CoupledTemporalRecallResult(Generic[T]):
    state: Vector
    temporal_state: Vector
    weights: List[float]
    top_index: int
    top_label: T
    temporal_items: List[tuple[T, float]]
    trace: List[CoupledTemporalStep[T]]
    converged: bool


class TemporalAssociationMemory(Generic[T]):
    """Stores each item with a bundle of nearby temporal neighbors."""

    def __init__(self, substrate: FHRR, window: int = 2):
        if window <= 0:
            raise ValueError("window must be positive")
        self.substrate = substrate
        self.window = window
        self.content = HopfieldMemory[T](substrate)
        self.labels: List[T] = []
        self.vectors: List[Vector] = []
        self.temporal_contexts: List[Vector] = []

    def store_sequence(self, labels: Sequence[T], vectors: Sequence[Vector]) -> None:
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        if not labels:
            raise ValueError("cannot store an empty sequence")
        self.labels = list(labels)
        self.vectors = list(vectors)
        self.content = HopfieldMemory[T](self.substrate)
        self.temporal_contexts = []
        for i, vector in enumerate(vectors):
            neighbors = [
                vectors[j]
                for j in range(max(0, i - self.window), min(len(vectors), i + self.window + 1))
                if j != i
            ]
            context = self.substrate.bundle(neighbors if neighbors else [vector])
            self.temporal_contexts.append(context)
            self.content.store(vector, label=labels[i])

    def recall(
        self,
        query: Vector,
        beta: float = 8.0,
        max_iter: int = 10,
        top_k: int = 5,
    ) -> TemporalRecallResult[T]:
        content_result = self.content.retrieve(query, beta=beta, max_iter=max_iter)
        temporal_state = self._weighted_temporal_context(content_result.weights)
        codebook = dict(zip(self.labels, self.vectors))
        temporal_items = self.substrate.top_k(temporal_state, codebook, k=top_k)
        return TemporalRecallResult(
            content=content_result,
            temporal_state=temporal_state,
            temporal_items=temporal_items,
        )

    def joint_recall(
        self,
        content_query: Vector,
        temporal_query: Optional[Vector],
        content_beta: float = 8.0,
        temporal_beta: float = 8.0,
        top_k: int = 5,
    ) -> JointTemporalRecallResult[T]:
        """Retrieve using a joint content and temporal-context compatibility score.

        This is the first controller-free approximation of a coupled energy read:
        candidate anchors are favored when they both resemble the content query
        and have a temporal context compatible with the surrounding cue.
        """
        if not self.vectors:
            raise ValueError("cannot retrieve from an empty temporal memory")

        content_scores = [self.substrate.similarity(content_query, vector) for vector in self.vectors]
        if temporal_query is None:
            temporal_scores = [0.0 for _ in self.temporal_contexts]
        else:
            temporal_scores = [
                self.substrate.similarity(temporal_query, context) for context in self.temporal_contexts
            ]
        joint_scores = [
            content_beta * content_score + temporal_beta * temporal_score
            for content_score, temporal_score in zip(content_scores, temporal_scores)
        ]
        weights = softmax(joint_scores)
        state = self.substrate.weighted_bundle(self.vectors, weights)
        temporal_state = self._weighted_temporal_context(weights)
        codebook = dict(zip(self.labels, self.vectors))
        temporal_items = self.substrate.top_k(temporal_state, codebook, k=top_k)
        top_index = max(range(len(joint_scores)), key=lambda i: joint_scores[i])
        return JointTemporalRecallResult(
            state=state,
            weights=weights,
            content_scores=content_scores,
            temporal_scores=temporal_scores,
            joint_scores=joint_scores,
            top_index=top_index,
            top_label=self.labels[top_index],
            temporal_state=temporal_state,
            temporal_items=temporal_items,
            entropy=normalized_entropy(weights),
        )

    def coupled_recall(
        self,
        content_query: Vector,
        temporal_query: Vector,
        content_beta: float = 8.0,
        temporal_beta: float = 8.0,
        feedback: float = 0.75,
        max_iter: int = 8,
        tol: float = 1e-3,
        top_k: int = 5,
    ) -> CoupledTemporalRecallResult[T]:
        """Iteratively settle content and temporal context together.

        The temporal cue is not just an input; it is updated by the temporal
        state implied by the current anchor distribution. This gives the system
        a trajectory we can inspect instead of a single scoring pass.
        """
        if not 0.0 <= feedback <= 1.0:
            raise ValueError("feedback must be between 0 and 1")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        current_temporal_query = temporal_query
        last_weights: Optional[List[float]] = None
        trace: List[CoupledTemporalStep[T]] = []
        weights: List[float] = []
        state = content_query
        temporal_state = temporal_query
        top_index = 0

        for iteration in range(1, max_iter + 1):
            result = self.joint_recall(
                content_query,
                current_temporal_query,
                content_beta=content_beta,
                temporal_beta=temporal_beta,
                top_k=top_k,
            )
            weights = result.weights
            state = result.state
            temporal_state = result.temporal_state
            top_index = result.top_index
            top_weight = weights[top_index]
            trace.append(
                CoupledTemporalStep(
                    iteration=iteration,
                    top_label=result.top_label,
                    top_weight=top_weight,
                    entropy=result.entropy,
                    top_joint_score=result.joint_scores[top_index],
                )
            )

            weight_delta = None
            if last_weights is not None:
                weight_delta = max(abs(a - b) for a, b in zip(weights, last_weights))
            last_weights = list(weights)
            if weight_delta is not None and weight_delta < tol:
                break

            current_temporal_query = self.substrate.weighted_bundle(
                [temporal_query, temporal_state],
                [1.0 - feedback, feedback],
            )

        codebook = dict(zip(self.labels, self.vectors))
        temporal_items = self.substrate.top_k(temporal_state, codebook, k=top_k)
        return CoupledTemporalRecallResult(
            state=state,
            temporal_state=temporal_state,
            weights=weights,
            top_index=top_index,
            top_label=self.labels[top_index],
            temporal_items=temporal_items,
            trace=trace,
            converged=len(trace) < max_iter,
        )

    def _weighted_temporal_context(self, weights: Iterable[float]) -> Vector:
        return self.substrate.weighted_bundle(self.temporal_contexts, list(weights))
