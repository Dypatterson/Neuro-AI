"""Torch-backed temporal association memory.

This mirrors the reference temporal memory for the hot path that benefits from
matrix scoring on CPU/MPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory._torch_math import torch_normalized_entropy
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class TorchCoupledStep:
    iteration: int
    top_label: str
    top_weight: float
    entropy: float
    top_joint_score: float


@dataclass(frozen=True)
class TorchCoupledResult:
    top_label: str
    temporal_items: List[tuple[str, float]]
    trace: List[TorchCoupledStep]
    converged: bool


class TorchTemporalAssociationMemory:
    def __init__(self, substrate: TorchFHRR, window: int = 2):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("TorchTemporalAssociationMemory requires torch") from _IMPORT_ERROR
        if window <= 0:
            raise ValueError("window must be positive")
        self.substrate = substrate
        self.window = window
        self.labels: List[str] = []
        self.vectors = None
        self.temporal_contexts = None

    def store_sequence(self, labels: Sequence[str], vectors) -> None:
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        self.labels = list(labels)
        self.vectors = torch.stack(list(vectors), dim=0)
        contexts = []
        for i in range(len(labels)):
            neighbors = [
                vectors[j]
                for j in range(max(0, i - self.window), min(len(labels), i + self.window + 1))
                if j != i
            ]
            contexts.append(self.substrate.bundle(neighbors if neighbors else [vectors[i]]))
        self.temporal_contexts = torch.stack(contexts, dim=0)

    def coupled_recall(
        self,
        content_query,
        temporal_query,
        content_beta: float = 80.0,
        temporal_beta: float = 4.0,
        feedback: float = 0.75,
        max_iter: int = 12,
        tol: float = 1e-3,
        top_k: int = 4,
    ) -> TorchCoupledResult:
        if self.vectors is None or self.temporal_contexts is None:
            raise ValueError("memory is empty")
        current_temporal_query = temporal_query
        last_weights = None
        trace: List[TorchCoupledStep] = []
        weights = None
        temporal_state = temporal_query
        top_index = 0

        for iteration in range(1, max_iter + 1):
            content_scores = self.substrate.similarity_matrix(content_query, self.vectors)
            temporal_scores = self.substrate.similarity_matrix(current_temporal_query, self.temporal_contexts)
            joint_scores = content_beta * content_scores + temporal_beta * temporal_scores
            weights = torch.softmax(joint_scores, dim=0)
            temporal_state = self.substrate.normalize((self.temporal_contexts * weights[:, None]).sum(dim=0))
            top_index = int(torch.argmax(joint_scores).detach().cpu())
            entropy = torch_normalized_entropy(weights)
            trace.append(
                TorchCoupledStep(
                    iteration=iteration,
                    top_label=self.labels[top_index],
                    top_weight=float(weights[top_index].detach().cpu()),
                    entropy=entropy,
                    top_joint_score=float(joint_scores[top_index].detach().cpu()),
                )
            )
            if last_weights is not None:
                delta = float((weights - last_weights).abs().max().detach().cpu())
                if delta < tol:
                    break
            last_weights = weights
            current_temporal_query = self.substrate.weighted_bundle(
                [temporal_query, temporal_state],
                [1.0 - feedback, feedback],
            )

        temporal_items = self.substrate.top_k(temporal_state, self.labels, self.vectors, k=top_k)
        return TorchCoupledResult(
            top_label=self.labels[top_index],
            temporal_items=temporal_items,
            trace=trace,
            converged=len(trace) < max_iter,
        )

