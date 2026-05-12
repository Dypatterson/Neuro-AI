"""Unified replay loop tying trajectory traces, the engagement-resolution
gate, replay re-settling, and Benna-Fusi consolidation.

The flow per the Phase 4 unified design:

  retrieve(query):
    1. Settle query through Hopfield, capturing trajectory
    2. Compute gate signal = engagement * (1 - resolution)
    3. If gate > store_threshold: trace enters replay store
    4. Reinforce u_1 of the winning pattern

  every K retrievals (replay cycle):
    5. Sample traces from store ∝ gate × age
    6. Re-settle each trace's query through current landscape
    7. If new resolution > resolve_threshold: emit candidate pattern,
       attach consolidation state at u_1 = novelty_strength
    8. Step Benna-Fusi dynamics once across all patterns
    9. Garbage-collect patterns whose u-chain has decayed below death
       threshold for death_window consecutive steps

All decisions distribute into local geometric dynamics — no supervisor
decides what to replay, what to consolidate, or what to prune. The
sampling distributions, threshold filters, and u-chain dynamics are the
decision-makers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory.torch_hopfield import TorchRetrievalResult
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.phase4.trajectory import (
    TracedHopfieldMemory,
    TrajectoryTrace,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

T = TypeVar("T")


@dataclass(frozen=True)
class ReplayConfig:
    """Tuning parameters for the unified replay loop."""

    store_threshold: float = 0.1
    store_capacity: int = 1000
    resolve_threshold: float = 0.7
    replay_every: int = 50
    replay_batch_size: int = 10
    max_age: int = 5
    novelty_strength: float = 1.0
    retrieval_gain: float = 0.1


class ReplayStore:
    """Bounded buffer of trajectory traces ranked by gate signal × age."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.traces: List[TrajectoryTrace] = []
        self.gate_signals: List[float] = []
        self._evicted = 0

    def add(self, trace: TrajectoryTrace, gate_signal: float) -> None:
        if len(self.traces) >= self.capacity:
            self._evict_lowest()
        self.traces.append(trace)
        self.gate_signals.append(gate_signal)

    def _evict_lowest(self) -> None:
        if not self.gate_signals:
            return
        idx = min(range(len(self.gate_signals)), key=lambda i: self.gate_signals[i])
        self.traces.pop(idx)
        self.gate_signals.pop(idx)
        self._evicted += 1

    def sample(
        self,
        n: int,
        generator: Optional["torch.Generator"] = None,
    ) -> List[int]:
        """Sample n indices weighted by gate_signal * (1 + age)."""
        if not self.traces:
            return []
        n_sample = min(n, len(self.traces))
        weights = torch.tensor([
            self.gate_signals[i] * (1.0 + self.traces[i].age)
            for i in range(len(self.traces))
        ])
        weights = weights.clamp(min=1e-9)
        weights /= weights.sum()
        idx = torch.multinomial(weights, n_sample, replacement=False, generator=generator)
        return idx.tolist()

    def get(self, idx: int) -> TrajectoryTrace:
        return self.traces[idx]

    def remove(self, idx: int) -> None:
        self.traces.pop(idx)
        self.gate_signals.pop(idx)

    def update_gate(self, idx: int, gate_signal: float) -> None:
        self.gate_signals[idx] = gate_signal

    def __len__(self) -> int:
        return len(self.traces)

    def stats(self) -> dict:
        if not self.traces:
            return {
                "size": 0, "capacity": self.capacity, "evicted": self._evicted,
                "mean_gate": 0.0, "max_gate": 0.0, "mean_age": 0.0,
            }
        return {
            "size": len(self.traces),
            "capacity": self.capacity,
            "evicted": self._evicted,
            "mean_gate": sum(self.gate_signals) / len(self.gate_signals),
            "max_gate": max(self.gate_signals),
            "mean_age": sum(t.age for t in self.traces) / len(self.traces),
        }


class UnifiedReplayMemory(Generic[T]):
    """Coordinates traced Hopfield retrieval, replay, and consolidation.

    Wraps a TracedHopfieldMemory and a ConsolidationState. Exposes:
      - retrieve_and_observe(query): does retrieval + trace + gate
      - run_replay_cycle(): runs the replay batch + steps consolidation
      - garbage_collect(): removes dead patterns from both memories
    """

    def __init__(
        self,
        substrate: TorchFHRR,
        memory: TracedHopfieldMemory,
        consolidation: ConsolidationState,
        config: ReplayConfig = ReplayConfig(),
        candidate_callback: Optional[Callable[[TrajectoryTrace, int], None]] = None,
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("UnifiedReplayMemory requires torch") from _IMPORT_ERROR
        self.substrate = substrate
        self.memory = memory
        self.consolidation = consolidation
        self.config = config
        self.store = ReplayStore(capacity=config.store_capacity)
        self._retrieval_count = 0
        self._candidate_count = 0
        self._candidate_callback = candidate_callback

    def attach_initial_patterns(self) -> None:
        """Initialize consolidation state for already-stored patterns.

        Called once after the underlying memory is populated with a
        landscape. Each existing pattern enters consolidation at u_1 = novelty_strength.
        """
        while self.consolidation.n_patterns < self.memory.stored_count:
            self.consolidation.add_pattern(
                novelty_strength=self.config.novelty_strength,
            )

    def retrieve_and_observe(
        self,
        query: "torch.Tensor",
        beta: float = 10.0,
        max_iter: int = 12,
        tol: float = 1e-8,
    ) -> Tuple[TorchRetrievalResult[T], TrajectoryTrace]:
        result, trace = self.memory.retrieve_with_trace(
            query=query, beta=beta, max_iter=max_iter, tol=tol,
        )
        gate = trace.gate_signal()
        if gate > self.config.store_threshold:
            self.store.add(trace, gate_signal=gate)
        if (
            trace.final_top_index is not None
            and trace.final_top_index < self.consolidation.n_patterns
        ):
            self.consolidation.reinforce(
                trace.final_top_index,
                magnitude=self.config.retrieval_gain,
            )
        self._retrieval_count += 1
        return result, trace

    def should_replay(self) -> bool:
        return (
            self._retrieval_count > 0
            and self._retrieval_count % self.config.replay_every == 0
        )

    def run_replay_cycle(
        self,
        beta: float = 10.0,
        max_iter: int = 12,
        candidate_handler: Optional[Callable[[TrajectoryTrace], Optional[int]]] = None,
    ) -> dict:
        """Sample traces, re-settle, emit candidates, step consolidation.

        candidate_handler: optional callback that takes the new trace and
        returns the pattern index where it was stored (or None if the
        caller doesn't want to add it to the memory). If None, candidates
        are counted but not stored — the caller is responsible for the
        actual pattern addition.

        Returns: dict with cycle stats.
        """
        if not self.store.traces:
            self.consolidation.step_dynamics()
            return {
                "sampled": 0, "candidates": 0, "decayed": 0,
                "store_after": len(self.store),
            }

        sampled_local = self.store.sample(self.config.replay_batch_size)
        # Sort descending so removes don't invalidate later indices
        sampled_local.sort(reverse=True)

        candidates = 0
        decayed = 0

        for local_idx in sampled_local:
            trace = self.store.get(local_idx)

            new_result, new_trace = self.memory.retrieve_with_trace(
                query=trace.query, beta=beta, max_iter=max_iter,
            )

            if new_trace.final_top_score >= self.config.resolve_threshold:
                if candidate_handler is not None:
                    new_idx = candidate_handler(new_trace)
                    if new_idx is not None:
                        while self.consolidation.n_patterns <= new_idx:
                            self.consolidation.add_pattern(
                                novelty_strength=self.config.novelty_strength,
                            )
                candidates += 1
                self.store.remove(local_idx)
            else:
                trace.age += 1
                new_gate = new_trace.gate_signal()
                if trace.age > self.config.max_age:
                    self.store.remove(local_idx)
                    decayed += 1
                else:
                    self.store.update_gate(local_idx, new_gate)

        self.consolidation.step_dynamics()
        self._candidate_count += candidates

        return {
            "sampled": len(sampled_local),
            "candidates": candidates,
            "decayed": decayed,
            "store_after": len(self.store),
        }

    def garbage_collect(self) -> List[int]:
        """Remove patterns whose u-chain has decayed below death threshold.

        Returns the list of removed pattern indices (relative to the
        memory's current state, before removal).
        """
        dead = self.consolidation.dead_indices()
        if not dead:
            return []
        for idx in sorted(dead, reverse=True):
            if idx >= len(self.memory._patterns):
                continue
            self.memory._patterns.pop(idx)
            if idx < len(self.memory.labels):
                self.memory.labels.pop(idx)
            self.consolidation.remove_pattern(idx)
        return dead

    def stats(self) -> dict:
        return {
            "retrievals": self._retrieval_count,
            "candidates_total": self._candidate_count,
            "store": self.store.stats(),
            "consolidation": self.consolidation.stats(),
            "memory_size": self.memory.stored_count,
        }
