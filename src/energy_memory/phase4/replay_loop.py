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
    # Idea 4a (Joo & Frank 2023): when a new trace's query is close to an
    # already-stored trace, increment that trace's tag_count rather than
    # appending a duplicate. Sampling weight becomes gate * tag_count.
    # Set to None to disable (preserves prior gate * (1 + age) weighting).
    tag_overlap_threshold: Optional[float] = 0.7
    # Idea 4c (Biderman SFMA, 2023): inhibition of return. Each time a trace
    # is sampled for replay, its suppression multiplier is scaled by
    # suppression_decay; between replay cycles, suppression recovers toward
    # 1.0 by suppression_recovery per cycle. This prevents the highest-gate
    # trace from monopolizing replay.
    #
    # Defaults are off (decay=1.0, recovery=0.0) per report 013: in the
    # project's current resolve-and-remove replay flow, sampled traces are
    # removed immediately on resolution, so suppression has no window to
    # bias subsequent samples. The mechanism is correct and tested; flip
    # these knobs on if/when the replay flow moves to keep-and-sweep.
    suppression_decay: float = 1.0
    suppression_recovery: float = 0.0


class ReplayStore:
    """Bounded buffer of trajectory traces ranked by gate_signal × tag_count × suppression.

    Per-trace state:
      - gate_signal: engagement × (1 - resolution) at last observation
      - tag_count:   how many times an overlapping query has been seen
                     (Joo & Frank 2023 — replay priority predicted by tag
                     count, not age)
      - suppression: inhibition-of-return multiplier; decays on each replay
                     attempt, recovers between cycles (Biderman SFMA 2023)

    Backward compatibility: when ``substrate`` is None or
    ``tag_overlap_threshold`` is None, ``add`` never collapses overlapping
    queries (tag_count stays 1 for every entry), so the prior behavior of
    one-trace-per-add is preserved. When ``suppression_decay >= 1.0`` and
    ``suppression_recovery == 0.0`` the inhibition-of-return mechanism is
    a no-op.
    """

    def __init__(
        self,
        capacity: int,
        *,
        substrate: Optional[TorchFHRR] = None,
        tag_overlap_threshold: Optional[float] = None,
        suppression_decay: float = 1.0,
        suppression_recovery: float = 0.0,
    ):
        self.capacity = capacity
        self.traces: List[TrajectoryTrace] = []
        self.gate_signals: List[float] = []
        self.tag_counts: List[int] = []
        self.suppression: List[float] = []
        self._evicted = 0
        self._substrate = substrate
        self._tag_overlap_threshold = tag_overlap_threshold
        self._suppression_decay = suppression_decay
        self._suppression_recovery = suppression_recovery

    def add(self, trace: TrajectoryTrace, gate_signal: float) -> None:
        # Idea 4a: overlap collapse. If the incoming trace's query is close
        # enough to an existing stored trace, bump that trace's tag_count
        # and refresh its gate signal rather than storing a duplicate.
        if (
            self._substrate is not None
            and self._tag_overlap_threshold is not None
            and self.traces
        ):
            overlap_idx = self._find_overlap(trace)
            if overlap_idx is not None:
                self.tag_counts[overlap_idx] += 1
                # Track the max gate signal across observations of this trace.
                if gate_signal > self.gate_signals[overlap_idx]:
                    self.gate_signals[overlap_idx] = gate_signal
                return

        if len(self.traces) >= self.capacity:
            self._evict_lowest()
        self.traces.append(trace)
        self.gate_signals.append(gate_signal)
        self.tag_counts.append(1)
        self.suppression.append(1.0)

    def _find_overlap(self, trace: TrajectoryTrace) -> Optional[int]:
        assert self._substrate is not None
        assert self._tag_overlap_threshold is not None
        sims = [
            float(self._substrate.similarity(trace.query, existing.query))
            for existing in self.traces
        ]
        if not sims:
            return None
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        if sims[best_idx] >= self._tag_overlap_threshold:
            return best_idx
        return None

    def _evict_lowest(self) -> None:
        if not self.gate_signals:
            return
        # Evict by priority (gate × tag × suppression), not raw gate, so a
        # frequently-tagged trace isn't displaced by a one-off high-gate hit.
        priorities = self._priorities()
        idx = min(range(len(priorities)), key=lambda i: priorities[i])
        self._pop(idx)
        self._evicted += 1

    def _priorities(self) -> List[float]:
        return [
            self.gate_signals[i] * self.tag_counts[i] * self.suppression[i]
            for i in range(len(self.traces))
        ]

    def sample(
        self,
        n: int,
        generator: Optional["torch.Generator"] = None,
    ) -> List[int]:
        """Sample n indices weighted by gate × tag_count × suppression.

        After sampling, each sampled index has its suppression multiplier
        scaled by ``suppression_decay`` (inhibition of return). Non-sampled
        indices recover toward 1.0 by ``suppression_recovery``.
        """
        if not self.traces:
            return []
        n_sample = min(n, len(self.traces))
        weights = torch.tensor(self._priorities(), dtype=torch.float32)
        weights = weights.clamp(min=1e-9)
        weights /= weights.sum()
        idx = torch.multinomial(weights, n_sample, replacement=False, generator=generator)
        sampled = idx.tolist()

        if self._suppression_decay < 1.0 or self._suppression_recovery > 0.0:
            sampled_set = set(sampled)
            for i in range(len(self.suppression)):
                if i in sampled_set:
                    self.suppression[i] *= self._suppression_decay
                else:
                    self.suppression[i] = min(
                        1.0, self.suppression[i] + self._suppression_recovery
                    )
        return sampled

    def get(self, idx: int) -> TrajectoryTrace:
        return self.traces[idx]

    def remove(self, idx: int) -> None:
        self._pop(idx)

    def _pop(self, idx: int) -> None:
        self.traces.pop(idx)
        self.gate_signals.pop(idx)
        self.tag_counts.pop(idx)
        self.suppression.pop(idx)

    def update_gate(self, idx: int, gate_signal: float) -> None:
        self.gate_signals[idx] = gate_signal

    def __len__(self) -> int:
        return len(self.traces)

    def stats(self) -> dict:
        if not self.traces:
            return {
                "size": 0, "capacity": self.capacity, "evicted": self._evicted,
                "mean_gate": 0.0, "max_gate": 0.0, "mean_age": 0.0,
                "mean_tag_count": 0.0, "max_tag_count": 0,
                "mean_suppression": 0.0,
            }
        return {
            "size": len(self.traces),
            "capacity": self.capacity,
            "evicted": self._evicted,
            "mean_gate": sum(self.gate_signals) / len(self.gate_signals),
            "max_gate": max(self.gate_signals),
            "mean_age": sum(t.age for t in self.traces) / len(self.traces),
            "mean_tag_count": sum(self.tag_counts) / len(self.tag_counts),
            "max_tag_count": max(self.tag_counts),
            "mean_suppression": sum(self.suppression) / len(self.suppression),
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
        self.store = ReplayStore(
            capacity=config.store_capacity,
            substrate=substrate if config.tag_overlap_threshold is not None else None,
            tag_overlap_threshold=config.tag_overlap_threshold,
            suppression_decay=config.suppression_decay,
            suppression_recovery=config.suppression_recovery,
        )
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
        # Saighi A_k: bias retrieval scores away from already-used attractors.
        # When inhibition_gain=0, A is all-zero and bias has no effect.
        bias = None
        if (
            self.consolidation.config.inhibition_gain > 0.0
            and self.consolidation.n_patterns == self.memory.stored_count
        ):
            bias = self.consolidation.inhibition_bias()
        result, trace = self.memory.retrieve_with_trace(
            query=query, beta=beta, max_iter=max_iter, tol=tol,
            score_bias=bias,
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
            # Saighi A_k accumulation: every successful retrieval of
            # attractor k increments A_k by inhibition_gain (no-op when 0).
            self.consolidation.accumulate_inhibition(trace.final_top_index)
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

        # Replay re-settling also respects Saighi A_k inhibition: the
        # re-settled trajectory is pushed off attractors that have already
        # accumulated dominance, biasing the discovery channel toward
        # under-used regions of the landscape. Re-fetched per iteration
        # because candidate_handler may have grown both memory and
        # consolidation between iterations.
        inhibition_active = self.consolidation.config.inhibition_gain > 0.0

        for local_idx in sampled_local:
            trace = self.store.get(local_idx)

            replay_bias = None
            if (
                inhibition_active
                and self.consolidation.n_patterns == self.memory.stored_count
            ):
                replay_bias = self.consolidation.inhibition_bias()

            new_result, new_trace = self.memory.retrieve_with_trace(
                query=trace.query, beta=beta, max_iter=max_iter,
                score_bias=replay_bias,
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
            if idx >= self.memory.stored_count:
                continue
            self.memory.remove_pattern(idx)
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
