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
    _coverage_redundancy_instantaneous,
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
    # Candidate B (dimensionality-preserving repulsion field).
    # Step size for the per-cycle substrate update under the
    # H_anti = -α·log(d_eff) gradient. At 0.0 (default) the substrate
    # patterns are static and behavior is bit-identical to baseline.
    # Per the load-bearing precondition (notes/notes/2026-05-20-...md
    # §"Candidate B"): this value is set ONCE at training start from
    # theoretical considerations, NOT tuned to land d_eff in a target
    # range. Companion to substrate.alpha_anti — both must be set
    # together for B to fire.
    repulsion_step_size: float = 0.0
    # Pair #4 (metastability ~ replay-prioritization).
    # κ — multiplicative gain on per-trace metastability m_trace in the
    # replay-store priority composition:
    #     priority(t) = gate(t) · tag_count(t) · suppression(t) · (1 + κ · m_trace)
    # At κ = 0 the composition is bit-identical to the pre-pivot baseline
    # (the load-bearing precondition for the falsification control).
    # μ_rep — decay applied to m_{i*(trace)} on each replay-sampling event:
    #     m_i ← (1 − μ_rep) · m_i
    # Both default off; values pre-committed per the design note before
    # any retrain (audit constraint #4: never adapted from observed
    # meta_stable_rate).
    # See notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md.
    metastability_gain: float = 0.0
    metastability_replay_decay: float = 0.0


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
        consolidation: Optional[ConsolidationState] = None,
        metastability_gain: float = 0.0,
        metastability_replay_decay: float = 0.0,
    ):
        self.capacity = capacity
        self.traces: List[TrajectoryTrace] = []
        self.gate_signals: List[float] = []
        self.tag_counts: List[int] = []
        self.suppression: List[float] = []
        # Pair #4: per-trace primary-atom index — the highest-similarity
        # stored atom for the trace's query. Used purely as a key into
        # ConsolidationState.metastability_ema (audit constraint #5: no
        # side effects beyond keying m). -1 sentinel = "no overlap data"
        # (e.g., substrate is None, no stored atoms, or below threshold).
        self.primary_atom: List[int] = []
        self._evicted = 0
        self._substrate = substrate
        self._tag_overlap_threshold = tag_overlap_threshold
        self._suppression_decay = suppression_decay
        self._suppression_recovery = suppression_recovery
        self._consolidation = consolidation
        self._metastability_gain = float(metastability_gain)
        self._metastability_replay_decay = float(metastability_replay_decay)

    def add(
        self,
        trace: TrajectoryTrace,
        gate_signal: float,
        *,
        primary_atom_idx: int = -1,
    ) -> None:
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
                # Pair #4: keep the most recently observed primary atom
                # for this collapsed trace (the freshest retrieval's
                # top_index). It is purely a key into metastability_ema.
                if primary_atom_idx >= 0:
                    self.primary_atom[overlap_idx] = primary_atom_idx
                return

        if len(self.traces) >= self.capacity:
            self._evict_lowest()
        self.traces.append(trace)
        self.gate_signals.append(gate_signal)
        self.tag_counts.append(1)
        self.suppression.append(1.0)
        self.primary_atom.append(int(primary_atom_idx))

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
        # Pair #4: at κ = 0 (the κ=0 control), m_factor = 1.0 for every
        # trace and the composition is bit-identical to the pre-pivot
        # baseline. This is the load-bearing precondition for the
        # falsification control. Audit constraint #6: same code path —
        # the multiplication is applied unconditionally rather than
        # gated by an ``if κ == 0`` branch.
        kappa = self._metastability_gain
        cons = self._consolidation
        if kappa > 0.0 and cons is not None and cons.n_patterns > 0:
            m_tensor = cons.metastability_ema
            n = cons.n_patterns
            m_factors: List[float] = []
            for i in range(len(self.traces)):
                idx = self.primary_atom[i]
                if 0 <= idx < n:
                    m_factors.append(1.0 + kappa * float(m_tensor[idx].detach().cpu()))
                else:
                    m_factors.append(1.0)
        else:
            m_factors = [1.0] * len(self.traces)
        return [
            self.gate_signals[i] * self.tag_counts[i] * self.suppression[i] * m_factors[i]
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

        # Pair #4: pay-down on replay sampling. m_{i*(trace)} ← (1 − μ_rep) · m_{i*(trace)}.
        # Audit constraint #3: lives inside sample() after multinomial, not
        # in a separate maintenance call. The trigger is the sampling event
        # itself, so the pay-down is the local response of m_i to its own
        # atom being drained.
        cons = self._consolidation
        mu_rep = self._metastability_replay_decay
        if cons is not None and mu_rep > 0.0 and cons.n_patterns > 0:
            factor = 1.0 - mu_rep
            for trace_idx in sampled:
                atom_idx = self.primary_atom[trace_idx]
                if 0 <= atom_idx < cons.n_patterns:
                    cons.metastability_payback(atom_idx, factor)
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
        self.primary_atom.pop(idx)

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
            consolidation=consolidation,
            metastability_gain=config.metastability_gain,
            metastability_replay_decay=config.metastability_replay_decay,
        )
        self._retrieval_count = 0
        self._candidate_count = 0
        self._candidate_callback = candidate_callback

    def attach_initial_patterns(self) -> None:
        """Initialize consolidation state for already-stored patterns.

        Called once after the underlying memory is populated with a
        landscape. Each existing pattern enters consolidation at u_1 = novelty_strength.

        A1 (notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md):
        when ``coverage_lambda > 0`` and the landscape has ≥ 2 patterns,
        each new atom's r_ema starts at its geometric equilibrium
        (``r_inst`` against the current substrate) instead of 0. For an
        initially-orthogonal landscape this is ≈ 0 across all atoms; for
        a landscape with near-duplicate patterns those atoms start at
        their geometric redundancy.
        """
        r_inst = self._compute_r_inst_for_init()
        while self.consolidation.n_patterns < self.memory.stored_count:
            idx = self.consolidation.n_patterns
            r_init = None if r_inst is None else float(r_inst[idx].detach().cpu())
            self.consolidation.add_pattern(
                novelty_strength=self.config.novelty_strength,
                r_ema_init=r_init,
            )

    def _compute_r_inst_for_init(self):
        """Compute per-atom r_inst on the current memory's pattern matrix
        for use as A1's r_ema initialization.

        Returns None when the geometric init is inert (``coverage_lambda
        = 0`` or fewer than 2 stored patterns). Callers pass the per-row
        scalar into ``ConsolidationState.add_pattern(r_ema_init=...)``.
        """
        if self.consolidation.config.coverage_lambda <= 0.0:
            return None
        if self.memory.stored_count < 2:
            return None
        pattern_matrix = self.memory._pattern_matrix()
        return _coverage_redundancy_instantaneous(pattern_matrix)

    def retrieve_and_observe(
        self,
        query: "torch.Tensor",
        beta: float = 10.0,
        max_iter: int = 12,
        tol: float = 1e-8,
    ) -> Tuple[TorchRetrievalResult[T], TrajectoryTrace]:
        bias = self._score_bias()
        result, trace = self.memory.retrieve_with_trace(
            query=query, beta=beta, max_iter=max_iter, tol=tol,
            score_bias=bias,
        )
        gate = trace.gate_signal()
        # Pair #4: primary-atom index for this trace is the retrieval's
        # top index — the highest-similarity stored atom for the query.
        # It is a pure key into ConsolidationState.metastability_ema
        # (audit constraint #5: no side effects). -1 when no retrieval
        # was made or top_index is out of consolidation range.
        primary_atom_idx = -1
        if (
            trace.final_top_index is not None
            and trace.final_top_index < self.consolidation.n_patterns
        ):
            primary_atom_idx = int(trace.final_top_index)
        if gate > self.config.store_threshold:
            self.store.add(trace, gate_signal=gate, primary_atom_idx=primary_atom_idx)
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
        # Pair #4 (Path 3): update per-atom metastability EMA from the
        # retrieval's trajectory-based c_i contribution, computed inside
        # retrieve()'s settling loop (audit constraint #8). No-op when
        # metastability_obs_rate == 0 (the κ=0 control baseline).
        if (
            result.metastability_contribution is not None
            and result.metastability_contribution.shape[0] == self.consolidation.n_patterns
        ):
            self.consolidation.update_metastability(result.metastability_contribution)
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
            # Even with no traces to replay, the substrate's slow-timescale
            # dynamics evolve: r_ema (Candidate A) updates from the current
            # geometry, and repulsion (Candidate B) flows in pattern space.
            self._step_substrate_dynamics()
            return {
                "sampled": 0, "candidates": 0, "decayed": 0,
                "store_after": len(self.store),
            }

        sampled_local = self.store.sample(self.config.replay_batch_size)
        # Sort descending so removes don't invalidate later indices
        sampled_local.sort(reverse=True)

        candidates = 0
        decayed = 0

        # Replay re-settling respects the same bias stack as the main
        # retrieval path: Saighi A_k inhibition (basin-narrowing toward
        # under-used attractors) plus the step-3 E_i-weighted retrieval
        # contribution (asymptotic-death visible at retrieval). Bias is
        # re-fetched per iteration because candidate_handler may have
        # grown both memory and consolidation between iterations.

        for local_idx in sampled_local:
            trace = self.store.get(local_idx)

            replay_bias = self._score_bias()

            new_result, new_trace = self.memory.retrieve_with_trace(
                query=trace.query, beta=beta, max_iter=max_iter,
                score_bias=replay_bias,
            )

            # Pair #4 (Path 3): update m_i from the replay retrieval's
            # trajectory-based c_i contribution. Replay retrievals are real
            # settling events through the substrate and produce per-atom
            # contributions that should accumulate into m_i. No-op at
            # metastability_obs_rate=0.
            if (
                new_result.metastability_contribution is not None
                and new_result.metastability_contribution.shape[0] == self.consolidation.n_patterns
            ):
                self.consolidation.update_metastability(new_result.metastability_contribution)

            if new_trace.final_top_score >= self.config.resolve_threshold:
                if candidate_handler is not None:
                    new_idx = candidate_handler(new_trace)
                    if new_idx is not None:
                        # A1: re-compute r_inst on the augmented pattern
                        # matrix (which now includes the just-added atom
                        # via candidate_handler) so each pending-add row
                        # starts at the EMA's geometric equilibrium for
                        # the current substrate. See
                        # notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md.
                        r_inst = self._compute_r_inst_for_init()
                        while self.consolidation.n_patterns <= new_idx:
                            next_idx = self.consolidation.n_patterns
                            r_init = (
                                None if r_inst is None
                                else float(r_inst[next_idx].detach().cpu())
                            )
                            self.consolidation.add_pattern(
                                novelty_strength=self.config.novelty_strength,
                                r_ema_init=r_init,
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

        self._step_substrate_dynamics()
        self._candidate_count += candidates

        return {
            "sampled": len(sampled_local),
            "candidates": candidates,
            "decayed": decayed,
            "store_after": len(self.store),
        }

    def _score_bias(self) -> Optional["torch.Tensor"]:
        """Build the retrieval score_bias for both main and replay paths.

        Sums two independent biases when their respective mechanisms are
        configured on:

        - **Saighi A_k inhibition** (``inhibition_gain > 0``): basin
          narrowing toward under-used attractors. ``inhibition_bias()``
          is the per-pattern A_k accumulator.
        - **Step 3 E_i-weighted retrieval contribution**
          (``coverage_lambda > 0``): atoms whose ``|E_i|`` decays toward
          zero contribute infinitesimally to retrieval. The bias is
          ``softplus((ε − |E_i|) / τ)`` per atom, equivalent to
          ``-log(σ((|E_i| − ε) / τ))``.

        Both are valid when ``consolidation.n_patterns ==
        memory.stored_count`` (the legacy alignment guard). Returns
        ``None`` when neither mechanism is active so the existing
        retrieval path is bit-identical to baseline.
        """
        if self.consolidation.n_patterns != self.memory.stored_count:
            return None
        components: List["torch.Tensor"] = []
        if self.consolidation.config.inhibition_gain > 0.0:
            components.append(self.consolidation.inhibition_bias())
        if self.consolidation.config.coverage_lambda > 0.0:
            components.append(self.consolidation.retrieval_weight_bias())
        if not components:
            return None
        if len(components) == 1:
            return components[0]
        return sum(components[1:], components[0])

    def _step_substrate_dynamics(self) -> None:
        """One slow-timescale substrate step: A's r_ema update + B's repulsion flow.

        Both A and B share the same pattern-matrix snapshot for this
        cycle. The repulsion update mutates the stored patterns and
        invalidates the underlying memory's matrix cache so subsequent
        retrievals see the new state.
        """
        pattern_matrix = None
        if self.memory.stored_count >= 2 and (
            self.consolidation.config.coverage_lambda > 0.0
            or (
                self.substrate.alpha_anti > 0.0
                and self.config.repulsion_step_size > 0.0
            )
        ):
            pattern_matrix = self.memory._pattern_matrix()

        self.consolidation.step_dynamics(pattern_matrix=pattern_matrix)

        if (
            pattern_matrix is not None
            and self.substrate.alpha_anti > 0.0
            and self.config.repulsion_step_size > 0.0
        ):
            force = self.substrate.repulsion_force(pattern_matrix)
            step = self.config.repulsion_step_size
            new_patterns = self.substrate.normalize(pattern_matrix + step * force)
            for i in range(self.memory.stored_count):
                self.memory._patterns[i] = new_patterns[i]
            self.memory.invalidate_cache()

    def garbage_collect(self) -> List[int]:
        """Remove patterns whose u-chain has decayed below death threshold.

        Returns the list of removed pattern indices (relative to the
        memory's current state, before removal).

        Becomes a **no-op** when the A+B continuous death dynamic is
        configured on (``consolidation.config.coverage_lambda > 0``):
        asymptotic decay of effective_strength under (1-r_i)-modulated
        reinforcement replaces binary deletion. Existing experiment
        scripts call ``garbage_collect()`` unconditionally; this guard
        ensures that turning A+B on does not silently run the binary
        controller it replaces. Per the anti-homunculus audit verdict
        and [design note §Implementation sketch]
        (../../../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md).
        """
        if self.consolidation.config.coverage_lambda > 0.0:
            return []
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
