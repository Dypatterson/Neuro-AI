---
date: 2026-05-12
project: neuro-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - phase-4
  - design
---

# Phase 4 Unified Design — Trajectory Trace × Benna-Fusi Consolidation

This document specifies the Phase 4 architecture, integrating:

- The trajectory-trace + engagement-resolution gate from
  [notes/notes/2026-05-02-trajectory-trace-and-replay.md](../notes/2026-05-02-trajectory-trace-and-replay.md)
- The Benna-Fusi (2016) multi-timescale synaptic consolidation model
- The sparse-update principle from SQ-HN (Alonso & Krichmar 2024)

Supersedes the earlier `notes/phase-4-mvp-design.md`, which built only the
metastability-driven tension signal and missed the design specified by the
2026-05-02 note.

## What this design does

Replay is the dynamical process that bridges fast-timescale settling and
slow-timescale consolidation. Specifically:

1. **Trajectory trace** captures the path of Hopfield settling (per-step
   co-activation snapshots), not just the endpoint.
2. **Engagement × low-resolution** gates which trajectories enter the replay
   store. Both axes are needed: high force (engagement) without clean
   commitment (low resolution) is the geometric signature of an unresolved
   trajectory.
3. **Replay** re-settles the trajectory through the current landscape. If
   it resolves, the resolution is a *new candidate pattern*.
4. **Benna-Fusi consolidation** routes candidates through a chain of m
   multi-timescale variables u_1, ..., u_m. New patterns enter at u_1 (fast,
   weak). Replay events drive bidirectional coupling u_k ↔ u_{k+1}. Patterns
   that consistently re-emerge consolidate to slow variables and become
   durable.

The "decisions" of what to keep, what to prune, what to consolidate distribute
into the dynamics. No supervisor.

## Architectural diagram

```
                    cue
                     │
                     ▼
             [Hopfield settle]
              │            │
              │            └──► trajectory trace
              │                  (per-step co-activations)
              ▼                       │
        settled state                 │
              │                       ▼
              │           [engagement × low-resolution gate]
              │                       │
              │                       ▼
              │                  replay store
              │                       │
              │                       ▼
              │              [re-settle through current landscape]
              │                       │
              │                       ▼
              │            ┌──── resolution? ─────┐
              │            │                      │
              │            ▼ yes                  ▼ no
              │      candidate pattern      stays in store / decays
              │            │
              ▼            ▼
         [u_1...u_m]  ──── Benna-Fusi chain ────► retrieval strength
              ▲              (bidirectional)
              │
         retrieval driving
       u_k ↔ u_{k+1} coupling
```

## Component specifications

### 1. Trajectory trace

**Data structure** (`TrajectorySnapshot`):
- `step`: iteration index (1, 2, ..., until convergence or max_iter)
- `top_k_indices`: sparse list of pattern indices with weight > snapshot_threshold
- `top_k_weights`: corresponding weights
- `entropy`: softmax entropy at this step

**Data structure** (`TrajectoryTrace`):
- `query`: original cue (kept for re-settling)
- `snapshots`: ordered list of `TrajectorySnapshot`
- `final_state`: settled state (FHRR vector)
- `final_top_score`: max cos similarity to any stored pattern (resolution)
- `final_top_index`: which pattern won (or none if unresolved)
- `converged`: bool

**Capture mechanism**: a `TracedHopfieldMemory` wrapper that overrides
`retrieve()` to record the trajectory. No interference with settling itself —
the trace is a passive observer.

### 2. Engagement and resolution metrics

**Resolution**: `final_top_score` (max cosine of settled state to any stored
pattern). High = clean basin lock-in (prototype mode). Low = state didn't
commit to any pattern (feature mode / metastable).

**Engagement**: mean entropy across settling snapshots, weighted by
iteration count.

```
engagement = (1 / n_steps) * Σ_t entropy(weights_t)
```

Rationale: high entropy = many patterns simultaneously active = strong
co-activation = "the landscape exerted force across multiple basins." Single
basin lock-in produces low entropy throughout (low engagement). Indecisive
oscillation between basins produces high entropy across steps (high
engagement).

**Gate signal**:

```
gate(trace) = engagement × (1 - resolution)
```

A trajectory enters the replay store if `gate > store_threshold`. The
threshold is *not* a controller — it's a passive filter on a geometric
property. Trajectories whose gate signal is high enough to cross the
threshold are by definition unresolved-with-engagement; those below are
either clean (low engagement) or already resolved (high resolution).

### 3. Replay mechanism

**Replay store**: bounded-size buffer. When full, lowest-gate entry is
evicted (or oldest, depending on policy — start with gate-ranked).

**Replay step** (triggered every K cues, or when store fills):
1. Sample a trace from the store, weighted by gate signal × age
2. Re-settle the trace's `query` through the *current* landscape (the
   landscape may have evolved since the trace was recorded — new patterns
   added, codebook drifted, consolidation states updated)
3. If new `final_top_score > resolve_threshold`:
   - Emit `final_state` as candidate pattern
   - Submit candidate to consolidation pipeline
   - Remove trace from store
4. Else if `gate > store_threshold` again:
   - Re-enqueue with incremented age
   - Below decay threshold → drop

**Decay**: replay store entries have an `age` counter. With each cycle the
trace doesn't resolve, age increments. Above `max_age` → drop. This is the
"trajectories that never resolve decay out" mechanism.

### 4. Benna-Fusi consolidation channel

Each stored pattern has `m` consolidation variables `u_1, ..., u_m`. Per
Benna-Fusi (2016) Eq. 10-11:

```
u_i(t+1) = u_i(t) + α * (-2 * u_i(t) + u_{i-1}(t) + u_{i+1}(t))      (Eq. 10)
u_1(t+1) = u_1(t) + I(t) + α * (-2 * u_1(t) + u_2(t))                (Eq. 11)
```

with boundary `u_{m+1}(t) = 0`.

Time constants grow exponentially with k. We discretize and use α ≈ 1/4 per
the paper.

**Inputs to u_1**:
- New candidate pattern enters with strong u_1 perturbation (e.g., +1)
- Each retrieval where this pattern is the top winner: u_1 += retrieval_strength
  (mild reinforcement, e.g., +0.1)

**Bidirectional coupling**: replay activity is what drives the u_k ↔ u_{k+1}
coupling. Implemented by stepping the entire u-variable chain every replay
cycle. Per Benna-Fusi (p. 1026): *"coupling between u_k variables could be
mediated by neuronal activity, such as replay activity."*

**Pattern effective strength** (used in retrieval):

```
strength = Σ_k w_k * u_k     where w_k = 2^(-k) (or similar weighting)
```

Equivalent: weighted sum that emphasizes fast variables for recent patterns,
slow variables for durable ones.

**Pattern death**: if `strength < death_threshold` for `death_window` consecutive
steps, the pattern is garbage-collected. This is not a controller — it's the
natural endpoint of the dynamics when no replay sustains the pattern.

### 5. Sparse-update principle (from SQ-HN)

When a candidate pattern's u variables are updated, modify *only* the u
variables for that pattern, not the whole chain across all patterns. SQ-HN's
anti-forgetting result depends on update sparsity.

When a new pattern is added, do not perturb the u variables of existing
patterns. New patterns compete by their own u-chain dynamics.

## Anti-homunculus check

- **Who decides what enters the trajectory trace?**
  Nobody — every retrieval generates one. Passive observation.

- **Who decides what enters the replay store?**
  Nobody — engagement × (1 - resolution) is a geometric property of the
  trace. The store threshold is a passive filter, not a rule.

- **Who decides when to replay?**
  Replay runs on a fixed cadence (every K cues) or when the store fills.
  Both are properties of the architecture, not supervisor decisions. Could
  later be replaced with a tension-driven schedule (e.g., replay triggers
  when total store gate signal crosses a threshold), still passing the
  filter.

- **Who decides what a candidate pattern is?**
  Nobody — it's whatever the re-settle dynamics resolve to. The dynamics
  are the decision.

- **Who decides when to consolidate vs prune?**
  Nobody — Benna-Fusi u-variable dynamics determine consolidation depth.
  Patterns whose dynamics decay below threshold get pruned. The dynamics
  are the decision.

- **Who decides effective retrieval strength?**
  Nobody — it's a weighted sum of u_k. The weights are architectural
  constants, not adjusted by any supervisor.

All decisions distribute into local geometric dynamics. ✓

## Data flow

```
retrieve(query):
    trace = TracedHopfield.retrieve(query)
    engagement = mean_entropy(trace.snapshots)
    resolution = trace.final_top_score
    gate_signal = engagement * (1 - resolution)
    if gate_signal > store_threshold:
        replay_store.add(trace, gate_signal=gate_signal)

    # Apply retrieval reinforcement to the winning pattern's u_1
    if trace.final_top_index is not None:
        u_state.reinforce(trace.final_top_index, magnitude=retrieval_gain)

    return trace.final_state

every K retrievals:
    for _ in range(replay_batch_size):
        trace = replay_store.sample()
        new_trace = TracedHopfield.retrieve(trace.query)
        if new_trace.final_top_score > resolve_threshold:
            # Candidate discovered
            candidate = new_trace.final_state
            memory.add_pattern(candidate)
            u_state.initialize(new_idx, u_1=novelty_strength)
            replay_store.remove(trace)
        else:
            new_gate = mean_entropy(new_trace) * (1 - new_trace.final_top_score)
            if new_gate > store_threshold and trace.age < max_age:
                trace.age += 1
                trace.gate_signal = new_gate
            else:
                replay_store.remove(trace)
    # Step the Benna-Fusi dynamics once per replay cycle
    u_state.step_dynamics(alpha)
    # Prune dead patterns
    for idx in u_state.dead_indices():
        memory.remove_pattern(idx)
```

## Headline metric for Phase 4

Per the project's headline-vs-drill-down structure:

> **Recall@K and cap-coverage on masked-token contextual completion,
> measured before vs. after N consolidation cycles, with active codebook
> drift between cycles.**

The drift simulates Phase 3's continuous codebook updates. With drift but
without replay, stored patterns become stale and accuracy degrades. With
drift + replay, the consolidation pipeline should keep durable patterns
aligned with the current codebook through repeated reinforcement of u_k
chains.

Drill-downs:
- Meta-stable rate over time (should stay lower with replay)
- Mean engagement over time (should drop as landscape stabilizes)
- Pattern age distribution (Benna-Fusi predicts: some short-lived, some
  long-lived)
- Number of discovered patterns per cycle
- u_k variable distributions (should show graduated consolidation)

Control conditions:
- No-replay baseline (pure decay under drift)
- Random-codebook control (the trajectory trace + replay mechanism should
  fail to produce meaningful discoveries without learned codebook structure)

## What's deferred from this design

- **Cross-pattern coupling.** When pattern A is consolidated, do related
  patterns' u variables also update? Defer to Phase 5 or later.
- **Adaptive m (chain length).** Benna-Fusi proves m ≈ log(T) where T is
  memory lifetime. Start with fixed m=6.
- **Adaptive store_threshold.** Defer; start with fixed threshold tuned
  empirically.
- **Multi-scale interaction.** With three scales (W=2, W=3, W=4) each with
  its own memory, do they share trajectory traces? Defer — start with
  per-scale replay and consolidation.
- **Sleep/wake cycles.** Defer — replay runs interleaved with retrieval at
  fixed cadence for now.

## Implementation files

- `src/energy_memory/phase4/trajectory.py` — `TrajectorySnapshot`,
  `TrajectoryTrace`, `TracedHopfieldMemory` wrapper
- `src/energy_memory/phase4/consolidation.py` — `ConsolidationState` (u
  variables per pattern), Benna-Fusi dynamics
- `src/energy_memory/phase4/replay_loop.py` — `ReplayStore`,
  `UnifiedReplayMemory` that ties trace + gate + replay + consolidation
- Tests for each component

## Implementation order

1. Trajectory trace (foundation — needed by everything else)
2. Consolidation state (independent of trajectory, can be built in parallel)
3. Replay loop (integration — depends on both)

Each component is testable in isolation before integration.
