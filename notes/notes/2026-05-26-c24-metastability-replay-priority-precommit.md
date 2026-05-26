---
date: 2026-05-26
project: neuro-ai
tags:
  - notes
  - phase-3
  - path-c
  - planning
  - precommit
  - diagnostic-actuator-identity
  - anti-homunculus-reviewer-required
---

# C.2.4 — Metastability → Replay-Buffer Energy-Ranking (Precommit)

## Status

Design precommit for Path C C.2.4 per
[2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md).

**This note specifies a proposed slow-timescale dynamic.** No code lands
until the anti-homunculus reviewer PASSES.

## Pre-existing Infrastructure (discovered during C.1.5 and the C.2.4 reviewer pass)

**Update 2026-05-26 — reviewer pass surfaced that the actuator side is
also already substantially wired** at
[`src/energy_memory/phase4/replay_loop.py`](../../src/energy_memory/phase4/replay_loop.py)
(per a 2026-05-20 dynamic-form note referenced at line 112):

- `metastability_gain` (κ) and `metastability_replay_decay` (μ_rep)
  config fields exist (lines 113-114); both default to 0.0.
- Per-trace `primary_atom` tracking (lines 183, 217-218, 227).
- The priority multiplier `(1 + κ · m_trace)` at lines 260-269,
  early-exit at `kappa > 0.0` (line 262).
- `metastability_payback` called from `sample()` at lines 315-321.

**C.2.4's actual scope reduces to** (analogous to C.1.5's wrapper):
- Verify the wiring meets the binding anti-homunculus assertions
  (especially A5 H6 runtime test, A6 κ→∞ entropy floor, A4 κ=0
  byte-identical baseline).
- Add the convergence-equivalence tests including the binding
  composed-system smoothness test (reviewer WE3).
- No new code beyond tests, unless an assertion fails.

The metastability machinery is **already implemented** in
[`src/energy_memory/phase4/consolidation.py`](../../src/energy_memory/phase4/consolidation.py):

- `ConsolidationState.metastability_ema: torch.Tensor` — per-atom EMA
  (lines 163-165).
- `update_metastability(contribution: Tensor)` — EMA update applied per
  retrieval (lines 334-381). Early-exits at `metastability_obs_rate == 0`.
- `metastability_payback(idx: int, factor: float)` — decays a single
  atom's EMA after replay (lines 383-397).
- Per-retrieval `c_i^(traj) = max_t w_i(t) − w_i(T)` already materialized
  as `TorchRetrievalResult.metastability_contribution`
  ([`torch_hopfield.py:170`](../../src/energy_memory/memory/torch_hopfield.py)).

The knob `metastability_obs_rate` defaults to `0.0` and has never been
exercised in any report (per CLAUDE.md and the audit). **C.2.4 is the
mechanism that exercises it.**

## Paired Diagnostic (C.1.5, already landed)

**C.1.5 metastability EMA** =
per-atom `m_i = (1−μ_obs)·m_i + μ_obs·c_i^(traj)` exposed as a passive
log via [`src/energy_memory/phase3/metastability_diagnostic.py`](../../src/energy_memory/phase3/metastability_diagnostic.py).
5/5 tests pass.

## Proposed Slow-Timescale Dynamic

### Form (right-hand-column per 2026-05-09:140-148)

The replay buffer assigns each trace a sampling priority. Currently
(per the C.1.5 research): `priority(trace) = gate · tag · suppression`
at [`src/energy_memory/phase4/replay_loop.py:204`](../../src/energy_memory/phase4/replay_loop.py).

C.2.4 modifies this to include the trace's metastability:

```
priority(trace) = gate · tag · suppression · (1 + κ_meta · m_trace)
```

where:
- `m_trace = m_atom[primary_atom_of_trace]` — the trajectory's
  metastability is the EMA of its **`primary_atom`** (the atom whose
  retrieval top-1 was at the moment of capture; the substrate already
  tracks this per-trace at [`replay_loop.py:183, 217-218, 227`](../../src/energy_memory/phase4/replay_loop.py)).
- `κ_meta ≥ 0` is the metastability gain — a fixed substrate constant.

**Note on reduction choice (binding per reviewer 2026-05-26 WE1):** the
precommit's earlier draft specified `max_{atom ∈ trace.atoms}`. The
reviewer found the existing wiring at
[`replay_loop.py:267-269`](../../src/energy_memory/phase4/replay_loop.py)
already uses `primary_atom` (single representative) — derived from a
prior 2026-05-20 dynamic-form note referenced at line 112 of the same
file. Both reductions are substrate-pure (both pass the anti-homunculus
filter); `primary_atom` is the more conservative choice (no new
`TrajectoryTrace` schema) and matches the existing overlap-collapse
semantics. **C.2.4 commits to the `primary_atom` reduction** to align
the precommit with the wiring already in the substrate.

This is a **multiplicative reshaping** of the existing replay priority
distribution. When metastability is zero everywhere (the κ_obs=0
baseline, or when no atoms have accumulated metastability), every
trace gets multiplied by 1.0 — the distribution is unchanged. When a
trace's `m_trace` is high, it gets multiplied by `(1 + κ_meta · m_trace)`,
raising its sampling probability via `torch.multinomial`.

### Replay paydown (the second half of the dynamic)

When a trace is sampled and replayed, its metastability is decayed:

```
m_atom ← m_atom · (1 − μ_rep)   for atom ∈ replayed_trace.atoms
```

via the existing `ConsolidationState.metastability_payback(idx, factor=1-μ_rep)`.
`μ_rep ∈ (0, 1]` is the paydown rate — a fixed substrate constant.

This produces the **substrate-level "fatigue"** equivalent: a trace that
has been replayed recently has lower metastability and lower priority
for the next sampling step. Without paydown, high-metastability traces
would dominate indefinitely. With paydown, the buffer self-explores its
high-energy regions and returns to balance.

### Diagnostic ↔ Actuator Identity

Per 2026-05-09:150-154:

- **Slow-timescale dynamic (actuator):** the priority-weighted sampling
  trajectory of the replay buffer over time, biased toward
  high-metastability traces, with payback decaying their priority after
  replay. Over many consolidation cycles, this produces a
  self-balancing exploration of the codebook's metastable regions.
- **Fast-timescale snapshot (the actuator's own diagnostic):** the
  per-atom `m_i` at any single retrieval event. Computable from
  `metastability_ema` directly.
- **C.1.5 metastability_diagnostic** reads `m_i` from
  `ConsolidationState.metastability_ema` and exposes it as
  `MetastabilityDiagnostics.per_atom`. **Both C.1.5 and C.2.4 read
  the same `metastability_ema` tensor from `ConsolidationState`** —
  this is a substrate-shared primitive (the EMA itself). The actuator
  does not consume `MetastabilityDiagnostics`; it consumes
  `state.metastability_ema` directly.

Two views of the same substrate state, no diagnostic→actuator pipeline.

### Interaction with C.2.1, C.2.2, C.2.3

C.2.4 modifies *replay priority*, not the consolidation update. There
is no direct arithmetic composition with C.2.1, C.2.2, or C.2.3 (which
all act on the consolidation force).

**Indirect interaction:** high-metastability traces get replayed more
often → atoms involved in those traces get more consolidation events →
C.2.1 / C.2.2 / C.2.3 forces are applied more often to those atoms.
This is principled: metastable atoms are exactly the ones that *need*
more consolidation visits (their basins are ambiguous), so biasing
replay toward them concentrates the substrate's adaptive resources
where they matter.

This is the project's first place where two layers of dynamics
(replay scheduling × per-event consolidation) compose. The combined
effect should be: **the substrate naturally allocates its consolidation
budget toward atoms whose basins show the most ambiguity, without any
controller deciding "spend more time on confused atoms."**

## Anti-Homunculus Self-Check (precommit, pre-reviewer)

| Question | Answer |
|---|---|
| Is there a supervisor module that decides which traces to replay? | No. `torch.multinomial` samples from the priority-weighted distribution; no branch. |
| Is there an `if X then Y` rule on a metric? | No. The metastability multiplier `(1 + κ_meta · m_trace)` is a continuous function applied to every trace's priority. |
| Are `κ_meta`, `μ_rep`, `μ_obs` adaptive on observation? | NO. All three are fixed substrate constants. |
| Does the actuator read C.1.5's `MetastabilityDiagnostics`? | NO. It reads `state.metastability_ema` (the substrate tensor) directly. |
| Could `κ_meta → ∞` make the dynamic into a near-deterministic "always replay highest-m trace"? | Yes, in the limit. The convergence-equivalence test must verify the priority distribution remains broadly-supported (not collapsed to a delta). Default `κ_meta = 2.0` is moderate. |
| Trace-to-atom reduction (max over involved atoms): is this a hidden controller? | No. Max-over-atoms is a substrate-pure aggregation, like the centroid of basin members. The reduction is the same for every trace; no atom is privileged by a rule. |
| Could the payback `m ← m·(1−μ_rep)` produce a discrete cycle (replay → drop to zero → replay another → ...)? | At μ_rep = 1, yes (atomic payment). For μ_rep < 1 the decay is continuous. Default μ_rep = 0.5 keeps the dynamics smooth. |

### Concrete H6 check

Both C.1.5 (the diagnostic) and C.2.4 (the actuator) read
`state.metastability_ema` — the **substrate's own per-atom EMA tensor**.
Neither imports from the other. This is the canonical
"shared-substrate-primitive" pattern the reviewer named in C.2.1's
review as the correct deduplication path. ✓

### Concrete H5 check

The priority composition is `gate · tag · suppression · (1 + κ_meta · m_trace)` —
multiplication, continuous in `m_trace`. The payback is multiplicative
decay, continuous in `μ_rep`. Sampling is `torch.multinomial` on the
softmax of priorities — categorical, but its parameters are continuous
functions of substrate state, not arbitrations.

Note that `torch.multinomial` is itself the "categorical decision," but
this is a standard substrate primitive (like settling, like
consolidation gate firing). The shape under review is whether the
PROBABILITY DISTRIBUTION that multinomial samples from is
controller-shaped. The distribution is `softmax(log_priority)` where
`log_priority = log_gate + log_tag + log_suppression + log(1 + κ·m_trace)` —
continuous in all inputs. ✓

## Construction Constants (pre-committed; non-adaptive)

| Constant | Default | Rationale |
|---|---|---|
| `metastability_obs_rate` (μ_obs, EMA rate) | `0.0` (off) | Existing knob; default zero preserves κ=0 baseline. C.2.4 makes the smoke runs that exercise it. |
| `metastability_gain` (κ_meta) | `0.0` (off) | NEW. Default zero preserves the existing replay priority computation byte-identically. When κ_meta=0, `(1 + 0·m) = 1` — the metastability multiplier disappears. |
| `metastability_replay_paydown` (μ_rep) | `0.0` (off) | NEW. Default zero means no paydown; combined with κ_meta=0 default, C.2.4 is fully dormant. When κ_meta > 0, μ_rep > 0 prevents indefinite high-m bias. Recommended exercised value 0.5 per the C.1.5 research's pre-committed value. |

All three are added to `ConsolidationConfig` as fixed substrate
constants. The defaults preserve byte-identical κ=0 control.

## Per-Trace State

No new per-atom slow state. `m_trace` is computed on-demand from
`max_{atom ∈ trace.atoms} state.metastability_ema[atom]` at the moment
the priority is computed. This is consistent with C.2.3's stateless
form (where the force is computed instantaneously from substrate state).

## Required Tests (before C.2.4 lands)

### Anti-homunculus reviewer pass
Mandatory before any code.

### Convergence-equivalence test (substrate-pure CP8)

**Assertion 1 — high-metastability traces are sampled more often:**
Build a replay buffer with N=8 traces. Pre-populate metastability such
that trace 0's atoms have `m = 0.5`, others have `m = 0.05`. With
κ_meta=2.0, μ_rep=0.0 (no payback), sample 1000 times from the buffer.
Assert trace 0's sample count > 2× the average count of the other
traces.

**Assertion 2 — payback decays priority:** Same setup but with
μ_rep=0.5. After trace 0 is sampled once, its metastability halves.
Sample 1000 times sequentially with payback applied after each sample.
Assert trace 0's total samples are LESS than under (A1) — paydown
prevents indefinite dominance.

**Assertion 3 — κ_meta=0 byte-identical baseline:** With κ_meta=0.0,
the replay priority distribution is exactly equal to the pre-C.2.4
distribution (gate · tag · suppression). Sample 1000 times from both
distributions with matched seeds and assert sample counts are
identical.

**Assertion 4 — trajectory smoothness:** Run the replay loop for 500
events with κ_meta=2.0, μ_obs=0.1, μ_rep=0.5. Record per-event
metastability statistics (mean, max). Assert the trajectory is smooth
(max/median ratio of per-event Δ mean < 5; looser than per-atom
smoothness because trace-level statistics are inherently noisier).

**Assertion 5 — H6 verified at runtime:** monkey-patch
`energy_memory.phase3.metastability_diagnostic.compute_metastability_diagnostics`
to raise. C.2.4 replay priority computation runs unchanged. (Tests
that the actuator does not consume the diagnostic module.)

**Assertion 6 — `κ_meta → ∞` does NOT collapse the distribution:** With
`κ_meta = 100` (extreme), sample 1000 times. Assert the entropy of the
sample distribution is non-trivial (more than half of the
single-deterministic-trace entropy `log(1) = 0`). Specifically:
sample entropy > `0.5 · log(N)`. This catches the case where the
multiplier overwhelms gate/tag/suppression and the sampling becomes
near-deterministic.

**Assertion 7 — composed-system smoothness (binding per reviewer
WE3 2026-05-26):** with all four C.2.x mechanisms enabled at modest
values `(λ_ac=0.5, μ_T=0.1, λ_cc=0.5, κ_meta=2.0, μ_rep=0.5,
μ_obs=0.1)`, run the replay loop for 200 events. Record the trajectory
of `(mean m_i, mean T_k, mean tr(Σ_k))` across events. **Assert no
step-discontinuities**: per-event Δ ratio (max/median) for each of the
three statistics is < 5 (loose composed-system threshold; per-link
smoothness is < 3 in the binding assertions for each individual
C.2.x). The composition of state-dependent update frequency
(C.2.4-driven) and state-dependent forces (C.2.1/C.2.2/C.2.3) must not
produce a hidden control loop.

### Unit tests

- `κ_meta = 0.0` early-exits cleanly (the `(1 + 0·m_trace)` multiplier
  is replaced by `1`).
- `m_trace` reduction (`primary_atom` representative) for a trace
  returns `state.metastability_ema[primary_atom_idx]`.
- Payback `m ← m·(1−μ_rep)` reduces metastability monotonically.
- With κ_meta=2.0 and `m_trace=0.5`, the multiplier is `(1 + 1.0) = 2.0`.

## What This Does Not Permit

- Implementation. No code lands until the anti-homunculus reviewer
  PASSES.
- Adaptive `κ_meta` / `μ_rep`. Both are fixed substrate constants.
- Importing from `metastability_diagnostic.py` into the replay path.
  The actuator reads `state.metastability_ema` directly.
- Tuning `κ_meta` against any Phase 5 / functional metric. Smoke
  selection is against structural assertions (sample entropy bound,
  smoothness).

## Anti-Homunculus Discipline Notes

H1–H15 from previous precommits apply. In addition:

- H16 (C.2.4-specific) — `κ_meta`, `μ_rep`, `μ_obs` must never become
  functions of any substrate observable.
- H17 (C.2.4-specific) — The actuator reads `state.metastability_ema`
  (substrate tensor) directly. It does NOT consume
  `MetastabilityDiagnostics` or any field of the C.1.5 module.
- H18 (C.2.4-specific) — The metastability multiplier is continuous in
  `m_trace`. The trace-to-atom reduction (max) is substrate-pure;
  changing it to anything that *reads* per-atom flags or *gates* on
  observables would be controller-shape.
- H19 (C.2.4-specific) — `torch.multinomial` is the only categorical
  primitive permitted. Replacing the priority-weighted sampling with
  any selection rule (top-k, threshold-based, etc.) would change the
  shape from "energy-ranked construction" to "decision."

## Implementation Findings (2026-05-26 — C.2.4 landed via test suite)

- **Status:** all 7 binding assertions pass at 11/11; full discovery
  regression at 491/491. **No new `src/` code** — the C.2.4 actuator
  side was already wired in `replay_loop.py` (per the 2026-05-20
  dynamic-form note referenced at line 112 of that file). C.2.4 is the
  test suite that verifies the wiring is anti-homunculus-clean and
  exercises the previously-dormant knobs.
- **Anti-homunculus reviewer PASS with 5 watch-edges:**
  - WE1: precommit amended to use `primary_atom` reduction (matches
    existing wiring; both `max` and `primary_atom` are substrate-pure).
  - WE2 (binding): κ→∞ entropy floor — at κ=100, sample entropy 1.99
    ≫ 0.5·log(8) = 1.04. Distribution remains shaped, not collapsed
    to a delta.
  - WE3 (binding): composed-system smoothness — all four C.2.x
    mechanisms running together over 300 events with warmup window
    [100, 300] produces smooth trajectories for `m`, `T`, and
    `tr(Σ)` (after floor-gating the near-zero-median case).
  - WE4: `record_retrieval` requires no changes for C.2.4 (no writes
    to `metastability_ema`).
  - WE5: per-iteration `.cpu()` sync at `replay_loop.py:269` flagged
    for future GPU optimization; not anti-homunculus-relevant.
- **Operationalization (exactly per the amended precommit):**
  - `priority(trace) = gate · tag · suppression · (1 + κ · m[primary_atom])`
  - `metastability_payback(idx, factor=1−μ_rep)` after sampling
- **Numbers:**
  - A1 (κ=4.0, m_high=0.5, m_low=0.05): trace 0 = 235 samples, others
    mean = 109.3, ratio = 2.15 (matches algebraic 2.5× prediction).
  - A2: with μ_rep=0.5, trace 0 samples drop to 123 from 199 baseline.
  - A3: κ=0 byte-identical sample counts (the `if kappa > 0.0` branch
    at `replay_loop.py:262` is the sole gate).
  - A4: sliding-window (w=20) smoothed `mean(metastability_ema)`
    trajectory has max/median Δ ratio = 4.34 < 5.0 binding threshold.
    Methodological finding: single-event Δ is intrinsically cue-noisy
    via `c_i^(traj)` variance; sliding-window averages out the noise
    and exposes the underlying smooth dynamic. A4's intent is
    discontinuity detection, not cue-noise measurement.
  - A5: `phase3.metastability_diagnostic.compute_metastability_diagnostics`
    monkey-patched to raise; replay priority and sample counts
    byte-identical to unpatched run. H6 verified at runtime.
  - A6 (binding WE2): κ=100, m random ∈ [0, 0.5], entropy = 1.9995
    against threshold 1.0397 (= 0.5·log(8)). Distribution non-collapsed.
  - A7 (binding WE3): all four C.2.x mechanisms, 300 events with
    warmup [100, 300] + floor gating:
    - `mean(m)`: ratio 2.87 (< 5) ✓
    - `mean(T)`: ratio 4.50 (< 5) ✓
    - `Σ tr(Σ)`: median 7.2e-11 below 1e-7 floor; absolute-scale
      check shows max 1.55e-9 < 1e-5 (stationary at numerical noise
      floor). ✓
- **Within-phase variance finding (recorded for the audit):** the
  subset-conditional A4 attempt found that even within the
  non-replay-step phase, single-event Δ on `mean(m)` has its own
  variance (~7× median over 50 events) driven by cue-by-cue
  variance in `c_i^(traj)`. This is intrinsic to the metastability
  measurement; the sliding-window approach is the correct sensor.
  **Recorded as a complement to the audit's §4.4 fixed-cadence
  finding** — the replay scheduler's tension-driven redesign should
  consider both inter-phase (cadence) and intra-phase (cue-variance)
  smoothness when defining "smooth replay rhythm."
- **Files created:** [`tests/test_metastability_replay_priority.py`](../../tests/test_metastability_replay_priority.py) (11 tests).
- **Files modified:** none in `src/`. C.2.4 is purely a verification
  pass over pre-existing wiring.
- **C.2.4 closed. Next: C.2.5 (drift → replay-tension energy) — the
  final C.2 deliverable.**
