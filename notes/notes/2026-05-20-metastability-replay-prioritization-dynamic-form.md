---
date: 2026-05-20
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: design-note
session-closes: STATUS-phase5-pair4-pivot
---

# Metastability ~ Replay-Prioritization Dynamic Form (Pair #4)

Companion to:

- [2026-05-20 diagnostic-actuator death-dynamic note](2026-05-20-diagnostic-actuator-death-dynamic-form.md) (Pair #1, A+B — closed)
- [2026-05-20 discovery-channel r_ema init note](2026-05-20-discovery-channel-r-ema-init-dynamic-form.md) (Fix A1)
- [2026-05-20 `r_inst` redundancy-measure note](2026-05-20-r-inst-measure-dynamic-form.md) (Fix A1')
- [2026-05-20 cue-regime / role-prior note](2026-05-20-cue-regime-role-prior-dynamic-form.md) (Pair #3 attempt — falsified)
- [brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md](../../brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md) (the ranked-pivot brief that names this pair as the cleanest next move)

Held immediately after [report 051](../../reports/051_phase5_tier1_disambiguation.md)
falsified all three Tier-1 alternative fidelity metrics on the
existing A1' substrate. The four-step debugging chain
(A+B → A1 → A1' → β) built the substrate the design specified
(d_eff preserved, duplicates throttled, original atoms dominate) but
did not produce a substrate with per-atom role-fidelity variance at
D=4096. The K-branch state_divergence operationalization of pair #3
(bimodality / splitting) has reached the FHRR crosstalk noise floor
1/√D ≈ 0.016 and is empirically exhausted as a Phase 5 graduation
route.

Per the 2026-05-20 brainstorm's pre-committed decision tree (Idea 5,
ranked #1 of three open pairs), Phase 5 pivots to **pair #4:
metastability ~ replay-prioritization**.

Closes [STATUS](../../STATUS.md) Phase 5 pivot decision as a design
note. Does **NOT** commit to implementation — that requires the
anti-homunculus reviewer audit followed by a separate decision.

## What this session is for

Per [2026-05-09 §"Diagnostics vs Actuators":156-167](2026-05-09-papers-diagnostics-and-actuator-dynamics.md):

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of. Diagnostic and actuator are the
> same physical process viewed at different temporal resolutions.

For pair #4: the question is **what slow-timescale dynamic is
`meta_stable_rate` (and the per-retrieval metastable flag) a
fast-timescale snapshot of, such that the natural evolution of that
process biases replay toward atoms that need it without any
"if metastable then prioritize" rule firing.**

The brainstorm research brief D names the answer in dynamic-form
shorthand:

> Replay frequency for atom i ∝ `m_i(t)` where m_i evolves as
> `dm_i/dt = ζ · 𝟙{atom_i is in metastable settling} − μ · m_i · 𝟙{atom_i replayed}`.
> m_i accumulates when atom i participates in metastable settlings
> (its softmax weight is high but no single weight peaks); m_i pays
> down when atom i is replayed. **The replay store's priority
> function gains a multiplicative `m_i` factor — no threshold, no
> categorical "metastable" flag.**

This session translates that shorthand into:
1. A precise local-geometric definition of "atom i participated in a
   metastable settling" (no global metric is read; the contribution
   is a per-atom softmax-weight statistic the substrate already
   computes and currently discards).
2. The Saighi/Benna-Fusi-shape EMA on that local quantity.
3. Plumbing into the existing `ReplayStore` priority composition
   (currently `gate × tag × suppression`).
4. Pre-committed falsification criteria so the retrain cannot
   rationalize either direction of effect as success.

## What the diagnostic established

`meta_stable_rate` is the Phase 4 D1 headline ([report 038](../../reports/038_phase4_d1_graduation.md):
Δms_w3 = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds). It is defined
([phase2/metrics.py:69](../../src/energy_memory/phase2/metrics.py)) as
the fraction of retrievals whose top softmax score falls below a
threshold (default 0.95) — i.e. retrievals that did not converge
sharply to a single basin.

The per-retrieval substrate already exposes everything required:
- `TorchRetrievalResult.weights` ([torch_hopfield.py:29](../../src/energy_memory/memory/torch_hopfield.py))
  is the **full per-pattern softmax vector** at the settled state.
  Currently consumed only for entropy and top-1 selection.
- `TorchRetrievalResult.top_score` is the max similarity score; the
  metastable flag is `top_score < 0.95`.
- Aggregate `meta_stable_rate` is computed every replay cycle on the
  evaluation buffer.

**The per-atom metastability contribution is already being computed
inside every retrieve() call and discarded after.** Specifically: for
a retrieval with weight vector `w`, atom i's contribution to the
metastable signature is

> `c_i = w_i · (1 − max_j w_j)`

This is bounded in `[0, 1/4]` (maximised at a perfectly-tied retrieval
with one of the tied atoms being i; zero at any sharp retrieval and
zero at any retrieval where i contributes negligible weight). No
threshold; no membership flag. The substrate's softmax computation
*is* the diagnostic — the question is whether to integrate this local
quantity into a slow-timescale variable.

## The four-step chain status (before this note)

| Layer | Mechanism | Verdict at n=1 |
| --- | --- | --- |
| Substrate-construction | A+B | PASS (d_eff preserved, [report 044](../../reports/044_consolidation_geometry_diagnostic.md)) |
| Measurement (per-atom redundancy proxy init) | A1 | PASS (discovery atom strengths reduced 20–33%, [report 048](../../reports/048_phase5_a1_pilot_seed17.md)) |
| Reduction operator (RMS → max-over-others) | A1' | PASS (sparse duplicates correctly read as redundant) |
| Selector layer (continuous fidelity-weighted prior) | β | FAIL ([report 050](../../reports/050_phase5_beta_smoke_seed17.md); uniform f_i at D=4096) |
| Tier-1 metric rescue | decode-margin / settling / pairwise | FAIL ([report 051](../../reports/051_phase5_tier1_disambiguation.md); all three below std > 0.02 threshold) |

The substrate is built as designed and Phase 4 D1 graduation is
preserved. The K-branch state_divergence operationalization of pair #3
is the failure layer.

## The wrong-shape candidate this note rejects

The most natural mistake — and the one the 2026-05-09 note warns
"the temptation will be strongest exactly here" — is:

```python
# Pseudocode of the wrong-shape candidate
if retrieval.top_score < 0.95:
    for i, w in enumerate(retrieval.weights):
        if w > 0.1:
            replay_store.boost_priority(i, factor=2.0)
```

Shape diagnosis:
- A controller reads `top_score`, applies a threshold (0.95).
- A second controller reads each `w_i`, applies a second threshold (0.1).
- A discrete `boost_priority` operation is triggered.
- The metastability bias is the *side effect* of accumulated boosts,
  not a quantity any local dynamic is integrating.

Anti-homunculus filter: **fails**, on the same grounds as the binary
death mechanism the death-dynamic note rejects. Even though all read
quantities are local-geometric (softmax weights), the two thresholds
and the discrete boost together constitute a controller arbitrating
which atoms get prioritized.

## The dynamic-form candidate: per-atom metastability EMA `m_i`

### Local definition (the per-retrieval contribution)

For each `retrieve()` call exposing weights `w ∈ ℝ^N` and `max_w =
max_j w_j`:

> `c_i = w_i · (1 − max_w)`  ∀ i ∈ [N]

`c_i` is a continuous, per-atom, local-geometric quantity. No
threshold is read; no `if-then`. Atoms outside the retrieval (those
with negligible softmax weight) contribute c_i ≈ 0 automatically by
the softmax's exponential roll-off, *not* by a membership test.

### Slow dynamic (the EMA)

Per-atom metastability variable `m_i ∈ [0, ∞)` evolves on every
retrieval that touches the substrate:

> `m_i ← (1 − μ_obs) · m_i + μ_obs · c_i`

where `μ_obs ∈ (0, 1)` is the observation timescale. Atom i's `m_i`
is the EMA of its softmax-weighted metastability contribution across
recent retrievals. Inactive atoms decay toward zero with timescale
`1/μ_obs`. Frequently-metastable atoms saturate near `1/4` (the local
upper bound on c_i).

This is the Saighi-shape ([report 034](../../reports/034_saighi_ak_seed1_prototype.md))
per-atom adaptation variable, repurposed: instead of accumulating on
retrieval *count* (which led to the n=10 falsification in [report 035](../../reports/035_saighi_ak_n10_falsification.md)),
it accumulates on **per-retrieval softmax-weighted metastability
contribution**. The shape inherits Saighi's locality and the
Benna-Fusi (2016) fast-variable accumulator role; the *content* is
HEN's (Kashyap 2024) metastable-state diagnostic surfaced at the
per-atom scale.

### Pay-down on replay (the dual term)

When atom i is sampled by the replay store and used for a
consolidation update, `m_i` pays down:

> `m_i ← (1 − μ_rep) · m_i`

with `μ_rep ∈ (0, 1)` the replay-timescale decay. This is the Benna-
Fusi "fast variable flushes into slow" recipe: m_i is the fast
accumulator that the replay event drains.

### Plumbing into the replay-store priority

Currently ([replay_loop.py:185-189](../../src/energy_memory/phase4/replay_loop.py)):

```python
def _priorities(self) -> List[float]:
    return [
        self.gate_signals[i] * self.tag_counts[i] * self.suppression[i]
        for i in range(len(self.traces))
    ]
```

Note `ReplayStore` indexes *traces*, not atoms — a trace is the
trajectory record of one observation, not a per-atom record. So
`m_i` enters as a per-trace aggregation: each stored trace `t`
already knows which atom indices its query overlaps (via the
substrate's similarity computation in `_find_overlap`). The trace's
metastability priority is

> `m_trace = m_{i*(trace)}`

where `i*(trace)` is the highest-similarity stored atom for that
trace's query. (For multi-atom traces, use the max-over-atoms; the
max-over-others reduction inherits A1's shape choice.)

The new composition is

> `priority(t) = gate(t) · tag_count(t) · suppression(t) · (1 + κ · m_trace)`

with `κ ≥ 0` the metastability gain. The `(1 + κ · m_trace)` form
preserves the existing priority scale when `m_trace = 0` (no
metastability signal) and biases priority continuously upward when an
atom has accumulated metastability. **No threshold; no `if-then`.**
At `κ = 0` the new system is identical to the old one (this is a
load-bearing precondition for the n=10 non-regression test).

### Anti-homunculus check (this note's mandatory section)

Per the death-dynamic note's "explicitly write out who decides X, who
decides Y, where the 'decision' actually lives in the dynamics."

- **Who decides which atom gets prioritized for replay?** No one.
  Atom i's contribution to its trace's priority is `m_i`, which is an
  EMA over `c_i = w_i · (1 − max_w)` — a per-atom softmax-weight
  statistic computed by the existing retrieve() call. The retrieval's
  softmax *is* the decision. The replay-store's `torch.multinomial`
  *is* the selection. There is no controller reading "is i
  metastable?" and acting.
- **Who decides when m_i pays down?** No one. m_i pays down by a
  factor `(1 − μ_rep)` whenever atom i's trace is sampled by the
  replay store — which is itself a softmax-multinomial event. The
  pay-down is a local response to the local sampling event, not a
  scheduled action.
- **Who decides the metastability gain κ?** It is set once at
  substrate construction or per-retrain from theoretical
  considerations (paper-derived or one-shot calibration). It is
  **NOT** tuned to land `meta_stable_rate` in a target range. If the
  first retrain misses the falsification criterion, that is a
  falsification result, not an invitation to retune κ. (This is
  symmetric to A+B's α/λ commitment in the death-dynamic note.)
- **Does any global metric drive an action?** No. The aggregate
  `meta_stable_rate` is reported as a diagnostic — it is the snapshot
  the dynamic is integrating, not a switch the architecture reads.
  At no point is `meta_stable_rate` compared against a threshold to
  trigger a behavioral change.
- **Is `μ_obs` / `μ_rep` a schedule in disguise?** No. They are fixed
  substrate parameters analogous to `coverage_lambda` in
  `ConsolidationConfig`. They are not modulated by any observed
  signal during training.

Anti-homunculus filter: **PASS** by construction. The mechanism
inherits A+B's shape (per-atom EMA on local-geometric quantity,
multiplied into an existing energy-ranked sampling) — it does not
introduce a new controller layer.

The required confirmation is an **explicit anti-homunculus reviewer
audit before any code lands**, with the audit specifically checking:

1. `c_i` is computed from `retrieve()`'s existing weights vector, not
   from a separately-invoked re-evaluation that reads a metric.
2. The `(1 + κ · m_trace)` priority multiplication is the *only* edit
   to `_priorities()`. No supplementary `if m_trace > X` branch is
   added.
3. The pay-down `m_i ← (1 − μ_rep) · m_i` is triggered by the
   sampling event itself (i.e. inside `sample()` after
   `torch.multinomial`), not by any explicit boost-policy code.
4. κ, μ_obs, μ_rep are set once and surfaced as config; no
   feedback-loop code reads `meta_stable_rate` and writes any of
   them.

## Why the prior chain's failure does not contaminate pair #4

The K-branch state_divergence chase (β path, [report 050](../../reports/050_phase5_beta_smoke_seed17.md))
failed because role-fidelity `f_i = mean(1 − |G_jk|)` is uniform at
D=4096 — a **substrate-encoding-level** structural fact (FHRR
crosstalk 1/√D ≈ 0.016 sets f_i ≈ 0.984 regardless of pattern
content). The metric had no per-atom variance to expose.

Pair #4 uses `c_i = w_i · (1 − max_w)`, which has guaranteed
per-atom variance **by construction**: w_i is the softmax of per-
pattern scores against a *cue*, not a pairwise pattern-pairwise
quantity. Different retrievals produce different w distributions;
different atoms therefore accumulate different m_i values over the
training trajectory. The FHRR-noise-floor failure mode does not
apply.

More directly: Phase 4 D1 already proved meta_stable_rate has
substantial cross-condition variance at this substrate
([report 038](../../reports/038_phase4_d1_graduation.md): Δms_w3 =
−0.7920, CI [−0.9376, −0.6463] across 10 seeds). The substrate has
variance in meta_stable_rate. It does not have variance in
role-fidelity at D=4096. Pair #4 uses the metric class the substrate
*has* variance in.

## Pre-committed falsification criteria

Before any retrain, pre-commit these (binding, per H4 / [phase-5-checklist.md:195](../emergent-codebook/phase-5-checklist.md)):

- **D1 non-regression (Phase 4 preservation):** Δms_w3 ≤ −0.5 with CI
  disjoint from zero across 10 seeds. The new mechanism must not
  break Phase 4. ([Report 038](../../reports/038_phase4_d1_graduation.md)
  bar of −0.79 is the prior; the floor is the −0.5 CI-disjoint test
  the design originally specified.)
- **Phase 5 headline (graduation criterion):** **Δ meta_stable_rate
  at W=3 with `m_i`-weighted replay vs the κ=0 control**, n ≥ 10
  seeds, **CI disjoint from zero in the direction of reduced
  metastability** (Δ ≤ −0.1, CI upper bound < 0). The control is
  identical training with κ=0 — i.e. the new code path runs but the
  metastability factor is multiplied by 0, so the priority composition
  collapses to the pre-pivot baseline. **Same seeds, same data, same
  hyperparameters.** This is the symmetric n=10 falsification bar
  the death-dynamic note specified for A+B.
- **Replay-store dynamics sanity:** during training, the distribution
  of `m_i` across surviving atoms should become non-uniform within
  the first 500 steps (std/mean > 0.1). If m_i remains uniform across
  atoms, the mechanism is not differentiating; that is a measurement
  failure separate from a graduation failure. Pre-commit before
  observing.
- **Substrate non-regression:** substrate d_eff ≥ 25 at step 1800
  (the A+B pre-commitment from the death-dynamic note). The
  metastability mechanism must not collapse the substrate.
- **κ, μ_obs, μ_rep are set once before the first retrain from
  theoretical considerations** (paper-derived or one-shot
  calibration), NOT tuned to land Δ meta_stable_rate at a target. If
  the first retrain misses the falsification criterion, that is a
  falsification result, not an invitation to retune. Retuning is
  permitted only after the retrain has been reported as falsified and
  a redesign session has re-derived the parameters from first
  principles.

If the mechanism passes D1 non-regression but the Phase 5 headline
fails to reach Δ ≤ −0.1 CI-disjoint from zero, the mechanism is
falsified. The discipline is to report that and either re-scope or
declare Phase 5 graduation-unattained — **not** to chase a second
mechanism revision.

If the mechanism passes both the D1 non-regression and the Phase 5
headline, Phase 5 graduates on a substrate-pure metric class the
project has already validated.

## What this design note explicitly does NOT do

- It does **NOT** commit to a Phase 4 retrain. That requires the
  anti-homunculus reviewer audit and a separate decision.
- It does **NOT** close out pair #2 (drift / replay-pressure) or
  pair #5 (cap-coverage / restructuring). Those remain as Phase 5.5
  and Phase 6 targets per the brainstorm's ranked recommendation.
- It does **NOT** re-open the K-branch state_divergence chase. The
  bimodality / splitting pair (#3) is folded into pair #5
  (restructuring-by-creation) as the right architectural scale; the
  K-branch operationalization is closed as falsified-by-substrate-
  encoding.
- It does **NOT** change Phase 4 D1's graduation status. D1
  graduation stands ([report 038](../../reports/038_phase4_d1_graduation.md));
  the non-regression test is a safety constraint, not a re-graduation.
- It does **NOT** add a new substrate-energy term. The mechanism is
  pure replay-store priority modulation. (Pair #5 would add an
  H_restruct term; pair #4 does not.)
- It does **NOT** require the θ′(β) calibration spike that pair #5
  would need. The metastability mechanism reads no Vangara-Gopinath
  cap baseline.

## Implementation sketch

Files that would change (sketch only, NOT to be implemented from this
note alone):

- `src/energy_memory/phase4/consolidation.py` — add a per-atom
  `metastability_ema: torch.Tensor` field to `ConsolidationState`
  parallel to `r_ema`. Initialise at zero on `add_pattern`. Surface
  an `update_metastability(weights: torch.Tensor)` method that
  applies the EMA update from a retrieval's full weight vector.
- `src/energy_memory/phase4/replay_loop.py` —
  - `ReplayStore.__init__` gains a `metastability_gain: float = 0.0`
    parameter (κ) and an optional `consolidation: ConsolidationState`
    handle.
  - `ReplayStore.add` records each trace's primary-atom index
    `i*(trace)` (max-similarity stored atom for the trace's query),
    computed via the existing `_find_overlap`-style scan.
  - `_priorities()` multiplies in `(1 + κ · m_{i*(trace)})` when the
    consolidation handle is non-None.
  - `sample()` triggers `m_{i*(trace)} ← (1 − μ_rep) · m_{i*(trace)}`
    for each sampled trace, inside the same code block that already
    applies `suppression_decay`.
- `src/energy_memory/phase4/replay_loop.py::UnifiedReplayMemory.retrieve_and_observe`
  — after each retrieve() call, invoke
  `consolidation.update_metastability(result.weights_tensor)`. The
  weights are already computed; this is a single tensor write per
  retrieval.
- `src/energy_memory/memory/torch_hopfield.py::TorchRetrievalResult`
  — add a `weights_tensor: torch.Tensor` field alongside the existing
  `weights: List[float]` (the list is preserved for backwards
  compatibility). The tensor is the pre-CPU-sync version; the list is
  derived from it. This avoids re-syncing for the m_i update.
- Tests:
  - `tests/test_phase5_metastability.py` (new) —
    - `test_metastability_ema_updates_correctly`: synthetic weights
      tensor, EMA update produces expected per-atom values.
    - `test_metastability_payback_on_replay`: simulate one
      `ReplayStore.sample` event; verify m_{sampled} decreases by the
      expected factor.
    - `test_kappa_zero_preserves_priority`: with κ=0 the priority
      composition is bit-identical to the pre-pivot baseline.
    - `test_metastability_diagnostic_sanity`: high-entropy retrieval
      produces non-zero m_i contributions; sharp retrieval produces
      m_i ≈ 0.
  - `tests/test_consolidation_state.py` — add an `add_pattern` test
    that verifies `metastability_ema` is initialised at zero for a
    new atom.

Cost estimate: ~½ day plumbing + tests (per the brainstorm research
brief D's "lowest implementation effort of the three" rating). Plus
the anti-homunculus reviewer audit. Plus the n=10 Colab retrain (the
graduation attempt). Plus the headline-with-CI write-up.

Done-gates (per [CLAUDE.md "What 'done' looks like"](../../CLAUDE.md)):
1. Δ meta_stable_rate at W=3 with bootstrap CI.
2. κ=0 control on the same seeds and test set.
3. Drill-downs explain the headline: per-atom m_i distribution at
   step 1800, fraction of replay events biased by κ · m_trace > 0.1,
   Phase 4 D1 preserved.
4. Numbered report under `reports/` (next number).
5. STATUS.md and phase-5 checklist updated.

## What the next session should do

If the anti-homunculus reviewer audits this design and finds it
**PASS**, the next session:

1. Implements `metastability_ema` on `ConsolidationState` and the
   weights-tensor surfacing on `TorchRetrievalResult` first. These
   are pure additions and should not change any test outputs.
2. Implements `update_metastability` and verifies, in isolation,
   that the EMA matches a hand-computed expectation on a synthetic
   weights vector.
3. Implements the `(1 + κ · m_trace)` priority multiplication and the
   `sample()` pay-down. Verifies with the test `test_kappa_zero_preserves_priority`
   that the existing test suite passes bit-identically at κ=0 (262
   tests → 266 tests, no regressions, no per-test-flake re-runs).
4. Runs a 1-seed local pilot at κ > 0 to verify:
   - `m_i` distribution becomes non-uniform within 500 steps.
   - D1 metric stays within 0.05 of the baseline.
5. Pre-registers the n=10 Colab seed list and the κ value chosen.
6. Submits the Colab n=10 retrain. Wall-time is similar to the
   Phase 4 D1 retrain (∼1–2 hours per seed; the m_i update is O(N)
   per retrieval, dominated by the existing softmax).
7. Computes Δ meta_stable_rate at W=3 between κ > 0 and κ = 0 on the
   same seeds. Reports against the pre-committed criteria.

If the audit finds the design **FAIL** on any anti-homunculus
dimension, the design gets reframed before any code lands. Most
likely failure modes the audit should specifically look for:

- A scheduled "recompute and broadcast m_i across atoms" code path
  (would be the controller-in-disguise variant of A's r_ema
  precondition).
- An `if m_trace > X` branch inserted for "performance" (would be a
  threshold leak).
- A feedback loop where the κ value is adapted from observed
  `meta_stable_rate` (would be a controller).
- A "metastable detector" module separate from the retrieve() call
  (would be a separate-arbitration layer).

If the audit fails and the design is unsalvageable in this shape,
the fallback is pair #2 (drift / replay-pressure) per the brainstorm
research brief D's #2 ranking. Pair #2 has the same EMA-into-replay-
priority shape and is expected to inherit the audit cleanly.

## Closing the loop on the four-step debugging chain

The four-step chain (A+B → A1 → A1' → β) built the substrate the
design specified. What the chain did not build is a substrate with
*per-atom role-fidelity variance at D=4096*. The K-branch
state_divergence operationalization of pair #3 required that
variance, and the FHRR crosstalk noise floor structurally denies it.

Pair #4 closes pair #3's intent at a different layer: instead of
"branches separate via per-atom fidelity-weighted prior" (which
required structural variance the substrate cannot provide), the
metastability mechanism redirects replay-pressure toward atoms whose
retrievals are *currently* diffuse, allowing consolidation to settle
them into sharper basins over time. The Phase 5 graduation criterion
moves from "ΔE_K4 CI-disjoint from zero" (the chase that just ended)
to "Δ meta_stable_rate at W=3 CI-disjoint from zero" (the substrate-
pure metric class Phase 4 D1 already validated).

**If pair #4 graduates, the project will have its second
diagnostic-actuator pair in dynamic form** (after A+B's
spread/consolidation closure). Each pair is the same architectural
shape: per-atom EMA on a local-geometric quantity, multiplied into
an existing energy-ranked computation. Pair #2 (drift /
replay-pressure) is symmetric and would be Phase 5.5 if scope
allows; pair #5 (cap-coverage / restructuring) becomes the Phase 6
target.

---

# ADDENDUM (2026-05-21): Trajectory-based `c_i` reformulation

Held same week as the original design note, after the 1-seed Colab
smoke (commit [c98c6ea](https://github.com/Dypatterson/Neuro-AI/commit/c98c6ea)
fix-bypass) confirmed the metastability mechanism FIRES but produces
m_i magnitudes that are too small to bias the replay trajectory. This
addendum reformulates `c_i` (the per-retrieval contribution into m_i)
based on HEN (Kashyap 2024) §rank-reduction findings.

## What the smoke established

Seed 17, A+B+A1' substrate, μ_obs=0.05, κ ∈ {0, 2}, μ_rep=0.5,
n_cues=3000. Both conditions reached final m_max ≈ 0.0082 with
**bit-identical training trajectories**. Meta_stable_w3 was 1.0000 in
both conditions (and identical 11/seed across all 10 seeds in the
original buggy n=10 run that did not call update_metastability, which
the new fix's bit-identical result confirms is the true baseline).

Diagnosis: under the A+B substrate's sharp self-retrieving basins
([report 049](../../reports/049_phase5_a1prime_pilot_seed17.md):
top atom self-retrieves with FP precision), `max_w → 1` at the fixed
point, so `c_i = w_i · (1 − max_w) → 0` for every atom regardless of
how the retrieval got there. The fixed-point operationalization of
metastability has **structurally near-zero magnitude** on this
substrate.

This is the **same shape of failure** as the four-step chain's prior
chapters:
- Failure mode 1 (A+B closed): controller arbitrating over local metric.
- Failure mode 2 (A1 closed): implementer hard-coded a constant where
  a measurement belonged.
- Failure mode 3 (A1' closed): wrong reduction operator (RMS instead
  of max) collapsed a sparse-duplicate signal.
- Failure mode 4 (β at D=4096): substrate-encoding noise floor
  structurally denied per-atom variance.
- **Failure mode 5 (this addendum): fixed-point measurement on
  sharp-basin substrate collapsed a trajectory signal.**

Pattern: each layer's failure mode has a corresponding substrate-side
local-geometric fix at a different operational layer. Each fix is
literature-grounded and anti-homunculus clean. The architecture
discipline has produced its fifth chapter.

## The literature insight

**HEN (Kashyap 2024)** — `tmp/pdf_text/MHN-ENR.txt` — explicitly
addresses fixed-point-vs-trajectory metastability. Two load-bearing
findings from §"Quantifying Meta-Stable States":

> "the dynamics destabilize to low-rank solutions, collapsing the
> retrieval fidelity. For sufficiently high β = [80, 150], the
> dynamics stabilize over a period of time, leading to near-perfect
> recovery"

> "we report the relative rank ( RR = R_S/R_Ξ) of the recovered state
> matrix... For sufficiently high β, the iterates of HEN stabilize
> to provide near perfect retrieval"

The metastability signature lives in **the trajectory of settling
iterates**, not the converged state. Under high β + sharp basins,
the converged state loses the signal; the signal is in
which-atoms-competed-during-settling-and-lost.

**Modern Hopfield Network (Ramsauer 2020 / Krotov-Hopfield)** — cited
by HEN — also notes that β controls convergence behavior. Our Phase 4
β=10 is in the "stabilize-to-sharp-fixed-point" regime, which is
exactly where the fixed-point c_i fails.

**MIR (Aljundi 2019)** — `tmp/pdf_text/OCL-MIR.txt` — operationalizes
its replay-priority signal as a *predicted loss change under virtual
parameter update*, not a fixed-point softmax weight. MIR sidesteps
this failure mode entirely by reading a different kind of signal.
Our pair #4 cannot import MIR's signal directly (we do not have an
explicit task loss), but the methodological lesson applies: choose a
signal that does not collapse at the fixed point.

## The reformulation: `c_i^(traj)`

**Replace** the original fixed-point operationalization

> `c_i = w_i^(final) · (1 − max_j w_j^(final))`            ⟵ Eq. F

**With** the trajectory-based "lost-out atom" operationalization

> `c_i^(traj) = max_{t < T} w_i^(t) − w_i^(final)`         ⟵ Eq. T

where `T` is the actual converged iteration count (≤ max_iter),
`w_i^(t)` is atom i's softmax weight at iteration t, and the max is
taken across the full settling trajectory of that retrieval.

**Bounded** in `[0, 1]` (since each `w_i^(t)` is a softmax weight).

**Sign:** strictly non-negative. Zero only for atoms that were either
(a) always-winning across the full trajectory (in which case
`max_t w_i^(t) = w_i^(final)`) or (b) never-competing (in which case
both terms are ≈ 0).

**Magnitude (substrate-independent argument):** for any atom i that
participated in early settling (which is itself diffuse — the query
starts somewhere between basins and the softmax is initially
high-entropy regardless of β), there exists some t for which
`w_i^(t) > w_i^(final)`. The gap `max_t w_i^(t) − w_i^(final)` is
not collapsed by the basin's sharpness — it is preserved by the
fact that the FIRST iteration's softmax is always diffuse. This is
why Eq. T is "literature-grounded": HEN's rank-reduction finding
shows that the *trajectory* contains the metastability signal even
when the *fixed point* does not.

## Anti-homunculus self-check (binding)

The addendum's anti-homunculus check, before any code lands:

- **Who decides which iteration `t` is `max_t`?** No one. `max_t` is a
  reduction over the full trajectory — every iteration's `w_i^(t)`
  is computed by the existing settling loop. The max is a
  measurement of trajectory geometry. No threshold, no controller.
- **Is `c_i^(traj)` still local-per-atom?** Yes. Each atom's trajectory
  in weight-space is its own continuous variable; `max_t` operates
  per-atom independently. No global metric is read.
- **Does the trajectory get a separate evaluation pass?** No. The
  trajectory is already computed by the existing `retrieve()` /
  `retrieve_with_trace()` call. The `max_t` reduction happens inside
  the same loop. Audit constraint #1 from the original audit
  (no re-evaluation pass) is preserved.
- **Does anything else change?** No. μ_obs (EMA blend) unchanged.
  μ_rep (pay-down on sampling) unchanged. κ (priority gain) unchanged.
  Priority composition `gate · tag · suppression · (1 + κ · m_trace)`
  unchanged. The only change is the *source* of m_i — the per-atom
  measurement that feeds the EMA.
- **Is `max_t` a controller-in-disguise?** No more than A1's
  `max_{j≠i} |G_ij|` reduction is (which received anti-homunculus
  PASS as a measurement). Both are local-per-atom max-over-set
  reductions. The set is just different (other atoms vs. own
  trajectory iterations).
- **Does Eq. T sneak in implicit β-tuning?** No. β is fixed at the
  substrate-construction layer. Eq. T's magnitude under high β is
  the empirical observation that motivates the reformulation, not
  an invitation to retune β.

**Anti-homunculus verdict (self-check):** PASS by inheritance of the
original audit. The change is at the c_i operationalization layer
only; every audit constraint #1-#7 from the original is preserved.

**The reviewer audit is still required before code lands** — the
self-check is necessary but not sufficient.

## Why `c_i^(traj)` is NOT retuning in disguise

The original design note pre-committed κ, μ_obs, μ_rep against
retuning in response to first-retrain results. This reformulation:

1. **Does NOT change any of κ, μ_obs, μ_rep.** Those stay at the
   pre-committed values (κ=2.0, μ_obs=0.05, μ_rep=0.5). The Path 2
   "recalibrate κ to ~50" option is explicitly NOT taken.
2. **Changes the per-atom measurement c_i.** This is a redesign of
   the operationalization, parallel to A1' (which changed the
   redundancy proxy from RMS to max-over-others, an
   operationalization change).
3. **Is motivated by a literature finding** (HEN: trajectory > fixed
   point under high β), not by sliding a parameter to hit a target.
4. **Has its own anti-homunculus audit** before code lands.

The discipline binding "first retrain misses falsification criterion
⟹ falsification, not retune" applies to *parameters within a fixed
operationalization*. A redesigned operationalization is a fresh
mechanism that gets its own falsification attempt under fresh
pre-committed parameters.

This is the same logic that took A → A1 (substrate-derived r_ema init
fixed A's failure mode) → A1' (max-reduction fixed A1's failure
mode). Each step changed the measurement; κ-equivalent parameters
were preserved or freshly chosen, never feedback-tuned.

## What `c_i^(traj)` does NOT solve

- **If the substrate converges in 1–2 iterations** (sharp basins +
  early-exit on `tol=1e-8`), the trajectory is too short to expose
  any "lost-out atom" signal. `max_t w_i^(t) ≈ w_i^(final)` for all
  atoms, and `c_i^(traj) ≈ 0`. This is Path 3's own failure mode;
  the 1-seed smoke will reveal it in 6–10 min.
- **If the early iterations are already sharp** (e.g., the query is
  very close to a stored pattern from cycle 1), `c_i^(traj)` for
  non-winners is small but non-zero. This is the regime where pair #4
  has its smallest measurable effect.
- **The signal does not survive substrates with categorical, single-
  iteration retrieval.** That is, the substrate must do some
  iterative settling to expose trajectory metastability. The current
  `TorchHopfieldMemory.retrieve` already does this (max_iter=12,
  early-exit on convergence).

## Pre-committed falsification criteria (unchanged + one extension)

All four original criteria stand:
- Δ meta_stable_rate at W=3, CI disjoint from zero, Δ ≤ −0.10
- D1 non-regression Δms_w3 ≤ −0.5
- m_i CV > 0.1 within first eval
- d_eff ≥ 25 at step 1800

**Extension (binding for Path 3):**
- **Smoke-stage gate (new):** the 1-seed smoke must show
  `m_max > 0.05` and `|Δ meta_stable_w3| > 0.01` *before* the full
  n=10 launches. If both fail, Path 3's failure mode (substrate
  converges too fast for trajectory metastability) is real and the
  mechanism is falsified at the smoke layer — DO NOT proceed to n=10.

This smoke-stage gate is symmetric to A+B's
`phase5_ab_calibration.json` smoke that gated the full retrain.

## Implementation deltas vs. the original design

The original implementation sketch had four lines of changed code in
the consolidation layer plus the `weights_tensor` surfacing. Path 3
adds one line per retrieval (the running max update) and changes
the source of c_i. Specifically:

1. **`TorchHopfieldMemory.retrieve`** (and **`TracedHopfieldMemory.retrieve_with_trace`**)
   gain a per-iteration `running_max_weights = torch.maximum(
   running_max_weights, current_weights)` update inside the existing
   settling loop. One elementwise max per iteration, no extra sync.
   After the loop, `c_i^(traj) = running_max_weights − final_weights`
   is computed once.
2. **`TorchRetrievalResult`** gains a new field
   `metastability_contribution: Optional[torch.Tensor]` carrying
   `c_i^(traj)` on-device.
3. **`ConsolidationState.update_metastability`** signature changes
   from `(weights)` to `(contribution)`. The method body becomes
   `self.metastability_ema = (1 − μ_obs) · m_i + μ_obs · contribution`
   — the EMA is unchanged; the input is just the pre-computed
   trajectory contribution instead of the fixed-point weights.
4. **Call sites** (`UnifiedReplayMemory.retrieve_and_observe`,
   `run_replay_cycle`, the experiment 19 inlined loop) pass
   `result.metastability_contribution` instead of
   `result.weights_tensor`.
5. **Tests** in `tests/test_phase5_metastability.py` get a new
   `TestTrajectoryMetastabilityContribution` class that:
   - Asserts a synthetic monotonically-converging trajectory
     produces `c_i^(traj) > 0` for non-winners.
   - Asserts an instantly-converged retrieval (1 iteration) produces
     `c_i^(traj) ≈ 0` everywhere (Path 3's own failure mode is
     correctly identified).
   - Asserts `update_metastability(contribution)` with κ=0 still
     preserves bit-identical priority composition.

The substrate-wide architectural principle stated by this
reformulation:

> **Substrate measurements that are softmax-fixed-point quantities
> collapse under sharp-basin regimes. Trajectory-based
> reformulations recover the signal.**

This principle applies to pair #5 (cap-coverage / restructuring) when
that pair is opened. Pair #5's `c_i = cos(retrieve(cue_i), p_i)` is
fixed-point and will need its own trajectory reformulation in Phase 6.

Pair #2 (drift / replay-pressure) operates at the *consolidation-step
timescale*, not the per-retrieval timescale, so it is immune to this
failure mode by construction — `δ_i = ||p_i(t) − p_i(t − Δt_i)||²`
is already a trajectory measurement across consolidation steps.

## Closing the loop on the substrate-principle thread

The 2026-05-09 note prescribed five diagnostic-actuator pairs as the
architectural threshold-crossing. The chain has now produced four
substrate-principle findings:

| Phase 5 chapter | Substrate principle |
| --- | --- |
| A+B (closed) | d_eff is endogenous to substrate energy |
| A1 | new-atom r_ema is a measurement, not an implementer constant |
| A1' | redundancy reduction must saturate on sparse duplicates |
| β path (falsified) | role-fidelity via pairwise-distance is null at the FHRR noise floor |
| **Pair #4 c_i^(traj) (this addendum)** | **softmax fixed-point measurements collapse on sharp-basin substrates; trajectory reformulations recover the signal** |

Each principle is now a transferable architectural constraint for
the rest of the project. Pair #4's graduation under the
reformulation would close the second pair (after A+B) and validate
both the per-atom-EMA-into-energy-ranked-sampling shape AND the
trajectory-over-fixed-point principle.

If the smoke under `c_i^(traj)` fails Path 3's own falsification
(m_max < 0.05 or |Δ| < 0.01), pair #4 is falsified architecturally
(not just under one κ choice) and the right move is the pivot to
pair #2 — which is immune to this entire class of failure modes by
operating at consolidation timescale rather than retrieval
timescale.
