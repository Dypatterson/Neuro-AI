# Report 052 — Phase 5 pair #4 falsified at smoke gate (both operationalizations)

**Date:** 2026-05-21
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Pair #4 (metastability ~ replay-prioritization) falsified at the
n=1 Colab smoke layer under BOTH the fixed-point operationalization
(Path 1: `c_i = w_i · (1 − max_w)`) AND the trajectory-based
reformulation (Path 3: `c_i^(traj) = max_t w_i^(t) − w_i^(final)`).
The pre-committed smoke gate from the [design addendum](../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md)
§ADDENDUM (audit constraint #10) is binding: per the design's own
escalation rule, the response is the **pair #2 (drift /
replay-pressure) pivot**, not a fourth c_i reformulation.
**Driver:** [scripts/colab_phase5_pair4_n10.ipynb](../scripts/colab_phase5_pair4_n10.ipynb)
against the A+B+A1' substrate at seed 17.
**Output:** the smoke `phase34_results.json` files under
`reports/phase5_pair4_smoke_{kappa0_control,pair4_active}_seed17/`.

## Experiment preamble

**Active phase:** 5

**Headline metric for this report:** the **smoke-stage gate** from
the design addendum:
- `m_max > 0.05` (mechanism produces non-trivial per-atom signal)
- `|Δ meta_stable_w3| > 0.01` (mechanism affects the training
  trajectory at all)

Both must hold for the n=10 graduation retrain to be launched. Per
audit constraint #10, failure means pivot to pair #2 — NOT a fresh
c_i reformulation.

**Required controls:** same-seed κ=0 control on the same training
data. Both conditions run with identical CLI except for
`--metastability-gain`.

**Last verified result:** [report 050](050_phase5_beta_smoke_seed17.md)
— β path falsified at n=1; root cause attributed to FHRR crosstalk
`1/√D ≈ 0.016` setting f_i ≈ 0.984 structurally at D=4096.
[Report 051](051_phase5_tier1_disambiguation.md) — three Tier-1
alternative fidelity metrics also failed. Brainstorm pivoted to
pair #4.

**Why this experiment now:** the [design note](../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md)
pre-committed κ=2.0, μ_obs=0.05, μ_rep=0.5. The first smoke (Path 1)
showed bit-identical training trajectories between conditions. The
[ADDENDUM](../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md#L514)
reformulated c_i via the HEN (Kashyap 2024) trajectory-based rank-
reduction insight, with three new anti-homunculus constraints and a
binding pre-committed smoke gate. This report tests that
reformulation.

## Headline result

Both operationalizations fail the pre-committed smoke gate.

| Operationalization | μ_obs | κ | m_max (final eval) | Δ meta_stable_w3 (active − control) |
| --- | ---: | ---: | ---: | ---: |
| Path 1 (fixed-point `w_i · (1−max_w)`) | 0.05 | 2.0 | 0.0082 | 0.0000 |
| Path 3 (trajectory `max_t w_i^(t) − w_i^(final)`) | 0.05 | 2.0 | **0.0014** | **0.0000** |
| **Pre-committed gate** | — | — | > 0.05 | > 0.01 |

**Both fail.** And — counter-intuitively — Path 3's m_max is *smaller*
than Path 1's. That is the central finding of this report and is
explained in §"Why Path 3 was SMALLER than Path 1" below.

## Drill-down: meta_stable_w3 + m_i evolution per condition

Path 3 smoke, seed 17:

```
kappa0_control:
  meta_stable_w3 (final eval): 1.0000
  m_mean (first eval):  0.0000
  m_std  (first eval):  0.0000
  m_mean (final eval):  0.0004
  m_max  (final eval):  0.0014

pair4_active:
  meta_stable_w3 (final eval): 1.0000
  m_mean (first eval):  0.0000
  m_std  (first eval):  0.0000
  m_mean (final eval):  0.0004
  m_max  (final eval):  0.0014

Delta meta_stable_w3 (active - control) = +0.0000
pair4_active m_max = 0.0014
```

The two conditions are **bit-identical**, including the m_i evolution
(m_max, m_mean equal to 4 decimal places). This means the κ multiplier
has zero measurable effect on the replay sampling at this scale of
m_i, even though the mechanism is firing in both conditions.

## Architectural finding: substrate geometry forecloses pair #4

Both Path 1 and Path 3 are downstream of the same substrate-level
structural property: **at D=4096 with the trained codebook,
retrievals are winner-take-all from iteration 1**. The settling
trajectory has no diffuse phase. Both operationalizations of c_i are
trying to extract a signal from softmax weights, and at this D the
weights are essentially one-hot at iteration 1 already.

The mechanism, in more detail:

- FHRR crosstalk noise floor (per [report 050](050_phase5_beta_smoke_seed17.md)):
  random pattern-pattern cosine similarity is `1/√D ≈ 0.016` at
  D=4096. A query that is the encoded form of one stored pattern has
  similarity ≈ 1.0 to that one and ≈ 0.016 to all others.
- At β=10, softmax `exp(β · sim)`:
  - For the matching pattern: `exp(10) ≈ 22026`
  - For non-matching: `exp(0.16) ≈ 1.17`
- With ≈ 2000 stored patterns at W=3, the matching pattern's softmax
  weight is `22026 / (22026 + 2000 · 1.17) ≈ 22026 / 24366 ≈ 0.904`
  in iteration 1.
- Each non-matching pattern's weight is `1.17 / 24366 ≈ 0.000048`.
- After iteration 2, the state moves toward the matching pattern,
  similarity tightens to ≈ 1.0, weights become `[~1, ~0, ~0, ...]`.

So **at iteration 1 the softmax is already 90% concentrated on one
atom**, and **by iteration 2 it is 99%+ concentrated**. There is no
"diffuse phase" for either operationalization to find a signal in.

Path 1 captures `w_i · (1 − max_w)`:
- For the winner: `~1 · ~0 = 0`
- For each non-winner: `~5e-5 · ~0.1 = ~5e-6` per retrieval
- After 3000 retrievals at μ_obs=0.05, EMA equilibrium is the mean c_i
  across all retrievals. Occasional queries that don't match any
  stored pattern produce more diffuse softmaxes; those spikes drive
  m_max to ~0.008.

Path 3 captures `max_t w_i^(t) − w_i^(final)`:
- For the winner: `max_t = w_i^(final)` so `c_i = 0`
- For each non-winner: `max_t w_i^(t) ≈ w_i^(1) ≈ 5e-5`, `w_i^(final) ≈ 0`
- `c_i ≈ 5e-5` per retrieval — *strictly smaller* than Path 1 in
  steady-state because Path 1 captures the fixed-point spread while
  Path 3 captures only the *change* across iterations
- After EMA blending, m_max ≈ 0.0014

This is the inverted-counter-intuitive finding from this report:
Path 3 magnitudes are *smaller* than Path 1's, because:
- Path 1 captures noise-floor mass at the fixed point (every atom
  has w_i ≥ 5e-5 even at convergence, and `w_i · (1 − max_w)` is
  dominated by the multiplication of these floor weights by the
  small non-zero residual `1 − max_w`).
- Path 3 captures only the *trajectory gap* `max_t − final`. At
  D=4096 with sharp basins, this gap is small *because there is no
  iteration where the non-winner is significantly above its final
  weight*.

The HEN (Kashyap 2024) finding that "metastability lives in the
trajectory" assumes the substrate produces *enough trajectory* to
have metastability. At D=4096 with the project's encoding pipeline,
the substrate does not produce that trajectory. Path 3's
reformulation is correct in principle but is **substrate-incompatible
in practice**.

## Connection to the broader four-step debugging chain

This is the **same root cause** as the β path's falsification in
[report 050](050_phase5_beta_smoke_seed17.md):

| Layer | Mechanism | What it tries to extract | Substrate's reply |
| --- | --- | --- | --- |
| β path (selector) | `f_i = mean(1 − \|G_jk\|)` | per-atom role-fidelity from pairwise distances | `f_i = 0.984 ± 0.000` (noise-floor uniform) |
| Path 1 c_i | `w_i · (1 − max_w)` | per-atom metastability at fixed point | `c_i ≈ 5e-6` (noise-floor mass × small residual) |
| Path 3 c_i^(traj) | `max_t w_i − w_i^(final)` | per-atom trajectory-competition signal | `c_i ≈ 5e-5` (no trajectory phase exists) |
| Tier-1 (report 051) | three alt fidelity metrics | per-atom variance in different metric class | all std < 0.02 (noise floor wins) |

**All four are different attempts to extract a per-atom signal from
softmax-derived quantities on the substrate's actual retrievals.**
None of them produce enough variance to be a useful selector. The
substrate has been engineered (by the encoding pipeline + A+B
dynamics) to produce **clean, sharp, fast-converging retrievals**.
That is a positive property for the project's contextual-completion
goal. But it forecloses ANY mechanism that needs softmax-derived per-
atom variance to differentiate atoms.

## What the smoke gate forecloses (no Path 4)

Per audit constraint #10 from the design addendum:

> #10. The smoke-stage gate is binding in the falsifying direction.
> If the smoke fails, the response is the pair-#2 pivot, NOT a
> Path-4 reformulation of c_i.

The discipline is real: if we slide to a fourth c_i operationalization
in response to this finding, we have papered over a **structural
finding about the substrate** (its geometry forecloses softmax-derived
per-atom signals at this D) with a sliding-toward-success
parameter/formula tweak. That is the failure mode the discipline rule
was guarding against.

**Pair #4 is falsified at n=1 smoke under the pre-committed gate.**
The mechanism class — "per-atom softmax-derived signal feeds an EMA
into replay priority" — is incompatible with the substrate's D=4096
sharp-basin geometry. This is not a calibration miss; it is an
architectural incompatibility.

## The pivot: pair #2 (drift / replay-pressure)

Pair #2 is the **only open pair** structurally immune to this entire
class of failure modes, because it operates at the consolidation-step
timescale rather than the retrieval timescale. The drift signal
`δ_i = ||p_i(t) − p_i(t − Δt_i)||²` measures how much an atom's pattern
has moved since its last replay event. This is a *cumulative
displacement* signal, not a *fixed-point softmax* signal:

- It does not read softmax weights of any retrieval.
- It does not depend on β or settling sharpness.
- Its magnitude is set by how much the codebook re-encoding pipeline
  + Hebbian online updates moved the atom over the inter-replay
  interval — which is a substrate-level integrative quantity, not a
  retrieval-pulse quantity.
- It is immune to the FHRR-noise-floor failure mode because it does
  not derive its signal from inter-atom similarities or softmax.

Per [research D §"Pair #2"](../brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md):

- The substrate already exposes `codebook_drift()` at the global
  level ([phase34/reencoding.py:107](../src/energy_memory/phase34/reencoding.py)).
- Per-atom drift is a one-snapshot reduction (per-atom delta against
  the previous-replay codebook).
- The MIR (Aljundi 2019) replay-priority literature ground transfers
  cleanly: drift-weighted replay reproduces the "items maximally
  interfered by recent updates" sampling that MIR shows reduces
  forgetting.
- The `m_i ← (1 − μ_obs)·m_i + μ_obs·δ_i / replay-payback / (1 + κ·m_trace)`
  pattern from pair #4 transfers verbatim to pair #2 — same shape,
  different signal source.

The pair #4 design's substrate-side scaffolding (m_i EMA on
ConsolidationState, ReplayStore primary_atom indexing, κ multiplier
on priority composition, payback on sample, exp 19 CLI knobs) is
**fully reusable for pair #2**. Only the signal source changes.

## Pre-committed binding on Path 3's audit constraint #10

The smoke gate was pre-committed as binding before observing this
result. The disciplined response is enforced by the design note's own
text:

> **Smoke-stage gate (new):** the 1-seed smoke must show
> `m_max > 0.05` and `|Δ meta_stable_w3| > 0.01` *before* the full
> n=10 launches. If both fail, Path 3's failure mode (substrate
> converges too fast for trajectory metastability) is real and the
> mechanism is falsified at the smoke layer — DO NOT proceed to n=10.

Both criteria failed (m_max=0.0014, |Δ|=0.0000). The mechanism is
falsified at the smoke layer. The next session opens pair #2 per the
addendum's escalation rule.

## What this report does NOT do

- It does **NOT** propose a Path 4 c_i operationalization. Per audit
  constraint #10, that is exactly the failure mode the smoke gate
  forecloses.
- It does **NOT** retune κ, μ_obs, or μ_rep. Those were pre-committed.
- It does **NOT** declare the substrate broken. The substrate's
  sharp-basin geometry is a positive property for the contextual-
  completion goal. The finding is that pair #4's mechanism class is
  *incompatible with* this substrate, not that the substrate is wrong.
- It does **NOT** rule out pair #4 architecturally on other
  substrates. If a future Phase had a substrate with more iterative
  settling (e.g. low-β regimes, or a different similarity kernel),
  pair #4 might work there. But this is the substrate the project
  has, and the substrate is right.
- It does **NOT** affect Phase 4 D1 graduation. That stands.
- It does **NOT** re-open the K-branch state_divergence chase.

## Sequencing

1. STATUS walk-back to record the pair #4 falsification + pair #2
   pivot.
2. Draft pair #2 design note following the same template as the
   pair #4 design note. Anti-homunculus reviewer audit. PASS
   required.
3. Implementation (most of the library scaffolding from pair #4 is
   directly reusable): per-atom δ_i state in ConsolidationState;
   drift-from-last-replay computation in the reencode cycle; priority
   composition gains `(1 + κ · δ_trace)` factor.
4. Re-run smoke on seed 17 against the **same** pre-committed
   smoke-gate structure (m_max > 0.05 AND |Δ| > 0.01) but with the
   gate's *m_max* substituted by *δ_max*. If pair #2 fails its own
   smoke gate, the mechanism class "consolidation-timescale signal
   feeds replay priority" is also falsified, and the next move is
   declining Phase 5 graduation and re-scoping — NOT proposing yet
   another mechanism.
5. If pair #2 smoke passes, full n=10 graduation retrain.

## Substrate-principle preservation

The architectural principle from the pair #4 ADDENDUM —
"softmax fixed-point measurements collapse on sharp-basin
substrates; trajectory reformulations recover the signal" — was
correct in the literature but **conditional on the substrate
producing a trajectory in the first place**. At D=4096 with this
encoding pipeline, the trajectory is essentially absent. The principle
transfers to pair #5 (cap-coverage / restructuring) only if pair #5's
retrieval-trajectory-based reformulation is itself substrate-
compatible — which would need its own smoke gate.

A refined version of the principle, after this report:

> **Substrate measurements that depend on softmax-derived per-atom
> variance collapse on substrates engineered for clean retrieval.
> Mechanisms that rely on such signals must verify their viability
> at the smoke-gate layer before committing to a graduation attempt.**

This applies to pair #5 by extension. Pair #2 is exempt because its
signal is not softmax-derived.
