---
date: 2026-05-27
project: personal-ai
phase: Path γ — Phase 3 mechanism redesign
mechanism: Γ1 — context-residual consolidation
status: precommit (design + reviewer gate; no code authorized yet)
tags:
  - notes
  - subject/cognitive-architecture
  - subject/path-gamma
  - subject/phase-3
---

# Path γ — Γ1 context-residual consolidation precommit

## Status

**Design-only, pre-code, pre-anti-homunculus-review.** No implementation
authorized by this document. Code lands only after the
`anti-homunculus-reviewer` agent records a PASS verdict against this
document and the verification protocol (§Required pre-code gates) is
satisfied.

## Lineage

This precommit closes the Path γ first deliverable named in
[Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md)
§"Path γ — recommended next research direction" and selects the leader
candidate **Γ1 (context-residual consolidation)** identified by the
[Path γ mechanism-family survey](../emergent-codebook/path-gamma-mechanism-family-survey.md).
Survey ranked Γ1 against four design constraints derived from the v3 → v6
wikitext walk-back:

1. Training-signal *shape* — not its strength — is the bottleneck. Any
   "Hebbian pull with a different multiplier" is in the same shape class
   as the closed C.2 mechanism and is predicted to hit the same noise
   floor.
2. Per-seed paired robustness ≥ 70% (not pooled CI-disjointness alone) is
   the bar per the revised exit criterion at
   [phase-3-deep-dive.md:188-205](../emergent-codebook/phase-3-deep-dive.md).
3. Substrate capacity is not the dial. D-curve at the closed C.2 mechanism
   reversed at D ∈ {1024, 2048, 16384}; pooled D=4096 vs D=8192 paired
   test showed only 8/20 seeds improve. No candidate whose justification
   routes through "more D" can be expected to graduate.
4. Anti-homunculus compatibility is binding. H1–H7 of the Path C
   precommit and the 2026-05-09 anti-homunculus filter apply.

The design grilled the leader through one
[mp-grill-with-docs](../../) session on 2026-05-27 covering terminology,
mathematical form, role-vector status, anti-homunculus framing, operating
point, sweep dimensions, and scope. The Γ1 design table in the survey
records the locked decisions; this document restates them in a form the
reviewer agent can audit.

## Architectural claim

The codebook lives on an energy landscape whose per-event term includes a
**repulsion between confused atom pairs**. The gradient descent on this
energy *is* the consolidation update. Confusions and repulsions are the
same physical event viewed at different temporal resolutions: the
classification confusion (substrate event: `predicted_id ≠ target_id`) is
the snapshot at fast timescale; the repulsion (substrate dynamic: the
codebook moves) is the same process at slow timescale.

Define the per-event context-residual energy:

> **E_cr(codebook | event) = − Σ_{j ∈ roles} 1[predicted_id_j ≠ target_id_j] · ½ · ‖ codebook[target_id_j] − codebook[predicted_id_j] ‖²**

The minus sign makes the term a repulsion: lowering `E_cr` means moving
confused atom pairs farther apart in atom-space. The indicator
`1[predicted_id_j ≠ target_id_j]` is a **property of the energy
function's support** — it states that the energy contribution from role
`j` is identically zero when the system retrieves correctly for that
role. It is not a runtime gate or a controller. (Same shape as the
existing push at [online_codebook.py:155-163](../../src/energy_memory/phase34/online_codebook.py),
which is gradient descent on `½ ‖ codebook[predicted] − slot_query ‖²`
and is identically zero on correct events for the same reason.)

The mechanism *is* gradient descent on `E_cr`, computed asymmetrically
(see §Mechanism). This is the **right-hand column** form per
[2026-05-09:140-148](../../notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md).

## Mechanism

### Gradient

For an event with roles `j` and codebook entries `codebook[target_id_j]`,
`codebook[predicted_id_j]`, the partial derivative of `E_cr` with respect
to `codebook[target_id_j]` is:

> ∂E_cr / ∂ codebook[target_id_j] = − 1[predicted_id_j ≠ target_id_j] · ( codebook[target_id_j] − codebook[predicted_id_j] )

Let `ε_j := codebook[target_id_j] − codebook[predicted_id_j]` (the
*atom-space* residual). Gradient descent on `E_cr` with respect to
`codebook[target_id_j]` only is:

> codebook[target_id_j] ← codebook[target_id_j] − lr_cr · ∂E_cr/∂ codebook[target_id_j]
>                       = codebook[target_id_j] + lr_cr · 1[predicted_id_j ≠ target_id_j] · ε_j

### Asymmetric descent

`codebook[predicted_id_j]` is treated as a **stop-gradient** reference for
the purposes of `E_cr`: no Γ1.c update is applied to it via this energy
term. (`codebook[predicted_id_j]` may still be moved by other concurrent
forces — specifically, the C.2.x dynamics listed in §Operating point —
which compose additively as substrate-pure vector forces.)

Asymmetric gradient descent on a symmetric energy is a partial descent
direction, not the full Newton/Euler step. The symmetric variant
(complete descent on `E_cr`) would update both `codebook[target_id_j]`
and `codebook[predicted_id_j]` by ε_j of opposite sign; it is named here
as a precommitted follow-up (§Pre-committed follow-ups) and is *not*
authorized by this precommit.

### Composition with C.2.x dynamics

The Γ1.c update replaces pull/push as the **base update** but does not
displace any of C.2.1–C.2.5, which continue to fire in the same order as
in Path C:

1. C.2.5 pre-snapshot of codebook (drift tension reference).
2. **Γ1.c context-residual update** (replaces pull/push at [online_codebook.py:144-164](../../src/energy_memory/phase34/online_codebook.py)).
3. C.2.1 anti-collapse force.
4. C.2.3 cap-coverage gradient.
5. C.2.2 splitting-tension multiplicative modulation on the net update.
6. C.2.5 drift-tension EMA update.

All five C.2.x dynamics retain Path C values (see §Operating point).
Step 5's multiplicative modulation now operates on the Γ1.c update +
C.2.1 + C.2.3 sum rather than on the pull/push + C.2.1 + C.2.3 sum.

### Role-vector status at Phase 3

Roles `j` in `E_cr` index positions in the window. At Phase 3, positions
are **fixed substrate-position vectors** built by
[phase2/encoding.py:18-26](../../src/energy_memory/phase2/encoding.py)
(`positions[j] = bind(positions[j−1], step)` chained from a frozen
`base`). The codebook holds only **filler atoms** (token →
hypervector). There is no learned role atom. Γ1.c updates only filler
atoms; `unbind(state, positions[j])` operates on a substrate-rooted
quantity.

Per-event labels:

- `target_id_j` is the supervised label from the event encoding (token
  index at position `j` in the corpus window — substrate-rooted: this
  is what the encoder placed at that position).
- `predicted_id_j` is the substrate state output of MHN cleanup applied
  to `slot_query_j = unbind(state, positions[j])` — i.e., the existing
  pipeline's prediction.

No C.1 diagnostic log is consumed by Γ1.c (H6).

### Disposition of existing pull/push

Pull/push at [online_codebook.py:144-164](../../src/energy_memory/phase34/online_codebook.py)
is **retained as default-on** to preserve Path C reproducibility under
the default-knob discipline named in
[CLAUDE.md §"Common failure modes"](../../CLAUDE.md). Γ1.c is gated by
two new flags on `ConsolidationConfig`:

- `use_context_residual: bool = False` — turn on Γ1.c.
- `use_pull_push: bool = True` — turn off pull/push when Γ1.c is on.

The Γ1 headline condition sets `use_context_residual=True,
use_pull_push=False`. The Path C baseline condition sets
`use_context_residual=False, use_pull_push=True` (default values,
byte-identical to the current code).

## Operating point

The headline gate uses the **Path C wikitext operating point** byte-
identical across all knobs except the two new flags and `lr_cr`. Holding
everything else constant isolates the mechanism-shape change.

| Knob | Value | Source |
|---|---|---|
| `corpus_source` | `wikitext` (wikitext-2-raw-v1) | Path C v3–v6 |
| `vocab_cap` | 1000 | Path C |
| `window_size` | 8 | Path C |
| `D` | 4096 | Path C primary D |
| `β` (Hopfield temperature) | 10 | Path C |
| `K` (Recall@K) | 5 | Path C |
| `landscape_size` | 64 | Path C |
| `n_consolidation_events` | 1000 | Path C (3000 nulled — Report 112) |
| `α_anti` | 0.01 | Path α (2026-05-26) |
| `repulsion_step_size` | 0.05 | Path α |
| `λ_ac` (C.2.1) | 0.5 | [c3_phase3_exit_criterion.py:119](../../experiments/c3_phase3_exit_criterion.py) |
| `μ_T`, `τ_T` (C.2.2) | 0.1, 0.5 | c3_phase3_exit_criterion.py:120 |
| `λ_cc` (C.2.3) | 0.5 | c3_phase3_exit_criterion.py:122 |
| `metastability_obs_rate` (C.2.4) | 0.1 | c3_phase3_exit_criterion.py:125 |
| `drift_ema_rate` (C.2.5) | 0.1 | c3_phase3_exit_criterion.py:128 |
| `use_context_residual` | **True** (new) | This precommit |
| `use_pull_push` | **False** (new) | This precommit |
| `lr_cr` | **0.1** (single value, no probe) | This precommit |
| `n_seeds` | 10 for headline | Path C protocol |
| Seeds | 0..9 for headline; 10..19 for any pooling | Path C protocol |

Wilson CIs and per-seed paired Δ are computed per the existing C.3
driver at [experiments/c3_phase3_exit_criterion.py](../../experiments/c3_phase3_exit_criterion.py).

## Anti-homunculus discipline

Per the Path C precommit's H1–H7 (binding for Path γ inheritor), and the
2026-05-09 anti-homunculus filter.

**H1 — No controller chooses storage/retrieval routes.**
✅ Γ1.c is a consolidation-layer mechanism. It does not change storage or
retrieval routing. Existing MHN energy descent and codebook lookup remain
the storage/retrieval mechanism unchanged.

**H2 — No metric-triggered replay sampler switching.**
✅ The replay sampler is unchanged. Γ1.c fires inside `_consolidate()`,
which is consolidation-time, not replay-sampling-time.

**H3 — No best-of-N condition selection as graduation evidence.**
✅ The headline gate runs a single `lr_cr=0.1` operating point. No probe
is run before the headline; no sweep is taken as graduation evidence.
If the headline fails, an explicitly-named follow-up precommit (§Pre-
committed follow-ups) authorizes an `lr_cr` sweep — but at smoke scale
(n=3), and the *follow-up* is itself a separate precommit, not the
headline. (See also §Falsifiable graduation criterion for the binding
single-condition framing.)

**H4 — Diagnostics are passive logs, not execution gates.**
✅ Γ1.c does not read any output from
[basin_diagnostics.py](../../src/energy_memory/phase3/basin_diagnostics.py),
[bimodality_diagnostic.py](../../src/energy_memory/phase3/bimodality_diagnostic.py),
[regime_diagnostic.py](../../src/energy_memory/phase3/regime_diagnostic.py),
[metastability_diagnostic.py](../../src/energy_memory/phase3/metastability_diagnostic.py),
or [theta_prime_calibration.py](../../src/energy_memory/phase3/theta_prime_calibration.py).
Inputs to `E_cr` are exclusively substrate state (`codebook[*]`,
`predicted_id_j`, `target_id_j`).

**H5 — Right-hand column expressibility (substrate dynamics, not
controller-style "X causes Y").**
✅ The mechanism is gradient descent on the per-event energy `E_cr`. The
prose framing is "the codebook lives on an energy landscape whose per-
event term repels confused atom pairs; the gradient descent IS the
consolidation update." The indicator `1[predicted_id_j ≠ target_id_j]`
is a property of `E_cr`'s support, not a runtime gate. Same shape as
existing push, which the Path C reviewer audit passed.

**H6 — Mechanism reads substrate state, not C.1 diagnostic logs.**
✅ As enumerated under H4: all inputs are substrate state. The MHN
cleanup that produces `predicted_id_j` is substrate dynamics (energy
descent + nearest-codebook-row, both substrate primitives). The
`target_id_j` is event supervision (corpus encoder output). No C.1
diagnostic log is read.

**H7 — `min_branch` aggregator and `q_bundle`/`q_greedy` dispatch are
Phase 5 CP7 sites, out of scope.**
✅ Γ1.c is Phase 3. It does not touch any Phase 5 dispatch.

### Subtleties flagged proactively for the reviewer

1. **The asymmetric descent has a "stop-gradient" character.** Reviewer
   should verify this is read as "partial gradient descent on a symmetric
   energy" (a standard SGD pattern) and not as "two different energy
   functions for the two atoms" (which would be a covert controller, with
   `codebook[target_id]` updating under one rule and
   `codebook[predicted_id]` updating under another).

2. **Composition with C.2.x is purely additive.** The Γ1.c update is one
   vector force on `codebook[target_id_j]`; C.2.1 (anti-collapse), C.2.3
   (cap-coverage) add as additional vector forces on the same atom;
   C.2.2 multiplicatively modulates the net by `1/(1 + T_k/τ_T)`. The
   *composition order* matches Path C exactly. No new gating, no new
   conditional firing.

3. **`predicted_id_j` is read once per event.** It is computed during
   query encoding (existing pipeline) and consumed by Γ1.c at the same
   `_consolidate()` step. Reviewer should verify the precommit does not
   ask `predicted_id_j` to be recomputed mid-update (which would create a
   coupled fixed-point ambiguity at the implementation layer; the
   straightforward implementation does not).

### Watch items for the implementation-PR reviewer (per 2026-05-27 anti-homunculus review)

The Γ1 precommit reviewer recorded these as non-blocking concerns for the
*implementation* PR (not for this precommit document). They are landed
here so the audit trail is preserved.

- **W1 — Indicator-as-mask vs if/branch (reviewer concern C3).** The
  mathematical justification at §"Architectural claim" treats
  `1[predicted_id_j ≠ target_id_j]` as a multiplicative mask in the
  energy's support. If the implementation realizes Γ1.c as `if
  predicted_id != target_id: apply_update()` rather than as "compute ε;
  multiply by indicator-mask; apply," the *runtime code* has an
  `if/then` shape even though the *math* doesn't. Both are equivalent
  gradient descent on `E_cr`, so the math is sound either way. But the
  implementation-PR reviewer should *prefer the multiplicative-mask form*
  where it costs no extra ops (it does in this case: a `(predicted_id !=
  target_id).float()` mask is cheap). Style preference, not correctness.

- **W2 — `predicted_id_j` provenance (reviewer concern C4).** The H4/H6
  framing requires `predicted_id_j` in the buffer to be the MHN-cleanup
  output (substrate dynamics), not (e.g.) a diagnostic-classifier
  output. Existing `_BufferedFailure` at
  [online_codebook.py:100-106](../../src/energy_memory/phase34/online_codebook.py)
  takes `predicted_id` from the caller; the caller must populate it
  from MHN cleanup. Implementation PR should verify by reading the
  caller site (likely in a `phase34/` driver or `phase4/replay_loop.py`).

- **W3 — F2 (symmetric Γ1.c) review is structurally cleaner, not
  harder (reviewer concern C2).** The symmetric variant is *full*
  gradient descent on the same symmetric `E_cr`; it has no
  stop-gradient asymmetry to defend. If F2 ever ships as a follow-up
  precommit, the anti-homunculus review on it should be easier, not
  harder, than this asymmetric precommit's review. Worth recording so
  future-Dylan doesn't budget the same review effort for F2.

## Required pre-code gates

All three must pass before any code lands. The `anti-homunculus-reviewer`
PASS is the binding gate on this document; the other two are gates on
the resulting implementation PR.

1. **Anti-homunculus reviewer PASS on this document.** Binding.
   `claude/agents/anti-homunculus-reviewer` returns PASS with no findings
   of arbitration / metric-triggered routing / supervisor module. Any
   findings are addressed in this document (with edits recorded inline)
   before code lands.

2. **Convergence-equivalence test for each C.2.x dynamic under the new
   Γ1.c base update.** Same protocol as Path C C.2: at a quasi-stationary
   substrate, the live diagnostic (the C.1.x measurement) must match the
   slow-timescale dynamic's fixed-point value to numerical tolerance.
   Path C tested this under pull/push base; under Γ1.c base, the
   fixed-point identities may differ. Re-running the equivalence tests
   establishes that C.2.1–C.2.5 remain anti-homunculus-clean under
   composition with Γ1.c. Per-dynamic tests under
   `tests/test_c2x_convergence_under_context_residual.py`.
   **Binding sub-clause (per 2026-05-27 reviewer concern C1):** if any
   C.2.x equivalence test fails under Γ1.c base, the diagnosis returns
   to anti-homunculus re-review *before* the failing dynamic is retuned.
   A C.2.x dynamic that re-tunes to recover equivalence may have
   silently changed its anti-homunculus-clean shape under composition;
   the reviewer must confirm the retuned form still occupies the
   right-hand column of [2026-05-09:140-148](2026-05-09-papers-diagnostics-and-actuator-dynamics.md).

3. **Baseline parity test.** With `use_context_residual=False,
   use_pull_push=True` (default values), the new code must produce
   byte-identical Recall@K, regime-stratified Δ, and per-seed paired Δ
   to the Path C C.3 second smoke configuration on a small seed set
   (e.g., seeds 0..2, n_consolidation_events=100, vocab_cap=200, D=4096).
   This preserves Path C reproducibility under the default-knob
   discipline. Test under
   `tests/test_consolidation_path_c_byte_identity.py`.

4. **Unit tests for the new `_consolidate()` branch** under
   `use_context_residual=True`. Minimum: (a) ε vector computed correctly
   for a synthetic 2-atom event; (b) update applied only to
   `codebook[target_id]`, not `codebook[predicted_id]`; (c) update
   skipped when `predicted_id == target_id`; (d) composition with C.2.1
   produces the additive sum, not a replacement.

## Falsifiable graduation criterion

Per the revised C.3 exit criterion at
[phase-3-deep-dive.md:188-205](../emergent-codebook/phase-3-deep-dive.md):

**Γ1.c graduates Phase 3 if and only if both clauses hold at the headline
operating point (§Operating point), with n=10 seeds 0..9 (or pooled n=30
seeds 0..29 if the n=10 result is borderline):**

1. Wilson CI on the standard-vs-shuffled-token-control Δ Recall@K is
   strictly disjoint in at least one regime stratum (`default/spread` or
   `calibrated/tight`).
2. Per-seed paired robustness ≥ 70%: at least 7/10 (or 21/30 pooled)
   seeds show stratum-pooled per-seed Δ > 0.

**If both clauses hold, Γ1.c is the graduated Phase 3 mechanism.** Phase
5′ is *not* automatically reopened by this graduation — the Phase 5
audit findings remain binding, and Phase 5′ reopening is a separate
decision against the
[Phase 5 audit](../../audit-phase5-2026-05-26.md) §9 failure modes.

**If exactly one clause holds (CI-disjoint but per-seed paired robustness
60–69%, or vice versa), Γ1.c is "partially-succeeded".** No graduation
claim. The pre-committed follow-up (§Pre-committed follow-ups) authorizes
either symmetric Γ1.c or Γ3 (SFA-head) composition as a separate
precommit.

**If neither clause holds, Γ1.c is rejected at this operating point.**
The pre-committed `lr_cr` sweep follow-up authorizes a smoke-scale
diagnostic; if the sweep also produces null, the Γ1 family is closed.

## What this precommit does not permit

- No `lr_cr` sweep at headline scale (would violate H3).
- No D sweep, β sweep, α_anti ablation, λ_ac ablation, or any C.2.x
  ablation at headline scale (these are post-graduation drill-downs).
- No symmetric Γ1.c variant (separate precommit if/when authorized).
- No composition with Γ3 (SFA-head) (separate precommit).
- No corpus other than wikitext-2-raw-v1 at headline scale.
- No Phase 5 work of any kind: ΔE, bridge, M2, full matrix, headline,
  graduation. Phase 5′ pause remains binding per
  [STATUS.md](../../STATUS.md) and the 2026-05-26 Phase 5 audit.
- No removal of pull/push from the codebase (default-on retained for
  Path C reproducibility; toggled OFF only via the new config flag).
- No removal or weakening of the convergence-equivalence tests for
  C.2.1–C.2.5 (re-run under Γ1.c base, not replaced).

## Pre-committed follow-ups (named, not authorized, not drafted)

These follow-ups are *named* here so the headline gate is not framed as
"Γ1 must pass at lr_cr=0.1 or the family is dead." Each requires its own
precommit document and (where it introduces a mechanism change) its own
anti-homunculus reviewer PASS before authorization.

- **F1 — `lr_cr` sweep at smoke scale.** If the headline gate fails
  (neither graduation clause holds), this follow-up sweeps `lr_cr ∈
  {0.01, 0.05, 0.1, 0.2, 0.5}` at n=3 (smoke). It is a diagnostic, not a
  graduation gate. If the sweep produces null at all magnitudes, the Γ1
  family is closed.

- **F2 — Symmetric Γ1.c.** If the headline gate partially-succeeds, the
  symmetric variant (full gradient descent on `E_cr`, updating both
  `codebook[target_id_j]` and `codebook[predicted_id_j]`) is a 5-line
  diff. Re-run the headline gate with `use_symmetric_context_residual=
  True`.

- **F3 — Γ1.c + Γ3 (SFA-head).** If the headline gate partially-
  succeeds, Γ3 composes as an additional local loss term during
  consolidation. Needs its own precommit + anti-homunculus reviewer.

- **F4 — α_anti / λ_ac / all-C.2.x ablations.** Post-graduation drill-
  downs. Each ablation is a separate report localizing how much of the
  headline signal needs the corresponding force.

- **F5 — D-curve drill-down.** If Γ1.c graduates, D ∈ {1024, 2048, 8192,
  16384} characterization to compare Γ1.c's D-curve to the pull/push
  D-curve from Report 112 §"v6 — D=8192 confirmation".

- **F6 — Corpus generalization.** wikitext-103, other natural corpora.
  Post-graduation.

## Implementation surface

Files expected to change (no code in this precommit):

- [src/energy_memory/phase34/online_codebook.py](../../src/energy_memory/phase34/online_codebook.py)
  — `_consolidate()` gains a branch on `use_context_residual` and
  `use_pull_push` flags. New method `_apply_context_residual(affected,
  event_predictions)`. Insertion order matches §Mechanism step list.
- `src/energy_memory/phase4/consolidation.py` — `ConsolidationConfig`
  gains `use_context_residual: bool = False`, `lr_cr: float = 0.1`,
  `use_pull_push: bool = True`. Default values preserve Path C
  byte-identity.
- `tests/test_context_residual_consolidation.py` — new tests per
  §Required pre-code gates #4.
- `tests/test_c2x_convergence_under_context_residual.py` — new tests
  per §Required pre-code gates #2.
- `tests/test_consolidation_path_c_byte_identity.py` — new tests per
  §Required pre-code gates #3.
- [experiments/c3_phase3_exit_criterion.py](../../experiments/c3_phase3_exit_criterion.py)
  — gains `--use-context-residual` CLI flag (Bool default False);
  `--lr-cr` CLI flag (float default 0.1); `--no-pull-push` CLI flag
  (Bool default False). Driver's existing test coverage extends to the
  new flags.

Expected LOC change: ~150 lines of mechanism + ~250 lines of tests.

## Implementation findings (running log)

### 2026-05-27 — Anti-homunculus reviewer: PASS

- **Verdict:** PASS, with four minor non-blocking concerns (landed
  inline into the precommit as binding sub-clauses and watch items;
  see "Binding sub-clause (per 2026-05-27 reviewer concern C1)" in
  §"Required pre-code gates" #2 and §"Watch items for the
  implementation-PR reviewer" for W1–W3).
- **Apparent decisions audited and their verdicts (per the reviewer
  agent):**
  - Indicator `1[predicted_id_j ≠ target_id_j]` inside E_cr →
    energy-support property, not runtime gate. PASS.
  - Asymmetric stop-gradient descent → partial gradient descent on a
    single symmetric energy, not two energies in disguise. PASS.
  - Composition with C.2.1–C.2.5 → additive vector forces with C.2.2
    multiplicative net modulation; composition order matches Path C
    exactly. PASS.
  - Flag-gated mechanism swap (`use_context_residual` /
    `use_pull_push`) → boundary-IO static config knob, not a runtime
    arbiter. PASS.
  - `predicted_id_j` consumed once per event → measurement input to a
    local force, no coupled fixed point. PASS.
- **H1–H7 status:** all addressed substantively (not checkbox-ticked).
  Right-hand-column form per [2026-05-09:140-148](2026-05-09-papers-diagnostics-and-actuator-dynamics.md)
  explicitly adopted at §"Architectural claim".
- **Gate status:** code is authorized to proceed past this gate,
  subject to the other two pre-code gates (convergence-equivalence
  re-run for C.2.1–C.2.5 under Γ1.c base; baseline parity test
  preserving Path C C.3 second smoke byte-identity). See
  §"Required pre-code gates" #2 and #3.

### 2026-05-27 — Γ1.c implementation landed; all pre-code gates green

- **Anti-homunculus reviewer gate:** ✅ (logged above).
- **Convergence-equivalence under Γ1.c base:** ✅
  [tests/test_c2x_convergence_under_context_residual.py](../../tests/test_c2x_convergence_under_context_residual.py)
  3/3 tests pass. Composition smoke verifies all five C.2.x dynamics
  fire under Γ1.c base, the codebook trajectory differs from Γ1.c-only
  (composition is real, not silent), per-pattern state arrays
  (splitting_tension, drift_tension, metastability_ema) survive shape-
  intact, and pull/push base also still composes (regression check).
- **Baseline parity (Path C byte-identity):** ✅
  [tests/test_consolidation_path_c_byte_identity.py](../../tests/test_consolidation_path_c_byte_identity.py)
  3/3 tests pass. With default flags
  (`use_pull_push=True, use_context_residual=False`), refactored
  `_consolidate()` produces codebook results byte-identical to inline-
  computed pull/push math at float32 atol=1e-7.
- **Unit tests for Γ1.c branch:** ✅
  [tests/test_context_residual_consolidation.py](../../tests/test_context_residual_consolidation.py)
  8/8 tests pass. Verifies: ε computed correctly from snapshot;
  asymmetric (predicted atom not updated); zero update when
  predicted == target (W1 indicator-as-mask form); mean aggregation
  across events; snapshot semantics across chained target/predicted
  overlap; unit modulus preserved; default flags route to pull/push;
  composition with C.2.1 anti-collapse is additive.
- **Full regression:** ✅ `tests.test_phase34` (14/14),
  `tests.test_anti_collapse`, `tests.test_splitting_tension`,
  `tests.test_cap_coverage_force`, `tests.test_metastability_replay_priority`,
  `tests.test_drift_replay_tension`, `tests.test_c3_phase3_exit_criterion`
  (12/12), plus the three new test files all green (109/109 tests in
  the combined sweep).
- **C.3 driver CLI:** ✅ [experiments/c3_phase3_exit_criterion.py](../../experiments/c3_phase3_exit_criterion.py)
  gains `--use-context-residual`, `--lr-cr`, `--no-pull-push`. Plumbed
  through 3 function layers down to the
  `OnlineCodebookUpdater` constructor. JSON summary records the three
  flag values. End-to-end smoke at D=64, vocab=10, n_consolidation_events=5
  executes cleanly with `--use-context-residual --no-pull-push`.

- **Watch item W2 (predicted_id provenance) verified.** [c3_phase3_exit_criterion.py:517](../../experiments/c3_phase3_exit_criterion.py)
  computes `predicted_id = int(scores.argmax())` over codebook
  similarities — i.e., MHN cleanup (nearest-codebook-row by cosine).
  Substrate-rooted, not a diagnostic-classifier output. H4/H6 framing
  upheld.
- **Watch item W1 (indicator-as-mask form) implemented.** [online_codebook.py:_apply_context_residual](../../src/energy_memory/phase34/online_codebook.py)
  uses no `if predicted_id != target_id:` branch in the runtime path.
  The indicator emerges as the energy-support property: `ε =
  codebook[target] − codebook[predicted]` is the zero vector when
  `target == predicted`, so the update naturally falls through to
  no-op for correct events.
- **Snapshot semantics implemented.** `_apply_context_residual()`
  takes `codebook_snapshot = self.codebook.detach().clone()` at the
  start and computes all ε from the snapshot (not the live codebook).
  Prevents coupled fixed-point ambiguity when the same atom appears
  as `target_id` of one entry and `predicted_id` of another in the
  same buffer batch.

**Status:** code authorized to proceed past all three pre-code gates
named in §"Required pre-code gates". The n=10 headline gate on
wikitext-2 at the operating point in §"Operating point" is now
authorized to run.

### 2026-05-27 — Headline gate FAIL at lr_cr=0.1; F1 lr_cr sweep is next

- **Verdict:** ❌ FAIL on both clauses of the revised C.3 criterion per
  [Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md).
- **Clause 1 (CI-disjoint at n=10):** failed. Γ1.c's largest Δ is
  +0.012 (default/spread), with CIs heavily overlapping. Same shape
  across all three populated strata.
- **Clause 2 (per-seed paired robustness ≥ 70%):** failed. 5/10 seeds
  with Δ > 0 = 50%.
- **Test-harness sanity** ✅: matched-seed PathC baseline (defaults,
  pull/push on) reproduces [Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md)
  v3 byte-identically (Δ_default/spread=+0.055 CI [0.241,0.280] vs
  [0.188,0.223] disjoint; Δ_calibrated/tight=+0.083 CI [0.350,0.401]
  vs [0.268,0.316] disjoint). Confirms the Γ1 refactor preserves
  experiment-output byte-identity on real wikitext-2.
- **Γ1.c is attenuated AND shape-different from pull/push.** Per-seed
  mean Δ +0.0115 ≈ 1/5 of PathC's +0.055. Per-seed σ ≈ 0.038 vs
  PathC's ≈ 0.169 — Γ1.c is 4.4× less seed-variance.
- **experiment-result-auditor 2026-05-27:** ✅ 5/5 done-gates PASS
  (headline+CI, control on same test set, drill-downs explain
  anomalies, markdown report, STATUS update landed this session).
- **Next move (per precommit's F1):** `lr_cr` sweep at {0.01, 0.05,
  0.1, 0.2, 0.5} at smoke scale n=3. Designed to distinguish
  effective-learning-rate-mismatch (atom-pair distances are typically
  smaller than cue magnitudes, so lr_cr=0.1 is effectively smaller
  than lr_pull=0.1) from atom-vs-atom-geometry-carries-less-signal
  (the corpus signal lives in cue geometry, not atom-pair geometry).
  F1 will be its own precommit document.
- **If F1 nulls at all magnitudes,** the Γ1 family closes and the
  next-candidate precommit (Γ2 bundle-first per the survey, or Γ3
  SFA-head) becomes the next deliverable.

*(Subsequent log entries: F1 precommit + results, follow-up precommits
F2–F6 as they land.)*

---

*See also: [path-gamma-mechanism-family-survey.md](../emergent-codebook/path-gamma-mechanism-family-survey.md),
[Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md),
[2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md),
[2026-05-09-papers-diagnostics-and-actuator-dynamics.md](2026-05-09-papers-diagnostics-and-actuator-dynamics.md),
[phase-3-deep-dive.md:188-205](../emergent-codebook/phase-3-deep-dive.md).*
