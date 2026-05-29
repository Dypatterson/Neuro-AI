# Q3 premise check — the σ-kill assumption (grill item 3)

**Date:** 2026-05-29 · verification artifact, no spec edits.

> ⚠️ **RELIABILITY WARNING.** An earlier draft fabricated a `PROJECT_PLAN:160-162`
> quote. Struck. Only verbatim-read lines below.

## Grill's Q3 claim
"The pivot assumes luck enters the slope only as intercept — but your own
charter ('better structure → faster later encoding') predicts luck hits the
slope too. If so, pairing cancels nothing and the pivot buys zero power.
1-hour smoke settles it."

## Verbatim ground truth (literally read)

**(a) Frame B's intercept-only assumption — CONFIRMED verbatim.**
`frame-b-exposure-slope-headline-design.md:30-32`: "The per-seed codebook
'luck' is **largely an additive intercept** on the exposure–recall curve; a
**within-seed slope differences it out**." Reinforced at `:66-67` ("invariant
to any additive per-seed constant") and `:112-113` (variance kill #1).

**(b) The charter "compounds with experience" claim — CONFIRMED verbatim,
but it is a GENERAL claim, not a slope-specific one.**
- `…continual-learning-…-finding.md:34-37`: "Phase 3's real claim is
  **continual learning** ('Growing Codebook', PROJECT_PLAN:160): atoms encode
  and refine from an experience stream **over time**."
- `PROJECT_PLAN:162`: "atoms that drift, stabilize, split, decay, and
  **consolidate from experience**."
- The grill's gloss "better structure → faster later *encoding*" (a slope
  claim) is the grill's INTERPRETATION; the docs say "refine over time /
  consolidate from experience," which is compatible with luck-as-intercept
  too. Neither doc explicitly says luck loads onto the slope.

## The finding (the grill's inference is legitimate and NOT doc-adjudicable)
Whether per-seed codebook luck loads only onto the intercept (Frame B's
assumption) or also onto the slope (the failure mode) is an **empirical**
question. No doc settles it. Item 3's smoke is the right instrument:
- measure `corr(intercept, slope)` across seeds,
- realized `σ(β_real)` vs the level `σ≈0.196`,
- whether real-vs-shuffle curves **fan out** (slope signal) or run
  **parallel** (signal is in the level → slope is the wrong estimand).

## Coupling to Q1
Q1 was approved **conditional on this smoke**. Q3 is therefore not a separate
gate — it is the *specification* of that smoke. Open user question: are the
smoke's thresholds (σ-reduction target + fan-vs-parallel test) set right, and
does it run before any Frame B build (grill's recommendation)?
