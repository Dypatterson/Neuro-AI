# Frame B pre-build smoke — DRAFT spec (item 3 / 7 / 8)

**Date:** 2026-05-29 · **DRAFT for review** · not binding, no spec edit.
**Decision basis:** user ruled "run the smoke BEFORE any Frame B build" (Q1
conditional + this session's 3-way decision). Build proceeds ONLY if the
smoke passes all gates below.

## Why this exists (verified, cross-checked)
Three grill items collapse into one empirical question the docs cannot settle:
- **Item 3:** Frame B asserts per-seed codebook luck is "largely an additive
  intercept" (`frame-b…design.md:30-32`, verbatim-confirmed) and never proves
  luck doesn't also load on the slope. If it does, within-seed pairing cancels
  nothing and the reframe buys ~0 power.
- **Item 7:** the DiD `Δβ=β_real−β_shuffle` is only non-vacuous if
  `E[β_shuffle]>0` (`…design.md:114, 151`, confirmed). Assumed, never argued.
  If `β_shuffle≈0`, clause-1 collapses into clause-2 and real−shuffle is
  decorative.
- **Item 8:** if the real-vs-shuffle gap is **parallel** (constant offset) not
  **fanning** (diverging), the signal lives in the LEVEL and the slope is the
  wrong estimand. Not answerable from committed data — Gate 0 ran only the
  endpoint (`STATUS.md:63`: DiD +0.019 CI [-0.121,+0.159], σ≈0.19).

## What the smoke measures (one pilot run, reuse existing machinery)
Reuse `experiments/gate0_frame_a.py` + the `c3` checkpoint-eval path; single
operating point (Γ1/Path C wikitext); 2 worlds (real, stream-shuffle);
**a HOLDOUT seed set distinct from the graduation seeds 0..9** (see item 6
fix — pilot seeds e.g. 1000..1009 so the σ threshold is not read on the
verdict seeds). Read-only checkpoint evals at the (corrected, item 9/10)
exposure schedule.

Per pilot seed `s`, both worlds, collect `r_w(s, e_m)` at the checkpoints,
fit per-world OLS `β_w(s)` and intercept `a_w(s)`. Then compute:

1. **corr(intercept, slope):** Pearson `corr(a_real(s), β_real(s))` across
   pilot seeds. *Tests item 3.* If strongly positive, luck loads on the slope
   too → pairing won't shrink σ as assumed.
2. **realized σ(β_real)** and **σ(Δβ)** vs the level σ≈0.196. *Tests the whole
   power premise.* Report implied honest-n via existing
   `variance_report_from_summary` / `n_for_80pct_power`.
3. **fan-out vs parallel:** is `[r_real(e)−r_shuffle(e)]` increasing in
   `log e` (fanning) or flat (parallel)? Slope of the gap-vs-log-e line, with
   CI. *Tests item 8.*
4. **E[β_shuffle]:** mean and CI of `β_shuffle(s)`. *Tests item 7.*

## PROPOSED pass gates (these are thresholds → user sign-off)
Proceed to Frame B build only if ALL hold on the pilot:
- **G1 (σ-kill real):** σ(Δβ) materially below σ_level≈0.196 — proposed
  σ(Δβ) ≤ ~0.05 (mirror the design's own line-264 bound; flagged underived in
  item 6, so treat as provisional and re-derive from the pilot itself).
- **G2 (intercept-only-ish):** corr(intercept, slope) not strongly positive —
  proposed |corr| ≤ ~0.4 (NEW threshold, needs your number).
- **G3 (fanning):** gap-vs-log-e slope CI excludes 0 and is positive.
- **G4 (non-vacuous control):** `E[β_shuffle]` CI excludes 0 (positive).

**Branch on failure:**
- G1/G2 fail → luck loads on the slope; pivot buys no power → **do NOT reframe;
  the LEVEL headline (Frame A) is correct** (this is exactly the Q1 conditional
  firing). 
- G3 fails (parallel) → signal is in the level → slope is wrong estimand →
  stay with endpoint DiD, escalate n there instead.
- G4 fails (β_shuffle≈0) → DiD is decorative → drop the real−shuffle outer
  difference or redesign the control.

## Cost
≈ pilot n (say 10) × 2 worlds = 20 procs, ~1 hr (same scale as the design's
own §Cost estimate). Pure read; no new substrate mechanism → no
anti-homunculus reviewer pass needed (consistent with the design's `:176-177`).

## Open user inputs before the smoke runs
- pilot n and seed offset (proposed n=10, seeds 1000..1009);
- G2 corr threshold (proposed |corr|≤0.4 — no precedent, your call);
- whether G1's 0.05 is re-derived from the pilot or kept as the line-264 value.
