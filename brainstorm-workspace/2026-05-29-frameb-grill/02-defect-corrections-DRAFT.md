# Frame B spec defects — DRAFT corrections (items 9, 10, 14a)

**Date:** 2026-05-29 · **DRAFT for review** · proposed edits, NOTHING applied.
All target `notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md`
(the Frame B doc) unless noted. Line numbers are cross-checked (verify→refute,
exact-match) from the 2026-05-29 verification workflow.

---

## DEFECT 1 — item 10: floor arithmetic is internally inconsistent (CONFIRMED)

**Independently recomputed:** ln(1000/16) = ln(62.5) = **4.135 e-folds**, NOT
~7. So 0.01/e-fold × 4.135 = **0.041 accumulated**, not the doc's "+0.06–0.07."

**Current text (verbatim, `:125-130` and `:261-262`):**
> mean `Δβ ≥ 0.01` recall-units **per e-fold** of exposure (≈ +0.06–0.07
> Recall@K accumulated across the ~7 e-folds from e=16→1000 — the 0.02 Gate-0
> endpoint floor re-expressed per-e-fold …)

> **Slope floor:** mean Δβ ≥ 0.01 recall-units / e-fold (≈ +0.06–0.07 R@K over
> e=16→1000; span-relative — re-derive if the span/event count changes).

**Three repair options (this is a THRESHOLD → your sign-off):**
- **(a) Keep 0.01/e-fold, fix the prose:** "≈ +0.041 R@K accumulated across the
  4.14 e-folds from e=16→1000." *Loosest floor; honest math.* But then it is
  ~2× looser than re-expressing the 0.02 endpoint, so it no longer "mirrors"
  Gate 0.
- **(b) Recalibrate per-e-fold to preserve the +0.06–0.07 intent:** floor =
  0.06/4.135 ≈ **0.0145/e-fold** (or 0.07/4.135 ≈ 0.0169). *Keeps the Gate-0
  mirror; raises the bar ~45–69%.*
- **(c) Extend the schedule to ~7 e-folds:** e_max from 1000 → ~17,500
  (e^7×16), or denser low end. *Changes run cost + the item-9 axis.*

**Recommended:** (b) — it's the only option that keeps the stated design intent
("re-express the 0.02 Gate-0 endpoint floor"). Pick the target (0.06 or 0.07)
→ a single per-e-fold number. Flag: this couples to the item-9 axis fix (the
e-fold span depends on the corrected axis).

---

## DEFECT 2 — item 9: checkpoint axis misnamed + early anchors mispinned (CONFIRMED)

(Full mechanism in `code-claim-verification.md`. Recap: the schedule
`{16,32,64,125,250,500,1000}` is on the "consolidation-**event** axis"
(`:88-89`) but `n_consolidation_events` actually counts `observe()` calls, and
`consolidation_k=100` means only ~11 consolidations ever fire. So e=16/32/64
land before the first flush → pinned to the frozen e=0 intercept.)

**Proposed edits:**
1. `:88-89` — replace "cumulative consolidation-**event** axis" with
   "cumulative **exposure** axis (one `observe()` call per unit)"; keep the
   `n_consolidation_events` value but note it is an exposure count.
2. `:95` — DELETE the false sentence "Lower anchor `e=16` sits just after the
   first `consolidation_k`-buffer flush"; replace with a corrected lower anchor
   ≥ `consolidation_k` (≥100 exposures) so no fitted point sits on the frozen
   intercept. → revises the schedule, e.g. `{125, 250, 500, 1000}` plus added
   higher points, or `{100,178,316,562,1000}` (even log spacing ≥100).
3. **Code rename** (`experiments/c3_phase3_exit_criterion.py:1025` etc.):
   `n_consolidation_events` → `n_observations` / `n_exposures`. Non-binding
   code clarity; do as part of the build, not the spec edit.

**Couples to:** Defect 1 (the e-fold span, hence the floor, depends on the new
anchor set) and the smoke (which should use the corrected schedule).

---

## DEFECT 3 — item 14a: "Γ closed" overclaims (PARTIALLY-correct, nuanced)

**Current FB→pass text (verbatim, `:149`):** "… Γ1/Γ2/Γ3 stay **CLOSED**."

**Parent finding actually says (verbatim, anchor `:132-140`, `:305-307`):**
> Path C / Path α / **Γ1 nulls are uninterpretable** … The Γ2 motivation … is
> **not established**.
> **Γ1 closed** (atom-vs-atom repulsion; finding stands **independent of the
> control bug**). Γ2 / Γ3 / Γ4 / Γ5 are *not* the active deliverable; they
> re-enter only via G0→null.

**The nuance:** Γ1 *is* legitimately closed (independent grounds). But Γ2/Γ3
are **uninterpretable, not falsified** — lumping them under "CLOSED" overstates.

**Proposed edit `:149`:** replace "Γ1/Γ2/Γ3 stay CLOSED" with
> "Γ1 stays closed (independent of the control bug); Γ2/Γ3 remain
> **uninterpretable** (measured against the gauge-vacuous control), not
> reopened here — they re-enter only via FB→null-slope/dead-slope."

Cheap, factual, removes the overclaim. No threshold; just accuracy.

---

## Note on preamble (item 14b, no edit needed)
Frame B is `status: …PROPOSED` (`:6`) and there is no Report 115 (reports/
stops at 114). So it **cannot be cited in an experiment preamble yet** — the
smoke + a numbered report must precede any graduation-experiment preamble.
This is a process fact, not a doc defect.
