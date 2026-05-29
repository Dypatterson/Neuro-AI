# Frame B precommit-v2 — DRAFT hardening (items 6, 12, 13, 4, 5c)

**Date:** 2026-05-29 · **DRAFT for review** · proposed additions, NOTHING
applied to binding docs. Consolidates the five "tighten before binding" items
into one precommit revision for your sign-off. Line refs cross-checked
(verify→refute exact-match) from the 2026-05-29 workflow.

---

## H1 — item 6: σ-gate circularity + same-seed contamination (grill-correct)

**The problem (verbatim):**
- `:235` "affordable *because* the slope's σ is small (the whole point)" —
  circular: you don't know σ is small until you run.
- `:263-264` "σ(Δβ) ≤ ~0.05 / honest-n ≤ ~30 as 'beaten'" — threshold
  **underived**.
- `:76-80` + `:117-119` + `:123`: σ(Δβ) is computed from the **same** n=10
  seeds (0..9) that produce the CI and robustness verdicts → non-independent.

**Proposed fix:** the **smoke (deliverable 01) already breaks the circularity**
— it estimates σ on HOLDOUT pilot seeds (1000..1009) *before* the graduation
run. So precommit-v2 should:
1. Make the σ-gate threshold **derived from the pilot**, not asserted: "σ(Δβ)
   must beat the pilot-measured intercept-dominated baseline by factor F"
   (propose F=2). Replaces the bare 0.05.
2. State explicitly that σ used for the **go/no-go to reframe** comes from the
   pilot seeds; σ reported alongside the n=10 verdict is a *consistency check*,
   not the gate. Removes the same-seed contamination for the load-bearing
   decision.

---

## H2 — item 12: branch table not disjoint + no precedence (grill-correct)

**The problem (verbatim, `:147-154` branch table):** FB→non-linear
(`|quad/lin|>1` OR <70% monotone), FB→weak (CI overlaps OR <70%), and
FB→null-slope (Δβ≈floor, both β>0) **can all be true at once** → analyst picks
the branch after seeing curves. Yet `:142-145` says "Verdict is a pure function
of stored per-seed stats … no post-hoc tuning" — so the overlap directly
violates the doc's own no-post-hoc rule.

**Proposed fix — a deterministic precedence order baked into
`_classify_verdict` (precommit, before any run):**
```
1. FB→confound      (gauge arm E fails / 4a not byte-identical)   # integrity first
2. FB→dead-slope    (β_real ≈ β_shuffle ≈ 0)
3. FB→non-linear    (curvature fail) -> triggers AUC re-read, then re-enter at step 4
4. FB→null-slope    (Δβ ≤ floor, both β > 0)
5. FB→pass          (both clauses + σ beaten)
6. FB→weak          (everything else)
```
Rationale: integrity gate dominates; structural failures (dead/non-linear)
before signal classification; pass before weak so weak is the true residual.
Encode as an ordered if/elif in `_classify_verdict`; add a unit test asserting
no input maps to two branches.

---

## H3 — item 13: sequential peeking, no α-spending (grill-correct)

**The problem (verbatim):** `:150` + `:259` + `:235` describe n=10→20→30
escalation on FB→weak, stopping when the CI clears — a 3-look sequential test.
No Pocock/O'Brien-Fleming/Bonferroni mentioned → real Type-I ~10–12% vs nominal
5%.

**Proposed fix:** adopt **O'Brien-Fleming α-spending** for the 3 looks
(conservative early, near-nominal at the final look) — preserves most power at
n=30 while controlling family-wise α at 0.05. Concretely: looks at n=10/20/30
with OBF boundaries (two-sided α=0.05 → nominal per-look ~0.0006 / 0.015 /
0.045). State that the **first** n=10 look uses the OBF boundary, not raw 95%.
Alternative if you prefer simplicity: Pocock (constant ~0.022 per look) or
Bonferroni (0.0167). OBF recommended (best power retention).

---

## H4 — item 4: estimand ADR has no itemized alternatives/cost table (partial)

**The problem (verbatim):** `:21-23` claims "4-way design panel … winner =
slope-DiD spine (29/30)" and `:25-36` gives the rationale, but there is **no
itemized score table** showing how slope beat the alternatives, nor the power
cost of each.

**Proposed fix:** add a short ADR table to the doc (or a sibling
`frame-b-estimand-adr.md`):

| estimand | power (σ / honest-n) | reversibility | faithfulness to charter | panel score |
|---|---|---|---|---|
| endpoint level-DiD (Frame A) | σ=0.196 / ~590 seeds | — (status quo) | direct | (the demoted baseline) |
| within-seed slope-DiD | (pilot-TBD) | hard to reverse | "compounds w/ exposure" | 29/30 |
| time-to-encode-novelty | not estimated | — | direct but discrete/noisy | (rejected) |
| AUC-above-intercept | ~slope | held as non-linear fallback | same | (fallback) |

Fill the slope σ row **from the pilot** (ties H1). Records the alternatives +
cost the grill asked for; makes the 29/30 auditable.

---

## H5 — item 5c: "shuffle" names 3 objects (grill-correct)

**The problem (verbatim):** "shuffle" = (1) global token-**stream** shuffle
(`:100-108`, the DiD control), (2) **within-window** shuffle (`:100-108`,
post-grad drill-down), (3) codebook **row-permutation / gauge** arm
(deep-dive `:182-189`, retired). Pure terminology hazard.

**Proposed fix (rename, no semantics change):**
- (1) → **`stream-shuffle`** (keep `world="shuffled"` code value; doc prose
  always "stream-shuffle").
- (2) → **`window-shuffle`**.
- (3) → **`gauge-permutation`** (never "shuffle").
Add a one-line glossary at the top of §Control. Cosmetic but removes a real
mis-read risk in a binding doc.

---

## Sequencing (how these compose)
1. **Smoke first** (deliverable 01) — gates the whole reframe; feeds σ to H1+H4.
2. If smoke passes → apply **defect corrections** (deliverable 02: floor math,
   axis, Γ language) + **precommit-v2** (this file: H1-H5) as ONE spec revision.
3. Then pin into `phase-3-deep-dive.md` §Headline + write Report 115.
All of step 2/3 is binding-doc work → returns to you for sign-off; nothing is
applied from this draft.
