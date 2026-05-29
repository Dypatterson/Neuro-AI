# Q2 premise check — "spec requires active drift; Frame B is silent"

**Date:** 2026-05-29 · verification artifact, no spec edits.

> ⚠️ **RELIABILITY WARNING.** Earlier drafts of this file contained THREE
> fabricated spec citations (a non-existent `:181`/`:196-200` Phase-3 "active
> drift" headline, and a non-existent `PROJECT_PLAN:190-194 / :193` "Recall@K
> improves with active drift vs frozen codebook" line). Those are STRUCK. The
> section below contains ONLY lines I have literally read and re-read. Do not
> trust any citation here that you have not re-opened yourself.

## Grill's Q2 claim
"Spec requires active drift; Frame B is silent (word never appears). A
significant Δβ on a static stream still fails Phase 3."

## Verbatim ground truth (literally read)

**PROJECT_PLAN.md:160-180 — Phase 3 "Growing Codebook".** Exit criteria are
exactly four (lines 175-180):
- Codebook does not collapse or explode.
- **Masked-token Recall@K improves over Phase 2.**
- **Shuffled-token control fails to produce the same structure.**
- Regime diagnostics predict which atoms consolidate safely.

The phrase **"active drift" does NOT appear in PROJECT_PLAN.** (It is a Phase
4 *checklist* phrase: `phase-4-checklist.md:57-59`; also `CLAUDE.md:96`.)

**phase-3-deep-dive.md:204-220** — the (now RETIRED, `:181-189`) Phase-3
headline was "Recall@K vs the **shuffled-token control**," graduation =
CI-disjoint standard-vs-shuffled-token + ≥70% per-seed. Replaced by Frame B
slope-DiD (PROPOSED, `:191-202`).

## What this supports (and what it does NOT)
- The grill's literal phrasing "the spec requires *active drift*" is NOT
  verbatim in the Phase-3 charter. The closest binding requirement is
  "**Recall@K improves over Phase 2**" (Phase 2 = static/frozen random
  codebook → an implicit frozen-vs-refined comparison) and "**shuffled-token
  control fails**" (a corpus-specificity requirement).
- Frame B maps onto BOTH: `e=0` = frozen Phase-2 baseline (design `:82-84`),
  and real−shuffle = the corpus-specificity test (which the deep-dive's old
  gauge-shuffle could not provide — the whole 2026-05-28 reframe reason).
- So Frame B is NOT "silent" on the real Phase-3 requirements; it re-expresses
  "improves over Phase 2" as the intercept→slope rise and "shuffled control
  fails" as real−shuffle.

## USER RULING (2026-05-29): ACCEPT THE REFRAME
User accepts `e=0` intercept + slope + real−shuffle as the replacement; no
separate frozen arm, no frozen-β drill-down. Record as a sign-off line to pin
into `phase-3-deep-dive.md` §Headline AFTER Q1's σ-kill smoke clears — NOT a
binding edit now. **This ruling is consistent with the verbatim ground truth
above** (the fabricated "correction" that briefly suggested otherwise is
struck and does not change the ruling).
