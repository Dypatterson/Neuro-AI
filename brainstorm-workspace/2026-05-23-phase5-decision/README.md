# Phase 5 Critical Decision Point — 2026-05-23

> **CURRENT STATE — READ THIS FIRST.** This README was generated at the
> START of the 2026-05-23 brainstorm session (Phase 1 context-gathering)
> and presents the pre-brainstorm framing. **It is HISTORICAL.** The
> brainstorm session itself ran a Tier 0 diagnostic trio and was then
> independently audited (Codex, two rounds), which materially changed
> the strategic decision tree. The "Path A vs Path B" two-path framing
> below is **superseded**.
>
> **Authoritative current decision-state:**
> [tier0-results.md](tier0-results.md) — has all walk-backs applied,
> reflects Codex audit corrections, and carries the corrected path
> ranking. Read this first.
>
> [STATUS.md banner at repo root](../../STATUS.md) — project-level
> summary of where the decision tree stands.
>
> **What changed since this README was written:**
> 1. Path A's premise ("D-sweep rescues floor") was wrong as STATUS
>    originally framed it. The D-lever at fixed N is only ~1.3× across
>    16× D range. Path A is only viable along a capacity-proportional
>    trajectory (D ↓ AND N ↓). With the corrected operating-point ΔE
>    (report 058 best cell +0.002478, not the older +0.0013), even
>    D=1024 with N=266 crosses the floor — Path A is **stronger**
>    than the brainstorm originally framed.
> 2. Path B as written (pair #4 metastability/replay, "½ day pivot")
>    is **foreclosed** — pair #4 trajectory-c_i has already failed
>    its own pre-committed smoke gate per report 054. The actionable
>    form is **Path B'** (close + pivot to surprise/PE-driven replay,
>    ~1 week, requires Phase 5 spec update).
> 3. A third path emerged from research — **Path C** (Varner log-prior
>    softmax bias spike) — but was initially over-sold based on a
>    misinterpreted Fisher diagnostic. Per Codex audit P1, the
>    diagnostic [scripts/fisher_separation_diagnostic.py](../../scripts/fisher_separation_diagnostic.py)
>    measures global pattern-pair separability, not the Varner per-cue
>    functional-subset/background PCA Fisher index. Path C is now
>    framed as a cheap (1-2 day) exploratory probe, not an
>    "empirically supported" next step.
>
> The decision tree below is preserved as written (historical record
> of the brainstorm's starting state). The corrected tree is in
> [tier0-results.md](tier0-results.md) §"Recommended next session".

---

This session is a **strategic brainstorm** at a critical juncture. Phase 5's graduation attempt has failed not due to parameter tuning or mechanism breakdown, but due to a fundamental **substrate-saturation finding**: the A+B+A1' substrate's clean-basin architecture produces sub-noise energy gaps between structurally-different priors at D=4096.

## Session Purpose

Before committing to the next major effort (lower-D redesign, ~2–3 weeks; or closure + pivot to a different target), explore:
- Whether the substrate-saturation finding is correctly diagnosed
- Whether any unexplored diagnostic or mechanism lever could bypass the need for lower-D redesign
- Which strategic direction is most generative for the architecture's long-term vision

## Context

**File:** `/context/phase5-current.md` (206 lines)

**Key sections:**
1. **What Phase 5 was trying to do** — the architectural hypothesis and mechanism
2. **What the substrate-saturation finding says** — six independent instances where clean-basin geometry forecloses advancement
3. **What's foreclosed empirically** — options 2, 3, 4 are dead; the evidence against each
4. **What's still live** — option 1 (lower-D redesign) or contingency (close + pivot to pair #4)
5. **Subtle constraints to remember** — audit #10, anti-homunculus filter, magnitude floor, K=4 vs K=1 dichotomy, ΔE/basin anti-correlation, random-prior pathology
6. **What the 2026-05-20 brainstorm already explored** — 12 ideas organized into tiers; which ones are still live vs deferred vs shelved
7. **Timeline and decision gates** — Path A (lower-D) vs Path B (closure + pair #4 pivot)

## Most Recent Reports (the empirical record)

- **Report 058** (2026-05-23): Cue-regime sweep, 24 cells, cross-seed. **Key finding:** ΔE and basin-membership metrics are **anti-correlated** across cue regimes. Best ΔE has zero basin hits; best basin-hit has negative ΔE.
- **Report 057** (2026-05-21): Cross-seed β sweep. Confirms β=10 is optimum; ΔE statistically significant but 4.2× below magnitude floor.
- **Report 053** (2026-05-21): n=10 headline run. Binding pre-commit gate: both CI-disjoint-from-zero AND magnitude ≥ 5.5e-3. Passes first, fails second. **Graduation-unattained.**
- **Report 044** (2026-05-20): Consolidation-geometry diagnostic. d_eff collapses 7×–16× from pre-death (~40) to post-death (~5). Explains K-branch divergence collapse.

## Decision Tree (HISTORICAL — see banner above; superseded by [tier0-results.md](tier0-results.md))

```
User decision (PRE-BRAINSTORM FRAMING — see walk-back at top of README):

Path A: Lower-D redesign  [STRENGTHENED post-Tier-0 — see tier0-results.md]
├─ D-sweep diagnostic (2 days)
│  ├─ If lower-D shows promise (ΔE within 1–2× floor at some D):
│  │  └─ Phase 4 retrain at lower D (1–2 weeks)
│  │     └─ Phase 5 retry with re-derived magnitude floor
│  └─ If lower-D doesn't rescue or anti-correlation persists:
│     └─ Close Phase 5, pivot to Path B
│
Path B: Close Phase 5 + Pair #4 pivot   [FORECLOSED post-Tier-0 — pair #4 smoke failed per report 054]
├─ Pair #4 (metastability/replay) implementation (~½ day)
├─ Pair #4 graduation run (n=10, 2 days)
└─ Document substrate-saturation finding as closure

Path C: Other (emerging during brainstorm)   [→ became Varner log-prior spike; downgraded by Codex audit]
└─ TBD
```

## Reading Order (CORRECTED)

1. **[tier0-results.md](tier0-results.md)** — authoritative current decision-state with all walk-backs applied
2. **[STATUS.md banner](../../STATUS.md)** — project-level summary
3. This README (historical context for what the brainstorm STARTED with)
4. `/context/phase5-current.md` — comprehensive pre-brainstorm context
5. `/brainstorm-phase5-decision.md` — the brainstorm doc (Phase 4 output of the brainstorm skill; also pre-walk-back)
6. `/research/*.md` — five deep-research briefs (binding operators, basin engineering, dimensionality scaling, metastability/replay, FEP/closure)
7. Latest three reports (058, 057, 053) for empirical specifics
8. The 2026-05-20 brainstorm for unexplored options

## Questions for Brainstorm (HISTORICAL — these were the starting questions, answered now by tier0-results.md)

1. Is the six-instance substrate-saturation diagnosis correct? Any alternative explanation?
2. Is lower-D redesign (Path A) worth 2–3 weeks, or are the blocking issues fundamental to branching-on-sharp-basins regardless of D?
3. Are there unexplored levers in the 12 ideas that could bypass substrate redesign?
4. If closing Phase 5 (Path B), is pair #4 (metastability/replay) the right next target, or should the brainstorm surface a different diagnostic-actuator pair?
5. What does the 5-year architectural vision suggest? Is Phase 5's closure + pivot philosophically consistent with the long-term design?

---

**Artifacts:**
- `context/phase5-current.md` — comprehensive context summary (pre-brainstorm)
- `context/prior-phases.md` — Phases 2/3/4 architectural carry-forward
- `context/research-base.md` — papers + cross-paper synthesis themes
- `research/*.md` — five deep-research briefs
- `brainstorm-phase5-decision.md` — Phase 4 brainstorm output (pre-walk-back)
- **`tier0-results.md`** — **authoritative current decision-state** (all walk-backs applied)
- `README.md` — this file (historical starting framing)

**Generate time:** 2026-05-23, ~30 min from raw reports and notes
**Status:** Brainstorm + Tier 0 + two Codex audit rounds complete; user decision pending on Path A vs B' vs C
