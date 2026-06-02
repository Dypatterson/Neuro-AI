# Report 131 — Capacity sweep: NULL-MONOTONE — capacity-tightness does NOT force abstraction (necessary-not-sufficient); adaptive-assignment is load-bearing; the local writer plateaus below the global ceiling regardless of capacity

**Status:** NULL-MONOTONE (the under-capacity-peak hypothesis is falsified). Banked. Confirms the capacity grounding's "necessary-not-sufficient."
**Date:** 2026-06-02. **Branch:** `experiment/tem-local-reachability-oracle`.
**Harness:** `experiments/72_online_local_on_build_s.py` (swept over k). **Precommit:** [phase-3-capacity-sweep-precommit.md](../../notes/emergent-codebook/phase-3-capacity-sweep-precommit.md). **Artifacts:** `k4/k64/k512.json`, `run.log` (k=32 = the exp72 anchor).

## Preamble

- **Active capability:** Codebook-growth (P3 structure), substrate-free oracle (in-scope).
- **Headline:** within-paradigmatic-set label-shuffle B-KILL of the adaptive online-local writer **vs capacity k** — does it PEAK in the under-capacity regime (the user's "fixed capacity forces abstraction" hypothesis)?
- **Required controls:** frozen-random arm (B); global ceilings (offline k-WTA / k-means / NMF); converged-calibration arm (decay=1.0 must reproduce the offline ceiling — confirms the writer isn't broken); grow_G floor; anchor +0.1092/0.222.
- **Last verified:** the capacity grounding (necessary-not-sufficient); Report 129 raw arrays (tighter budget made the local writer worse).

## Result (build_S, D=4096, n=10; anchor valid)

| k (capacity) | adaptive-local **A** (online_bounded) | frozen-random **B** | A−B | local-no-budget (converged) | offline ceiling |
|---|---|---|---|---|---|
| 4   | +0.130 | +0.035 | +0.095 | +0.150 | ~+0.14–0.25 |
| 32 (exp72) | ~+0.179 | — | — | ~+0.224 | ~+0.23 |
| 64  | +0.181 | +0.051 | +0.130 | +0.225 | +0.227 |
| 512 | +0.182 | +0.058 | +0.124 | +0.225 | +0.235 |

## Verdict — NULL-MONOTONE

The adaptive local writer's paradigmatic B-KILL **rises then plateaus at ~+0.18** (k=4 → 512); it does **not** peak in the under-capacity regime — the *tightest* capacity (k=4) gave the **lowest** score. The precommit's PASS-COMPRESSION required a non-monotone peak at k\*≪V; we observe monotone-rising-then-flat. **The "tighter capacity forces more abstraction" hypothesis is falsified.**

What **is** confirmed (and load-bearing):
- **Adaptive-assignment matters at every capacity:** A ≫ B (frozen-random), gap +0.095→+0.124. The 129 reconciliation holds — adaptive assignment under the budget is the real ingredient, not capacity-tightness.
- **The local-bounded writer plateaus *below* the global ceiling regardless of k** (ratio ~0.77–0.80; converged−bounded across-seed CI-lo>0 at every k). The locality cost is robust and capacity-independent.
- The converged-calibration arm reproduces the offline ceiling (writer not broken).

## What this banks — the spec clause

**Capacity is necessary (global methods reach king/queen at k=8–32) but not sufficient — tightness does not force abstraction; it *hurt* the local writer at the tightest budget.** The load-bearing pieces are (i) adaptive-assignment, and (ii) *something beyond a single bounded-local pass* — the local writer cannot close the gap to global by capacity alone. This **closes the "just squeeze the capacity" branch** and points, with the frustrated-phase null (Report 130, single-projection is channel-invariant), at the same place: **depth + adaptive nonlinear capture + a local temporal write** — the one direction that isn't single-projection-anything.

## Honest notes

- g3 (corr<0.15) remains diagnostic-only at n≤40 (the 129/exp65 lesson); B-KILL is the arbiter.
- Over-capacity (k=512) did not fake a positive here (Report 126's saturation warning) — the plateau is flat, not a spurious spike; the memory-validity held.
- Scope: single-layer online-local-VQ on build_S. The null is "capacity-tightness is not the lever for a single bounded-local pass," NOT "compression is irrelevant" — the depth route (Report 132 / `experiments/77`) holds the residual across layers, which capacity-on-a-flat-code cannot.

## Disposition

Capacity-tightness branch **closed**. Adaptive-assignment confirmed load-bearing but insufficient alone. → **depth + local-temporal (habituation) write** ([phase-3-depth-habituation-precommit.md], `experiments/77`) is the converged next build.
