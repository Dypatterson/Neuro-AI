# Report 030 — Phase 3+4 integration with discovered-pattern reencoding fix (st03_rfix)

**Date:** 2026-05-14
**Active phase:** 4 (verification)
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift.
**Required controls:** No-replay baseline (A) ✓; phase3-only (B) ✓.
**Last verified result:** Report 029 — ΔR@10 = +0.0145 [−0.005, +0.034], 4/5 positive, but Δtop1 = −0.0258 (4/5 negative).
**Why this experiment:** Report 029 surfaced a top1 regression and identified the *stale-discovered-patterns* hypothesis (discovered patterns stored with `source_windows=None` are never re-encoded against the drifting codebook). This run implements `reencode_discovered_patterns()` — re-settles each cached cue query through the current memory at each periodic reencode pass — and 5-seed sweeps to test whether the fix recovers top1 without sacrificing R@10.

Compute: Colab Pro A100, ~10 min.
Source: [scripts/colab_phase34_5seed.ipynb (rfix tag)](../scripts/colab_phase34_5seed.ipynb), aggregation by hand against report-029 baseline.
Per-seed JSON: `reports/phase34_integrated_hebbian_st03_rfix_seed{17,11,23,1,2}/phase34_results.json`.

---

## Headline result: **the fix did not work.**

| Metric | 029 (no fix) | **030 (rfix)** | Δ vs 029 | Verdict |
| --- | ---: | ---: | ---: | --- |
| **ΔR@10** | +0.0145 | **+0.0109** | **−0.0036** | Fix *erodes* the headline |
| **Δtop1** | −0.0258 | **−0.0229** | +0.0029 | Within noise; regression persists |
| **Δcap_t05** | +0.0064 | **−0.0043** | **−0.0107** | Fix flips capt5 net-negative |
| ΔR@10 5-seed t-CI | [−0.005, +0.034] | [−0.005, +0.027] | tighter, still includes 0 | n=5 still insufficient at this variance |
| ΔR@10 pooled-Wilson | +0.0144 (388/1111 vs 372/1111) | +0.0108 (384/1111 vs 372/1111) | −0.0036 | sample-level Δ smaller |

The fix moves all three headline-relevant metrics in the wrong direction or sideways. Top1 barely improved (still 4/5 negative). Cap-coverage swung from slight-positive to slight-negative.

## Per-seed (030)

| Seed | n | ΔR@10 | Δcap_t05 | Δtop1 | cons (C) | drift (C) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 17 | 225 | +0.009 | +0.009 | −0.027 | 376 | 2.5e-5 |
| 11 | 210 | +0.014 | **+0.086** | −0.038 | 433 | 2.4e-5 |
| 23 | 229 | **−0.009** | **−0.044** | −0.004 | 212 | 1.2e-5 |
| 1 | 227 | +0.026 | −0.013 | +0.000 | 9 | 6.3e-8 |
| 2 | 220 | +0.014 | **−0.059** | −0.045 | 232 | 1.3e-5 |

- **Seed 23: 4th consecutive negative-outlier appearance.** Reports 026, 028, 029, and now 030 all identify seed 23 as the cap_t05 / R@10 outlier. Continued tolerance without diagnosis is the bottleneck on tightening CIs.
- **Seed 1 (low-drift): R@10 advantage holds.** With only 9 consolidations and ~zero drift, seed 1 still produces ΔR@10 = +0.026, similar to its 029 result (+0.035). This is consistent with the post-hoc reading below: when the codebook isn't moving, discovered patterns retain their value with or without the fix.
- **Seeds 11, 2: capt5 split.** Seed 11 capt5 lifts strongly (+0.086) but seed 2 capt5 collapses (−0.059). Bimodal even within the higher-drift seeds.

## Why the fix didn't work — in retrospect, predictable

A discovered pattern is the *settled state of an unresolved retrieval* — high engagement, low resolution. It sits **outside** existing attractor basins; that's the whole reason the gate fired and the pattern was added.

Re-settling the cached query against the current memory pulls the pattern toward whichever attractor the current landscape favors. The current landscape now contains:
- Refreshed original patterns (post `reencode_patterns`)
- Other discovered patterns (some refreshed, some not yet)

The re-settled state is therefore drawn toward an **existing** attractor rather than re-establishing the unresolved meta-stable point. This destroys the geometric property the discovery channel was trying to capture.

Empirical support for this reading:
- **C − B advantage at R@10 *shrinks* with the fix** (+0.0062 → +0.0026). Phase 4 adds less over phase3-only when discovered patterns are refreshed.
- **C − B advantage at capt5 turns more negative** (−0.0171 → −0.0278). Refreshed discovered patterns compete with — rather than complement — the refreshed originals.
- **Hebbian consolidations rise modestly with the fix** (1090 total → 1262 total, +16%). The fix produces *more* partial-success retrievals to learn from, but those partial successes don't translate into improved evaluation metrics.

The pre-experiment internal note flagged this risk verbatim:

> "Re-settling the discovered pattern through the current memory will pull it toward the *current* memory's local attractor. So the re-settled discovered pattern will tend to collapse toward existing patterns... which defeats the discovery channel's purpose."

The right move was to honor that read. We did not. The fix was shipped anyway; this report is the cost.

## What this run *does* tell us

1. **The top1 regression is not caused by representation staleness of discovered patterns.** Refreshing them doesn't recover top1. The regression must come from somewhere else.
2. **The R@10 gain in 029 was partially structural** — it depended on discovered patterns occupying their original unresolved-state positions in vector space. When refreshed, those positions move and the discovery-channel's R@10 contribution shrinks.
3. **The right fix is *death*, not refresh.** Stale discovered patterns should be culled, not relocated.

## Implications for blockers

**Blocker #1 (Phase 3+4 integration with active drift, ΔR@10 verified):**
The conditional close from report 029 stands. ΔR@10 = +0.0145 4/5 positive is the verified-best result. The rfix variant produced +0.0109 — also positive 4/5, also CI-includes-0, but slightly smaller. Either run supports the design's claim that Phase 4 helps R@10 under sanctioned online drift; the rfix attempt didn't strengthen that claim.

**Blocker #6 (discovered-pattern reencoding gap):**
**Closing as "wrong-shaped fix, hypothesis revised."** Refresh-via-resettle is not the right primitive. Replacing with:

**Blocker #6′ (new): top1 regression under drift — root cause is *not* representation staleness of discovered patterns.** Hypothesis revision: top1 collapse comes from the Hebbian-updated codebook itself reshaping atom geometry, not from stale stored patterns. Test: run condition B (phase3_reencode only, no Phase 4) for 1500 cues and observe its top1 trajectory at this drift rate. If B_top1 also crashes from 0.110 → 0.081 in seed 11 *without any Phase 4*, then top1 regression is a Phase 3 property, not a Phase 4 one. Looking at the existing reports/phase34_integrated_hebbian_st03_seed11/phase34_results.json, B_top1 trajectory is 0.110 → 0.148 → 0.157 → 0.119 → 0.081 → 0.081 — **so yes, the regression is happening in B too.** Top1 collapse is Hebbian-driven codebook reshaping, not stale Phase 4 patterns.

This is a meaningful re-frame:
- **Phase 4 *also* hurts top1** (C_top1 < B_top1 in 4/5 seeds for both 029 and 030)
- **But the bulk of the top1 collapse comes from Phase 3 / Hebbian alone**
- Therefore the top1 regression is **not a Phase 4 architectural gap** — it's a Phase 3 learning-rate / objective issue, or an inherent tradeoff of online learning at this drift rate

**Blocker #2 (death mechanism vacuous):** This is now load-bearing. If the death mechanism fired, stale discovered patterns would self-cull, and Phase 4's C−B advantage might lift cleanly. Fix order:
1. Set `death_threshold=0.05`, `death_window=20` (audit-recommended values).
2. Rerun st=0.3 5-seed without the rfix.
3. Compare to 029.

If with-death produces ΔR@10 ≥ 029 *and* Δtop1 stops regressing in condition C relative to condition B, blocker #1 closes outright.

## Code disposition for the rfix change

The `reencode_discovered_patterns` function and `--reencode-discovered` flag are kept in the codebase but the default should flip to `False`. Reasoning: the primitive isn't useless — it might be the right tool for a future variant (e.g. selective refresh of patterns whose query atoms have drifted), but as a default it harms the headline.

**Recommended diff:** in [experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py), change `default=True` → `default=False` on the `--reencode-discovered` flag, with a comment pointing here.

## Anti-homunculus check

- The rfix primitive itself is a local geometric dynamic (re-run retrieval). No homunculus. ✓
- The death mechanism is also a local dynamic (Benna-Fusi u-chain → garbage_collect when below threshold). No homunculus. ✓
- The right primitive to fix the top1-vs-R@10 tension is death, not refresh. The architecture already specifies the right tool; we just haven't exercised it.

Anti-homunculus discipline holds.
