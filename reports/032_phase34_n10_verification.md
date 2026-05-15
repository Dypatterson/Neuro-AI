# Report 032 — Phase 3+4 integration, n=10 verification

**Date:** 2026-05-14
**Active phase:** 4 (verification)
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift.
**Required controls:** No-replay baseline (A) ✓; phase3-only (B) ✓.
**Last result:** Report 029 — n=5 ΔR@10 = +0.0145 [−0.005, +0.034], 4/5 positive, CI just barely includes 0 due to seed-23 outlier.
**Why this run:** Report 031 diagnostic showed seed 23 is not architecturally pathological — its negative result is a dynamics-trajectory outcome, not a substrate issue. The clean response is to add seeds. This run adds 5 new seeds {3, 5, 7, 13, 19} for a total n=10 verification.

Compute: Colab Pro Blackwell A100, ~10 min.
Source: [scripts/aggregate_phase34_n10.py](../scripts/aggregate_phase34_n10.py).
Per-seed JSON: `reports/phase34_integrated_hebbian_st03_seed{3,5,7,13,19,17,11,23,1,2}/phase34_results.json`.

---

## Headline: report 029 was sample-lucky.

| Subset | mean ΔR@10 | 95% CI (t) | Signs | Pooled-Wilson Δ |
| --- | ---: | --- | --- | ---: |
| Original 5 (rep 029) | **+0.0145** | [−0.005, +0.034] | 4+ / 0 / 1− | +0.0144 |
| New 5 (this run) | **+0.0041** | [−0.018, +0.027] | 1+ / 3 / 1− | +0.0045 |
| **All 10 combined** | **+0.0093** | **[−0.003, +0.021]** | **5+ / 3 / 2−** | **+0.0095** |

The new 5 seeds attenuate the effect by ~3.5×. Combining all 10 seeds, the per-seed t-CI is [−0.003, +0.021] — **still includes zero**, though n=10 is tighter than n=5 was at this true effect size. The mean ΔR@10 is positive (+0.009) and 5/10 seeds are positive, but the result is not disjoint from a null effect at standard confidence.

This is the strongest evidence to date that **the true ΔR@10 under online Hebbian drift is small** — likely in the +0.005 to +0.015 range — rather than the +0.015+ that report 029 suggested.

## Per-seed n=10 detail

| Seed | n | ΔR@10 | Δtop1 | Δcap_t05 | cons C | drift C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 225 | +0.013 | −0.027 | +0.004 | 275 | 1.6e-5 |
| 11 | 210 | +0.014 | −0.057 | +0.081 | 359 | 2.0e-5 |
| **23** | 229 | **−0.009** | −0.004 | −0.044 | 220 | 1.3e-5 |
| 1 | 227 | **+0.035** | +0.000 | +0.026 | 12 | 1.0e-7 |
| 2 | 220 | +0.018 | −0.041 | −0.036 | 224 | 1.2e-5 |
| **3** | 218 | **−0.014** | −0.005 | −0.037 | 248 | 1.5e-5 |
| 5 | 214 | +0.000 | −0.005 | −0.005 | 97 | 6.1e-6 |
| **7** | 232 | **+0.034** | −0.078 | +0.086 | 305 | 2.0e-5 |
| 13 | 215 | +0.000 | +0.014 | −0.009 | 85 | 5.7e-6 |
| 19 | 223 | +0.000 | +0.022 | −0.013 | 113 | 5.1e-6 |

**Pattern emerging:** the distribution of per-seed ΔR@10 is bimodal-with-a-fat-middle. A few seeds (1, 7) produce strong positive results (+0.034 to +0.035); a few (23, 3) produce negative; the middle 6 cluster near zero with small magnitudes. This is consistent with the report-031 finding that *seed-level variance is dynamics-trajectory variance*, not substrate variance.

## Δtop1 regression: confirmed across n=10

| Metric | n=10 mean | per-seed-t 95% CI | Signs |
| --- | ---: | --- | --- |
| Δtop1 | **−0.018** | [−0.041, +0.005] | 2+ / 1(0) / 7− |
| Δcap_t05 | +0.005 | [−0.028, +0.039] | 4+ / 0 / 6− |
| C − B R@10 | +0.004 | [−0.005, +0.013] | 4+ / 3(0) / 3− |

The top1 regression we identified in report 030 (as a Phase-3-driven property of Hebbian codebook reshaping, not a Phase 4 architectural gap) holds at n=10: **7 of 10 seeds show negative Δtop1.** Mean is −0.018, CI nearly excludes zero on the negative side.

C − B R@10 at +0.004 is the cleanest test of "what does Phase 4 add over phase3-only" — mean is positive (4/10 seeds), but tiny and CI clearly includes 0.

## What this run tells us

1. **Phase 4 under online Hebbian drift produces a small positive expected ΔR@10**, but the effect size is small and seed-variance is high.
2. **The report-029 result was sample-lucky.** The lucky seeds (1, 11, 2, 17) drove a misleadingly-confident headline at n=5.
3. **The top1 regression is robustly Phase-3-driven** — 7/10 seeds show negative Δtop1, persistent across our extended seed range. This is an inherent online-Hebbian property at this learning-rate regime, not a Phase 4 gap.
4. **The discovery channel's contribution is small but not zero.** C − B R@10 = +0.004 mean is positive sign, but the magnitude is below what would be a clearly-useful architectural addition.

## What this run does NOT tell us

- Whether Phase 4 would help more under different drift regimes (lower drift, longer runs, different success_threshold values).
- Whether the architecture's design intent — that Phase 4 should *compound* with codebook drift, not just survive it — is achievable with current parameters.
- Whether the bimodal per-seed distribution reflects two qualitatively-different Phase 4 outcomes (good vs neutral) or a single distribution with high variance.

## Implications for blocker #1

**Blocker #1 does not cleanly close at n=10.** The verified ΔR@10 result is:

> **+0.009, 95% CI [−0.003, +0.021], 5/10 positive, n=10**

Honest read: Phase 4 under sanctioned online drift produces a small positive R@10 expectation, but the effect is not disjoint from zero at standard confidence with n=10. This is *not the same* as "Phase 4 doesn't work" — it is "Phase 4's R@10 contribution is small enough that ~10 seeds aren't sufficient to confirm it crosses zero." Three honest options:

1. **Accept current evidence** — report n=10 result, declare Phase 4's R@10 contribution as "weakly positive expectation, individual-seed reliability low," and graduate phase on this evidence. Acceptable if we treat per-seed variance as a real architectural property.
2. **Run n=20 or n=50** to nail down the CI. Each new seed costs ~2 min on A100; we could do this for ~$5 worth of compute. The downside is that if the true effect is +0.005 (genuinely tiny), no reasonable n will produce a confidently-disjoint CI.
3. **Reframe the headline.** R@10 may not be the right metric. The C − B R@10 difference (+0.004 mean, +0.0095 pooled-Wilson) is the cleaner architectural test, but even it doesn't cleanly exclude zero. We could pivot to a different metric where Phase 4 produces a larger relative effect — but we'd need a principled reason for the change, not metric-shopping.

## Recommended next move

**Honest publication of current evidence and decision on the headline framing.** Two productive next steps in priority order:

1. **Decide whether the "discovered patterns" R@10 lift is the right Phase 4 success metric, or whether the meta-stable-rate reduction (from D1 aggregation, Δ = −0.51 at W=3 with std 0.038) is closer to what Phase 4 architecturally *does*.** The D1 result is far cleaner than R@10 ever has been. If the Phase 4 design intent is "compress meta-stable states into committed retrievals via consolidation," then meta-stable rate reduction *is* the headline. R@10 is a secondary downstream metric.

2. **If R@10 is to remain the headline, run n=20** (10 more seeds) to bring per-seed-t-CI to roughly [+0.001, +0.018] or similar — likely disjoint from zero but margin will be tight.

## Pattern check across reports

| Report | Config | n | ΔR@10 | CI excludes 0? |
| --- | --- | ---: | ---: | --- |
| 026 | Frozen + synthetic drift 0.30 | 5 | +0.010 | **Yes** |
| 028 | Online Hebbian st=0.5 (no drift) | 5 | +0.004 | No |
| 029 | Online Hebbian st=0.3 (real drift) | 5 | +0.0145 | No (barely) |
| 030 | + rfix | 5 | +0.011 | No |
| 031 (diagnostic) | — | — | — | — |
| **032 (this)** | **Online Hebbian st=0.3, n=10** | **10** | **+0.009** | **No** |

The progression is: synthetic-drift result (+0.010) was the most robust we've seen, online-Hebbian results land lower (+0.004 to +0.015 depending on configuration), and at n=10 with no special tricks we land at +0.009. The mean has stayed positive across every variant, which is meaningful, but the size is consistent with "small effect."

## Anti-homunculus check

- All experimental conditions remain local geometric dynamics. ✓
- No supervisor decides anything. ✓
- The bimodal seed distribution is emergent, not arbitrated. ✓

Anti-homunculus discipline holds. The honest conclusion is just that the effect we set out to verify is smaller than we thought.
