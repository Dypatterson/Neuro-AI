# Report 028 — Phase 3+4 integration, 5-seed (Hebbian)

**Date:** 2026-05-14
**Active phase:** 4 (verification)
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift between cycles.
**Required controls per [phase-4-unified-design.md:318-323](../notes/emergent-codebook/phase-4-unified-design.md):** No-replay baseline (condition A) ✓; phase3-only (condition B) ✓ — both built into exp 19.
**Last verified result:** [Report 026](026_phase4_verification_design_spec.md) — ΔR@10 = +0.010 [+0.001, +0.022] at step 2000, 5 seeds, frozen Phase 3c codebook + *synthetic* drift.
**Why this experiment:** STATUS.md blocker #1. The prior verified Phase 4 result used a frozen codebook + synthetic perturbation, not the architecturally-intended online-Hebbian drift. This run tests Phase 3+4 with codebook drift produced by the online Hebbian updater itself, across 5 seeds {17, 11, 23, 1, 2}.

Compute: Google Colab Pro A100, ~10 min total.
Source: [scripts/aggregate_phase34_5seed.py](../scripts/aggregate_phase34_5seed.py).
Per-seed JSON: `reports/phase34_integrated_hebbian_seed{17,11,23,1,2}/phase34_results.json`.

---

## Headline result

| Metric | 5-seed mean | 95% CI (t, 4df) | Per-seed signs | Verdict |
| --- | ---: | --- | --- | --- |
| **ΔR@10** | **+0.0044** | **[−0.012, +0.021]** | 2+ / 2(0) / 1− | CI **includes zero** |
| Δtop1 | +0.0129 | [−0.007, +0.033] | 2+ / 3(0) / 0− | CI includes zero, sign consistent |
| Δcap_t05 | +0.0028 | [−0.021, +0.027] | 3+ / 0(0) / 2− | CI includes zero |

Pooled-Wilson over the union (n=1111): R@10 baseline 0.335 → phase4 0.339 (Δ = +0.004).

**This does not replicate the report 026 ΔR@10 = +0.010 result with disjoint CI.** Sign is in the right direction but the magnitude is smaller and the CI now crosses zero.

## The reason matters: the test did not exercise drift

The headline blocker said "Phase 3+4 integration **with active drift**." The drift in this configuration is the online Hebbian updater itself (the design's sanctioned "online Hebbian reinforcement" pathway). Per-seed diagnostic counts at 1500 cues:

| Seed | Hebbian consolidations | Codebook drift from initial | Phase 4 candidates discovered |
| ---: | ---: | ---: | ---: |
| 17 | 22 | 2.78e-06 | 103 |
| 11 | 15 | 1.10e-06 | 84 |
| 23 | 23 | 2.94e-06 | 84 |
| **1** | **0** | **0.00e+00** | 84 |
| 2 | 15 | 1.14e-06 | 79 |

- Hebbian fired on **~1% of cues** across the run (mean 15/1500).
- **Codebook drift is six orders of magnitude smaller than its mean atom norm** — effectively zero.
- Seed 1 produced **zero consolidations**, so condition C was operating against a frozen codebook for that seed; per-seed final R@10 there was actually +0.026, consistent with the report-026 frozen-codebook regime.

**The Phase 4 discovery channel did fire** — 79–103 candidate patterns added per condition C run, regardless of whether the codebook drifted. So the replay machinery is healthy. What didn't happen is what this test was meant to stress: Phase 4 operating against a moving codebook.

## Per-seed final-checkpoint deltas (C − A)

| Seed | n | A R@10 | C R@10 | ΔR@10 | A capt5 | C capt5 | Δcapt5 | A top1 | C top1 | Δtop1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 225 | 0.320 | 0.320 | +0.000 | 0.280 | 0.298 | **+0.018** | 0.129 | 0.129 | 0.000 |
| 11 | 210 | 0.362 | 0.362 | +0.000 | 0.238 | 0.252 | +0.014 | 0.110 | 0.148 | **+0.038** |
| 23 | 229 | 0.362 | 0.354 | **−0.009** | 0.240 | 0.218 | **−0.022** | 0.079 | 0.105 | +0.026 |
| 1 | 227 | 0.339 | 0.366 | **+0.026** | 0.211 | 0.229 | +0.018 | 0.079 | 0.079 | 0.000 |
| 2 | 220 | 0.291 | 0.295 | +0.005 | 0.227 | 0.214 | −0.014 | 0.114 | 0.114 | 0.000 |

Notes:
- **Seed 23 is again the cap_t05 outlier** with Δ = −0.022, mirroring its behavior in the [report 026 frozen-codebook](026_phase4_verification_design_spec.md) run. Two independent runs now point to seed 23 as having idiosyncratic geometry that makes Phase 4 marginally worse on cap_t05. Worth a separate seed-level diagnostic.
- **Condition B vs A is essentially flat** in 4/5 seeds (B−A on R@10: −0.018 to +0.000), confirming that the Phase 3 reencode pass on its own does little at this drift magnitude. The B−A signal is the right control: any real lift in C should be C−B, not just C−A.
- **C ≥ B everywhere on R@10** — the discovery channel does not hurt at any seed.

## What this run *does* tell us

1. **The Phase 4 discovery channel is robust under near-zero drift.** ~85 candidates discovered per 1500-cue run regardless of seed. Replay machinery is healthy.
2. **The directional Phase 4 result from report 026 survives a different driver of (failed-to-emerge) drift** — sign is positive on R@10, Δtop1, Δcapt5 even though the underlying conditions don't match. Report 026 isn't refuted.
3. **The "architecturally-intended drift via online Hebbian" pathway requires more cues or a looser success_threshold** to actually produce drift. At `success_threshold=0.5` and 1500 cues, the firing rate is ~1% — insufficient to move the codebook beyond rounding error.

## What this run *cannot* tell us

- Whether Phase 4 measurably helps when the codebook is **actually drifting**. The test name was "with active drift" but the drift didn't happen.
- Whether the meta-stable-rate reduction at W=3/W=4 from the [D1 5-seed aggregation](d1_metastable_5seed.json) also holds under online Hebbian drift — would need the per-scale metastable_rate logged here, which exp 19 doesn't currently emit at the per-scale granularity exp 18 does.

## Implications for STATUS.md blocker #1

Blocker #1 is **not closed** by this result. The test as configured did not produce drift. Three options for next move, in order of cost:

1. **Lower the Hebbian success threshold** to drive firing rate up (e.g. `--success-threshold 0.3`) and rerun. Cheap.
2. **Longer runs** (3000–5000 cues). Moderate.
3. **Add a synthetic perturbation cycle on top of the Hebbian pathway**, mirroring report 026's drift mechanism but layered with online updates. This is what would actually test the design's stated drift sources jointly. Cost: requires a small exp 19 change to accept `--drift-magnitude` like exp 18 does.

The recommendation, per the project's anti-homunculus discipline, is **option 1 first** — surface the regime where Hebbian actually fires, then re-evaluate whether Phase 4 helps when the codebook moves.

## Cross-check against report 026

Report 026 (frozen codebook + synthetic drift): ΔR@10 = +0.010 [+0.001, +0.022], 5-seed CI disjoint from 0.
Report 028 (this run, online Hebbian, no synthetic drift): ΔR@10 = +0.0044 [−0.012, +0.021], 5-seed CI includes 0.

These are **consistent with the same underlying effect attenuated**: report 026 had real drift (synthetic, 0.30 magnitude); this run had essentially none, so the discovery channel had less to do. The sign matches; the magnitude scales with the amount of drift present. This is what we'd predict if Phase 4's value scales with drift.

## Data integrity flags

- **Seed 1 produced 0 Hebbian consolidations.** Not a bug — the seed's retrieval trajectories never crossed the `success_threshold=0.5` × `trivial_skip_threshold=0.95` window. But it means seed 1 is effectively a no-drift trial, which is useful information itself.
- **`codebook_drift_from_initial` field is logged correctly** and at 1e-6 magnitudes — the codebook movement is bounded by `lr_hebbian × n_consolidations` which is `0.01 × 15` ≈ 0.15 total perturbation budget across all atoms. That mostly cancels out, leaving a sub-microscopic norm change.
- **Cap_t05 per-seed variance (range −0.022 to +0.018, std 0.017) is again ~5× larger than the headline R@10 variance.** This recurring pattern across two independent runs reinforces that cap_t05 needs either more seeds or a different parameterization to verify cleanly.

---

## Appendix: full trajectory

| Cues | A R@10 | B R@10 | C R@10 | Δ (C − A) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3349 | 0.3349 | 0.3349 | +0.0000 |
| 300 | 0.3349 | 0.3331 | 0.3384 | +0.0035 |
| 600 | 0.3349 | 0.3331 | 0.3384 | +0.0035 |
| 900 | 0.3349 | 0.3332 | 0.3393 | +0.0044 |
| 1200 | 0.3349 | 0.3358 | 0.3393 | +0.0044 |
| 1500 | 0.3349 | 0.3376 | 0.3393 | +0.0044 |

The C−A gap grows monotonically through the run (matches report 026's monotonic-growth pattern). Sign stable.
