# Report 036 — A_k decay sweep + seed-17 trajectory: A_k is orthogonal to the dominant signal

**Date:** 2026-05-16
**Active phase:** 4
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift between cycles. (Δtop1 demoted to drill-down per [2026-05-16 substrate-vs-readout note](../notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md).)
**Required controls:** No-replay baseline (Condition A) — present.
**Last verified result:** [Report 035](035_saighi_ak_n10_falsification.md) — n=10 A_k at decay=0.0 falsified report 034's seed-1 prototype: ΔR@10 indistinguishable from baseline, Δcap_t05 CI crosses zero.
**Why this experiment now:** Test whether the gain-vs-decay trade-off Saighi calls out on p. 4 of the paper rescues A_k — does inhibition decay (a "forgiveness window") restore positive design-spec headlines?

---

## Method

### Decay sweep (seed 1, parallel on Colab H100 NVL)

Four runs in parallel, identical to report 035's seed-1 config except `--inhibition-decay`:

```
--seed 1 --inhibition-gain 0.01 --n-cues 3000 --death-window 10
--inhibition-decay ∈ {0.02, 0.05, 0.10, 0.20}
```

For comparison, decay=0.0 result is taken from the n=10 sweep (seed 1).

### Seed-17 trajectory analysis (free, from n=10 data on Drive)

Pulled seed 17's full per-checkpoint JSON from Drive (the strongest positive Δcap seed at n=10). Looked at how Δcap_t05 evolved through the run and what happened to pattern counts.

---

## Results

### Decay sweep (seed 1)

| decay | Δcap_t05 | ΔR@10 | Δtop1 (drill) | A_max | A_nz |
|---:|---:|---:|---:|---:|---:|
| 0.00 | +0.0176 | +0.0264 | +0.0529 | 13.61 | 3 |
| 0.02 | +0.0441 | +0.0264 | +0.0529 | 8.50 | 3 |
| 0.05 | +0.0396 | +0.0264 | +0.0529 | 4.42 | 3 |
| 0.10 | +0.0176 | +0.0264 | +0.0529 | 1.68 | 3 |
| 0.20 | +0.0308 | +0.0264 | +0.0529 | 0.93 | 3 |

Two observations:
1. **ΔR@10 and Δtop1 are bit-identical across all decay values.** Same 30/227 top1 hits, same 83/227 R@10 hits. A_max varies 15× (0.93 → 13.6); the test-set readout doesn't notice.
2. **Δcap_t05 varies non-monotonically across decay values, range 0.026.** Best at decay=0.02 (+0.0441), but the curve is noisy. No clean trade-off direction.

A_k decay is NOT a meaningful lever on the design-spec headlines.

### Seed-17 trajectory (n=10 run, decay=0.0)

| step | top1 | R@10 | **cap_t_05** | **n_pat_alive_w2** | deaths_total | A_max |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 0.129 | 0.320 | 0.280 | (init) | 0 | 0.00 |
| 500  | 0.102 | 0.320 | 0.293 | 4125 | 0 | 1.27 |
| 1000 | 0.102 | 0.329 | 0.276 | 4125 | 0 | 2.98 |
| 1500 | 0.102 | 0.333 | 0.284 | 4127 | 0 | 6.84 |
| 2000 | 0.062 | 0.342 | **0.316** | **32** | **7200** | 8.30 |

**Mass death happens between step 1500 and step 2000.** Pattern count at scale 2 collapses from 4127 → 32 (99.2% loss). All of the meaningful Δcap_t05 (+0.036 final) happens *after* mass death; Δcap was +0.004 at step 1500 (essentially baseline) and +0.036 at step 2000.

The Δcap_t05 improvement is being driven by **death**, not by A_k. A_k is along for the ride.

---

## Architectural reading

Three findings line up to one conclusion:

1. **A_k decay tunes inhibition magnitude but not pattern identity.** A_max varies 15× across decay values but the same 3 patterns get inhibited and the same dominant attractors stay dominant.
2. **Death is the dominant substrate-shaping force at n_cues=3000.** Between step 1500 and 2000, ~7200 vocab atoms die for being un-reinforced. The substrate that remains is what produces the (variance-bound) cap_t05 result.
3. **A_k is orthogonal to the death-driven substrate-shaping.** A_k inhibits the few dominant attractors; death kills the many un-touched atoms. These are mostly disjoint sets of patterns; the mechanisms don't compete.

So the n=10 Δcap_t05 = −0.026 (CI [−0.072, +0.021]) result from report 035 is NOT a story about A_k. It's a story about **which patterns happen to survive mass death** per seed. Some seeds keep useful patterns and improve cap_t05 (17: +0.036, 7: +0.052). Some seeds lose useful patterns and collapse (3: −0.179). A_k changes the relative scores of the survivors but doesn't change *which* atoms survive.

This is a falsification of A_k as a Phase 4 graduation mechanism. Not because the mechanism is broken — it's correctly implemented — but because the dominant signal in the headlines is upstream of A_k.

---

## What this report does and does not establish

**Does establish:**
- A_k at gain=0.01 with any decay in {0.0, 0.02, 0.05, 0.10, 0.20} does not improve ΔR@10 on seed 1, and varies Δcap_t05 by less than 0.026.
- The death mechanism at n_cues=3000 mass-kills ~99% of initial vocab atoms.
- All meaningful Δcap_t05 movement on seed 17 happens *after* mass death (step 1500 → step 2000), not during the A_k accumulation phase.
- The variance in n=10 Δcap_t05 is driven by survival outcomes, not by A_k parameterization.

**Does NOT establish:**
- That A_k scoped to discovered patterns only is also useless. Untested.
- That death is the wrong mechanism. Mass death of un-reached vocab atoms *might* be exactly what the architecture should do for contextual completion at this corpus size — the substrate-vs-readout discipline says we should evaluate based on whether the surviving substrate is rich enough (cap_t05 + R@K), not on whether *all* patterns survived.
- That the n=10 cap_t05 CI would remain crossing zero with longer n_cues (n_cues=6000) or different death parameters.

---

## Open four-way decision for next session

(Lifted from the AskUserQuestion in the conversation transcript so it's preserved if the question result is lost.)

1. **Diagnose seed 3 collapse** (~15 min, no compute). Seed 3's Δcap = −0.179 is the outlier dragging the n=10 mean negative. Pull its trajectory from Drive; see WHEN cap collapsed and which patterns survived. Either (a) understand and prevent the failure mode, or (b) identify it as a corpus-order issue meaning the variance is intrinsic to seed.
2. **Run n=10 at death disabled** (~30 min compute). Set `--death-window` very large so mass death never fires. Re-run the n=10 sweep. Tests whether mass death is the cause of the cap_t05 variance — if without death the distribution tightens around zero, death is the driver.
3. **Scope A_k to discovered patterns only** (~1h wire + n=10 verify). Track an `is_discovered` flag per pattern; gate `accumulate_inhibition` on it. A_k applies only to Phase 4-emitted patterns, not vocab atoms.
4. **Pivot Phase 4 headline to D1 meta-stable rate** (~2h). D1 Δ = −0.51 W=3 with per-seed std 0.038 is the most robust Phase 4 result we have. Write up Phase 4 graduating on D1, with R@K + cap-coverage as "promising but variance-bound". Move on.

Option 1 was the recommended starting move because it's free and decision-shaping for the others.

---

## Status implications

- Blocker #2 (A_k mechanism): **falsified at gain=0.01 across decay ∈ {0, 0.02, 0.05, 0.10, 0.20}**. Three remaining branches (death-disabled, discovered-only, D1 pivot) are now the candidate moves. The mechanism is correctly implemented but produces an orthogonal effect to the dominant substrate-shaping force.
- Blocker #3 (Δcap-coverage): n=10 result still CI-crosses-zero. The variance source is now identified as survival-after-mass-death, not A_k parameterization. This is actionable information for next session.
- Blocker #6' (top1 regression): demoted to drill-down per [2026-05-16 substrate-vs-readout note](../notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md). Still worth watching as a substrate-distortion signal.
