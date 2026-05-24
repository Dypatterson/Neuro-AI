# Report 060 - Phase 5 Log-Prior n=10 Colab Confirmation

**Date:** 2026-05-24
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:269-292](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior - E_role-prior, paired per cue. Positive means the role-prior branch lands at lower final-state energy than the content-prior branch.
**Required controls per [phase-5-unified-design.md:296-304](../notes/emergent-codebook/phase-5-unified-design.md):** random-prior, K=1, no-prior, no-schema-store. This confirmation preserves the random-prior and K=1 readouts but does not run the no-prior or no-schema-store controls, so it is not a Phase 5 graduation result.
**Precursor:** [Report 059](059_phase5_log_prior_spike_local_smoke.md) found gain 1 as the cleanest local-smoke candidate on seeds 17/11/23, with gains 2/4 moving energy more but worsening random-prior behavior on some seeds.

This is an n=10 diagnostic confirmation of the C-first log-prior spike. It confirms that the per-pattern log-prior boost can move the locked energy-margin headline above the pre-committed magnitude floor on the existing A+B+A1' substrate, but it does not yet establish structural retrieval or graduate Phase 5.

## Setup

- Runner: `scripts/colab_phase5_log_prior_n10_runner.py`
- Commit run in Colab: `e3f746c`
- Runtime: Colab T4, Google Drive-mounted output
- Seeds: `17, 11, 23, 1, 2, 3, 5, 7, 13, 29`
- Fixed operating point: `beta=10`, `gamma=0.5`, `K=1`, `formulation=per_pattern`, `binding_noise_std=0.05`, `content_distortion=0.6`
- Gain grid: `log_prior_gain ∈ {0, 1, 2, 4}`
- Cues: fixed harness cue set per seed
- Created UTC: `2026-05-24T02:49:43.166540+00:00`

## Aggregate Results

Magnitude floor: `5.5e-3`.

| gain | mean ΔE | 95% CI | seeds positive | ΔE/floor | random_lowest | Δ random_lowest vs gain 0 | role<content | role<content<random | hit_role | rank_role | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.002398 | [+0.00142, +0.00349] | 9/10 | 0.436 | 0.339 | +0.000 | 0.449 | 0.068 | 0.003 | 462.2 | false |
| 1 | +0.008883 | [+0.00645, +0.01164] | 10/10 | 1.615 | 0.372 | +0.033 | 0.486 | 0.083 | 0.003 | 443.3 | true |
| 2 | +0.015097 | [+0.01072, +0.01930] | 10/10 | 2.745 | 0.376 | +0.037 | 0.518 | 0.086 | 0.004 | 417.8 | true |
| 4 | +0.012791 | [+0.00598, +0.02010] | 9/10 | 2.326 | 0.370 | +0.031 | 0.417 | 0.097 | 0.004 | 331.7 | true |

## Per-Seed ΔE

| seed | gain 0 | gain 1 | gain 2 | gain 4 |
|---:|---:|---:|---:|---:|
| 17 | +0.00319 | +0.00847 | +0.02371 | +0.01576 |
| 11 | +0.00176 | +0.01126 | +0.01695 | +0.00907 |
| 23 | +0.00290 | +0.00618 | +0.02433 | +0.03326 |
| 1 | +0.00112 | +0.00771 | +0.00851 | -0.00164 |
| 2 | +0.00365 | +0.00805 | +0.00573 | +0.00093 |
| 3 | +0.00215 | +0.00561 | +0.01727 | +0.01472 |
| 5 | +0.00231 | +0.01685 | +0.01801 | +0.00456 |
| 7 | -0.00027 | +0.00542 | +0.00504 | +0.00045 |
| 13 | +0.00624 | +0.01584 | +0.02147 | +0.02409 |
| 29 | +0.00093 | +0.00344 | +0.00995 | +0.02670 |

## Reading

**Gain 1 confirms the local-smoke direction.** It clears the magnitude floor at n=10, has CI fully above zero and above the floor, and is positive on all 10 seeds. This is the cleanest candidate if Path C continues because it moves ΔE without the larger gain cells' more obvious seed-level instability.

**The random-prior caveat remains.** Gain 1 raises `random_lowest` from 0.339 to 0.372. Gains 2 and 4 move the energy margin more, but their random-prior deltas are not better than gain 1 and their per-seed behavior is less stable. This means the spike is not yet a clean structural-retrieval result; it is an energy-margin intervention that also makes random-prior competition more prominent.

**Basin retrieval still does not recover.** Role-target hit rate remains near zero (`0.003-0.004`). Gain 4 improves mean role-target rank more than gain 1, but rank 331.7 is still not a credible basin-retrieval claim. The effect is still primarily visible in energy margin, not in role-target basin membership.

**This is not Phase 5 graduation.** The run did not include the no-prior or no-schema-store controls, and the random-prior readout is not clean enough to treat the positive ΔE as established structural retrieval.

## Decision Read

Path C is confirmed as capable of moving the locked Phase 5 energy-margin headline on the existing substrate. Gain 1 is the preferred follow-up cell because it is positive on 10/10 seeds and clears the magnitude floor with the least aggressive intervention.

Path C has not solved the core Phase 5 problem yet. Before any graduation claim, the next C-continuation step would need to run the missing required controls at minimum:

- gain 1 no-prior (`gamma=0`) at the same locked cue regime
- gain 1 no-schema-store control
- explicit random-prior gate language, because `random_lowest` rises at n=10

If those controls fail, this should be written up as a useful but insufficient prior-bias rescue: it moves ΔE, but does not recover structural basin retrieval.

## Artifacts

- Aggregate JSON: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_n10_gain_sweep_20260524/cross_seed_aggregate.json`
- Aggregate Markdown: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_n10_gain_sweep_20260524/cross_seed_aggregate.md`
- Per-seed/log directory: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_n10_gain_sweep_20260524`
