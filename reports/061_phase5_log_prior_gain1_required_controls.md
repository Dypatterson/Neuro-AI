# Report 061 - Phase 5 Log-Prior Gain-1 Required Controls

**Date:** 2026-05-24
**Active phase:** 5
**Status:** v2 runner implemented locally; full Colab validation pending.
**Decision:** Do not claim Phase 5 graduation. Path C remains useful but insufficient unless the corrected v2 controls pass.
**Headline metric:** `DeltaE = E_content-prior - E_role-prior`, paired per cue. Positive means the role-prior branch lands at lower final-state energy than the content-prior branch.
**Precursor:** [Report 060](060_phase5_log_prior_n10_colab_confirmation.md) confirmed the gain-1 log-prior spike at n=10 but deferred the no-prior and no-schema-store controls.

This report has been revised after code review. The first control pass used `gamma=0` while leaving `log_prior_gain=1`; that is a useful gamma-gating diagnostic, but it is not a true no-prior control because the additive log-prior spike remains active. The corrected v2 runner separates true no-prior from the gamma-only diagnostic and adds per-seed run-identity metadata so stale Drive outputs cannot be silently reused.

## Setup

- Runner: `scripts/colab_phase5_log_prior_gain1_controls_runner.py`
- Runner version: `v2_true_no_prior_identity_guard_20260524`
- New run tag: `phase5_log_prior_required_controls_v2_20260524`
- Local harness adaptation: `scripts/phase5_frozen_snapshot_audit.py` accepts `--log-prior-schema-source {slow_store,full_codebook}` for local/CLI reruns.
- Seeds: `17, 11, 23, 1, 2, 3, 5, 7, 13, 29`
- Fixed operating point unless a control explicitly changes it: `beta=10`, `K=1`, `gamma=0.5`, `log_prior_gain=1`, `content_distortion=0.6`, `binding_noise_std=0.05`, `cue_seed=117`, `n_cues=100`, `formulation=per_pattern`
- True no-prior control: `gamma=0.0`, `log_prior_gain=0.0`, `schema_source=slow_store`
- Gamma-only diagnostic: `gamma=0.0`, `log_prior_gain=1.0`, `schema_source=slow_store`
- No-schema-store control: `gamma=0.5`, `log_prior_gain=1.0`, `schema_source=full_codebook`
- Magnitude floor: `5.5e-3`

## Baseline For Comparison

Report 060 gain 1:

| mean DeltaE | 95% CI | seeds positive | DeltaE/floor | random_lowest | role<content | role<content<random | hit_role | rank_role |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +0.008883 | [+0.00645, +0.01164] | 10/10 | 1.615 | 0.372 | 0.486 | 0.083 | 0.003 | 443.3 |

## Corrected V2 Control Matrix

Full n=10 Colab metrics are pending because the local environment blocked transmitting the updated runner code into Colab. The implemented v2 runner will emit this table after validation:

| control | gamma | log_prior_gain | schema_source | expected result | current status |
|---|---:|---:|---|---|---|
| true_no_prior | 0.0 | 0.0 | slow_store | effect removed | pending v2 Colab run |
| gamma0_log_prior_ablation | 0.0 | 1.0 | slow_store | tests whether the additive spike bypasses gamma | v1 diagnostic already observed |
| gain1_no_schema_store | 0.5 | 1.0 | full_codebook | effect removed or shrunk if slow-store source is load-bearing | v1 diagnostic already observed |

## Superseded V1 Diagnostic Results

These numbers came from the first Colab run. They are retained because they are informative, but the `gamma=0, gain=1` row must not be read as the required no-prior control.

| diagnostic | mean DeltaE | 95% CI | seeds positive | DeltaE/floor | vs Report 060 gain 1 | random_lowest | role<content | role<content<random | hit_role | rank_role | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gamma0_log_prior_ablation | +0.007305 | [+0.00415, +0.01093] | 10/10 | 1.328 | -0.001578, 0.82x baseline | 0.366 | 0.484 | 0.066 | 0.003 | 470.6 | true |
| gain1_no_schema_store full-codebook | +0.091513 | [+0.08603, +0.09711] | 10/10 | 16.639 | +0.082630, 10.30x baseline | 0.464 | 0.934 | 0.018 | 0.009 | 440.5 | true |

## Superseded Per-Seed DeltaE

| seed | gamma0 log-prior diagnostic | no-schema-store |
|---:|---:|---:|
| 17 | +0.00692 | +0.09891 |
| 11 | +0.01081 | +0.08445 |
| 23 | +0.00065 | +0.10058 |
| 1 | +0.01044 | +0.07781 |
| 2 | +0.00676 | +0.10710 |
| 3 | +0.00290 | +0.08446 |
| 5 | +0.01592 | +0.09909 |
| 7 | +0.00094 | +0.08808 |
| 13 | +0.01571 | +0.08250 |
| 29 | +0.00200 | +0.09215 |

## Reading

**True no-prior remains the missing required v2 result.** The corrected runner now disables both prior channels (`gamma=0`, `log_prior_gain=0`) for `true_no_prior`. If that row comes back near zero as expected, it shows the additive prior mechanism is doing the energy-margin work.

**The gamma-only diagnostic shows the spike bypasses gamma.** The superseded v1 `gamma=0, gain=1` row preserved most of the Report 060 effect (`+0.007305`, 10/10 seeds positive), which means `gamma=0` alone is not a true ablation in the log-prior-spike implementation.

**The no-schema-store diagnostic remains the main Path C problem.** The v1 full-codebook row amplified the effect to mean DeltaE `+0.091513`, about `10.30x` the Report 060 gain-1 baseline. If this repeats under v2 identity-guarded output, Path C is not bounded to the slow-store schema source.

**The random-prior and basin caveats remain unresolved.** The v1 diagnostic rows had weak ordered readouts (`role<content<random` of `0.066` and `0.018`) and role-target basin hit near zero (`0.003` and `0.009`). V2 must continue reporting these alongside energy.

## Recommendation

Do not claim Phase 5 graduation.

Treat Path C as not yet viable under the current graduation criteria until the corrected v2 Colab run exists. If v2 true no-prior removes the effect but no-schema-store still preserves or amplifies it, close Path C as useful but insufficient and choose Path B' closure/pivot. A Path A scale-spanning closure probe is only worth doing if the immediate goal is mechanistic explanation of the energy-margin artifact, not another unbounded graduation route.

## Expected V2 Artifacts

- Aggregate JSON: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_required_controls_v2_20260524/cross_seed_aggregate.json`
- Aggregate Markdown: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_required_controls_v2_20260524/cross_seed_aggregate.md`
- Per-seed/log directory: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_required_controls_v2_20260524`
