# Report 061 - Phase 5 Log-Prior Gain-1 Required Controls

**Date:** 2026-05-24
**Active phase:** 5
**Status:** v2 Colab validation complete.
**Decision:** Do not claim Phase 5 graduation. The 2x2 (gamma, log_prior_gain) ablation shows the log-prior channel is the dominant lever on the slow-store substrate; the full-codebook no-schema-store row is an arbitration-shape positive control per the anti-homunculus filter, not portable evidence; basin/random-prior caveats remain unresolved.
**Headline metric:** `DeltaE = E_content-prior - E_role-prior`, paired per cue. Positive means the role-prior branch lands at lower final-state energy than the content-prior branch.
**Precursor:** [Report 060](060_phase5_log_prior_n10_colab_confirmation.md) confirmed the gain-1 log-prior spike at n=10 but deferred the no-prior and no-schema-store controls.

This report has been revised after code review and validated in Colab with the v2 identity-guarded runner. The first control pass used `gamma=0` while leaving `log_prior_gain=1`; that is a useful gamma-gating diagnostic, but it is not a true no-prior control because the additive log-prior spike remains active. The corrected v2 runner separates true no-prior from the gamma-only diagnostic and adds per-seed run-identity metadata so stale Drive outputs cannot be silently reused.

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
- Colab aggregate: [cross_seed_aggregate.md](https://drive.google.com/file/d/17PRaIo8GsJAo1ENHI9ws_CTbOSODvRO4/view?usp=drivesdk)
- Drive folder: [phase5_log_prior_required_controls_v2_20260524](https://drive.google.com/drive/folders/1GMQw5gZ3JtNdEZbLjOEVtBG9bI6WPTV2)

## Baseline For Comparison

Report 060 gain 1:

| mean DeltaE | 95% CI | seeds positive | DeltaE/floor | random_lowest | role<content | role<content<random | hit_role | rank_role |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +0.008883 | [+0.00645, +0.01164] | 10/10 | 1.615 | 0.372 | 0.486 | 0.083 | 0.003 | 443.3 |

## Corrected V2 Control Matrix

Identity-guarded n=10 Colab metrics:

| control | gamma | log_prior_gain | schema_source | mean DeltaE | bootstrap 95% CI | seeds+ | DeltaE/floor | random_lowest | role<content | role<content<random | hit_role | rank_role | magnitude gate |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| true_no_prior | 0.0 | 0.0 | slow_store | +0.000000 | [+0.00000, +0.00000] | 0/10 | +0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 489.0 | no |
| gamma0_log_prior_ablation | 0.0 | 1.0 | slow_store | +0.007318 | [+0.00404, +0.01080] | 10/10 | +1.331 | 0.366 | 0.486 | 0.065 | 0.003 | 470.5 | yes |
| gain1_no_schema_store | 0.5 | 1.0 | full_codebook | +0.091510 | [+0.08603, +0.09725] | 10/10 | +16.638 | 0.465 | 0.934 | 0.018 | 0.009 | 440.5 | yes |

## Per-Seed DeltaE

| seed | true no-prior | gamma0 log-prior | no-schema-store |
|---:|---:|---:|---:|
| 17 | +0.00000 | +0.00692 | +0.09887 |
| 11 | +0.00000 | +0.01081 | +0.08444 |
| 23 | +0.00000 | +0.00051 | +0.10058 |
| 1 | +0.00000 | +0.01051 | +0.07782 |
| 2 | +0.00000 | +0.00676 | +0.10709 |
| 3 | +0.00000 | +0.00314 | +0.08446 |
| 5 | +0.00000 | +0.01592 | +0.09909 |
| 7 | +0.00000 | +0.00090 | +0.08807 |
| 13 | +0.00000 | +0.01572 | +0.08253 |
| 29 | +0.00000 | +0.00199 | +0.09216 |

## Per-Seed Random Lowest

| seed | true no-prior | gamma0 log-prior | no-schema-store |
|---:|---:|---:|---:|
| 17 | 0.00 | 0.43 | 0.55 |
| 11 | 0.00 | 0.33 | 0.49 |
| 23 | 0.00 | 0.35 | 0.51 |
| 1 | 0.00 | 0.45 | 0.37 |
| 2 | 0.00 | 0.36 | 0.60 |
| 3 | 0.00 | 0.32 | 0.42 |
| 5 | 0.00 | 0.52 | 0.45 |
| 7 | 0.00 | 0.27 | 0.42 |
| 13 | 0.00 | 0.35 | 0.38 |
| 29 | 0.00 | 0.28 | 0.46 |

## Reading

**True no-prior is a structural sanity check, not load-bearing evidence.** Disabling both prior channels (`gamma=0`, `log_prior_gain=0`) removes the energy-margin effect: mean DeltaE is exactly `+0.000000`, bootstrap interval `[+0.00000, +0.00000]`, `0/10` seeds positive. This is **by construction** rather than discovery: with both prior channels off, the role-, content-, and random-prior branches run identical settling dynamics (no per-condition logit or settling-time difference), so per-cue ΔE = 0 to FP precision and the bootstrap CI collapses to a point. The row confirms the experimental framework has no per-condition leak, but does not by itself prove the prior machinery is "load-bearing" — that requires the 2x2 ablation below.

**Two-by-two (gamma, log_prior_gain) channel ablation.** Combining Report 060's gain=0 baseline (`gamma=0.5, gain=0`) with the three v2 rows yields a clean 2x2 over the two prior channels at the locked operating point:

| | log_prior_gain = 0 | log_prior_gain = 1 |
|---|---:|---:|
| **gamma = 0.0** | `+0.000000` (this report, true no-prior) | `+0.007318` (this report, gamma-only diagnostic) |
| **gamma = 0.5** | `+0.002398` ([Report 060](060_phase5_log_prior_n10_colab_confirmation.md) gain 0 baseline) | `+0.008883` ([Report 060](060_phase5_log_prior_n10_colab_confirmation.md) gain 1) |

The log-prior channel carries most of the effect: turning it on at `gamma=0.5` moves ΔE by `+0.006485` (`+0.008883 - +0.002398`); turning it on at `gamma=0` moves ΔE by `+0.007318`. The gamma channel alone (`gamma=0.5, gain=0`) is sub-floor at `0.436x` floor; the log-prior channel alone (`gamma=0, gain=1`) is above floor at `1.331x` floor. The two channels are approximately additive at this operating point. This is the load-bearing test the v1 control was trying to do, and it shows the log-prior spike is the dominant lever.

**The gamma-only diagnostic shows the spike bypasses gamma.** The v2 `gamma=0, gain=1` row preserves most of the Report 060 gain-1 effect: mean DeltaE `+0.007318`, bootstrap CI `[+0.00404, +0.01080]`, and `10/10` positive seeds. This confirms `gamma=0` alone is not a true ablation in the log-prior-spike implementation, and is consistent with the 2x2 reading above.

**The no-schema-store diagnostic is an arbitration-shape positive control, not a candidate mechanism.** The v2 `gamma=0.5, gain=1, schema_source=full_codebook` row amplifies the effect to mean DeltaE `+0.091510`, about `10.30x` the Report 060 gain-1 baseline. Anti-homunculus reading: with the slow-store filtered to the full codebook, the upstream selector collapses into roughly `argmax(content_sim) -> boost-that-atom` with no consolidated-schema dynamic backing it. That is the classic metric-reads-then-acts shape the anti-homunculus filter rules out as a candidate mechanism. The `+0.0915` magnitude is therefore evidence that the spike CAN drive arbitrary energy margins when paired with a bare top-K selector — i.e. it is a positive control on the mechanism class, not evidence that Path C generalizes off the slow-store substrate. This row must not be cited as Path-C support.

**The random-prior and basin caveats remain unresolved.** The v2 diagnostic rows still have weak ordered readouts (`role<content<random` of `0.065` and `0.018`) and role-target basin hit near zero (`0.003` and `0.009`). The no-schema-store row raises `random_lowest` to `0.465` and role-target rank stays poor (`440.5`), so the energy-margin result must not be laundered into a retrieval or basin-success claim.

## Recommendation

Do not claim Phase 5 graduation.

Close Path C as a useful controlled energy-margin diagnostic but insufficient under the current Phase 5 graduation criteria. The 2x2 ablation shows the log-prior channel is the load-bearing lever on the slow-store substrate; the no-schema-store amplification is an arbitration-shape positive control and not portable evidence; and near-zero role-target basin hit/rank prevent any structural retrieval interpretation. Choose Path B' closure/pivot if the immediate goal is honest Phase 5 closure. A Path A scale-spanning closure probe is only worth doing if the goal is mechanistic explanation of the energy-margin artifact, not another graduation route.

## V2 Artifacts

- Drive folder: [phase5_log_prior_required_controls_v2_20260524](https://drive.google.com/drive/folders/1GMQw5gZ3JtNdEZbLjOEVtBG9bI6WPTV2)
- Aggregate Markdown: [cross_seed_aggregate.md](https://drive.google.com/file/d/17PRaIo8GsJAo1ENHI9ws_CTbOSODvRO4/view?usp=drivesdk)
- Aggregate JSON: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_required_controls_v2_20260524/cross_seed_aggregate.json`
- Per-seed/log directory: `/content/drive/MyDrive/neuro-ai/results/phase5_log_prior_required_controls_v2_20260524`

## Superseded V1 Diagnostic Note

The first Colab control pass is superseded by the v2 identity-guarded run above. Its `gamma=0, gain=1` row remains useful historical evidence that the additive spike bypasses gamma, but it must not be read as the required true no-prior control.
