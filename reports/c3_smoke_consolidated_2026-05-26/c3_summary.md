# C.3 — Phase 3 Exit Criterion Re-run

**DIAGNOSTIC, NOT GRADUATION EVIDENCE.** This run used n_seeds=3 (< 10). The Phase 3 exit criterion requires n ≥ 10 seeds; the smoke output below is diagnostic-only and CIs are wide.

## Run configuration

- D = `4096` (Phase 2 baseline envelope)
- landscape_size = `64`
- window_size = `8`
- vocab_size = `200`
- n_train_windows = `1000`
- n_test_windows = `200`
- β = `10.0`
- K = `5`
- seeds = `[0, 1, 2]`
- theta_prime_modes = `['default', 'calibrated']`
- standard_mode = `consolidated`
- n_consolidation_events = `1000`
- C.2 dynamics config = `{'lambda_ac': 0.5, 'mu_T': 0.1, 'tau_T': 0.5, 'lambda_cc': 0.5, 'theta_cc': 0.5, 'tau_cc': 0.1, 'metastability_obs_rate': 0.1, 'metastability_gain': 2.0, 'metastability_replay_decay': 0.5, 'drift_ema_rate': 0.1, 'drift_replay_gain': 1.0}`
- device = `cpu`
- wall_clock = `1137.6s`

## Headline table — per-mode, per-stratum

| Mode | Stratum | Standard Recall@K [Wilson CI] | Shuffled-control Recall@K [Wilson CI] | Δ (std − ctrl) | CI-disjoint (std lower > ctrl upper)? | n_std | n_ctrl |
|---|---|---|---|---:|:--:|---:|---:|
| default | tight | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |
| default | spread | 0.023 [0.014, 0.039] | 0.032 [0.020, 0.049] | -0.008 | no | 600 | 600 |
| default | borderline | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |
| calibrated | tight | 0.022 [0.013, 0.039] | 0.000 [0.000, 0.000] | +0.022 | YES | 537 | 0 |
| calibrated | spread | 0.032 [0.009, 0.109] | 0.032 [0.020, 0.049] | +0.000 | no | 63 | 600 |
| calibrated | borderline | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |

## Regime classifier agreement diagnostic

Per the C.1.4 calibration finding (`1/β` off by 2-3 orders at low β), the regime classifier *can* disagree between the `default` (1/β) and `calibrated` modes. At β=10 the calibration JSON does not cover the operating point (its grid is β ∈ {0.01, 0.1, 1.0}), so the calibrated loader falls back to 1/β with a stderr warning; the two modes therefore agree numerically at this β.

## Regime distribution BEFORE vs AFTER consolidation (STD rows only)

| Seed | Mode | Before (t/s/b) | After (t/s/b) | Δtight | Δspread | Δborderline | cons fired |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | default | 0/200/0 | 0/200/0 | +0 | +0 | +0 | 10 |
| 1 | default | 0/200/0 | 0/200/0 | +0 | +0 | +0 | 10 |
| 2 | default | 0/200/0 | 0/200/0 | +0 | +0 | +0 | 10 |
| 0 | calibrated | 0/200/0 | 193/7/0 | +193 | -193 | +0 | 10 |
| 1 | calibrated | 0/200/0 | 174/26/0 | +174 | -174 | +0 | 10 |
| 2 | calibrated | 0/200/0 | 175/25/0 | +175 | -175 | +0 | 10 |

## Per-cell rows (per-seed)

| Seed | Substrate seed | Control? | Mode | tight/spread/borderline counts | tight Recall@K | spread Recall@K | borderline Recall@K |
|---:|---:|:--:|---|---|---:|---:|---:|
| 0 | 0 | STD | default | 0/200/0 | 0.000 (0) | 0.040 (200) | 0.000 (0) |
| 0 | 10000 | CTRL | default | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 1 | STD | default | 0/200/0 | 0.000 (0) | 0.015 (200) | 0.000 (0) |
| 1 | 10001 | CTRL | default | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 2 | 2 | STD | default | 0/200/0 | 0.000 (0) | 0.015 (200) | 0.000 (0) |
| 2 | 10002 | CTRL | default | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 0 | 0 | STD | calibrated | 193/7/0 | 0.036 (194) | 0.167 (6) | 0.000 (0) |
| 0 | 10000 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 1 | STD | calibrated | 174/26/0 | 0.018 (166) | 0.000 (34) | 0.000 (0) |
| 1 | 10001 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 2 | 2 | STD | calibrated | 175/25/0 | 0.011 (177) | 0.043 (23) | 0.000 (0) |
| 2 | 10002 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
