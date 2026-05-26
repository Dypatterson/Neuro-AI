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
- device = `cpu`
- wall_clock = `9.2s`

## Headline table — per-mode, per-stratum

| Mode | Stratum | Standard Recall@K [Wilson CI] | Shuffled-control Recall@K [Wilson CI] | Δ (std − ctrl) | CI-disjoint (std lower > ctrl upper)? | n_std | n_ctrl |
|---|---|---|---|---:|:--:|---:|---:|
| default | tight | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |
| default | spread | 0.030 [0.019, 0.047] | 0.032 [0.020, 0.049] | -0.002 | no | 600 | 600 |
| default | borderline | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |
| calibrated | tight | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |
| calibrated | spread | 0.030 [0.019, 0.047] | 0.032 [0.020, 0.049] | -0.002 | no | 600 | 600 |
| calibrated | borderline | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 | no | 0 | 0 |

## Regime classifier agreement diagnostic

Per the C.1.4 calibration finding (`1/β` off by 2-3 orders at low β), the regime classifier *can* disagree between the `default` (1/β) and `calibrated` modes. At β=10 the calibration JSON does not cover the operating point (its grid is β ∈ {0.01, 0.1, 1.0}), so the calibrated loader falls back to 1/β with a stderr warning; the two modes therefore agree numerically at this β.

## Per-cell rows (per-seed)

| Seed | Substrate seed | Control? | Mode | tight/spread/borderline counts | tight Recall@K | spread Recall@K | borderline Recall@K |
|---:|---:|:--:|---|---|---:|---:|---:|
| 0 | 0 | STD | default | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 0 | 10000 | CTRL | default | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 1 | STD | default | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 10001 | CTRL | default | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 2 | 2 | STD | default | 0/200/0 | 0.000 (0) | 0.040 (200) | 0.000 (0) |
| 2 | 10002 | CTRL | default | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 0 | 0 | STD | calibrated | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 0 | 10000 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 1 | STD | calibrated | 0/200/0 | 0.000 (0) | 0.025 (200) | 0.000 (0) |
| 1 | 10001 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
| 2 | 2 | STD | calibrated | 0/200/0 | 0.000 (0) | 0.040 (200) | 0.000 (0) |
| 2 | 10002 | CTRL | calibrated | 0/200/0 | 0.000 (0) | 0.035 (200) | 0.000 (0) |
