# Phase 5 M1 Provenance Audit

Status: **FAIL**

- Snapshot: `reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`
- Rows: 1064
- Roles: 4
- Weight source: geometric
- Row alignment: True
- Role coverage: 4/4
- Representative Phase 5 evidence: False
- Evidence scope: Seed 17 is wiring/provenance/degen smoke only; it is not representative Phase 5 evidence.
- Geometric mode: same_role_filler_density
- Mean normalized row entropy: 1.0000000042014552
- Uniform row fraction: 1.0

## Failure Reasons
- geometric_mean_role_entropy_degenerate
- geometric_uniform_role_rows_degenerate

## Warnings
- mask_token_participates_in_row_provenance
- count_role_weights_degenerate

## Role Counts
{
  "role_counts": [
    266.0,
    266.00006103515625,
    265.99981689453125,
    266.0001525878906
  ],
  "role_fractions": [
    0.25,
    0.25000005736386866,
    0.24999982790839403,
    0.25000014340967164
  ]
}
