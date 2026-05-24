# Phase 5 M1 Provenance Audit

Status: **FAIL**

- Snapshot: `reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`
- Rows: 1064
- Roles: 4
- Weight source: count
- Row alignment: True
- Role coverage: 4/4
- Mean normalized row entropy: 1.0
- Uniform row fraction: 1.0

## Failure Reasons
- mean_role_entropy_degenerate
- uniform_role_rows_degenerate

## Warnings
- mask_token_participates_in_row_provenance
- geometric_role_weights_degenerate

## Role Counts
{
  "role_counts": [
    1064.0,
    1064.0,
    1064.0,
    1064.0
  ],
  "role_fractions": [
    0.25,
    0.25,
    0.25,
    0.25
  ]
}
