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
- Geometric mode: codebook_prior_density
- Codebook: `/private/tmp/phase5_m1_codebook_prior_seed17_scrambled_codebook.pt`
- Codebook SHA-256: `771649bac28fe2d9da969fbdb3459e33244347180424c7baf7abc153a8046980`
- Codebook size bytes: 67176406
- Codebook repo-relative path: `None`
- Codebook registry: `config/phase5_m1_codebook_registry.json`
- Codebook registry match: False
- Mean normalized row entropy: 0.9996689968091205
- Uniform row fraction: 1.0

## Failure Reasons
- geometric_mean_role_entropy_degenerate
- geometric_uniform_role_rows_degenerate

## Warnings
- mask_token_participates_in_row_provenance
- codebook_path_outside_repo
- codebook_not_in_registry
- count_role_weights_degenerate

## Role Counts
{
  "role_counts": [
    261.35504150390625,
    267.532470703125,
    267.8243408203125,
    267.28814697265625
  ],
  "role_fractions": [
    0.24563443750367128,
    0.25144029201421525,
    0.25171460603412826,
    0.2512106644479852
  ]
}
