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
- Codebook: `/private/tmp/phase5_m1_codebook_prior_seed17_random_unit_codebook.pt`
- Codebook SHA-256: `2feb8dda399f755372855a6b3d49d5bce4fc85c5ddf20851fed24fa1720aa643`
- Codebook size bytes: 67176420
- Codebook repo-relative path: `None`
- Codebook registry: `config/phase5_m1_codebook_registry.json`
- Codebook registry match: False
- Mean normalized row entropy: 0.9996990434321246
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
    267.265380859375,
    268.0457458496094,
    263.51214599609375,
    265.1767272949219
  ],
  "role_fractions": [
    0.2511892677249765,
    0.251922693467678,
    0.24766179134971217,
    0.24922624745763333
  ]
}
