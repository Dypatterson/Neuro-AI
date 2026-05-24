# Phase 5 M1 P1 Geometric Probe

Evidence scope: seed 17 is wiring/provenance/degen smoke only. It is not
representative Phase 5 evidence, and this report does not support n=3, n=10, or
Phase 5 path claims by itself.

## Question

P1 tested whether row-domain geometric role weights can break the full-window
role symmetry that made count provenance degenerate:

- `unbind_density`: existing cross-role unbound filler pool.
- `same_role_filler_density` / `per_role_pool`: compare each row's role-r filler
  only against role-r fillers from other rows.
- `codebook_prior_density`: compare each row's role-r filler directly against an
  explicit codebook passed to the audit with `--codebook`.

## Synthetic Fixture

The role-specialized fixture uses 9 rows and 3 roles. Each row has one injected
dense codebook role target and sparse fillers in the other roles.

- Existing `unbind_density` remains near-uniform at k=8:
  mean normalized entropy `0.994735836982727`.
- `same_role_filler_density` is invariant for full-window rows:
  mean normalized entropy `1.0`, max row score spread
  `5.21540641784668e-08`.
- `codebook_prior_density` differentiates the injected role target:
  mean normalized entropy `0.00659200781956315`, minimum row max weight
  `0.9985371828079224`, argmax roles `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.

## Seed-17 Audit Smoke

Snapshot:
`reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`

Explicit codebook:
`/Users/dypatterson/Desktop/Neuro-AI/reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt`

Results:

- Count default: FAIL as expected with
  `mean_role_entropy_degenerate`, `uniform_role_rows_degenerate`.
- `same_role_filler_density`: FAIL with
  `geometric_mean_role_entropy_degenerate`,
  `geometric_uniform_role_rows_degenerate`.
- `codebook_prior_density`: PASS as a seed-17 audit smoke. Geometric mean
  normalized entropy `0.8866774142628774`, uniform row fraction
  `0.02537593984962406`, role coverage `4/4`.

## Interpretation

The same-role pool probe closes the cheap per-role-pool hypothesis: it stays
row-uniform because same-role unbinding preserves pairwise geometry of the full
pattern matrix.

The explicit codebook-prior probe is non-degenerate on both the synthetic
fixture and the seed-17 audit smoke. It should remain opt-in until reviewed and
until the next protocol level is explicitly authorized. Seed 17 is still only a
wiring/provenance/degen check.
