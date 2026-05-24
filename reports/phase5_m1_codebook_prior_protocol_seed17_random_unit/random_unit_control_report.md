# Phase 5 M1 Codebook-Prior Protocol - Random/Unit Control

Date: 2026-05-24

Branch: `codex/phase5-m1-codebook-prior-protocol`

## Scope

This is the Step C non-load-bearing control for the opt-in M1
`codebook_prior_density` diagnostic. It uses a shape-matched IID FHRR
unit-codebook generated with no corpus, no vocabulary, no Phase 3c training,
and no M1 snapshot rows.

Generated control:

- Script: `scripts/phase5_m1_codebook_random_control.py`
- Method: `iid_unit_complex_phases`
- Seed: `20260524`
- Shape: `[2050, 4096]`
- Dtype: `torch.complex64`
- Output `.pt`: `/private/tmp/phase5_m1_codebook_prior_seed17_random_unit_codebook.pt`
- Output SHA-256:
  `2feb8dda399f755372855a6b3d49d5bce4fc85c5ddf20851fed24fa1720aa643`
- Output size bytes: `67176420`
- Manifest:
  `phase5_m1_codebook_prior_seed17_random_unit_manifest.json`

The generated `.pt` remains outside the repo. The manifest is committed only
as metadata.

## Audit Result

Status: FAIL, as expected for a non-load-bearing independent/random control.

Failure reasons:

- `geometric_mean_role_entropy_degenerate`
- `geometric_uniform_role_rows_degenerate`

Warnings:

- `mask_token_participates_in_row_provenance`
- `codebook_path_outside_repo`
- `codebook_not_in_registry`
- `count_role_weights_degenerate`

Metrics:

- `geometric_entropy.mean_normalized = 0.9996990434321246`
- `geometric_entropy.uniform_row_fraction = 1.0`
- `geometric_role_fractions = [0.2511892677249765, 0.251922693467678, 0.24766179134971217, 0.24922624745763333]`

The audit artifacts are:

- `m1_provenance_audit_codebook_prior_random_unit.json`
- `m1_provenance_audit_codebook_prior_random_unit.md`
- `phase5_m1_codebook_prior_seed17_random_unit_manifest.json`

Each report artifact was byte-checked against the corresponding `/private/tmp`
dry-run output before final write.

## Comparison

| Reference | Status | Mean normalized entropy | Uniform row fraction |
|---|---:|---:|---:|
| Same-lineage Phase 3c codebook | PASS | 0.8866774142628774 | 0.02537593984962406 |
| Rowwise-coordinate scrambled Phase 3c codebook | FAIL | 0.9996689968091205 | 1.0 |
| Shape-matched random/unit codebook | FAIL | 0.9996990434321246 | 1.0 |

The random/unit control behaves like the scrambled control, not like the
same-lineage Phase 3c reference. This argues against the codebook-prior PASS
being a generic consequence of any shape-matched FHRR unit codebook or row-norm
distribution. The non-degenerate seed-17 result depends on alignment with the
actual Phase 3c reference bytes.

## Interpretation

This strengthens the Step A/Step B interpretation:

- The Step A scrambled FAIL shows that preserving shape and row norms is not
  enough.
- The Step C random/unit FAIL shows that a fresh IID FHRR unit-codebook is not
  enough.
- The same-lineage Phase 3c PASS is therefore specific to the standing Phase
  3c codebook geometry.

It does not make the same-lineage Phase 3c result lineage-independent. Seed 17
remains wiring/provenance/degen smoke only.

## Decision

Do not run n=3 or n=10 from this branch state. The codebook-prior probe is now
better characterized as a same-lineage diagnostic with two seed-17 negative
controls. A future escalation should first choose one of two routes:

- Accept the Phase 3c same-lineage reference as the explicit load-bearing
  gate input for Phase 5 M1, with the same-lineage limitation documented.
- Or generate a separate Phase 3c-style artifact from an independent seed/source
  lineage and test whether it preserves the non-degenerate gate behavior.
