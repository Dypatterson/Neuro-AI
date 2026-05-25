# phase5_log_prior_controls_local_smoke

Scratch outputs from a 2026-05-24 Path C log-prior controls smoke at
seed=17, n=5 cues. Local-only artefacts; not promoted to a numbered
report because the Path C / log-prior thread was superseded by the
substrate-shape diagnosis chain in [Report 064](../064_phase5_m1_retrieval_smoke_cross_seed_null.md)
→ [Report 065](../065_mqar_external_architecture_gate.md)
→ [Report 066](../066_mqar_ghrr_bundle_first_discriminator.md).

## What's here

- `seed17_full_codebook_n5.json` — log-prior controls with full codebook
  prior. Records large energy movement during settling but `hit_role = 0.0`
  on the 5-cue probe. Random-prior behavior is unstable.
- `seed17_no_prior_n5.json` — γ=0 / no-prior baseline. Essentially null;
  consistent with the broader Path C nulls in [Report 063](../063_phase5_spike_e1_centered_log_prior.md).

## Status

- Snapshot used: `phase3_phase4_w4_step1800_AB_A1prime.pt` (seed 17,
  pre-S1 provenance).
- Not load-bearing: the Path C thread is closed by Report 063, and the
  larger substrate-shape diagnosis (Reports 065 / 066) makes log-prior
  retuning irrelevant — the substrate's elementwise binding lacks
  key-only basins regardless of prior shape.
- Kept as scratch for traceability. If cleanup is desired, delete the
  directory — no downstream artefact references these files.

## Why this README exists

The 2026-05-24 session left these JSONs untracked under `reports/`. The
README is the explicit decision (per [CLAUDE.md "don't trust a 'never
built / never run' flag"](../../CLAUDE.md)) to mark them as discovered,
explained, and intentionally not promoted — so a future session can't
grep `reports/` for them and infer they're a missing-report blocker.
