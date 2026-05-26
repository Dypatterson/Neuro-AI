# Report 110: Phase 5' Range-Shaped Replay Downstream Precommit

**Date:** 2026-05-26
**Scope:** fresh Phase 5' lane precommit after Report 109 bridge closure
**Status:** precommitted; no retrieval or replay result is claimed by this report

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This lane is
not a Phase 5 headline run and does not claim graduation.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store for a Phase 5
headline run. This report does not authorize those runs.

**Last verified result:** Report 109 closes the current bundle-first Delta E
bridge/readout path as `not_viable_current_bridge`.

**Why this lane now:** Checklist F9 remains open for downstream Phase 4
consolidation comparison, while checklist I12 requires a fresh Phase 5' lane
precommit after bridge closure.

## Frozen Bridge Boundary

Report 109 at commit `6c05685` is frozen as the current bridge conclusion:

```text
current bundle-first Delta E bridge/readout path = not viable
```

Do not widen that path to n=3, n=10, gate, full matrix, M2, headline, or
graduation scale. This report switches lanes; it does not continue the bridge.

## Lane

The selected lane is:

```text
range_shaped_replay_downstream_f9_v1
```

The lane tests whether range-shaped replay has a useful downstream consequence
in Phase 4 consolidation, beyond candidate/provenance count. The decision will
be bounded to this repo-sample regenerated-codebook setting and will not
authorize Phase 5 `Delta E` unless a future fresh precommit is written.

## Fixed Inputs

Source and checkpoint inputs:

```text
corpus_source = repo_sample
codebook = reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt
codebook lineage = Report 072 regenerated repo-sample Phase 3C reconstruction
vocab/checkpoint support =
  reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_vocab.json
  reports/phase5_prime_phase3c_repo_sample_codebook/04_phase3c_reconstruction.md
```

The precommit JSON records the exact codebook SHA-256.

Precommit artifact:

```text
reports/phase5_prime_range_replay_downstream/precommit.json
SHA-256: e971879d4c468213b601f57a536e085d6fc3da82721169c5bdd920b47c55082b
payload SHA-256: 2ee639a863f08e51aad76fab77f15910de1154e0248694c926a73e1474929d4a
```

## Fixed Conditions

| Condition | Sampler | Fallback | Rebind | Insertion source | Role |
| --- | --- | --- | --- | --- | --- |
| `standard` | whole trace | none | none | post-settle final state | matched baseline |
| `range_postsettle` | `range_shaped` | `rebind` | `window_preserving` | post-settle final state | current production-compatible range-shaped path |
| `range_presettle` | `range_shaped` | `rebind` | `window_preserving` | pre-settle query | diagnostic positive control only |

`range_presettle` is not a production mechanism. It is included only to test
whether preserved novelty is sufficient to move downstream retrieval. If it
moves novelty/d_eff but not held-out retrieval, the lane stops instead of
escalating to Phase 5.

## Fixed Seeds And Parameters

Smoke seed:

```text
[17]
```

Decision seeds:

```text
[17, 11, 23, 1, 2, 3, 5, 7, 13, 29]
```

Fixed decision parameters:

```text
D = 128
max_vocab = 128
scales = [2, 3, 4]
landscape_size = 16
eval_window_size = 4
n_cues = 120
test_samples = 40
decode_k = 10
replay_every = 5
replay_batch_size = 4
store_capacity = 500
resolve_threshold = 0.2
smoothing_alpha = 0.0
beta = 10.0
near_duplicate_threshold = 0.95
```

Seed 17 is a wiring/smoke seed only. The n=10 run is a bounded lane decision,
not Phase 5 verification or graduation evidence.

## Metrics

Primary downstream metrics:

```text
heldout_top1
heldout_topk
heldout_cap_t05
```

Drill-down metrics:

```text
d_eff_final
query_near_existing_rate
final_near_duplicate_rate
stored_near_duplicate_rate
candidate count
provenance_cells
provenance_rectangularity_kl
query/final/stored winner diversity
seed-paired bootstrap CI over condition deltas
```

Candidate count alone is not useful evidence.

## Stop Criteria

The current production-compatible path `range_postsettle` is viable for a
fresh follow-up only if the same W=3 or W=4 cell shows all of:

```text
candidate/provenance support increases over standard
stored_near_duplicate_rate_mean <= 0.25
d_eff_final seed-paired mean delta >= 0
heldout_top1 delta >= +0.02 OR heldout_topk delta >= +0.05 OR cap_t05 delta >= +0.02
bootstrap CI lower bound >= 0 for the moving downstream metric
```

Stop the current lane if range-shaped replay only increases candidate count or
provenance support without useful novelty and held-out retrieval movement.
Also stop if the pre-settle positive control preserves novelty/d_eff but still
does not move held-out retrieval; that would mean novelty alone is not yet a
useful downstream consequence in this lane.

## Commands

Precommit artifact:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_range_replay_downstream_precommit.py
```

Local smoke:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/47_phase34_presettle_novelty.py \
  --seeds 17 \
  --conditions standard range_postsettle range_presettle \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --corpus-source repo_sample --max-vocab 128 --dim 128 \
  --scales 2 3 4 --landscape-size 8 --n-cues 20 --test-samples 8 \
  --replay-every 5 --replay-batch-size 4 --smoothing-alpha 0.0 \
  --out reports/phase5_prime_range_replay_downstream/smoke_seed17.json
```

Decision run:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/47_phase34_presettle_novelty.py \
  --seeds 17 11 23 1 2 3 5 7 13 29 \
  --conditions standard range_postsettle range_presettle \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --corpus-source repo_sample --max-vocab 128 --dim 128 \
  --scales 2 3 4 --landscape-size 16 --n-cues 120 --test-samples 40 \
  --replay-every 5 --replay-batch-size 4 --smoothing-alpha 0.0 \
  --out reports/phase5_prime_range_replay_downstream/results_n10.json
```

Analysis:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_range_replay_downstream_analysis.py \
  --precommit reports/phase5_prime_range_replay_downstream/precommit.json \
  --results reports/phase5_prime_range_replay_downstream/results_n10.json \
  --out reports/phase5_prime_range_replay_downstream/analysis_n10.json
```

If local smoke passes but the n=10 job is too slow or unavailable locally, run
the exact decision command in Colab/Safari and bring the JSON back to the same
`results_n10.json` path before analysis. Do not alter seeds, conditions, or
stop criteria in Colab.

## Anti-Homunculus Check

Pass. Conditions are fixed before execution. Diagnostics are read after the run
and never switch sampler, insertion source, or replay parameters inside a run.

## What This Report Does Not Authorize

- no bridge-path widening
- no Phase 5 graduation claim
- no headline replacement
- no M2 escalation
- no adaptive or metric-triggered routing
- no Phase 5 `Delta E` run from candidate-count-only evidence
