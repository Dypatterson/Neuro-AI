# Report 074: Phase 3/4 Pre-Settle Novelty Diagnostic

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5' precommit follow-up to Report 073
**Status:** Diagnostic only. No Phase 5 graduation claim.

## Question

Report 073 showed that `range_shaped + rebind + window_preserving` replay
creates off-support query vectors, but the current post-settle candidate
insertion stores the Hopfield-cleaned endpoint. This report asks whether
candidate novelty exists before settling, and whether an auxiliary positive
control that stores the pre-settle query can preserve it.

This is a Phase 3/4 drill-down. It does not run Phase 5 and does not evaluate
the Phase 5 `Delta E` headline.

## Implementation

Added `experiments/47_phase34_presettle_novelty.py`.

The experiment uses the same regenerated repo-sample Phase 3C tensor as Report
073:

`reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt`

It compares three static conditions:

| Condition | Replay config | Insertion source |
| --- | --- | --- |
| `standard` | standard whole-trace replay | post-settle final state |
| `range_postsettle` | `range_shaped + rebind + window_preserving` | post-settle final state |
| `range_presettle` | `range_shaped + rebind + window_preserving` | pre-settle replay query |

`range_presettle` is an auxiliary positive control. It uses the existing
`UnifiedReplayMemory.run_replay_cycle()` path and stores `trace.query` in the
candidate handler when the predeclared static condition is active. It is not a
controller, not metric-triggered routing, and not a production mechanism.

For each condition, seed, and scale, the diagnostic records:

- pre-settle query similarity to existing memory
- post-settle final similarity to existing memory
- near-duplicate rate before settle, after settle, and for the vector inserted
- encoder-provenance cells and KL-to-factored rectangularity
- query/final/stored winner diversity
- `d_eff` before and after insertion
- cheap held-out masked-token retrieval on the local test windows

## Commands

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile experiments/47_phase34_presettle_novelty.py
```

```bash
PYTHONPATH=src:. .venv/bin/python experiments/47_phase34_presettle_novelty.py \
  --seeds 17 11 23 \
  --conditions standard range_postsettle range_presettle \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --out reports/phase5_prime_presettle_novelty/results_n3.json
```

```bash
PYTHONPATH=src:. .venv/bin/python experiments/47_phase34_presettle_novelty.py \
  --seeds 17 11 23 1 2 3 5 7 13 29 \
  --conditions standard range_postsettle range_presettle \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --out reports/phase5_prime_presettle_novelty/results_n10.json
```

## n=10 Aggregate Results

Means over 10 seeds:

| Condition | W | Candidates | Provenance cells | Rect KL | Query near | Final near | Stored near | Query top | Final top | Final d_eff | Held-out top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 2 | 9.7 | 8.2 | 0.6931 | 0.000 | 1.000 | 1.000 | 0.193 | 1.000 | 5.75 | 0.013 |
| range_postsettle | 2 | 22.8 | 12.0 | 0.0755 | 0.000 | 1.000 | 1.000 | 0.185 | 1.000 | 4.57 | 0.013 |
| range_presettle | 2 | 40.7 | 27.6 | 0.1443 | 0.269 | 0.977 | 0.269 | 0.597 | 0.995 | 12.48 | 0.014 |
| standard | 3 | 9.9 | 17.0 | 0.9605 | 0.000 | 1.000 | 1.000 | 0.191 | 1.000 | 7.30 | 0.025 |
| range_postsettle | 3 | 27.0 | 29.8 | 0.1974 | 0.000 | 1.000 | 1.000 | 0.168 | 1.000 | 5.28 | 0.025 |
| range_presettle | 3 | 45.7 | 62.3 | 0.3251 | 0.004 | 0.974 | 0.004 | 0.419 | 0.993 | 22.62 | 0.011 |
| standard | 4 | 12.7 | 31.7 | 1.1375 | 0.000 | 1.000 | 1.000 | 0.180 | 1.000 | 7.57 | 0.013 |
| range_postsettle | 4 | 33.3 | 48.5 | 0.2167 | 0.000 | 1.000 | 1.000 | 0.161 | 1.000 | 5.20 | 0.000 |
| range_presettle | 4 | 45.9 | 89.2 | 0.4080 | 0.000 | 1.000 | 0.000 | 0.340 | 1.000 | 26.46 | 0.000 |

Raw outputs:

- `reports/phase5_prime_presettle_novelty/results_n3.json`
- `reports/phase5_prime_presettle_novelty/results_n10.json`

## Findings

1. **The off-support replay queries are novel before settling.** Under
   `range_postsettle`, query-near-existing remains `0.000` while final-near
   and stored-near are both `1.000`. The query vectors are not already
   existing basins; the stored endpoints are.

2. **Hopfield settling is the erasure point.** The `range_presettle` positive
   control stores query vectors with much lower stored-near rates: W=2
   `0.269`, W=3 `0.004`, W=4 `0.000`. But the same traces still settle to
   near-duplicates after cleanup: final-near is W=2 `0.977`, W=3 `0.974`,
   W=4 `1.000`.

3. **Pre-settle insertion preserves geometric diversity in this local cell.**
   Final `d_eff` rises under `range_presettle` instead of falling: W=2
   `12.48` vs `4.57` for range post-settle, W=3 `22.62` vs `5.28`, W=4
   `26.46` vs `5.20`. Stored winner diversity likewise tracks the pre-settle
   query channel rather than the collapsed final-state channel.

4. **No held-out retrieval win is shown.** Cheap held-out top1 remains at or
   near zero for every condition and scale. The positive control proves a
   novelty-preservation possibility, not useful downstream learning.

5. **Candidate counts still are not evidence by themselves.** The range
   conditions emit more candidates and more provenance cells, but only the
   insertion source changes whether the stored vectors remain novel.

## Interpretation

Report 073's collapse diagnosis is refined: range-shaped window-preserving
rebind does synthesize novel off-support query vectors, but the current
Phase 4 replay loop inserts candidates after Hopfield cleanup has projected
those vectors onto existing attractors.

The auxiliary pre-settle condition is therefore a positive control for
candidate-novelty preservation. It shows novelty can survive if the query is
captured before cleanup, but it does not show that such captured candidates
are useful memory grooves, stable over larger corpora, or beneficial for Phase
5 structural retrieval.

## What Remains Unverified

- No Phase 5 `Delta E` run was performed.
- No Phase 5' bundle-first n>=10 control grid was rerun.
- No WikiText, pretrained-codebook, or production-scale corpus run was
  performed.
- No downstream held-out retrieval improvement was shown.
- No production mechanism for pre-settle storage was proposed or validated.
- The positive-control path does not graduate Phase 5 and does not supersede
  bundle-first as the leading Phase 5' signal.

## Next Work

The next useful discriminator is a fixed, non-adaptive mechanism that can
present or store pre-settle synthesized windows without turning diagnostics
into a controller. Candidate options include a static auxiliary candidate bank
or a training-time presentation path, both evaluated against held-out retrieval
before any Phase 5 `Delta E` run.
