# Report 073: Phase 3/4 Candidate-Quality Diagnostic

**Date:** 2026-05-25  
**Branch:** `phase5-m1-role-energy-stack`  
**Scope:** Phase 5' precommit follow-up to Reports 070-072  
**Status:** Diagnostic only. No Phase 5 graduation claim.

## Question

Reports 070-072 showed that static `range_shaped + rebind + window_preserving`
replay produces more Phase 4 candidates than standard replay, but held-out
retrieval stayed flat. This report asks whether the extra candidates are useful
novel attractors or whether they settle back into existing basins.

The diagnostic is intentionally local to Phase 3/4. It does not run Phase 5 and
does not evaluate the Phase 5 headline `Delta E`.

## Implementation

Added `experiments/46_phase34_candidate_quality.py`.

The experiment compares `standard` replay against static
`range_rebind_window` replay using the regenerated repo-sample Phase 3C tensor
from Report 072:

`reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt`

For each condition, seed, and scale, it records:

- candidate count
- encoder-provenance cells and KL-to-factored rectangularity
- final settled winner diversity
- near-duplicate rate against the pre-insertion memory
- query similarity to existing patterns before settling
- `d_eff` before and after candidate insertion

The mechanism remains fixed by static config. There is no metric-triggered
routing, no controller, and no best-of-N selection.

## Commands

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile experiments/46_phase34_candidate_quality.py
```

```bash
PYTHONPATH=src:. .venv/bin/python experiments/46_phase34_candidate_quality.py \
  --seeds 17 11 23 \
  --conditions standard range_rebind_window \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --out reports/phase5_prime_candidate_quality/results_n3.json
```

```bash
PYTHONPATH=src:. .venv/bin/python experiments/46_phase34_candidate_quality.py \
  --seeds 17 11 23 1 2 3 5 7 13 29 \
  --conditions standard range_rebind_window \
  --codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt \
  --out reports/phase5_prime_candidate_quality/results_n10.json
```

## n=10 Aggregate Results

| Condition | W | Candidates | Provenance cells | Rect KL | Winner unique | Near duplicate | Query near existing | Final d_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 2 | 9.7 | 8.2 | 0.6931 | 6.3 | 1.000 | 0.000 | 5.75 |
| range_rebind_window | 2 | 21.9 | 10.9 | 0.0689 | 8.4 | 1.000 | 0.000 | 4.73 |
| standard | 3 | 9.9 | 17.0 | 0.9605 | 7.0 | 1.000 | 0.000 | 7.30 |
| range_rebind_window | 3 | 26.3 | 26.6 | 0.1822 | 6.4 | 1.000 | 0.000 | 5.38 |
| standard | 4 | 12.7 | 31.7 | 1.1375 | 8.4 | 1.000 | 0.000 | 7.57 |
| range_rebind_window | 4 | 35.3 | 55.3 | 0.2522 | 9.6 | 1.000 | 0.000 | 5.29 |

Raw outputs:

- `reports/phase5_prime_candidate_quality/results_n3.json`
- `reports/phase5_prime_candidate_quality/results_n10.json`

## Findings

1. Range-shaped replay still does what Report 068 and Report 070 predicted at
   the provenance layer. It increases candidate count and makes encoder-term
   support less rectangular: at W=4, candidates increase `12.7 -> 35.3`,
   provenance cells increase `31.7 -> 55.3`, and rectangularity drops
   `1.1375 -> 0.2522`.

2. The extra candidates are not novel attractors in this regime. Near-duplicate
   rate is `1.000` for every condition and scale. The query vectors themselves
   are not near existing patterns (`query_near_existing_rate = 0.000`), but the
   Hopfield settle step snaps them back onto existing basins before insertion.

3. Final diversity does not rescue the result. At W=3, range-window produces
   nearly 3x more candidates (`9.9 -> 26.3`) while final winner uniqueness is
   slightly lower (`7.0 -> 6.4`). At W=4, winner uniqueness rises only
   modestly (`8.4 -> 9.6`) despite almost 3x more candidates.

4. Effective dimension moves the wrong way after insertion. Final `d_eff` is
   lower under range-window at every scale: W=2 `5.75 -> 4.73`, W=3
   `7.30 -> 5.38`, W=4 `7.57 -> 5.29`.

## Interpretation

The range-shaped sampler plus window-preserving rebind integration is working
as a fixed replay geometry intervention, but the current Phase 4 settling
pipeline collapses the synthesized combinations into pre-existing attractors.

This explains the Report 071/072 pattern: more candidates and more W2 patterns
are visible, but held-out retrieval remains flat. In the present local
repo-sample regime, "more replay candidates" is insufficient because candidate
insertion happens after cleanup has already erased the off-support combination.

## What Remains Unverified

- No Phase 5 `Delta E` run was performed.
- No WikiText or production pretrained-codebook run was performed.
- No downstream retrieval improvement was shown.
- No n>=10 Phase 5' bundle-first controls were rerun here.
- The result does not falsify bundle-first structural memory.
- The result does not falsify range-shaped replay as a training-time or
  pre-settle data presentation mechanism; it only shows the current
  post-settle candidate insertion path does not preserve novelty.

## Next Work

The next useful discriminator is not another larger local candidate-count run.
The mechanism question is where to preserve the off-support synthesized pair:

1. test pre-settle or auxiliary-storage candidate capture before Hopfield
   cleanup erases the combination;
2. compare against a positive-control insertion that stores the synthesized
   window before settle;
3. only if candidate novelty survives, rerun held-out Phase 3/4 and then the
   Phase 5 `Delta E` headline with n>=10 controls.

Until then, Phase 5 remains ungraduated and bundle-first remains the leading
Phase 5' positive signal by Reports 066/067.
