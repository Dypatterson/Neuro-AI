# Report 111: Phase 5' Range-Shaped Replay Downstream Viability

**Date:** 2026-05-26
**Scope:** fixed Report 110 range-shaped replay downstream lane
**Status:** completed; current range-shaped downstream lane is not viable

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This report is
not a Phase 5 headline run and does not claim graduation.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store for a Phase 5
headline run. This report does not run or authorize that control matrix.

**Last verified result:** Report 110 precommitted this lane after Report 109
closed the current bundle-first Delta E bridge/readout path.

**Why this diagnostic now:** Checklist F9 remained open for a downstream Phase
4 consolidation comparison after range-shaped replay support was wired.

## Implementation

Added:

```text
scripts/phase5_prime_range_replay_downstream_precommit.py
scripts/phase5_prime_range_replay_downstream_analysis.py
tests/test_phase5_prime_range_replay_downstream.py
reports/phase5_prime_range_replay_downstream/precommit.json
reports/phase5_prime_range_replay_downstream/smoke_seed17.json
reports/phase5_prime_range_replay_downstream/results_n10.json
reports/phase5_prime_range_replay_downstream/analysis_n10.json
```

The fixed lane compares:

| Condition | Meaning |
| --- | --- |
| `standard` | whole-trace replay baseline |
| `range_postsettle` | current production-compatible `range_shaped + rebind + window_preserving`, storing post-settle final states |
| `range_presettle` | diagnostic positive control storing the pre-settle query |

The decision run used seeds `[17,11,23,1,2,3,5,7,13,29]`, W scales
`[2,3,4]`, `D=128`, `repo_sample`, the Report 072 regenerated codebook,
`n_cues=120`, `test_samples=40`, `replay_every=5`,
`replay_batch_size=4`, and `smoothing_alpha=0.0`.

## Artifacts

```text
reports/phase5_prime_range_replay_downstream/precommit.json
SHA-256: e971879d4c468213b601f57a536e085d6fc3da82721169c5bdd920b47c55082b

reports/phase5_prime_range_replay_downstream/smoke_seed17.json
SHA-256: fdabe2035f3ffa286f8b9bdb79f37c6e6c69729292ddaf79800e2e214f49f11d

reports/phase5_prime_range_replay_downstream/results_n10.json
SHA-256: 00d78a2afac4874d546d7f5aca3551ec70f054338a50dae7e5f5d9de94817bba

reports/phase5_prime_range_replay_downstream/analysis_n10.json
SHA-256: 3ab9653c405b27a2ebc313c306eb14826627c16788605e0b83e3208813a15f53
```

## Decision

```text
decision_id = not_viable_current_range_replay_downstream_novelty_without_retrieval
bounded_viability = not_viable_current_lane
phase5_delta_e_run_authorized = false
graduation_claim_authorized = false
bridge_path_reopened = false
```

The current production-compatible range path, `range_postsettle`, fails the
precommitted stop criteria at W=3 and W=4:

| W | Candidate support | Stored novelty | d_eff delta | Held-out movement | Decision |
| ---: | --- | --- | --- | --- | --- |
| 3 | pass | fail (`stored_near=1.000`) | fail (`-1.56`, CI `[-2.07,-1.06]`) | fail (`top1 +0.000`, `topk +0.0137`, CI crosses 0) | not viable |
| 4 | partial/pass | fail (`stored_near=1.000`) | fail (`-1.50`, CI `[-2.23,-0.73]`) | fail (`top1 -0.0121`, `topk -0.0065`) | not viable |

The diagnostic `range_presettle` positive control preserves novelty and raises
`d_eff`, but does not produce useful held-out retrieval movement:

| W | Stored near | d_eff delta | Held-out top1 delta | Held-out topk delta | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | `0.024` | `+19.41`, CI `[+18.41,+20.34]` | `-0.0017`, CI crosses 0 | `-0.0345` | novelty without utility |
| 4 | `0.014` | `+27.85`, CI `[+26.44,+29.28]` | `-0.0057`, CI crosses 0 | `+0.0010`, CI crosses 0 | novelty without utility |

## Findings

1. `range_postsettle` still changes the provenance geometry, not the useful
   memory geometry. At W=3, candidates rise `13.1 -> 28.9`, provenance cells
   rise `23.9 -> 32.7`, and rectangularity drops `0.9889 -> 0.1627`; however
   stored near-duplicates remain `1.000` and final `d_eff` falls.

2. W=4 is the same failure with slightly noisier support: candidates rise
   `14.7 -> 28.2`, rectangularity drops `1.1664 -> 0.2918`, but stored
   near-duplicates remain `1.000`, `d_eff` falls, and held-out top1/topk are
   worse than standard.

3. `range_presettle` proves that novelty can be preserved in the stored-vector
   channel: W=3 stored-near is `0.024` and W=4 is `0.014`, with large positive
   `d_eff` movement. But retrieval does not follow. This closes the "maybe
   just preserve novelty" version of the lane for this local cell.

4. Candidate/provenance count is not a sufficient signal. The exact stop
   condition triggered: range-shaped replay increases candidates/support, but
   the current path stores near-duplicates and the positive control does not
   convert novelty into held-out retrieval.

## Boundary

Stop the current range-shaped replay downstream lane. Do not run Phase 5
`Delta E`, n=10 headline, M2, full matrix, bridge widening, or graduation from
this evidence. A future replay idea would need a fresh precommit that changes
the mechanism, not just scale or candidate count.

## Anti-Homunculus Check

Pass. Conditions were fixed before execution. `range_presettle` is a static
diagnostic positive control, not metric-triggered routing. The analysis reads
artifacts after the run and never changes execution.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_range_replay_downstream_precommit.py \
  scripts/phase5_prime_range_replay_downstream_analysis.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_range_replay_downstream -v
```

```text
Ran 3 tests in 0.020s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_range_replay_downstream_precommit.py
```

```text
wrote reports/phase5_prime_range_replay_downstream/precommit.json
```

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

```text
passed; wrote 9 rows
```

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

```text
passed; wrote n=10 results
```

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_range_replay_downstream_analysis.py \
  --precommit reports/phase5_prime_range_replay_downstream/precommit.json \
  --results reports/phase5_prime_range_replay_downstream/results_n10.json \
  --out reports/phase5_prime_range_replay_downstream/analysis_n10.json
```

```text
decision not_viable_current_range_replay_downstream_novelty_without_retrieval
```
