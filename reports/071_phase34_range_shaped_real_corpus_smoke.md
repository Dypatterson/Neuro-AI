# Report 071 — Phase 3/4 Range-Shaped Real-Corpus Smoke

**Date:** 2026-05-25
**Active phase:** 5′ precommit
**Status:** Real-corpus wiring smoke completed. **No Phase 5 graduation claim.**
This report checks `experiments/19_phase34_integrated.py` can run with static
range-shaped replay config; it does not establish held-out improvement or Phase
5 `Delta E` evidence.

## What Changed

`experiments/19_phase34_integrated.py` now exposes static Phase 4 replay knobs:

```text
--replay-sampler standard|range_shaped
--range-shaped-fallback skip|closest|rebind
--range-shaped-rebind-mode single_binding|window_preserving
--range-shaped-smoothing-alpha FLOAT
--range-shaped-window-size INT
```

For each scale, the harness passes `replay_position_vectors=slots_c[s].positions`
and `replay_codebook=cb_c[0]` into `UnifiedReplayMemory`, so
`range_shaped + rebind` can synthesize replay traces inside the existing
Phase 3/4 path.

Also fixed a local harness crash: `_empty_device_cache()` no longer touches
`torch.mps.empty_cache()`. On this macOS/PyTorch/Python 3.13 stack, calling the
MPS cache helper during CPU-only runs segfaulted after condition A. CUDA cache
cleanup remains available for Colab/GPU runs.

## Smoke Setup

The default pretrained Phase 3C codebook file was not present locally, and
WikiText may require dataset/network availability. The smoke therefore used the
local repo-sample corpus and a random codebook, strictly to verify wiring:

```text
--corpus-source repo_sample --random-codebook
--dim 128 --device cpu --max-vocab 128
--scale-landscape 2:16,3:16,4:16
--eval-window-size 4 --n-cues 30 --checkpoint-every 15 --test-samples 10
--replay-every 5 --replay-batch-size 4
--store-threshold 0.0 --resolve-threshold 0.2 --reencode-every 15
```

Four matched runs were executed:

| output dir | updater | replay config |
| --- | --- | --- |
| `phase5_prime_phase34_range_standard_smoke` | none | `standard` |
| `phase5_prime_phase34_range_window_smoke` | none | `range_shaped + rebind + window_preserving` |
| `phase5_prime_phase34_range_standard_hebbian_smoke` | hebbian | `standard` |
| `phase5_prime_phase34_range_window_hebbian_smoke` | hebbian | `range_shaped + rebind + window_preserving` |

## Results

Final checkpoint for condition C (`phase3_phase4`):

| run | top1 | topk | cap_t05 | candidates | W2 patterns |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard, updater none | 0.000 | 0.000 | 0.000 | 26 | 21 |
| range window, updater none | 0.000 | 0.333 | 0.000 | 59 | 30 |
| standard, Hebbian | 0.000 | 0.000 | 0.000 | 27 | 21 |
| range window, Hebbian | 0.000 | 0.000 | 0.000 | 65 | 36 |

Raw outputs:

```text
reports/phase5_prime_phase34_range_standard_smoke/phase34_results.json
reports/phase5_prime_phase34_range_window_smoke/phase34_results.json
reports/phase5_prime_phase34_range_standard_hebbian_smoke/phase34_results.json
reports/phase5_prime_phase34_range_window_hebbian_smoke/phase34_results.json
```

## Interpretation

**F-1 — Experiment 19 is now wired for static range-shaped replay.** The
`standard` and `range_shaped + rebind + window_preserving` paths both complete
through all three experiment 19 conditions.

**F-2 — The range-shaped path emits more Phase 4 candidates.** In the matched
tiny smoke, candidate counts rise from `26/27` to `59/65`, and W2 live patterns
rise from `21` to `30/36`. This is consistent with Report 070's synthetic
downstream result that window-preserving rebind carries more support into
candidate generation.

**F-3 — No held-out improvement is established.** Held-out `top1` and
`cap_t05` remain zero. One non-Hebbian range-shaped run shows `topk=0.333`, but
this is a 10-sample smoke and not evidence.

**F-4 — The Hebbian runtime updater did not fire.** With random codebook and
default `success_threshold=0.5`, success rate stayed `0.000` and `upd=0`.
This means the Hebbian-labeled smoke is still mostly a replay-wiring check, not
a meaningful Phase 3 learning comparison.

## What Remains Unverified

- No pretrained-codebook real-corpus run has been executed locally.
- No WikiText or larger repo-corpus n>=10 comparison has been run.
- No held-out recall/codebook geometry improvement has been shown.
- No Phase 5 `Delta E` movement has been measured.
- No Colab run was needed yet; the current smoke is local and tiny.

## Bottom Line

The real-corpus harness now accepts the range-shaped static config and can run
`range_shaped + rebind + window_preserving` end to end. The next useful run is
a larger real-corpus comparison using an available pretrained codebook or a
forced/validated Phase 3 learning setup, then n>=10 on Colab only if held-out
or geometry readouts move.
