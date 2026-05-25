# Report 072 — Phase 3/4 Range-Shaped Smoke With Regenerated Codebook

**Date:** 2026-05-25
**Active phase:** 5′ precommit
**Status:** Local regenerated-codebook smoke completed. **No Phase 5 graduation
claim.** No held-out improvement or Phase 5 `Delta E` evidence.

## Why This Run

Report 071 verified `experiments/19_phase34_integrated.py` can run static
range-shaped replay, but it used a random codebook because the expected local
Phase 3C tensor was missing:

```text
reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt
```

This report regenerates a small repo-sample Phase 3A/3C codebook in new
Phase 5′ artifact directories, then reruns the matched Experiment 19 pair.
Historical Phase 3 report directories were not overwritten.

## Codebook Regeneration

Phase 3A:

```text
PYTHONPATH=src:. .venv/bin/python experiments/03_phase3a_hebbian_codebook.py \
  --corpus-source repo_sample --dim 128 --device cpu --max-vocab 128 \
  --train-window-size 4 --epochs 2 \
  --window-sizes 4 --landscape-sizes 16 --betas 10 \
  --mask-counts 1 --mask-positions center --test-samples 10 \
  --output-dir reports/phase5_prime_phase3a_repo_sample_codebook
```

Phase 3C:

```text
PYTHONPATH=src:. .venv/bin/python experiments/04_phase3c_reconstruction.py \
  --corpus-source repo_sample --dim 128 --device cpu --max-vocab 128 \
  --phase3a-dir reports/phase5_prime_phase3a_repo_sample_codebook \
  --train-window-size 4 --train-landscape-size 16 --train-probe-size 40 \
  --consolidation-k 10 --epochs 1 \
  --window-sizes 4 --landscape-sizes 16 --betas 10 \
  --mask-counts 1 --mask-positions center --test-samples 10 \
  --output-dir reports/phase5_prime_phase3c_repo_sample_codebook
```

Artifacts:

```text
reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt
reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_hebbian.pt
reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_random.pt
```

The regenerated Phase 3C mini-eval still had zero held-out generalization on
this tiny repo-sample cell, so the codebook is an integration artifact, not a
validated pretrained baseline.

## Experiment 19 Pair

Shared settings:

```text
--corpus-source repo_sample
--codebook-path reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_codebook_reconstruction.pt
--dim 128 --device cpu --max-vocab 128
--scale-landscape 2:16,3:16,4:16
--eval-window-size 4 --n-cues 60 --checkpoint-every 30 --test-samples 20
--replay-every 5 --replay-batch-size 4
--store-threshold 0.0 --resolve-threshold 0.2 --reencode-every 15
--updater-kind hebbian --success-threshold 0.0
```

Compared:

| output dir | replay config |
| --- | --- |
| `phase5_prime_phase34_phase3c_standard_smoke` | `standard` |
| `phase5_prime_phase34_phase3c_window_smoke` | `range_shaped + rebind + window_preserving` |

Final condition-C checkpoint:

| run | top1 | topk | cap_t05 | Hebbian events | candidates | W2 patterns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 0.000 | 0.000 | 0.000 | 90 | 54 | 22 |
| range window | 0.000 | 0.000 | 0.000 | 91 | 88 | 35 |

Raw outputs:

```text
reports/phase5_prime_phase34_phase3c_standard_smoke/phase34_results.json
reports/phase5_prime_phase34_phase3c_window_smoke/phase34_results.json
```

## Findings

**F-1 — The missing codebook blocker is resolved for local smoke scale.** The
branch now has a repo-sample Phase 3C reconstruction tensor that Experiment 19
can load via `--codebook-path`.

**F-2 — Range-shaped replay again increases candidate production.** With the
regenerated codebook, `range_shaped + rebind + window_preserving` emits `88`
condition-C candidates versus `54` for standard replay and leaves more W2 live
patterns (`35` vs `22`).

**F-3 — Held-out behavior still does not move.** `top1`, `topk`, and `cap_t05`
remain `0.000` in both runs. The increased candidate channel is not yet a
useful downstream win.

**F-4 — The current repo-sample codebook is too weak to justify Colab n>=10.**
The Phase 3C mini-eval is itself flat at zero. Escalating this exact cell would
mostly measure candidate inflation, not improved memory behavior.

## What Remains Unverified

- No WikiText/pretrained-codebook comparison has been run.
- No real held-out recall or codebook-geometry improvement has been shown.
- No Phase 5 `Delta E` run is warranted yet.
- No Colab run was needed for this cell.

## Bottom Line

The branch can now regenerate and load a Phase 3C codebook for local
Experiment 19 range-shaped comparisons. The result remains evidence-bounded:
window-preserving range-shaped replay increases Phase 4 candidate production,
but the current tiny regenerated-codebook cell shows no held-out improvement.
The next test should use a stronger pretrained codebook or a larger corpus
setup before spending Colab time or running Phase 5 headline metrics.
