# Report 077: Phase 5′ Strict Context-Anchor Diagnostic

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5′ bundle-first follow-up to Report 076
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 076 showed that a full substrate-derived `context_bundle` anchor is
sufficient to rescue the hard skewed bundle-first cells. That result was still
not production-clean because the full context trace included the queried
role/filler pair.

This follow-up asks whether stricter, query-side partial context still works:

1. `context_bundle_exclude_query_role`: full scene context except the queried
   role/filler pair.
2. `context_bundle_observed_prefix`: only a fixed observed subset of non-query
   roles.
3. `deranged_role`: a no-fixed-point role control.

The results below were pasted from the Colab notebook summary output. Raw JSON
was not provided in this session.

## Run Configuration

n=10 seeds, 512 queries per seed, `D=4096`, `N=512`, `cue_noise=0.15`,
skewed co-occurrence, `scene_token=1`, `scene_token_pool_size=0`.

Swept:

- `K_roles in {4, 8, 16}`
- `scene_token_weight in {0.1, 0.25, 0.5, 1.0}`
- `scene_token_source in {context_bundle_exclude_query_role,
  context_bundle_observed_prefix}`
- observed-prefix `context_roles`:
  - K=4: `{1, 2, 3}`
  - K=8: `{1, 2, 4}`
  - K=16: `{1, 2, 4}`
- conditions: `candidate`, `random_role`, `shuffled_role`,
  `deranged_role`, `content_cleanup_positive`

## Candidate Result

### Exclude Queried Role

Removing the queried role/filler from the query context still leaves near
ceiling performance.

| K | w=0.1 | w=0.25 | w=0.5 | w=1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.9029 | 0.9996 | 0.9998 | 0.9998 |
| 8 | 0.9764 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9918 | 1.0000 | 1.0000 | 1.0000 |

This is the main positive result: the strict context trace does not need to
include the queried role/filler to recover the correct scene and content.

### Observed Prefix

One observed role is not enough context; it stays at the skewed baseline.
Two observed roles already produce a strong graded rescue, and four observed
roles saturate most cells.

| K | context_roles | w=0.1 | w=0.25 | w=0.5 | w=1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.2338 | 0.2338 | 0.2338 | 0.2338 |
| 4 | 2 | 0.7584 | 0.9631 | 0.9654 | 0.9664 |
| 4 | 3 | 0.9029 | 0.9996 | 0.9998 | 0.9998 |
| 8 | 1 | 0.2283 | 0.2283 | 0.2283 | 0.2283 |
| 8 | 2 | 0.5625 | 0.9064 | 0.9686 | 0.9676 |
| 8 | 4 | 0.8150 | 0.9996 | 1.0000 | 1.0000 |
| 16 | 1 | 0.2316 | 0.2316 | 0.2316 | 0.2316 |
| 16 | 2 | 0.4461 | 0.7650 | 0.9590 | 0.9717 |
| 16 | 4 | 0.6385 | 0.9793 | 1.0000 | 1.0000 |

The failure and success modes are scene-identification failures/successes:
`scene_tix` tracks `content_tix` and top1 closely in all candidate cells.

## Controls

`content_cleanup_positive` is `1.0000` for all listed K/source/weight/context
settings, confirming cleanup is not the bottleneck.

`random_role` remains effectively zero even when scene identification is high.
Representative upper values:

- K=4 observed-prefix context_roles=1: top1 `0.0010`.
- K=8 observed-prefix context_roles=1: top1 `0.0014`.
- K=16 observed-prefix context_roles=1: top1 `0.0008`.
- Exclude-query-role cells: top1 `0.0000` across the pasted K/weight grid.

`deranged_role` is the cleaner negative control. It remains near zero even
when `scene_tix` reaches `1.0000`:

- K=4 exclude-query-role, w=1.0: top1 `0.0000`, scene_tix `0.9998`.
- K=8 exclude-query-role, w=1.0: top1 `0.0000`, scene_tix `1.0000`.
- K=16 exclude-query-role, w=1.0: top1 `0.0000`, scene_tix `1.0000`.
- K=16 observed-prefix context_roles=4, w=1.0: top1 `0.0000`,
  scene_tix `1.0000`.

`shuffled_role` remains a useful but dirtier negative control because random
permutations can preserve some role structure. Its residual rises with scene
identification, but `deranged_role` rules out the fixed-role-map leak as an
explanation for the candidate result.

## Interpretation

This is a strict-context pass.

The result supports the Phase 5′ bundle-first decomposition:

`partial context/scene completion -> role unbinding -> content cleanup`

The key upgrade over Report 076 is that partial substrate-derived context works.
The queried role/filler pair is not required in the context trace. Observed
prefixes show a graded context-size threshold: one observed role is
insufficient, two roles are often enough at moderate weights, and four roles
saturate the hard cells.

Correct role unbinding remains mandatory. High scene identification under
`random_role` or `deranged_role` does not recover the target content.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No leave-one-seed-out sensitivity.
- No natural-corpus or Phase 3/4 learned-codebook version.
- No learned/replay-derived context trace yet.

## Next Work

Do not run the full matrix yet. First run a targeted K=16 observed-prefix
context-size curve:

- `context_roles in {1, 2, 3, 4, 6, 8}`
- `scene_token_weight in {0.1, 0.25, 0.5}`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`
- same hard cell: `D=4096`, `K=16`, `N=512`, `noise=0.15`, skewed
  co-occurrence

That run should locate the context-size threshold before spending time on the
full Phase 5′ matrix.
