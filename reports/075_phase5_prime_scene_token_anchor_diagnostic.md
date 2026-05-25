# Report 075: Phase 5′ Scene-Token Anchor Diagnostic

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5′ bundle-first follow-up to Report 069
**Status:** Diagnostic only. No Phase 5 graduation claim.

## Question

Report 069's hard-cell bundle-first result showed that a scene token can rescue
the difficult skewed co-occurrence cell:

`D=4096`, `N=512`, `K_roles=16`, `cue_noise=0.15`, `scene_token=1`,
`cooccurrence=skewed`.

This report records the pasted Colab follow-up results and asks three narrower
questions:

1. Is the scene token load-bearing under skew, or was the hard-cell result a
   fluke?
2. Does a generic/global token help, or must the anchor be scene-specific?
3. Does the result survive role controls, or is it identity leakage that
   bypasses the role-binding substrate?

The pasted results are recorded here because the Colab runtime disconnected
and file upload was unreliable. Raw JSON artifacts have not yet been committed
for this follow-up.

## Harness Change

Updated `experiments/44_phase5_prime_bundle_first.py` with two static scene
anchor knobs:

- `--scene_token_weight`: sweeps the anchor contribution in both stored scene
  bundles and query cues.
- `--scene_token_source {random,context_bundle}`:
  - `random` is the previous random scene-anchor condition.
  - `context_bundle` uses the scene's role-filler bundle itself as a
    substrate-derived context trace.

The existing `--scene_token_pool_size` now distinguishes:

- `0`: one unique random anchor per scene.
- `16`: anchors reused across a small pool.
- `1`: one global anchor shared by every scene.

The Colab notebook now includes a context-bundle-anchor follow-up cell. That
cell is the next run; the results below are the already pasted random-anchor
diagnostics.

## Hard-Cell Control Result

n=10 seeds, 512 queries per seed, `D=4096`, `K_roles=16`, `N=512`,
`cue_noise=0.15`, `scene_token=1`, `token_weight=0.5`, unique random scene
anchors, skewed co-occurrence.

| Condition | Top1 | Wilson CI | scene_tix | content_tix | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `candidate` | 0.8812 | [0.8721, 0.8898] | 0.8797 | 0.8812 | hard cell remains strong |
| `bundle_positive` | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 | bundle/unbind ceiling intact |
| `content_cleanup_positive` | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 | content cleanup is not the bottleneck |
| `perfect_cue` | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 | storage/cleanup ceiling intact |
| `random_role` | 0.0000 | [0.0000, 0.0007] | 0.8797 | 0.0000 | correct scene alone is insufficient |
| `shuffled_role` | 0.0107 | [0.0083, 0.0140] | 0.1012 | 0.0107 | role structure is load-bearing |

The key split is `random_role`: it keeps scene identification high
(`scene_tix=0.8797`) while destroying content retrieval (`content_tix=0.0000`).
That rules out the strongest identity-leak interpretation. The mechanism still
needs the role-specific unbinding path.

## Candidate Grid Extremes

The worst pasted candidate cells were all the no-scene-token skewed cells at
high load. Representative examples:

| Cell | Top1 | Wilson CI | scene | content |
| --- | ---: | ---: | ---: | ---: |
| `D=4096,K=4,N=512,noise=0.0,scene_token=0,skewed` | 0.2145 | [0.1990, 0.2308] | 0.2023 | 0.2145 |
| `D=4096,K=8,N=512,noise=0.15,scene_token=0,skewed` | 0.2242 | [0.2085, 0.2408] | 0.2113 | 0.2242 |
| `D=4096,K=16,N=512,noise=0.15,scene_token=0,skewed` | 0.2273 | [0.2115, 0.2440] | 0.2145 | 0.2273 |

The best pasted cells were dominated by `scene_token=1` conditions, often at
or near `1.0000` top1. This confirms that the anchor is not a cosmetic change
under skew; it moves the scene-MHN identification bottleneck.

## Weight and Pool Sweep

Candidate condition only, n=10 seeds, `D=4096`, `N=512`, `noise=0.15`,
`scene_token=1`, skewed co-occurrence.

| K | token pool | w=0.0 | w=0.1 | w=0.25 | w=0.5 | w=1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | unique (`0`) | 0.2314 | 0.2510 | 0.3824 | 0.8812 | 1.0000 |
| 16 | pool of 16 | 0.2324 | 0.2480 | 0.3652 | 0.7646 | 0.8480 |
| 16 | global (`1`) | 0.2338 | 0.2342 | 0.2365 | 0.2346 | 0.2324 |
| 4 | unique (`0`) | 0.2316 | 0.2807 | 0.5975 | 1.0000 | 1.0000 |
| 4 | pool of 16 | 0.2299 | 0.2738 | 0.5611 | 0.8566 | 0.8562 |
| 4 | global (`1`) | 0.2299 | 0.2289 | 0.2287 | 0.2258 | 0.2328 |
| 8 | unique (`0`) | 0.2314 | 0.2592 | 0.4643 | 0.9824 | 1.0000 |
| 8 | pool of 16 | 0.2318 | 0.2641 | 0.4354 | 0.8361 | 0.8480 |
| 8 | global (`1`) | 0.2311 | 0.2324 | 0.2322 | 0.2332 | 0.2297 |

Three facts matter:

1. **Zero weight is the skewed baseline.** All K values cluster around
   `0.23`, matching the no-scene-token failure regime.
2. **Unique anchors are strongest.** The hard `K=16` cell rises from `0.2314`
   to `0.8812` at weight `0.5` and to `1.0000` at weight `1.0`.
3. **A global anchor does nothing.** `token_pool=1` stays near `0.23` at every
   weight. The effect is not generic extra norm, temperature, or an arbitrary
   shared bias vector.

Pool-of-16 anchors are intermediate. They help because they add discriminative
context, but collisions cap the benefit relative to unique per-scene anchors.

## Control Sweep Notes

The pasted control sweep shows:

- `content_cleanup_positive` is `1.0000` across every shown K, weight, and
  pool setting. Cleanup is solved once the right content query is presented.
- `random_role` stays near zero across the sweep, even when `scene_tix` is
  high. Correct scene identification does not by itself recover the queried
  filler.
- `shuffled_role` remains low in content top1, but it rises when unique random
  scene anchors dominate. Examples: `K=4`, unique, `w=1.0` gives top1
  `0.2166`; `K=8`, unique, `w=1.0` gives top1 `0.2398`; `K=16`, unique,
  `w=1.0` gives top1 `0.0799`. This is expected leakage from very strong
  scene identification under skew, and it is why role controls must remain
  matched in every future grid.

## Interpretation

The random scene-anchor result is real diagnostic signal, but it is not yet a
production mechanism.

It supports the Phase 5′ bundle-first decomposition:

`context/scene completion -> role unbinding -> content cleanup`

The scene-MHN bottleneck is the failure point under skew. Content cleanup and
algebraic unbinding are intact when the correct scene bundle is recovered.
The global-token null is especially important: the rescue requires
anchor-specific context, not a generic vector added to every cue.

The remaining problem is architectural cleanliness. A random unique anchor is
label-like if promoted directly. It violates the spirit of "personality/self
emerges from substrate, not pretraining or external IDs" and is not yet an
anti-homunculus production story. The next diagnostic therefore replaces the
random ID vector with `context_bundle`: a context trace derived from the same
role-filler substrate.

## Core-Principle Check

- **No controller / anti-homunculus:** Pass for this diagnostic. All
  conditions are fixed before execution; no metric-triggered route selection
  or best-of-N graduation claim is used.
- **Context completion rather than next-token prediction:** Strengthened. The
  successful path completes a scene/event context, then unbinds a queried role.
- **Continuous learning across domains:** Not tested here. This remains a
  static synthetic MQAR-style diagnostic and must later connect to Phase 3/4
  learning and replay.
- **Emergent self/personality from substrate:** Not tested here. Random scene
  anchors are explicitly not accepted as a production answer. The
  `context_bundle` follow-up is the first substrate-derived replacement.
- **Passive diagnostics:** Preserved. `scene_tix`, `content_tix`, entropy, and
  margins are audit signals only.

## What Remains Unverified

- The new `context_bundle` anchor follow-up has not yet been run at Colab
  scale.
- No Phase 5 `Delta E` headline run was performed.
- No full n>=10 verification matrix has been run across all K, N, cue noise,
  controls, and co-occurrence regimes.
- No leave-one-seed-out sensitivity has been reported.
- No natural corpus or Phase 3/4 learned-codebook version exists yet.
- Random unique anchors should not be treated as a production substrate.

## Next Work

1. Run the Colab context-bundle-anchor follow-up:
   `K_roles in {4,8,16}`, `N=512`, `noise=0.15`, skewed co-occurrence,
   weights `{0.1,0.25,0.5,1.0}`, matched candidate and controls.
2. If `context_bundle` retains the random-anchor effect while controls remain
   bounded, promote it from diagnostic to a fixed architectural candidate:
   substrate-derived context trace plus role unbinding plus content cleanup.
3. Only after that, run the full n>=10 Phase 5′ matrix with matched controls
   and then map the accepted mechanism back to the Phase 5 `Delta E` headline.
