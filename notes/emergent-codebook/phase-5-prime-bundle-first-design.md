---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - design
---

# Phase 5′ Bundle-First Design

Phase 5′ replaces the M1 rescue path with a precommitted bundle-first
structural-memory architecture plus range-shaped replay as a complementary
data-side intervention. This is a design precommit, not a graduation claim.

## Architectural Claim

Structural retrieval should be organized around scene-level bundled memories:

```text
scene_bundle_n = normalize(sum_j bind(role_j, filler_{n,j}))
```

The scene bundle is the stored item. Retrieval first identifies the scene,
then unbinds the queried role, then cleans the resulting content estimate.
The expected bottleneck is scene-MHN identification over scene bundles, not
algebraic role unbinding or content cleanup.

Reports 066 and 067 are diagnostic gates supporting this shape:

- Report 066: bundle-first is the only tested rescue with genuine basin
  retrieval; GHRR-native does not rescue key-only MHN retrieval.
- Report 067: multi-role bundle-first survives MQAR cleanup at diagnostic
  scale, with `scene_tix` and `content_tix` matching to within one trial in
  almost every cell.

Those reports are n=3 MQAR diagnostics. They are not Phase 5 graduation
evidence and they do not measure the Phase 5 ΔE headline.

## Mechanism

### 1. Scene-MHN Identification

Store one normalized bundle per scene/event. A partial structural cue, such as
one known `(role, filler)` pair, is used as the query to a scene-level Modern
Hopfield memory. The scene-MHN settles by energy to one scene bundle.

Diagnostics:

- `scene_tix`: whether the scene-MHN top index is the ground-truth scene
- scene entropy and margin
- degradation with `N` scenes
- robustness under cue noise and scene-token variants

### 2. Role Unbinding

Given a retrieved scene state and query role, compute:

```text
content_estimate = unbind(scene_state, role_query)
```

This is fixed algebra, not a selected route. The query specifies the role; no
controller chooses which role to unbind.

Diagnostics:

- direct similarity rank of `content_estimate` against the content codebook
- failure split between wrong-scene and wrong-content cases

### 3. Content Cleanup

Feed the unbound content estimate into a content-codebook MHN. The cleanup head
settles to a content atom.

Diagnostics:

- `content_tix`: whether the content-MHN top index is the ground-truth filler
- content entropy and margin
- `scene_tix == content_tix` invariant, used to detect whether cleanup is
  deterministic given correct scene identification

## Range-Shaped Replay Integration

Range-shaped replay is a Phase 4 data-side layer that samples from factored
role and atom marginals rather than the buffer's natural co-occurrence joint.
It is complementary to bundle-first storage:

- bundle-first changes what Phase 5 stores and retrieves
- range-shaped replay changes which `(role, atom)` combinations Phase 4
  consolidation rehearses

Report 068 validates the sampler algorithm only. Phase 4 integration requires
static-config wiring plus rebind-on-the-fly synthesis for missing pairs:

- single-binding: `bind(position_vectors[role], codebook[atom])`
- window-preserving: sample a full factored window, then call
  `encode_window_with_provenance()`

The sampler can optionally use atom-support smoothing to reach atoms with zero
buffer count. This is a fixed prior, not a metric-triggered switch.

## Headline Metric

For the diagnostic MQAR harness, the headline is:

```text
top1 over queried fillers, with Wilson 95% CI across seeds x queries
```

The load-bearing diagnostics are:

- `scene_tix`
- `content_tix`
- scene/content entropy
- scene/content margin
- confidence intervals

For any Phase 5 graduation-style claim, the legacy Phase 5 headline remains
binding unless explicitly replaced by user-approved spec amendment:

```text
Delta E = E_content-prior - E_role-prior
```

No Phase 5′ MQAR result should be described as graduation unless the relevant
spec and checklist are updated and n>=10 controls pass.

## Required Controls

Run controls on the same seeds, N values, role counts, cue-noise settings, and
query schedule as the candidate condition:

- random-role control: query with a random role not tied to the scene
- shuffled-role control: permute role labels across scenes or queries
- perfect-cue control: cue with the full scene bundle or exact content target
- bundle positive control: direct bundle/unbind path without scene-MHN failure
- content cleanup positive control: noisy content vector into content-MHN
- no scene-token vs scene-token condition, if scene tokens are included
- co-occurrence skew/natural-statistics condition, if cheap

Controls must remove, bound, or explain the effect. They must not be used to
select the best condition post hoc.

## Verification Standard

Diagnostic gates may run at n=3 to decide whether a mechanism is worth
building. Verified status requires:

- n_seeds >= 10
- confidence intervals on the headline
- required controls on the same test set
- seed-level and leave-one-seed-out readout
- explicit report of what remains unverified

Reports 066 and 067 pass diagnostic gates only. Report 068 passes an algorithm
gate only. None of these graduate Phase 5.

## Anti-Homunculus Check

This design uses fixed geometry, energy, algebra, and sampling dynamics:

- scene-MHN settles by energy; no controller chooses the scene
- role unbinding is deterministic algebra specified by the query
- content-MHN cleanup settles by energy; no metric chooses a content head
- range-shaped replay samples from static marginals and fixed smoothing alpha
- diagnostics are logged passively and never route execution

Forbidden:

- metric-triggered sampler switching
- best-of-N condition selection as evidence
- controller modules deciding whether scene-MHN, unbinding, or cleanup "wins"
- changing fallback mode in response to observed ΔE/top1 during a run
