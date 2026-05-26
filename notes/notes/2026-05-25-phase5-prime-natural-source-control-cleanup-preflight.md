---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Natural Source Control-Cleanup Preflight

## Status

Reports 089-091 close the first non-synthetic native provenance source cycle:

- Report 089 preflights `trajectory_native_provenance_context_trace` and shows
  support/query/geometry pass at the K=16 hard cell.
- Report 090 runs the allowed gate and finds high candidate recovery
  (`0.9539`) but dirty role-negative controls.
- Report 091 localizes every dirty-control hit to exact target equality at the
  retrieved scene's unbound wrong-role atom. The residual is dominated by
  `<UNK>`, high-frequency atoms, same-row repeats, and shuffled-role fixed
  points.

This is positive evidence for a non-synthetic native source, but it is not a
clean solved source mechanism and not Phase 5 graduation.

## Decision

Do not run another candidate/control gate until a source/control hygiene
preflight shows that a stricter natural-source protocol is feasible.

The next run is preflight-only. It may inspect source rows, query plans, atom
frequencies, same-row duplicates, and control exact-match opportunities. It
must not run scene-MHN retrieval, content cleanup, top1, or a new gate.

## Proposed Cleanup Protocol

Use the existing Report 089 source rows and selected observed-role plans.

Preflight fixed candidate query schedules under these constraints:

- target atom is not special (`<UNK>` or `<MASK>`);
- target atom appears only once in its 16-role source row;
- target source-row frequency is capped by a fixed grid:
  `none`, `512`, `256`, `128`, `64`, `32`;
- query role is held out from its own observed context;
- role-negative controls are evaluated for exact-match opportunity before
  retrieval:
  - `random_role`: `(query_role + 1) % K_roles`;
  - `deranged_role`: fixed derangement, no fixed points;
  - `fixedpoint_free_shuffled_role`: fixed derangement generated from the
    shuffled-control seed family if a future gate is proposed.

The existing Report 090 `shuffled_role` permutation should be reported as a
legacy comparison only, because Report 091 showed fixed points inflate that
control for natural sources.

## Pass Criteria

For a candidate cleanup protocol to be eligible for a later gate precommit:

- every seed has at least `n_queries=512` eligible query triples;
- selected query plans contain exactly `512` queries per seed;
- selected targets have `special_fraction=0.0000`;
- selected targets have `same_row_duplicate_fraction=0.0000`;
- same-scene exact-match opportunity is `0.0000` for `random_role`,
  `deranged_role`, and `fixedpoint_free_shuffled_role`;
- source artifact SHA and gate artifact SHA are recorded.

This preflight may recommend the strictest passing frequency cap, but that
recommendation is not permission to run a gate. A separate gate precommit is
required before any candidate/control top1 run.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No adaptive source selection or metric-triggered fallback.
- No changed-source or changed-control gate until this preflight is reported
  and a new gate precommit is written.
