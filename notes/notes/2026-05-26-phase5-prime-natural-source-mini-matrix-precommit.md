---
date: 2026-05-26
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Natural-Source Mini-Matrix Precommit

## Status

Report 095 ended the cleaned natural-source protocol drill-down chain. The
Report 093 gate now has byte-identical reusable-path parity, so another local
cleanup parity diagnostic would be scope drift.

The next bounded follow-up is a broader natural-source mini-matrix preflight
for the same cleaned protocol, not a full matrix, M2, Phase 5 headline run, or
graduation claim.

## Decision

Add a preflight-only planner before any new retrieval gate.

The planner should freeze:

- source rows from `trajectory_native_provenance_context_trace`;
- the cleaned `non_special_unique_target_freq_le_32` selected query plan;
- cue-noise values `{0.0, 0.05, 0.10, 0.15}`;
- the fixed hard cell `D=4096`, `K_roles=16`, `N=512`,
  `context_roles=4`, `scene_token_weight=0.25`;
- candidate, role-negative, positive-control, and no-scene-token baseline
  conditions.

## Scope

The preflight artifact may contain support checks, selected query plans,
control exact-opportunity summaries, condition/cell declarations, and source
SHA metadata.

It must not run scene-MHN retrieval, content cleanup, top1 scoring, or the
Phase 5 `Delta E` headline.

## Gate Policy

A passing mini-matrix preflight permits a separate fixed gate precommit for
the exact planned cells. It does not itself authorize:

- a full all-controls matrix;
- M2;
- a Phase 5 `Delta E` headline run;
- a graduation claim;
- adaptive source selection;
- metric-triggered fallback;
- best-of-N condition selection.

## Anti-Homunculus Check

The planner only declares fixed source rows, fixed query plans, fixed controls,
and fixed cue-noise values. No metric chooses a route, source, fallback, or
condition during a run. Later retrieval, if executed, must consume this plan as
a static design rather than changing behavior based on observed top1 or
energy.
