---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Cleaned Natural Source Integration

## Status

Report 093 is the first clean non-synthetic native provenance recovery:

`trajectory_native_provenance_context_trace`, protocol
`non_special_unique_target_freq_le_32`

Candidate remains high (`0.9512`, CI `[0.9449,0.9567]`) while role-negative
controls are clean:

- `random_role=0.0000`;
- `deranged_role=0.0004`;
- `fixedpoint_free_shuffled_role=0.0010`;
- `content_cleanup_positive=1.0000`.

This resolves the Report 090 dirty-control caveat for the cleaned natural
source protocol. It is still diagnostic evidence, not Phase 5 graduation.

## Decision

Integrate the cleaned natural-source protocol as reusable Phase 5′ machinery
before any broader experiment.

The next work should be a code integration and parity smoke, not a new
candidate/control gate and not a matrix.

## Integration Scope

Move report-local protocol code into reusable, tested modules:

1. **Natural source query planning**
   - build eligible query triples from fixed source rows and selected observed
     roles;
   - exclude special targets (`<UNK>`, `<MASK>`);
   - exclude same-row duplicate target atoms;
   - apply a fixed target-frequency cap;
   - select deterministic `n_queries` plans per seed.
2. **Fixed role-negative controls**
   - retain `random_role`;
   - retain `deranged_role`;
   - add fixed-point-free shuffled control generated from the shuffled-control
     seed family;
   - keep legacy shuffled-role only as an explicit diagnostic comparison, not
     the primary natural-source control.
3. **Manifest and parity checks**
   - record source artifact SHA;
   - record query-plan/protocol SHA;
   - require parity against Report 092 support/exact-opportunity counts;
   - require parity against Report 093 aggregate top1/CI when the gate is run
     from the reusable path.

## First Implementation Target

Add a reusable module under `src/energy_memory/phase5/`, for example:

`src/energy_memory/phase5/natural_source_protocol.py`

The module should not depend on report paths. Report scripts can import it,
but production-style code should receive source rows, selected observed roles,
and protocol config as data.

## Required Tests

- Unit tests for special-token exclusion.
- Unit tests for same-row duplicate exclusion.
- Unit tests for deterministic query selection.
- Unit tests that fixed-point-free shuffled controls have no fixed points.
- Unit tests for same-scene exact-opportunity rates.
- A parity smoke that reproduces Report 092 selected-plan support statistics.

## Gate Policy

No new gate is allowed until the reusable path reproduces Report 092 support
and exact-opportunity diagnostics.

If a gate is rerun after refactor, it must be a parity gate for Report 093:

- same source rows;
- same selected query plan;
- same seeds;
- same five conditions;
- same result readout;
- no matrix, no M2, no headline, no graduation claim.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No adaptive source selection or metric-triggered fallback.
- No learned or production-memory claim from Report 093 alone.
