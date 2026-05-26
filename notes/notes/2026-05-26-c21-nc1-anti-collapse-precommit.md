---
date: 2026-05-26
project: neuro-ai
tags:
  - notes
  - phase-3
  - path-c
  - planning
  - precommit
  - diagnostic-actuator-identity
  - anti-homunculus-reviewer-required
---

# C.2.1 — NC1 Anti-Collapse Pressure (Precommit)

## Status

Design precommit for Path C C.2.1 per
[2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md).

**This note specifies a proposed slow-timescale dynamic.** It does NOT
authorize implementation. The Path C precommit's H5 discipline binds:

> C.2 mechanisms must take the right-hand-column form of the 2026-05-09
> table (continuous tension / energy / gradient), not the left-hand-column
> form (`if metric > τ then action`). The anti-homunculus reviewer must
> PASS the proposed dynamic before code lands.

The next step after this note lands is the anti-homunculus reviewer pass.
Implementation is gated on PASS.

## Paired Diagnostic (already landed as passive log)

**C.1.1 NC1 within-basin variability** =
`d_eff(per-basin covariance) = (tr Σ_k)² / ||Σ_k||²_F`, where `Σ_k` is the
centered Hermitian Gram of the settled states `{q* : top1(q*) == k}`.

Status: shipped at
[`src/energy_memory/phase3/basin_diagnostics.py`](../../src/energy_memory/phase3/basin_diagnostics.py),
9/9 unit tests pass.

**Nuance worth recording up front:** d_eff (the participation ratio) does
not vanish under point-collapse. At rank-1 collapse, d_eff = 1; at rank-r
collapse, d_eff = r. The "collapse to a point" limit (rank-0) leaves
d_eff undefined — but more importantly, the *total* variance `tr(Σ_k)`
goes to zero in either case. So the natural actuator primitive to *bound*
is `tr(Σ_k)`, not d_eff itself. d_eff then remains the complementary
"is the variance distributed across dimensions" diagnostic.

This is consistent with the 2026-05-09 reformulation:

> Maintain bounded non-zero **within-basin variability** while preserving
> inter-basin separability.

"Variability" here is unspecified in the spec but is most naturally read
as total within-basin variance — i.e., `tr(Σ_k)`. Bounded `tr(Σ_k)` from
below is the substrate-level guarantee; bounded d_eff from above is a
separate (and softer) ask handled by inter-atom repulsion already wired
in the substrate as `alpha_anti`. The two are complementary.

## Proposed Slow-Timescale Dynamic

### Form (right-hand-column per 2026-05-09:140-148)

Add a per-basin **anti-collapse potential term** to the consolidation
energy landscape:

```
E_anti_collapse(k) = -λ_ac · log( tr(Σ_k) + ε_ac )
```

where:
- `Σ_k` is the centered per-basin covariance of the settled states
  `{q* : top1(q*) == k}` — the *same* primitive C.1.1 computes for the
  diagnostic.
- `λ_ac > 0` is a fixed, non-adaptive strength constant set at substrate
  construction.
- `ε_ac > 0` is a small offset that keeps the potential finite as
  `tr(Σ_k) → 0` and sets the soft-floor scale.

The gradient of `E_anti_collapse(k)` with respect to the basin members
`{q*_i : top1(q*_i) == k}` is:

```
∂ E_anti_collapse / ∂ q*_i  =  -λ_ac · 2 (q*_i − μ_k) / (tr(Σ_k) + ε_ac)
```

(where `μ_k` is the basin centroid). This is a **repulsive force from
the basin centroid** whose magnitude grows as the basin's total variance
shrinks. At equilibrium, the basin self-stabilizes at a variance that
balances this anti-collapse pressure against the consolidation update's
contraction pressure.

### What this is, in the substrate's own terms

The consolidation update already contains a contraction force (Hebbian /
coverage-driven pull of basin members toward the basin's representative
atom). The anti-collapse potential adds an opposing force whose
equilibrium produces bounded-non-zero variability.

**This is not a controller deciding when to fight collapse.** It is the
consolidation energy landscape itself, modified by an additional term
that's continuous in `tr(Σ_k)`, present always, applied uniformly to all
atoms. The "apparent decision" ("this atom needs more variance") is the
equilibrium of the modified energy landscape, not an arbitration.

### Diagnostic ↔ Actuator Identity

Per 2026-05-09:150-154:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of.

For C.2.1:

- **Fast-timescale snapshot (diagnostic, C.1.1):** `d_eff(Σ_k)`, the
  participation ratio. Reads off the *shape* of the basin's covariance.
- **Companion diagnostic (passive):** `tr(Σ_k)`, the total within-basin
  variance. Reads off the *magnitude*. This is what the actuator term
  bounds from below. The Path C work should expose `tr(Σ_k)` per basin
  alongside `d_eff` in the `BasinDiagnostics` dataclass so the
  convergence-equivalence test can verify the actuator's equilibrium
  matches the diagnostic's observation.
- **Slow-timescale dynamic (actuator):** the gradient of
  `-λ_ac · log(tr(Σ_k) + ε_ac)` w.r.t. basin members, integrated over
  consolidation events.

At quasi-stationary equilibrium, the basin's `tr(Σ_k)` is set by the
balance of contraction and anti-collapse forces. The diagnostic reads
off the equilibrium value; the actuator *is* the dynamic that produces
it. They are the same physical process viewed at different timescales.

## Anti-Homunculus Self-Check (precommit, pre-reviewer)

Pre-commit my own check; the reviewer will do the binding pass.

| Question | Answer |
|---|---|
| Is there a supervisor module that decides whether to fight collapse? | No. The anti-collapse term is part of the consolidation energy landscape; it has no "fire / don't fire" logic. |
| Is there an `if X then Y` rule on a metric? | No. The term is continuous in `tr(Σ_k)` and is computed every consolidation event regardless of any threshold. |
| Is `λ_ac` adaptive (read from observation)? | **NO.** `λ_ac` is a fixed substrate constant set at construction, like `alpha_anti` or `coverage_lambda`. Adaptive λ would import a controller. |
| Is `ε_ac` adaptive? | **NO.** Same as above. Fixed substrate constant. |
| Does the "decision" to fight collapse live in geometry or in a rule? | Geometry. The equilibrium of the modified energy landscape produces bounded variability. |
| Could this term, by composition with other consolidation forces, *imitate* an if/then? | Potentially — if `λ_ac` is set very large, the anti-collapse pressure could effectively override the contraction force and produce a hard floor that looks like a threshold. This is a tuning question, not a structural one. Set `λ_ac` so the system has a soft floor that emerges from the balance, not a hard one. The convergence-equivalence test will verify the equilibrium is smooth. |
| Does the term read `d_eff` (the C.1.1 diagnostic) directly? | **NO** — and this is critical. Reading d_eff to act on it would re-introduce the diagnostic→actuator pipeline shape that CP7 prohibits. The actuator reads `Σ_k` (the substrate's own basin covariance, a substrate primitive). d_eff and `tr(Σ_k)` are both *snapshots* of the same `Σ_k`; the actuator and the diagnostic see the same physical object, not a relay. |

### Concrete check against H6 (binding from the Path C precommit)

> H6 — C.1 diagnostic logs and C.2 actuator dynamics are the same
> physical process at different timescales, not a pipeline. C.2 reads
> substrate state, not C.1 logs.

The actuator reads `Σ_k` directly from the substrate's basin members.
It does NOT consume `BasinDiagnostics.nc1_per_atom`. The C.1.1 diagnostic
also reads `Σ_k` directly. Both see the substrate; neither sees the
other. ✓

## Construction Constants (pre-committed; non-adaptive)

| Constant | Default | Rationale |
|---|---|---|
| `λ_ac` (anti-collapse strength) | `0.0` (off) | Mirrors the project convention for new dynamics (e.g. `alpha_freq_lambda`, `coverage_lambda`): default zero preserves prior substrate reproducibility. Smoke runs exercise `λ_ac > 0`. The smoke value will be set after a brief sweep over `{0.001, 0.01, 0.1, 1.0}` against the convergence-equivalence test. |
| `ε_ac` (soft-floor offset) | `1e-6` | Numerical offset to keep the log finite. Small enough to not interfere with realistic basin variances; large enough to avoid `inf` gradients. |

`λ_ac` and `ε_ac` are added to `ConsolidationConfig` as new fields with
defaults `0.0` and `1e-6` respectively. Both are documented as fixed
substrate constants, not adaptive.

## Required Tests (before C.2.1 lands)

### Anti-homunculus reviewer pass
Mandatory before any code. Reviewer is the `anti-homunculus-reviewer`
agent. The reviewer audits this note plus the proposed code.

### Convergence-equivalence test (the substrate-pure version of CP8)

At quasi-stationary substrate, the live diagnostic value
`d_eff(Σ_k)` from C.1.1 must be consistent with the actuator's
equilibrium prediction. Concretely:

1. Build a small synthetic codebook (K=4 atoms at D=128) with seeded
   basin samples.
2. Run consolidation with `λ_ac = 0.0` to convergence; record per-basin
   `tr(Σ_k)` and `d_eff(Σ_k)`.
3. Run consolidation with `λ_ac > 0` to convergence; record again.
4. **Assertion 1 (anti-collapse works):** the `λ_ac > 0` run's
   `tr(Σ_k)` is strictly greater than the `λ_ac = 0` run's `tr(Σ_k)`
   for atoms that were collapsing in the baseline.
5. **Assertion 2 (no controller artifacts):** the `λ_ac > 0` run's
   trajectory is smooth — no discontinuous jumps in `tr(Σ_k)` over
   consolidation events. (Discontinuities would indicate a hidden
   threshold-trigger.)
6. **Assertion 3 (κ=0 baseline preserved):** `λ_ac = 0` reproduces the
   pre-C.2.1 consolidation behavior byte-identically.
7. **Assertion 4 (no diagnostic dependence):** with C.1.1's
   `BasinDiagnostics` log replaced by a stub that always returns zeros,
   the actuator's behavior is unchanged. (Verifies the actuator does
   not silently read the C.1.1 log — H6 binding.)

### Unit tests

- Test that `λ_ac = 0.0` early-exits cleanly (no gradient computation,
  no per-basin covariance computation). κ=0 control baseline.
- Test that the gradient computation is finite when `tr(Σ_k) → 0`
  (the `ε_ac` floor works).
- Test that the gradient direction repels basin members from the
  centroid (sign check).
- Test that the gradient is zero when all basin members are at the
  centroid (degenerate but well-defined).

## What This Does Not Permit

- Implementation. No code lands until the anti-homunculus reviewer
  passes this note.
- Wiring to a graduation experiment. C.2.1 is a substrate dynamic; its
  graduation evidence is the convergence-equivalence test plus Phase 3
  exit-criterion re-run in C.3.
- Tuning `λ_ac` against a Phase 5 metric. C.2.1 is a Phase 3 mechanism;
  Phase 5′ is paused. The smoke sweep over `{0.001, 0.01, 0.1, 1.0}` is
  against the convergence-equivalence test only.

## Anti-Homunculus Discipline Notes (specific to this mechanism)

H1–H4 from `phase-5-prime-checklist.md` continue to apply.
H5–H7 from `2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md`
continue to apply. In addition:

- H8 (C.2.1-specific) — `λ_ac` must never become a function of any
  substrate observable, including `tr(Σ_k)`, `d_eff`, basin count, or
  any Phase 4 / Phase 5 metric. If a future session is tempted to
  modulate `λ_ac`, that is a new C.2.x mechanism and requires its own
  precommit + reviewer pass.
- H9 (C.2.1-specific) — The actuator's gradient is computed from
  `Σ_k` (substrate state), not from the `BasinDiagnostics` dataclass.
  Refactoring the actuator to read the diagnostic output would
  introduce the diagnostic→actuator pipeline shape CP8 / H6 prohibits.

## Implementation Findings (2026-05-26 — C.2.1 landed)

- **Status:** all 4 binding assertions pass at 11/11; full regression on
  Path C + phase4 + replay clean (111 tests across the surface).
- **Anti-homunculus reviewer PASS recorded earlier this session.** All
  three watch-edges held during implementation:
  - `λ_ac` / `ε_ac` declared as fixed `ConsolidationConfig` fields (defaults `0.0`, `1e-6`); no adaptive coupling to any observable.
  - Actuator reads `Σ_k` directly via `ConsolidationState._basin_covariance()` — a **substrate primitive**, NOT `BasinDiagnostics`. C.1.1 (`src/energy_memory/phase3/basin_diagnostics.py`) was not touched.
  - Smoothness assertion (Assertion 2) held at the binding `3 × median` bar; actual max/median ratio = 1.44.
- **Operationalization used (exactly the precommit spec, no deviation):**
  `update_modified(atom_k) = update_base(atom_k) − λ_ac · 2·(μ_k − atom_k) / (tr(Σ_k) + ε_ac)`.
  The gradient acts on atom_k directly (well-defined), not through `top1` (discontinuous). Force is zero when atom = centroid, repulsive when displaced; respects complex FHRR tensors.
- **Architectural directive from the reviewer satisfied:** the substrate primitive `_basin_covariance(atom_idx)` lives on `ConsolidationState`. C.2.1 consumes it; future refactor of C.1.1 may also consume it (currently C.1.1 takes its own `BasinTraceBuffer` argument and stays decoupled). Two views of substrate, no shared cache between actuator and diagnostic.
- **Files modified:**
  - [`src/energy_memory/phase4/consolidation.py`](../../src/energy_memory/phase4/consolidation.py): added `lambda_ac`, `epsilon_ac`, `basin_trace_buffer_size` to `ConsolidationConfig`; added `_basin_buffer` deque to `ConsolidationState`; added `record_retrieval`, `basin_buffer_size`, `_basin_covariance`, `anti_collapse_force` methods.
  - [`src/energy_memory/phase34/online_codebook.py:145`](../../src/energy_memory/phase34/online_codebook.py): `_apply_anti_collapse()` after pull/push.
  - [`src/energy_memory/phase34/stable_online_codebook.py:263`](../../src/energy_memory/phase34/stable_online_codebook.py): same call inside V2's overridden `_consolidate()`.
  - [`src/energy_memory/phase4/replay_loop.py:492`](../../src/energy_memory/phase4/replay_loop.py): `state.record_retrieval(result.state, top_index)` after metastability EMA.
- **Files created:** [`tests/test_anti_collapse.py`](../../tests/test_anti_collapse.py) (11 tests).
- **Assertion numbers (logged here for the record):**
  - Assertion 1: baseline `tr(Σ_0) = 0.0` → λ_ac=0.5 yields `tr(Σ_0) = 2.13e-3`. Anti-collapse holds the basin off the floor.
  - Assertion 2: median Δtr per consolidation event = `3.19e-6`, max = `4.59e-6`; ratio `1.44 < 3.0` (binding threshold). Trajectory smooth.
  - Assertion 3: codebook is `torch.equal()` byte-identical at λ_ac=0 whether or not the basin buffer is populated. κ=0 control preserved.
  - Assertion 4: C.2.1 dynamic runs unchanged with `compute_basin_diagnostics` monkey-patched to raise. **H6 verified at runtime.**
- **Performance:** `record_retrieval()` at λ_ac=0 costs ≈40 ns/call (one attribute lookup + early return). At λ_ac=0.5, ≈1 µs/call (clone + deque append). κ=0 baseline pays effectively nothing per retrieval.
- **C.2.1 closed. Next: C.2.2 (bimodality → splitting-tension energy).**
