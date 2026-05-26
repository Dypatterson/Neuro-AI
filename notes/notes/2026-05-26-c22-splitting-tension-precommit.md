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

# C.2.2 — Bimodality → Splitting-Tension Energy (Precommit)

## Status

Design precommit for Path C C.2.2 per
[2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md).

**This note specifies a proposed slow-timescale dynamic.** It does NOT
authorize implementation. The Path C precommit's H5 / H6 discipline
binds; anti-homunculus reviewer must PASS before code lands.

The next step after this note lands is the reviewer pass.

## Paired Diagnostic (already landed)

**C.1.2 bimodality** =
hand-rolled Hartigan dip test (primary) or 1D GMM-BIC (fallback) on the
consecutive-cosine signal derived from a per-atom `context_bag_history`.
Persistent bimodality flag: ≥3/5 rejections in a rolling window of
recent consolidation events.

Shipped at [`src/energy_memory/phase3/bimodality_diagnostic.py`](../../src/energy_memory/phase3/bimodality_diagnostic.py),
10/10 tests pass. Standalone — accepts `ContextBagHistory` as input,
does not assume substrate-side history.

## Proposed Slow-Timescale Dynamic

### Form (right-hand-column per 2026-05-09:140-148)

Per-atom **splitting-tension scalar** `T_k` is an EMA accumulator over a
substrate-native bimodality signal. The signal source: the **ratio of
the top two eigenvalues of the per-basin covariance**:

```
bimodality_signal(k) = λ_2(Σ_k) / (λ_1(Σ_k) + ε_T)
```

where `λ_1 ≥ λ_2 ≥ … ≥ 0` are the eigenvalues of `Σ_k` (the centered
Hermitian Gram of basin members assigned to atom k — the same primitive
C.2.1 reads). `ε_T > 0` is a numerical offset that prevents division by
zero when the basin is collapsed.

Interpretation: when the basin is *unimodal* (one dominant cluster),
`λ_1 ≫ λ_2`, so the ratio is near zero. When the basin is *bimodal*
(two clusters of comparable mass along different axes), `λ_1 ≈ λ_2`, so
the ratio approaches 1. This is the **spatial** bimodality measurement —
distinct from C.1.2's **temporal** dip test on consecutive context
cosines. They are two views of the same "the basin wants to bifurcate"
structure.

The tension accumulates:

```
T_k ← (1 − μ_T) · T_k + μ_T · bimodality_signal(k)
```

per consolidation event. `μ_T ∈ (0, 1]` is a fixed substrate constant
EMA rate. `T_k` lives in `[0, 1]` and grows with sustained spatial
bimodality.

### Consolidation modulation (the actuator's effect on dynamics)

Atom k's consolidation update is **attenuated** by a continuous factor
that decreases as `T_k` grows:

```
update_modulated(atom_k) = update_base(atom_k) · (1 / (1 + T_k / τ_T))
```

where `τ_T > 0` is a fixed substrate constant scale. When `T_k → 0` the
factor is 1 (no attenuation, atom consolidates normally). When
`T_k → 1` the factor approaches `1 / (1 + 1/τ_T)`; with `τ_T = 0.5` this
is `1/3` — consolidation slows to a third of baseline. **No discrete
threshold; the modulation is smooth in `T_k`.**

The physical interpretation: an atom under persistent bimodality stress
becomes *more plastic* — it resists rapid consolidation, leaving room
for future Phase 5 splitting to act on a less-frozen representation.
This is the build-up half of the actuator. The split-actuator itself is
deferred to Phase 5 per
[`phase-3-deep-dive.md:147-171`](../emergent-codebook/phase-3-deep-dive.md)
("Phase 3 only tracks the signal — splitting itself fires in Phase 5").

### Diagnostic ↔ Actuator Identity

Per 2026-05-09:150-154:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of.

For C.2.2:

- **Slow-timescale dynamic (actuator, this precommit):** the time-EMA
  of the spatial bimodality signal `λ_2(Σ_k) / λ_1(Σ_k)`, modulating
  atom k's consolidation rate.
- **Fast-timescale snapshot (the actuator's own diagnostic):** the
  bimodality signal itself at any single consolidation event,
  computable directly from `Σ_k`. **This is the snapshot the actuator
  is the slow-timescale dynamic of.** Already available via C.2.1's
  `_basin_covariance` substrate primitive; no new primitive is needed.
- **Complementary diagnostic (C.1.2, NOT the actuator's own snapshot):**
  the Hartigan dip on the *temporal* sequence of context bags
  (`ContextBagHistory`). This measures a structurally different
  bimodality phenomenon — time-domain bimodality of context
  trajectories — versus C.2.2's space-domain bimodality of basin
  geometry. **The 2026-05-09 strict identity is between C.2.2 and its
  own spatial snapshot, NOT between C.2.2 and C.1.2.** C.1.2 is a
  complementary measurement of a motivationally-aligned but
  statistically distinct phenomenon. The actuator's anti-homunculus
  credentials rest on H6 (reads `Σ_k`, not the diagnostic) and on the
  smooth-modulation form — not on a strict identity with C.1.2.

**Watch-edge recorded by the anti-homunculus reviewer 2026-05-26:** the
earlier draft of this section overstated the C.1.2-C.2.2 identity. The
above wording is the corrected version; the structural soundness was
never in question, only the rhetorical framing.

The actuator does not read either diagnostic. It reads `Σ_k` from
substrate, computes eigenvalues, accumulates the EMA, and modulates
consolidation — all using substrate primitives.

## Anti-Homunculus Self-Check (precommit, pre-reviewer)

| Question | Answer |
|---|---|
| Is there a supervisor module that decides whether to attenuate consolidation? | No. Attenuation is `1 / (1 + T_k / τ_T)`, a continuous function. Always applied; no "fire / don't fire" branch. |
| Is there an `if X then Y` rule on a metric? | No. The signal, the EMA, and the modulation are all continuous functions of `Σ_k`. |
| Is `μ_T` or `τ_T` or `ε_T` adaptive? | NO. All three are fixed substrate constants. |
| Does the actuator read C.1.2's `BimodalityDiagnostics`? | NO. It reads `Σ_k` (substrate primitive). |
| Does the actuator share a cache with C.1.2? | NO. C.1.2 reads its own `ContextBagHistory`; the actuator reads `Σ_k`. Distinct primitives. |
| Could compositions with other consolidation forces produce a hidden if/then? | Borderline: if `τ_T` is set very small, the attenuation factor approaches a step function at `T_k = τ_T`. The smoothness assertion in the convergence-equivalence test must verify the trajectory does not show discontinuous jumps. |
| Does this risk an interaction with C.2.1? | C.2.1's anti-collapse force is *additive* to the update; C.2.2's modulation is *multiplicative*. When both fire, the modulated update is `(update_base + anti_collapse_force) · attenuation`. C.2.1's force is anti-collapse (pushes atom away from centroid); C.2.2's modulation slows the *net* update including C.2.1. There's no direct interference: the two mechanisms operate on different aspects of the dynamics. |

### Concrete H6 check

`Σ_k` is read directly via `ConsolidationState._basin_covariance(atom_idx)`
(introduced in C.2.1 as a substrate primitive). C.1.2 reads its own
`ContextBagHistory` argument; it does not consume `ConsolidationState`.
Two views of substrate, no shared cache between actuator and diagnostic. ✓

### Concrete H5 check (right-hand-column form)

The dynamic is `T_k ← (1−μ_T)·T_k + μ_T · signal`; the modulation is
`1 / (1 + T_k/τ_T)`. Both are continuous functions, applied always. No
threshold-trigger; the build-up of plasticity is a force-field shaping,
not a rule. ✓

## Construction Constants (pre-committed; non-adaptive)

| Constant | Default | Rationale |
|---|---|---|
| `mu_T` (tension EMA rate) | `0.0` (off) | Mirrors the project convention: default zero preserves prior substrate reproducibility byte-identically. Smoke runs exercise `mu_T > 0`. |
| `tau_T` (attenuation scale) | `0.5` | Sets the attenuation curve: at `T_k = 0.5`, update is attenuated by factor `0.5`. Default chosen so the modulation is meaningful but never reaches a step function. Smoke sweep will confirm trajectory smoothness; sweep is **structural**, not against any Phase 5 metric. |
| `epsilon_T` (numerical offset) | `1e-6` | Prevents `0/0` when basin is collapsed. Identical role to `ε_ac` in C.2.1. |
| `min_basin_for_signal` | `4` | Minimum basin members required to compute `λ_2`. With fewer members the signal returns 0 (treat as unimodal). Per the C.1.1 finite-sample finding, even 4 samples is the floor where eigenvalue ratios are meaningfully resolvable. |

`mu_T`, `tau_T`, `epsilon_T`, and `min_basin_for_signal` are added to
`ConsolidationConfig`. Defaults preserve κ=0 control.

## Per-Atom State

A new per-atom scalar tensor on `ConsolidationState`:

```
self.splitting_tension: torch.Tensor  # shape [n_patterns], float32
```

Initialized to zeros on `add_pattern()`. Pruned with the atom on
`prune()`. Recorded in the aggregate diagnostics dict alongside
`metastability_ema_mean`.

A new method on `ConsolidationState`:

```
def update_splitting_tension(self) -> None:
    """EMA-update T_k from substrate-side basin covariance.
    Early-exit at mu_T == 0 (κ=0 byte-identical baseline)."""
```

Called once per consolidation event (after pull/push, before
anti-collapse, so anti-collapse's effects are reflected in next event's
`Σ_k`).

A new method on `ConsolidationState`:

```
def splitting_tension_modulation(self, atom_idx: int) -> float:
    """Return the attenuation factor `1 / (1 + T_k / τ_T)` for atom k.
    Returns 1.0 when mu_T == 0 (early-exit)."""
```

The orchestrator multiplies the consolidation update by this factor at
the per-atom application site.

## Required Tests (before C.2.2 lands)

### Anti-homunculus reviewer pass
Mandatory before any code.

### Convergence-equivalence test (substrate-pure CP8)

1. Build a synthetic codebook (K=4 atoms at D=128) with atom 0's basin
   constructed to be **bimodal** (two equal-mass clusters at distinct
   angles); atoms 1-3 have unimodal basins.
2. Run consolidation with `mu_T = 0.0` for 50 events; record per-atom
   `T_k` (should remain 0) and codebook trajectory.
3. Reset; run with `mu_T = 0.1, tau_T = 0.5` for 50 events; record per-atom
   `T_k` and trajectory.

**Assertion 1 — tension accumulates on bimodal atoms:** `T_0` after 50
events with `mu_T = 0.1` is strictly greater than `T_0` with `mu_T = 0.0`
(which stays at 0). Also strictly greater than `T_1, T_2, T_3` (which
have unimodal basins).

**Assertion 2 — tension stays low on unimodal atoms:** `T_1, T_2, T_3`
each stay below `0.2` after 50 events (some natural noise allowed).

**Assertion 3 — no controller artifacts:** per-event Δ`T_0` trajectory
across the 50 events is smooth (max/median ratio < 3, same heuristic as
C.2.1 Assertion 2).

**Assertion 4 — κ=0 byte-identical baseline:** with `mu_T = 0.0`, codebook
state after 50 events is exactly equal (float32) to the pre-C.2.2 codebook
state after 50 events.

**Assertion 5 — H6 verified at runtime (binding per reviewer 2026-05-26):**
monkey-patch BOTH
`energy_memory.phase3.bimodality_diagnostic.compute_bimodality_diagnostics`
AND `energy_memory.phase3.bimodality_diagnostic.ContextBagHistory`
to raise. C.2.2 dynamic runs unchanged in both cases. (The dual patch is
binding because H11 also forbids reading `ContextBagHistory`, not just
`BimodalityDiagnostics`. The actuator must not depend on either symbol.)

**Assertion 6 — consolidation modulation is continuous:** at moderate
`T_k`, the modulation factor is between 0 and 1; no step. Specifically,
across `T_k` values `{0, 0.1, 0.2, 0.5, 0.9}`, modulation factors should
be monotonically decreasing and the differences between consecutive
factors should differ by no more than 2× (smoothness check).

**Assertion 7 — C.2.1 × C.2.2 composition has no hidden phase
transition (binding per reviewer 2026-05-26, watch-edge #1):** Construct
a **pathological basin** for atom 0 — two tight clusters placed far
apart (small total `tr(Σ_0)` from each cluster being tight, *but* high
`λ_2/λ_1` from inter-cluster mass split). Run a 2×2 ablation over
`(λ_ac, μ_T) ∈ {(0, 0), (0.5, 0), (0, 0.1), (0.5, 0.1)}` for 50
consolidation events each. Record `tr(Σ_0)` and `T_0` trajectories per
cell. **Assertion:** in the `(0.5, 0.1)` (both-on) cell, the
per-event Δ for both `tr(Σ_0)` and `T_0` shows no max/median ratio
greater than the single-mechanism cells' max ratio + 1.0 (i.e., the
composition does not introduce a phase transition the single mechanisms
don't have). If a phase transition emerges, fail — the composition is
not principled.

### Unit tests

- `mu_T = 0.0` early-exits cleanly (no eigenvalue computation, no
  EMA update, no modulation read).
- Tension stays in `[0, 1]` after many updates (EMA convergence to a
  bounded signal).
- Modulation factor is exactly 1.0 when `T_k = 0`.
- Modulation factor is in `(0, 1)` when `T_k > 0`.
- Eigenvalue computation handles complex (FHRR) tensors correctly (use
  Hermitian eigendecomposition — `torch.linalg.eigh` on the Gram matrix
  yields real eigenvalues for the centered Hermitian Gram).

## What This Does Not Permit

- Implementation. No code lands until the anti-homunculus reviewer
  passes this note.
- Per-atom context_bag_history substrate primitive. C.2.2's actuator
  uses the eigenvalue-ratio path, NOT a substrate-side parallel of
  C.1.2's history. (If a future C.2.2.b proposes the history path, it
  is a separate precommit + reviewer pass.)
- Splitting itself. Per the spec, splitting is deferred to Phase 5.
  C.2.2 builds the tension; Phase 5 consumes it.
- Tuning `mu_T` or `tau_T` against any Phase 5 / functional metric.
  The structural smoke sweep is against trajectory smoothness only.

## Anti-Homunculus Discipline Notes (specific to this mechanism)

H1–H4 from `phase-5-prime-checklist.md` apply.
H5–H9 from the Path C precommit and C.2.1 precommit apply. In addition:

- H10 (C.2.2-specific) — `mu_T`, `tau_T`, `epsilon_T`, and
  `min_basin_for_signal` must never become functions of any substrate
  observable.
- H11 (C.2.2-specific) — The actuator must not read C.1.2's
  `BimodalityDiagnostics` or `ContextBagHistory`. It reads `Σ_k`
  (substrate primitive shared with C.2.1).
- H12 (C.2.2-specific) — The consolidation modulation must be applied
  via multiplicative attenuation of the existing update, not as a new
  additive force. Replacing the update entirely (rather than
  attenuating it) would change the actuator's character from
  "make-plastic" to "override," which is closer to controller-shape.

## Implementation Findings (2026-05-26 — C.2.2 landed)

- **Status:** all 7 binding assertions pass at 14/14; full regression
  on Path C + C.2.1 + phase4 + replay clean (111 tests).
- **Anti-homunculus reviewer PASS recorded with 6 watch-edges**; all 6
  held during implementation. The two reviewer-binding watch-edges
  (A5 dual-patch H6; A7 2×2 ablation) were translated into binding test
  assertions with no relaxation.
- **A1 (tension accumulates on bimodal atom):** bimodal atom 0 T_0 =
  0.619 at end of 50 events; unimodal atoms 1-3 T_k ∈ [0.013, 0.020].
  ~30× contrast.
- **A3 (trajectory smoothness):** median Δ`T_0` = 1.29e-3, max = 3.33e-3,
  ratio = 2.58 < 3.0 (binding threshold).
- **A4 (κ=0 byte-identity):** confirmed by `torch.equal()` regardless
  of whether `splitting_tension` is pre-populated; modulation early-exits
  to 1.0 when μ_T=0.
- **A5 (dual-patch H6):** confirmed — with BOTH
  `compute_bimodality_diagnostics` AND `ContextBagHistory` monkey-patched
  to raise, the codebook and `T_k` trajectory are byte-identical to the
  unpatched run. H11 verified at runtime.
- **A6 (modulation continuous):** factors at T ∈ {0, 0.1, 0.2, 0.5, 0.9}
  are [1.000, 0.833, 0.714, 0.500, 0.357]; monotone decreasing,
  consecutive ratios ≤ 2×.
- **A7 (2×2 ablation, principled composition):**
  - `tr(Σ_0)` max/median ratios across cells `(λ_ac, μ_T)`:
    `(0,0)=11.729, (0.5,0)=11.729, (0,0.1)=11.729, (0.5,0.1)=11.729`.
    The eigenvalue ratio is a feature of basin geometry, invariant to
    which dynamic is active. **No phase transition.**
  - `T_0` max/median ratios:
    `(0,0)=1.0, (0.5,0)=1.0, (0,0.1)=12.55, (0.5,0.1)=12.54`.
    Both-on is *slightly lower* than T-only (12.54 vs 12.55) —
    anti-collapse and splitting-tension compose principally, not
    pathologically.
- **Operationalization used (exactly the precommit spec):**
  - Signal: `λ_2(Σ_k) / (λ_1(Σ_k) + ε_T)` via
    `torch.linalg.eigvalsh` (on-device throughout).
  - EMA: `T_k ← (1−μ_T)·T_k + μ_T·signal`.
  - Modulation: `update *= 1 / (1 + T_k / τ_T)`.
- **Files modified:**
  - [`src/energy_memory/phase4/consolidation.py`](../../src/energy_memory/phase4/consolidation.py):
    added `mu_T`, `tau_T`, `epsilon_T`, `min_basin_for_signal` to
    `ConsolidationConfig`; `splitting_tension` tensor state with
    grow/prune; `_spatial_bimodality_signal`,
    `update_splitting_tension`, `splitting_tension_modulation`
    methods; aggregate diagnostics keys.
  - [`src/energy_memory/phase34/online_codebook.py:136, 165`](../../src/energy_memory/phase34/online_codebook.py):
    `update_splitting_tension` called once/event (line 136);
    modulation applied to combined update via pre-state
    snapshot + post-update blend (`_apply_splitting_tension` at
    line 215).
  - [`src/energy_memory/phase34/stable_online_codebook.py:235, 271`](../../src/energy_memory/phase34/stable_online_codebook.py):
    parallel wire-up in V2's overridden `_consolidate`.
- **Files created:** [`tests/test_splitting_tension.py`](../../tests/test_splitting_tension.py) (14 tests).
- **Geometry interpretation noted by subagent:** "unimodal" in C.2.2's
  spatial context = rank-1 (λ_1 ≫ λ_2, signal ≈ 0). "Bimodal" =
  rank-≥2 (λ_1 ≈ λ_2, signal ≈ 1). This is the right reading for the
  eigenvalue-ratio measurement; the precommit's "bimodal basin"
  terminology refers to spatial rank, not temporal switching of
  context bags.
- **C.2.2 closed. Next: C.2.3 (cap-coverage failure → local error
  gradient on consolidation).**
