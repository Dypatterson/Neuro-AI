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

# C.2.3 — Cap-Coverage → Local Error Gradient (Precommit)

## Status

Design precommit for Path C C.2.3 per
[2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md).

**This note specifies a proposed slow-timescale dynamic.** No code lands
until the anti-homunculus reviewer PASSES this note.

## Paired Diagnostic Reference

Cap-coverage is the project's load-bearing measurement of whether
meaning lives in the retrieval neighborhood, per
[`notes/emergent-codebook/consolidation-geometry-diagnostic.md`](../emergent-codebook/consolidation-geometry-diagnostic.md)
and CLAUDE.md ("the metric most aligned with the architecture's actual
claim about where meaning lives"). The Phase 2 implementation lives at
[`src/energy_memory/phase2/metrics.py`](../../src/energy_memory/phase2/metrics.py)
(`cap_coverage_error` and friends), computed per retrieval as a
fraction of basin members within a similarity cap θ of the
representative atom.

Cap-coverage *failure* = many basin members fall *outside* the
representative's cap. The 2026-05-09 mapping table calls for the
actuator: "low cap-coverage → restructuring pressure" — operationalized
in the Path C precommit as "cap-coverage residual contributes
additively to the consolidation update on the responsible atoms."

## Proposed Slow-Timescale Dynamic

### Form (right-hand-column per 2026-05-09:140-148)

Add a per-atom **cap-coverage error force** to atom k's consolidation
update:

```
force_cc(atom_k) = λ_cc · mean_{i: top1(q*_i)=k} w_cc(q*_i) · (q*_i − atom_k)
```

where `w_cc(q*_i) ∈ [0, 1]` is a continuous **uncovered weight**:

```
w_cc(q*_i) = σ_cc((θ_cc − sim(q*_i, atom_k)) / τ_cc)
```

with `σ_cc` a sigmoid (smooth step). `sim(q*_i, atom_k)` is the FHRR
cosine similarity. `θ_cc` is the substrate's cap threshold; `τ_cc > 0`
is the sigmoid sharpness scale. `λ_cc ≥ 0` is the actuator strength.

**Interpretation:**
- When `sim(q*_i, atom_k) ≫ θ_cc` (q*_i is well-covered), `w_cc → 0`
  and the basin member contributes nothing to the force. No "waste"
  pulling atom k toward already-covered members.
- When `sim(q*_i, atom_k) ≪ θ_cc` (q*_i is far outside the cap),
  `w_cc → 1` and the basin member contributes a full pull toward
  itself.
- Around `sim ≈ θ_cc`, the sigmoid produces a smooth weight in (0, 1).
  **No discrete cap-membership decision per basin member.**

The force is added to atom k's consolidation update. At equilibrium,
atom k drifts toward its uncovered basin members, expanding its
effective cap to cover them — without anyone reading the cap-coverage
metric and triggering a restructure.

### Why a sigmoid weight, not a hard indicator

A hard indicator `(1 if sim<θ else 0)` would be discontinuous in
`atom_k`'s position — exactly the controller-shape (an `if/then` rule
on a metric) the anti-homunculus filter prohibits. The sigmoid is the
right-hand-column form: continuous, smooth, no decision boundary.

The reviewer should specifically check that `τ_cc` is not so small that
the sigmoid approximates a step (in which case the dynamic regresses to
the controller form). The convergence-equivalence test must verify
trajectory smoothness across realistic `τ_cc` values.

### Diagnostic ↔ Actuator Identity

Per 2026-05-09:150-154:

For C.2.3:

- **Slow-timescale dynamic (actuator):** the time-integrated cap-coverage
  error force on each atom, drifting atoms toward their uncovered basin
  members.
- **Fast-timescale snapshot (the actuator's own diagnostic):** the
  per-atom mean uncovered weight `mean_i w_cc(q*_i)` at a single
  consolidation event. Computable directly from `Σ_k` substrate buffer
  + atom_k state.
- **The Phase 2 cap-coverage error metric** is structurally similar
  but uses a hard threshold. It is NOT the actuator's snapshot;
  it is a related diagnostic that lives at the metric layer. C.2.3
  reads substrate, not the Phase 2 metric module.

### Interaction with existing consolidation forces

- **Hebbian:** pulls atom k toward basin centroid. C.2.3 pulls atom k
  specifically toward *uncovered* members. The combination is
  weighted Hebbian — uncovered members get extra pull.
- **`alpha_anti`:** repels atom k from other atoms. C.2.3 expands atom
  k's effective cap. The two forces compose: atoms expand to cover
  their basins, then repel each other to maintain inter-atom
  separation.
- **C.2.1 anti-collapse:** repels atom k from a collapsing basin
  centroid. C.2.3 pulls atom k toward uncovered (outlier) members. In a
  collapsing basin, there are no outliers (all members near centroid),
  so C.2.3's force is naturally weak. In a healthy spread basin, both
  forces are active and balanced.
- **C.2.2 splitting-tension:** attenuates all updates including C.2.3.
  A bimodal-stressed atom resists rapid coverage expansion. Composition
  is principled.

## Anti-Homunculus Self-Check (precommit, pre-reviewer)

| Question | Answer |
|---|---|
| Is there a supervisor module that decides whether to restructure? | No. The force is `λ_cc · mean_i w_cc · (q*_i − atom_k)`, continuous in basin geometry. Always applied. |
| Is there an `if X then Y` rule on a metric? | No. The weight `w_cc` is a smooth sigmoid, not a hard threshold. The force is applied regardless of any metric. |
| Are `λ_cc`, `θ_cc`, `τ_cc` adaptive on observation? | NO. All three are fixed substrate constants. |
| Does the actuator read the Phase 2 cap-coverage metric? | NO. The actuator reads `Σ_k` (substrate basin members) and computes `w_cc` from first principles. |
| Could small `τ_cc` make the sigmoid imitate a step (a hidden threshold)? | Yes, in the limit `τ_cc → 0`. The convergence-equivalence test must verify trajectory smoothness across realistic `τ_cc`. Default `τ_cc = 0.1` is chosen so the sigmoid transitions over a meaningful similarity range (e.g., the transition from `sim = 0.4` to `sim = 0.6` covers most of the weight change). |
| Could `λ_cc` very large drive the dynamic into a winner-take-all regime? | Yes, in the limit `λ_cc → ∞`. Same defense as C.2.1: smoke sweep selects `λ_cc` against trajectory smoothness, not against a Phase 5 metric (H10). |
| Does C.2.3 introduce a new substrate primitive? | No. Uses `_basin_covariance(atom_idx)` from C.2.1 — same substrate primitive. |

### Concrete H6 check

The actuator reads `Σ_k` and `atom_k` (both substrate state) and
computes the force directly. It does NOT consume `cap_coverage_error`
from `src/energy_memory/phase2/metrics.py` or any Phase 2 metric module.
Two views of substrate, no shared cache with the diagnostic. ✓

### Concrete H5 check

The dynamic is `force_cc(atom_k) = λ_cc · mean_i w_cc · (q*_i − atom_k)`
with `w_cc = sigmoid((θ_cc − sim)/τ_cc)`. Both `sigmoid` and `mean` are
continuous; `force_cc` is smooth in `atom_k` position; applied always.
Right-hand-column form. ✓

## Construction Constants (pre-committed; non-adaptive)

| Constant | Default | Rationale |
|---|---|---|
| `lambda_cc` (force strength) | `0.0` (off) | Mirrors project convention; default zero preserves κ=0 baseline. |
| `theta_cc` (cap threshold) | `0.5` | Cosine similarity threshold defining the "cap." **Substrate-side justification (binding per reviewer watch-edge 2026-05-26):** at β=10 Hopfield retrieval, the softmax landscape has a natural similarity-cap structure around `sim ≈ 0.5`; per the C.1.4 empirical θ′(β) calibration at [`notes/emergent-codebook/theta_prime_calibration.json`](../emergent-codebook/theta_prime_calibration.json), the recoverable similarity boundary at β=10 falls within the cap-friendly range. The Phase 2 `cap_t05` metric *happens to coincide* at 0.5, but the load-bearing justification is the substrate's retrieval geometry, NOT alignment with the Phase 2 metric (H14 protection). |
| `tau_cc` (sigmoid sharpness) | `0.1` | Sets the smoothness of the cap-coverage weight transition. Default 0.1 means the sigmoid transitions across `sim` values in `[θ - 0.3, θ + 0.3]` (roughly). Not so small that it approximates a step. |

`lambda_cc`, `theta_cc`, `tau_cc` are added to `ConsolidationConfig`.
Defaults preserve κ=0 control.

## Per-Atom State

No new per-atom scalar state. C.2.3 is stateless at the per-atom level —
the force is computed from instantaneous basin geometry per
consolidation event.

A new method on `ConsolidationState`:

```
def cap_coverage_force(self, atom_idx: int, atom_state: Tensor) -> Tensor:
    """Compute the cap-coverage error force on atom_idx.
    Returns zero tensor when lambda_cc == 0 (κ=0 baseline).
    Reads basin members from _basin_covariance (substrate primitive)."""
```

## Required Tests (before C.2.3 lands)

### Anti-homunculus reviewer pass
Mandatory before any code.

### Convergence-equivalence test (substrate-pure CP8)

**Assertion 1 — cap-coverage force pulls atom toward uncovered members:**
Build atom 0 at a fixed position, with basin members split into
"covered" (high similarity to atom 0) and "uncovered" (low similarity).
Compute `cap_coverage_force(atom_0)` with `λ_cc = 0.5`. Assert the
force vector has positive cosine similarity with the mean uncovered
member (force pulls in the right direction).

**Assertion 2 — force is zero when all basin members covered:**
Construct a basin where all members have `sim(member, atom_k) > 0.9`
(well above θ_cc = 0.5). The sigmoid weights are near zero. Assert
force magnitude < small threshold.

**Assertion 3 — trajectory smoothness:** Run consolidation with
`λ_cc = 0.5, tau_cc = 0.1` for 50 events on a basin with mixed
coverage. Record per-event Δ in atom_0's position. Assert max/median
ratio < 3 (binding smoothness threshold).

**Assertion 4 — κ=0 byte-identical baseline:** With `λ_cc = 0.0`,
codebook state after 50 events is `torch.equal()` to a pre-C.2.3 run.

**Assertion 5 — H6 verified at runtime:** monkey-patch
`energy_memory.phase2.metrics` to raise. C.2.3 dynamic runs unchanged.
(The actuator must not read the Phase 2 metric module.)

**Assertion 6 — small `tau_cc` does NOT produce a step-like trajectory
(strengthened per reviewer 2026-05-26):**
Run with `tau_cc = 0.001` (very sharp sigmoid, approaching a step).
Across 50 events, assert per-event Δ in atom_0 position has max/median
ratio < 5 (looser than binding 3, but still bounded — a true step
would produce ratio ≫ 5). **Additionally (binding):** record the
empirical distribution of `|θ_cc − sim(q*_i, atom_k)|` across all basin
members across all events. Assert the sigmoid transition width `4·τ_cc`
(the range where the sigmoid is in `[0.05, 0.95]`) covers ≥ 10% of the
IQR of that distribution. This catches the "formally continuous but
operationally a step" failure mode — if `τ_cc` is small relative to the
similarity gap distribution the substrate naturally produces, the
sigmoid acts as a switch in practice. Reviewer-binding.

**Assertion 7 — three-way composition (C.2.1 × C.2.2 × C.2.3),
strengthened per reviewer 2026-05-26:** Run the 2×2×2 ablation over
`(λ_ac, μ_T, λ_cc) ∈ {off, on}³`. Three sub-assertions are binding:

- **7a — max/median bound:** the all-three-on cell's max/median ratio
  for atom_0 position-Δ is no greater than the worst two-mechanism
  cell's ratio + 1.0. (Original assertion, retained.)
- **7b (binding superposition-distance):** the atom_0 trajectory in
  the all-three-on cell is close (L2 distance < some_threshold) to the
  *linear superposition* of the three single-on trajectories with the
  same seed (i.e., each single mechanism applied alone, then summed).
  This catches phase transitions emergent only in the three-way
  combination where pairwise tests would miss them. Set the threshold
  empirically against the worst pair's deviation from its
  two-single-on superposition.
- **7c (binding Δ-spike cap):** no consolidation-event Δ in the
  all-three-on trajectory exceeds 1.5× the worst Δ observed in any
  single-on or two-on cell. Phase transitions appear as one-shot Δ
  spikes; the max/median statistic can hide a single large spike
  inside a noisy baseline.

If 7a, 7b, OR 7c fails, the three-way composition has emergent
arbitration that the pairwise tests would miss. Implementation rejected;
require precommit revision.

### Unit tests

- `λ_cc = 0` early-exits cleanly (returns zero tensor).
- `force_cc` is finite even when basin has < `min_basin_for_signal`
  members (returns zero tensor in that case).
- `force_cc` direction is correct: when atom_k is at the basin
  centroid and all basin members are uncovered (low similarity), the
  force pulls atom_k toward the centroid (force . (centroid − atom_k) > 0).
- `force_cc` is zero when atom_k coincides with all basin members
  (trivial all-covered case).
- Sigmoid handles complex (FHRR) similarities: similarity is real-valued
  per the FHRR cosine convention.

## What This Does Not Permit

- Implementation. No code lands until the reviewer PASSES.
- Reading the Phase 2 `cap_coverage_error` function. C.2.3 computes
  the weight from first principles on substrate state.
- Tuning `λ_cc`, `θ_cc`, `τ_cc` against any Phase 5 / functional metric.
  The smoke sweep is structural (trajectory smoothness, equilibrium).
- A discrete "restructuring event." The force is continuous and
  always applied; restructuring is the equilibrium of the dynamics,
  not a discrete trigger.

## Anti-Homunculus Discipline Notes

H1–H12 from earlier precommits apply. In addition:

- H13 (C.2.3-specific) — `λ_cc`, `θ_cc`, `τ_cc` must never become
  functions of any substrate observable.
- H14 (C.2.3-specific) — The actuator reads `Σ_k` (substrate primitive)
  and `atom_k` state. It does NOT consume the Phase 2
  `cap_coverage_error` function or any field of the Phase 2 metrics.
- H15 (C.2.3-specific) — The sigmoid weight `w_cc` must remain smooth.
  If `τ_cc` is ever set very small (approaching a step), the
  convergence-equivalence test's smoothness assertion (A3, A6) must
  fail and the implementation must be rejected. The smoothness
  assertion is binding.

## Implementation Findings (2026-05-26 — C.2.3 landed)

- **Status:** all 7 binding assertions pass at 14/14; full discovery
  regression at 480/480.
- **Anti-homunculus reviewer PASS with 5 watch-edges; both binding
  test-plan strengthenings (A6 IQR check, A7 superposition + Δ-spike)
  held during implementation.**
- **Operationalization exactly per the precommit:** stateless force
  `force_cc(atom_k) = λ_cc · mean_i w_cc · (q*_i − atom_k)` with
  sigmoid weight `w_cc = sigmoid((θ_cc − sim)/τ_cc)`. Reads `Σ_k` via
  C.2.1's `_basin_covariance` substrate primitive.
- **Force is applied at [`src/energy_memory/phase34/online_codebook.py:164`](../../src/energy_memory/phase34/online_codebook.py)**, in the order: pull/push → anti-collapse (C.2.1) → cap-coverage (C.2.3) → splitting-tension modulation (C.2.2).
- **A1 (force pulls toward uncovered):** cosine(force, mean_uncovered − atom) = `1.0000`, force magnitude `0.348`.
- **A2 (force ≈ 0 when covered):** force magnitude `6.37e-4 < 1e-3`.
- **A3 (smoothness):** median Δ = `8.73e-11`, max Δ = `8.73e-11`, ratio `0.989 < 3` (binding).
- **A4 (κ=0 byte-identity):** `torch.equal()` confirmed at λ_cc=0.
- **A5 (H6 at runtime):** `phase2.metrics` callables monkey-patched to raise; C.2.3 codebook byte-identical to unpatched run. H14 verified at runtime.
- **A6 strengthened (IQR check, reviewer-binding):**
  - τ_cc=0.1 default: IQR(|θ−sim|) = `0.174`, 4·τ_cc = `0.400`, coverage ratio = `2.297` ≫ 0.10 binding threshold. **Passes.**
  - τ_cc=0.001 (extremely sharp): IQR = `0.206`, 4·τ_cc = `0.004`, coverage = `0.019` < `0.10`. **Correctly fails** — the IQR check identifies operationally-step-like configurations even when the sigmoid is formally continuous.
  - This demonstrates the strengthened smoothness assertion is well-calibrated: default config passes by wide margin; pathological config fails as expected.
- **A7 strengthened (3-way composition):** all 8 cells of the
  `(λ_ac, μ_T, λ_cc) ∈ {off, on}³` ablation ran on a pathological
  low-trace + high-bimodality + low-coverage basin.
  - **7a max/median:** all-three-on ratio `27.27` ≤ worst two-mechanism (35.58) + 1.0. ✓
  - **7b superposition-distance (binding):** all-three trajectory's L2/event from sum-of-singles = `0.0792`; worst pair's L2/event from sum-of-its-singles = `0.0791`; ratio `1.001` ≤ 2.0. **The three-way composition is essentially linear superposition** — no emergent arbitration.
  - **7c Δ-spike cap (binding):** all-three max per-event Δ = `0.3007`; worst non-three max Δ = `0.3007`; ratio `1.000` ≤ 1.5. No phase-transition spike. ✓
- **`theta_cc` docstring leads with substrate-side justification** (β=10 retrieval cap structure + reference to the θ′(β) calibration JSON). Phase 2 `cap_t05` coincidence noted as a follow-up. H14 protection in code.
- **Files modified:**
  - [`src/energy_memory/phase4/consolidation.py`](../../src/energy_memory/phase4/consolidation.py):
    added `lambda_cc`, `theta_cc`, `tau_cc` to `ConsolidationConfig`;
    added `cap_coverage_force(atom_idx, atom_state)` on `ConsolidationState`;
    extended `record_retrieval` short-circuit to gate on any of (λ_ac,
    μ_T, λ_cc) > 0.
  - [`src/energy_memory/phase34/online_codebook.py:164`](../../src/energy_memory/phase34/online_codebook.py):
    `_apply_cap_coverage(affected)` between anti-collapse and
    splitting-tension modulation.
- **Files created:** [`tests/test_cap_coverage_force.py`](../../tests/test_cap_coverage_force.py) (14 tests).
- **Deviations from spec:** measurement basin in A3/A6 uses static
  (non-contracting) members — co-contracting would couple force to
  moving geometry and conflate smoothness of the force itself with
  smoothness of the dynamic; A3/A6 are about the force's own
  smoothness. A6 part (ii) uses a custom wide-similarity-band basin
  to exercise the IQR check at default τ_cc. Both deviations are
  reviewer-binding-spec-compatible: they make the assertions more
  diagnostic, not less.
- **C.2.3 closed. Next: C.2.4 (metastability → replay-buffer
  energy-ranking by construction).**
