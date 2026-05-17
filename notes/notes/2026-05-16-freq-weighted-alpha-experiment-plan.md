# Freq-weighted Benna-Fusi α — experiment plan (Phase-4→Phase-5 bridge)

**Date:** 2026-05-16
**Source:** [Brainstorm idea 5](../../brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md)
**Why now:** Phase 4 graduated on D1 ([report 038](../../reports/038_phase4_d1_graduation.md)); the brainstorm doc names this as *"the key experiment for the architecture's compression → abstraction claim."* It bridges the graduated Phase 4 consolidation mechanism to Phase 5's structure-and-abstraction premise.

---

## Hypothesis (what we expect to see)

Replace the global coupling coefficient α in the Benna-Fusi cascade with a per-pattern α_eff that scales with retrieval frequency:

```
α_eff(k) = α_base × (1 + λ × normalized_retrieval_count(k))
normalized_retrieval_count(k) = retrieval_count(k) / max_retrieval_count
```

For λ = 0 the behavior is identical to the current implementation. For λ > 0, frequently-retrieved patterns transfer faster through the cascade (u_1 → u_2 → … → u_m); patterns that are consolidated once and never retrieved again stall in fast variables and decay.

**Predicted effects, in order of confidence:**

1. **(Confidence revised after impl)** Final-checkpoint correlation between u_m and retrieval_count *shifts* with λ. The brainstorm's framing implied correlation rises monotonically, but the implementation reveals a transient/equilibrium asymmetry (see "Architectural caveat" below): in the cascade transient, high-α patterns reach u_m faster; at long horizons under continuous input, the Dirichlet boundary at u_{m+1}=0 makes high-α cascades leak more mass per unit time, so equilibrium u_m is *lower* for higher α. Production runs at n_cues=3000 with replay_every=50 do ~60 cascade ticks, which (depending on α and replay rate) may be in the transient or near equilibrium. The *direction* of the correlation shift under λ>0 is what this experiment will resolve.
2. **(Medium confidence)** Slow-variable distribution (u_m across patterns) becomes more concentrated: a smaller fraction of atoms account for most of the slow-variable mass under freq-weighting. Measurable as Gini coefficient of |u_m| values.
3. **(Medium confidence)** D1 (Phase 4 graduation headline) preserved across all λ values. If the substrate stops being decisive under freq-weighting, that's a falsification of the mechanism, not a feature.
4. **(Low confidence)** R@K improves under freq-weighting. The brainstorm predicts this on Dury under-capacity → abstraction grounds, but Phase 4 integration regime has R@K already variance-bound, so the signal-to-noise here is unclear.

### Architectural caveat surfaced during implementation

The brainstorm describes the freq-weighting effect as "patterns retrieved frequently get faster transfer through the chain. Patterns that consolidate but are never retrieved again stall in fast variables and eventually decay." This is correct for the cascade *transient*: at finite time t before steady state, the high-α cascade has propagated more mass to u_m.

But the Benna-Fusi cascade in this codebase has a Dirichlet boundary at u_{m+1}=0 (see `step_dynamics`'s boundary handling). At cascade *equilibrium* with continuous input I at u_1 (the case after many ticks), `u_m = I / (5α)` for m=4. So higher α gives *lower* equilibrium u_m — mass leaks out the boundary faster than it accumulates.

This makes the experiment's headline outcome a *measurement* rather than a *check*: the correlation we observe depends on whether the production regime sits in the transient (high-α → high u_m) or near equilibrium (high-α → low u_m). Both outcomes are informative; we just don't know which one we're in without running.

A unit test (`test_lambda_changes_steady_state_distribution_of_u_m`) pins the steady-state behavior so this asymmetry doesn't get rediscovered later.

---

## Mechanism details (what changes in code)

| File | Change |
|---|---|
| `src/energy_memory/phase4/consolidation.py` | Add `alpha_freq_lambda: float = 0.0` to `ConsolidationConfig`. Add `retrieval_count` tensor to `ConsolidationState`. Increment it on `reinforce()`. Keep `add_pattern()` / `remove_pattern()` in sync. In `step_dynamics()`, when `alpha_freq_lambda > 0`, compute per-row α_eff and broadcast over the Laplacian. Add `retrieval_count_max` / `retrieval_count_mean` to `stats()`. |
| `experiments/19_phase34_integrated.py` | Add CLI `--alpha-freq-lambda` (default 0.0). Wire into `ConsolidationConfig`. Add `retrieval_u_m_correlation` to checkpoint payload. |
| `tests/test_consolidation.py` (new or extend existing) | Verify λ=0 is bit-identical to current behavior. Verify λ>0 with one high-retrieval pattern produces a higher final u_m than λ=0 on the same pattern. |

The Saighi A_k inhibition mechanism already provides a "retrieval count" proxy via `A_k`, but A_k can decay and is conceptually about basin-narrowing, not transfer rate. A separate `retrieval_count` int counter is cleaner and avoids coupling two orthogonal mechanisms.

---

## Conditions (single-seed smoke, then n=10 if smoke is informative)

| Condition | λ | What it tests |
|---|---|---|
| baseline-fixed-α | 0.0 | Equivalent to graduated Phase 4. Should reproduce report 038's numbers exactly. |
| weak-freq-α | 0.5 | Max-retrieval atom gets 1.5× α_base. Modest filter. |
| strong-freq-α | 1.0 | Max-retrieval atom gets 2× α_base. Strong filter; should make the mechanism observable. |

Run on the same Phase 3+4 integration harness as report 038: online Hebbian, st=0.3, n_cues=3000, A_k off, m=6, α_base=0.25. The only changed parameter is `--alpha-freq-lambda`.

**Smoke test:** seed 17 only, MPS device, exp 19. Check that:
- λ=0 reproduces report 038's seed-17 numbers (R@10=0.342, ms_w3=0.000, deaths=7201, etc.) → if not, the patch broke something.
- λ=1.0 produces a measurably different u_m distribution. If u_m looks identical, the wiring is broken.

**Full validation (only if smoke is informative):** n=10 seeds {17, 11, 23, 1, 2, 3, 5, 7, 13, 19} × 3 λ values = 30 runs on Colab H100 NVL. With ~14 min/seed in parallel, ~14 min × 3 batches = ~45 min wall time. Cost ~$15 in credits.

---

## Headline metric

**Δ correlation(u_m, retrieval_count) at step 2000 vs baseline-fixed-α, multi-seed, CI disjoint from 0.**

This is the direct mechanism-check: does freq-weighting actually filter the slow variable as designed? It's substrate-pure (per the [2026-05-16 discipline note](2026-05-16-substrate-vs-readout-metric-discipline.md)), seeded by an architectural prediction, and falsifiable.

---

## Drill-downs

- D1 (Δ meta_stable_w3 vs baseline_static): must stay graduated. If it drops, the freq-weighting broke the substrate; investigate.
- u_m sparsity: Gini coefficient of |u_m| across patterns. Predicted: rises with λ.
- u_m mass concentration: fraction of |u_m| total held by the top-10% of patterns. Predicted: rises with λ.
- Atom-count surviving death: predicted to drop with λ, because un-retrieved atoms decay faster through fewer cascade contributions.
- ΔR@10 vs baseline-fixed-α: drill-down. Brainstorm predicts improvement but this regime is variance-bound.
- u_m correlation with retrieval_count: the headline at full precision.

---

## Anti-homunculus check

- α_eff is a function of a local quantity (per-pattern retrieval count). No subsystem reads a metric and triggers a behavior.
- The cascade dynamics are otherwise unchanged: Eq 10/11 still applies; the only modification is that the coupling coefficient is row-dependent.
- The retrieval-count input is a count of `reinforce()` calls — a pure measurement, like the existing `step_count`.

The mechanism is "fast variables of well-retrieved patterns drain faster," which is local geometry. No homunculus.

---

## What this experiment does NOT establish

- It does not establish that freq-weighting *generalizes* (transfer to held-out structural episodes). That requires a different eval set than wikitext-2 validation provides. This experiment establishes the *mechanism is working as designed*; generalization is a Phase-5 follow-up.
- It does not establish that the architecture's "compression → abstraction" claim holds, only that the mechanism the claim rests on is functional and tunable by λ.
- A null result (no correlation rise) would falsify the mechanism *as currently parameterized* but not the broader compression-as-frequency-filter hypothesis (e.g., a different freq-weighting form might still work).

---

## Decision gate before promoting to Phase 5

The experiment produces three kinds of outcomes:

1. **Mechanism works as predicted** (correlation rises with λ; D1 preserved; u_m concentration measurable). → Strong evidence for the compression-as-frequency-filter hypothesis. Proceed to Phase 5 with this mechanism wired in.
2. **Mechanism is null** (no measurable difference between λ=0 and λ=1.0 at n=10). → The compression-as-frequency-filter hypothesis as currently formulated is not load-bearing in this regime. Re-evaluate the brainstorm-idea-5 framing; Phase 5 may need to seek the compression mechanism elsewhere (e.g., death-rate gating, capacity-pressure annealing).
3. **Mechanism works but breaks something** (correlation rises but D1 degrades or R@K regresses substantially). → Document the tradeoff; redesign the mechanism (e.g., bound λ_eff so the slowest variables retain a floor of capacity).

In all three outcomes the result is informative for Phase 5 design.
