---
date: 2026-05-24
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: spike-result
session-closes: D3-Lyapunov-check
---

# Spike D3 — Cross-K Softmax Lyapunov Analytical Pass

Companion to the
[Phase 5 rescue brainstorm](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)
(Tier-0 Idea D3, "Slot-style cross-branch softmax during K-branch
settling"). Closes the **Lyapunov commit gate** flagged by the
reviewer-agent audit of the brainstorm (2026-05-24, CONDITIONAL on
joint-energy existence and monotone descent).

## Question (from the audit)

> Write down the joint energy `E(s_1,…,s_K)` whose gradient flow IS
> the proposed update; prove (or numerically demonstrate) that it
> decreases monotonically under the proposed step; confirm the
> fixed-point structure isn't degenerate. If no such joint energy
> exists, the update is implementing a competitive *rule* (not a
> gradient) and D3 is arbitration-shape, regardless of how
> local-looking each pointwise step is.

## Notation

- `K` branches, each with state `s_k ∈ ℂ^D` (FHRR unit-magnitude
  per-element)
- `N` stored patterns `{x_p}_{p=1}^N` packed as codebook matrix `X ∈ ℂ^{N×D}`
- `β` inverse temperature
- Standard MHN per-branch logits: `ℓ_k(p) := β · Re⟨x_p, s_k⟩`
  (real part of FHRR inner product, the project's existing convention
  in `torch_hopfield.py`)
- Per-branch softmax (over patterns): `π_k(p) := softmax_p(ℓ_k(p))`
- Cross-branch softmax (over K, per pattern): `α_k(p) := softmax_k(ℓ_k(p))`

## The brainstorm's proposed update — and why the multiplicative form FAILS

The brainstorm wrote (D3, Idea 1):

> `α_k(p) = softmax_k( ℓ_k(p) )    [softmax over the K-branch axis, for each pattern p]`
> Then each branch's update mixes patterns weighted by `α_k(p) · softmax_p(ℓ_k(p))`.

Reading "weighted by" as multiplicative gives the proposed update:

```
s_k ← Σ_p [α_k(p) · π_k(p)] · x_p           (1)  [multiplicative form]
```

**Claim**: (1) is NOT a gradient of any scalar energy. Therefore D3
in the multiplicative form fails the Lyapunov commit gate.

### Proof sketch (cross-partial symmetry)

For an update `s_k ← g_k(s_1,…,s_K)` to be gradient descent on a
scalar energy `E`, the vector field `g_k` must be a gradient field —
equivalently the cross-partials `∂g_k/∂s_j` and `∂g_j/∂s_k` must
satisfy the integrability (symmetry) condition.

Compute for `g_k(p, s_1,…,s_K) := α_k(p) · π_k(p) · x_p` (per-pattern
contribution), letting `j ≠ k`:

```
∂α_k(p)/∂s_j = -β · α_k(p) · α_j(p) · x_p              (cross-K coupling)
∂π_k(p)/∂s_j = 0   for j ≠ k                            (π_k depends only on s_k)
```

Therefore:

```
∂[α_k(p)π_k(p)]/∂s_j · x_p
  = [∂α_k(p)/∂s_j · π_k(p)] · x_p
  = -β · α_k(p) · α_j(p) · π_k(p) · x_p · x_pᵀ
```

By symmetry of the role swap (j ↔ k):

```
∂[α_j(p)π_j(p)]/∂s_k · x_p
  = -β · α_j(p) · α_k(p) · π_j(p) · x_p · x_pᵀ
```

Summing over `p` gives the cross-partial blocks of the candidate
"Hessian." For integrability we need

```
Σ_p α_k(p) α_j(p) π_k(p) x_p x_pᵀ  =  Σ_p α_j(p) α_k(p) π_j(p) x_p x_pᵀ
```

i.e. `π_k(p) = π_j(p)` for all `p`. **This is false in general** —
`π_k` and `π_j` are softmaxes over the *different* states `s_k` and
`s_j`, so unless `s_k = s_j` (the degenerate branch-collapse case
Phase 5 is trying to avoid), they differ.

**Conclusion**: the multiplicative update (1) is not a gradient
field. There is no Lyapunov function for it. The brainstorm's D3 in
multiplicative form is arbitration-shape — it implements a
competitive *rule*, not a flow.

## The additive form PASSES

Re-read the brainstorm's "mixes patterns weighted by α_k(p) · π_k(p)"
as additive (which is more honest to slot attention's actual
mechanism — see §"Comparison to slot attention" below):

```
s_k ← Σ_p [π_k(p) + α_k(p)] · x_p           (2)  [additive form]
```

**Claim**: (2) IS a gradient flow on the scalar energy

```
E(s_1,…,s_K) = -(1/β) Σ_k lse_p(ℓ_k(p))                (per-branch terms)
              -(1/β) Σ_p lse_k(ℓ_k(p))                  (cross-K terms)
              + (1/2) Σ_k ‖s_k‖²                        (Ramsauer regularizer)
```

### Proof

Compute the gradient component-wise:

```
∂E/∂s_k = -Σ_p softmax_p(ℓ_k(p)) · x_p       (from per-branch term)
          -Σ_p softmax_k(ℓ_k(p)) · x_p        (from cross-K term)
          + s_k

        = -Σ_p [π_k(p) + α_k(p)] · x_p + s_k
```

Gradient descent `s_k ← s_k - η ∂E/∂s_k` at fixed-point (η → 1, or
treating the regularizer as the Ramsauer self-recovery `s_k ←
new_state`) gives exactly update (2):

```
s_k ← Σ_p [π_k(p) + α_k(p)] · x_p
```

Since (2) is the gradient of a single scalar `E`, and `E` is bounded
below (each LSE is bounded by `max_·` plus `log(N)/β` or `log(K)/β`;
the quadratic regularizer dominates at large `‖s‖`), gradient flow
under (2) decreases `E` monotonically and converges to a stationary
point. ✅ Lyapunov-clean.

### Fixed-point non-degeneracy

The cross-K LSE term `-(1/β) Σ_p lse_k(ℓ_k(p))` is strictly convex in
`{s_k}` only on a quotient by the permutation-symmetry of branches.
Concretely, two failure cases to check:

1. **All branches collapse to the same state** (`s_k = s_∀ k`). Then
   `α_k(p) = 1/K` for all `(k, p)`, the cross-K term reduces to a
   constant `-(N/β) log K`, and the per-branch terms drive each branch
   to the same MHN basin. **This is precisely the failure mode D3 is
   trying to avoid** — it is a legitimate fixed point of (2), so D3
   needs branch initialization to break the symmetry.

   Resolution: K-branch initial conditions in Phase 5 already seed
   branches with different priors (role vs content). The cross-K term
   `-Σ_p lse_k` then preserves the asymmetry rather than erasing it,
   because moving any one `s_k` away from another reduces the LSE
   coupling (lower energy = better). So the branch-distinct asymmetric
   initial conditions are stable under (2), and (2) actively pushes
   collapsed states apart along the gradient direction of the cross-K
   term.

2. **All-zero state** (`s_k = 0 ∀ k`). Then `α_k(p) = 1/K` and
   `π_k(p) = 1/N`. Update gives `s_k ← (1/K + 1/N) Σ_p x_p` for all
   branches, identical across branches — drives back to collapse.
   The Ramsauer regularizer `+½‖s‖²` prevents this from being a
   strong attractor on the FHRR unit-magnitude manifold.

### Comparison to slot attention

Slot attention's exact update (Locatello 2020 eq. 1, with `M`
attention matrix and `WK` value projection) is:

```
M_{kp} = softmax_k(⟨s_k, k_p⟩)                  [cross-K softmax]
W_{kp} = M_{kp} / Σ_{p'} M_{kp'}                 [pixel-axis renormalization]
s_k    ← Σ_p W_{kp} · v_p                        [weighted aggregation]
```

The row-renormalization (`W = M / row_sum(M)`) is *not* a gradient
step on any clean energy — slot attention's contraction comes from
end-to-end training of `WK, WQ, WV` against a permutation-invariant
decoder loss, not from a Lyapunov property of the cell.

For Phase 5 we are *not* training projections end-to-end (FHRR is
substrate-fixed). So the proper "slot-attention-style" graft is to
keep the *competitive primitive* (cross-K softmax) but realize it
as an *additive energy term* alongside the existing per-branch MHN
term. That additive composition is exactly (2), and it IS
Lyapunov-clean — unlike the row-renormalized slot-attention update.

**This is a constructive correction to the brainstorm**: rather than
porting slot attention's row-renormalization (which is not gradient
flow), port slot attention's *cross-K softmax primitive* as an
*additive log-sum-exp energy term*. The result preserves slot
attention's competitive shape while inheriting MHN's Lyapunov
discipline.

## Practical implementation form

Rewriting (2) as a per-step "augmented MHN retrieve":

```python
# Standard MHN per-branch logits
logits = beta * X @ s_k                          # shape (N,)
pi_k = softmax(logits, dim=-1)                   # per-branch (over patterns)

# Cross-K logits — need all branches' s_j to compute α_k(p)
all_logits = beta * X @ S                        # shape (N, K), S=[s_1,…,s_K]
alpha = softmax(all_logits, dim=-1, axis_over_K) # softmax over K, per pattern

# Combined update — additive
s_k_new = (pi_k + alpha[:, k]) @ X               # shape (D,)
```

Note both `pi_k` and `alpha[:, k]` are length-N probability-like
vectors over patterns. Their sum lies in `[0, 2]`; the resulting
state magnitude may exceed unit-magnitude on the FHRR manifold.
Two options:

1. **Average instead of sum**: `s_k_new = 0.5·(pi_k + alpha[:, k]) @ X`
   — preserves the additive Lyapunov structure (constant scaling
   doesn't change minimizer), keeps unit-magnitude scale closer to
   standard MHN.
2. **Tunable mix coefficient**: `s_k_new = ((1-λ)·pi_k + λ·alpha[:, k]) @ X`
   with `λ ∈ [0, 1]` — recovers standard MHN at `λ=0`, pure cross-K
   competition at `λ=1`. Same Lyapunov property as long as `λ` is a
   global substrate parameter (NOT per-cue, NOT thresholded — see
   anti-homunculus failure modes below).

The smoke-test implementation should default to **option 2 with
`λ=0.5`** as a balanced starting point and sweep `λ ∈ {0.25, 0.5, 0.75}`
to characterize the regime.

## Anti-homunculus self-check on the corrected D3

| Substep | Where decision lives | Verdict |
|---|---|---|
| Per-branch softmax `π_k(p)` | MHN retrieval gradient on per-branch term of additive energy | ✅ |
| Cross-K softmax `α_k(p)` | Gradient on cross-K LSE term of same additive energy | ✅ |
| Additive combination `[π_k + α_k]` | Gradient of sum equals sum of gradients (linearity); single energy descent | ✅ |
| Mix coefficient `λ` | Substrate parameter (same shape as `β`); fixed global, not per-cue | ✅ |
| Branch initialization | Already substrate-determined in Phase 5 (role-prior vs content-prior seeding) | ✅ |
| K-branch fixed-point breaking | Asymmetric init + cross-K LSE pushing distinct branches apart | ✅ |

**Failure modes that would re-introduce arbitration**:

- Per-cue `λ` (e.g. "`λ=0.7` when cue entropy high"). Controller. Keep `λ` global.
- Early-exit `if max_k α_k(p) > τ then commit branch k to pattern p`. Selector. Run to convergence.
- Multiplicative form (1) — proven NOT Lyapunov in §"multiplicative form FAILS"; refuse.
- Slot-attention-style row-renormalization `α_k(p) / Σ_{p'} α_k(p')`. Not gradient flow.

## Verdict for D3 unblocking

| D3 commit gate from reviewer audit | Status |
|---|---|
| Joint energy `E(s_1,…,s_K)` exists | ✅ derived above |
| Monotone descent under proposed update | ✅ gradient of `E` |
| Fixed-point non-degenerate | ✅ subject to asymmetric init (already true in Phase 5) |
| Anti-homunculus shape | ✅ all six substeps clean |

**D3 unblocks for implementation** in the additive form (option 2 with
tunable `λ`). The implementation must NOT use the multiplicative form
or slot-attention row-renormalization — both fail Lyapunov.

## Implementation budget

Modification to existing K-branch HAM settling loop:

- **Per-step update**: ~10 LOC change in
  `phase5/ham_aggregator.py` or `phase5/ham_with_layer2.py` (whichever
  is the canonical K-branch retrieve path — verify in D3
  implementation task).
- **Mix coefficient `λ`**: new entry in Phase 5 config, default 0.5.
- **Drill-down logging**: per-branch contribution from `π` term vs
  `α` term, plus joint energy `E` per step (substrate-pure, for
  Lyapunov sanity-check in smoke tests).
- **Tests**: monotone-descent assertion on synthetic K-branch settling
  (energy decreases each step within numerical tolerance).

Total: ~50 LOC including drill-downs and tests.

## Linked notes

- [Phase 5 rescue brainstorm — D3](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)
- Sibling spike: [S1 replay-trace schema check](2026-05-24-spike-S1-replay-trace-schema.md)
- Reference: slot attention (Locatello et al. NeurIPS 2020, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055))
- Reference: MHN energy (Ramsauer et al. 2020, [arXiv:2008.02217](https://arxiv.org/abs/2008.02217))
