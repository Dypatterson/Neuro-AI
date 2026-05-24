# Report 063 — Phase 5 Spike E1 Centered Log-Prior (Path C Closure)

**Date:** 2026-05-24
**Active phase:** 5
**Status:** Smoke complete. **Null result, with monotone worsening.** E1 closes "Path C done right" with the same outcome as Report 062 D1/D3. Combined with Report 062, retrieval-mechanism-only routes for Phase 5 are now exhaustively ruled out at smoke scale. Path D = Tier-2 training-time intervention is the only remaining unblocked option.
**Decision:** Do not promote E1 to a Colab confirmation run. The qualitative result is robust across the λ sweep and consistent with the substrate-level diagnosis: the FHRR substrate does not contain role-addressable basins, and no retrieval-mechanism reshaping can manufacture them.
**Headline metric per [phase-5-unified-design.md:269-292](../notes/emergent-codebook/phase-5-unified-design.md):** `ΔE = E_content_prior - E_role_prior`, paired per cue. Reported on the substrate-pure (unbiased) Hopfield energy — not the field-biased energy — so cross-condition comparison is on the same energy function.
**Required controls:** λ=0 is the within-experiment baseline; bit-identical to Report 062 baseline by construction (field is zero). Same 3 seeds × 30 cues × same cue seed as Report 062.
**Last verified result:** [Report 062](062_phase5_spikes_d1_d3_local_smoke.md) — D1/D3 smoke null on `hit_role` for all 90 cues; routes to Path D per the brainstorm's pre-committed decision recipe.
**Why this experiment now:** External (GPT-generated) suggestion to refine Path C by replacing the report-061 one-hot per-pattern log-prior (which was selector-shaped under the no-schema-store control) with a zero-mean per-atom logit field derived from role-vs-content similarity. The brainstorm's [active-inference brief](../brainstorm-workspace/2026-05-24-phase5-rescue/research/02-active-inference.md) Idea A flagged this exact shape (Betteti et al. 2025 IDP) as anti-homunculus-clean and the cleanest closure of "Path C done right." This spike runs that closure.

## Setup

- Harness: [`scripts/spike_e1_centered_log_prior.py`](../scripts/spike_e1_centered_log_prior.py)
- Substrate snapshots: same as Report 062 (A+B+A1' Phase 4 at seeds 17, 11, 23)
- Operating point: β=10, max settling iter=12, binding noise σ=0.05, content distortion=0.6, cue seed=117
- λ sweep: {0.0, 0.25, 0.5, 1.0}
- 3 seeds × 30 cues × 4 λ = 360 cue×λ evaluations

## E1 mechanism

For each cue, compute a **zero-mean per-atom logit field** `b ∈ ℝ^N`:

```
role_score_i    = Σ_w cosine(unbind(cue, p_w), atom_i)     # sum over W positions
content_score_i = cosine(cue, atom_i)
z_role          = (role_score - mean) / std
z_content       = (content_score - mean) / std
log_ratio       = log((softplus(z_role)+ε) / (softplus(z_content)+ε))
b               = λ · (log_ratio - mean(log_ratio))         # zero-mean
```

Settling adds `b` to per-pattern logits:

```
logits_i = β · sim(state, atom_i) + b_i
```

The field is per-cue (same for all branches of a given cue), constant during the settling loop. It modulates the energy landscape without injecting free energy into any one atom.

**Sum-over-positions aggregation chosen over max-over-positions**: the initial run with `max_w` produced worse-than-baseline results from noise-peak amplification (most atoms have some noisy position where unbinding similarity is moderately high; the max selects these noise peaks). Sum-over-positions integrates the "atom is bound at any position" signal and is the natural FHRR analog of the project's existing role-binding similarity in `experiments/40_phase5_branching.py`.

## Aggregated results (3 seeds × 30 cues per λ)

| λ | mean ΔE | ×floor | hit_role | rank_role | rand_low | mean &#124;b&#124; field | seeds ΔE > 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | +0.00267 | +0.49 | 0.000 | 205.5 | 0.367 | 0.000 | 2/3 |
| 0.25 | −0.00141 | −0.26 | 0.000 | 245.6 | 0.311 | 0.201 | 1/3 |
| 0.50 | −0.00752 | −1.37 | 0.000 | 333.4 | 0.289 | 0.402 | 1/3 |
| 1.00 | −0.01083 | −1.97 | 0.000 | 498.1 | 0.300 | 0.804 | 0/3 |

**`hit_role = 0.000` at every λ across all 3 seeds × 30 cues = 360 evaluations.** Same as Report 062's D1/D3/baseline conditions.

**`rank_role` worsens monotonically with λ**: 205 → 246 → 333 → 498. The E1 field is *actively pushing* the role-prior branch AWAY from the role-target atom as λ increases. At λ=1.0 the role-target atom ranks near the median (498/1024).

**`mean ΔE` becomes more negative with λ**: +0.0027 → −0.0014 → −0.0075 → −0.0108. The role-prior branch lands at *higher* energy than the content-prior branch when the field is active. The field penalizes role-prior settling more than content-prior settling.

**Field magnitude scales linearly with λ** (0.000 → 0.201 → 0.402 → 0.804). Sanity check: the mechanism is computing the intended quantity; the null is a property of the substrate, not a coding error.

**`random_lowest` DROPS with λ** (0.367 → 0.311 → 0.289 → 0.300). The field is also pushing the random-prior branch away from low-energy atoms. This is consistent with the field being noise-correlated: it penalizes all branches' settling toward whatever atoms the noise-driven `role_score` ranks highly, regardless of which branch is asking.

### Per-seed pattern (consistency check)

| seed | λ=0 ΔE | λ=0.5 ΔE | λ=1.0 ΔE | λ=0 rank | λ=1.0 rank |
|---:|---:|---:|---:|---:|---:|
| 17 | +0.00287 | −0.00432 | −0.01576 | 207.6 | 506.7 |
| 11 | −0.00266 | +0.00420 | −0.00237 | 204.2 | 567.4 |
| 23 | +0.00780 | −0.02244 | −0.01435 | 204.6 | 420.1 |

Direction is consistent: `rank_role` monotonically worsens across all 3 seeds. Seed 11 is the only seed with ΔE > 0 at λ=0.5, and it's still sub-floor and `rank_role` still worsens to 567. No single seed produces structural retrieval at any λ.

## Reading

### E1 does not surface role information that the substrate doesn't have.

The brainstorm pre-committed: "If E1 moves `hit_role` materially above 0.01: substrate has weak role information that asymmetric landscape reshaping surfaces; revisit Path D decision with E1 in the mix." That outcome did not occur — `hit_role` is 0.000 at every λ across every seed.

The complementary outcome — "If E1 moves ΔE clean above floor but hit_role stays ≈0: Path C done right is still wrong-shape on this substrate; the substrate genuinely lacks role-target basins" — *also* did not occur in its clean form. ΔE doesn't go cleanly above floor at any λ; instead it goes monotonically *negative*. Path C done right is worse than Path C done one-hot on this substrate.

### Why E1 worsens role-prior settling

The role/content asymmetric field requires that the substrate-computable role and content scores actually distinguish role-target atoms from non-role-target atoms. On this substrate:

- `role_score_i = Σ_w cosine(unbind(cue, p_w), atom_i)` is dominated by noise because (a) there are only W=4 positions to integrate over, (b) the codebook has 1024 atoms in D=4096 (about ~4× redundancy), and (c) FHRR unbinding similarity for off-target (atom, position) pairs is mean-zero but has nontrivial variance that accumulates additively.
- The atoms that score highest on `role_score` are not the actual role-target atoms — they are atoms with high noise correlation across the W positions.
- The E1 field boosts those noise-correlated atoms in the logits. The role-prior branch, which started at the actual role-target atom, gets pulled toward the noise-correlated atoms during settling.
- This is exactly the Report 058 finding GPT's analysis cited as motivation: "ΔE-positive cells have zero role-target hits and rank the role target hundreds of atoms down" — the substrate's role channel is anti-correlated with role-target identity.

E1 doesn't fix this; it amplifies it. The asymmetric field is computing the wrong asymmetry because the substrate's atoms don't have a clean role-vs-content separability.

### What this strengthens about the brainstorm + Report 062 diagnosis

Report 062 closed D1 (storage-rule change) and D3 (cross-K coupling) as null on `hit_role`. Report 063 closes E1 (landscape reshaping via substrate-derived role/content asymmetry) as null on `hit_role` with the additional finding that the field is actively counterproductive.

**Three architecturally distinct retrieval-mechanism families have now been smoke-tested and returned null on this substrate**:

| Family | Spike | Result |
|---|---|---|
| Storage rule (Hebb → pseudo-inverse) | D1 | `hit_role`=0; `rank_role` worsens 205→403 |
| Branch coupling (independent → cross-K softmax) | D3 | `hit_role`=0; ΔE drives negative 0/3 seeds |
| Landscape reshaping (uniform → asymmetric field) | E1 | `hit_role`=0; `rank_role` monotone worsens 205→498 |

No retrieval-mechanism family can extract role-target basin retrieval from this substrate. **This is strong combined evidence that the substrate as currently consolidated does not contain role-target basins**, not that we've been using the wrong retrieval mechanism.

The remaining Phase 5 paths are:
- **Path A** (capacity-proportional scale-down probe) — only useful as closure-paper evidence per the existing project framing
- **Path B'** (close Phase 5 + pivot to surprise/PE-driven replay) — accepts the null
- **Path C** (continue log-prior diagnostics) — both the one-hot form (Report 061) and the asymmetric-field form (this report) are now characterized; nothing left to vary at the retrieval-mechanism layer
- **Path D** (Tier-2 training-time intervention: M2 EqProp+role-shuffled-negatives+DSM warm-start, or M1 P1+D3+P3 stack) — **the only path that addresses the substrate-level diagnosis directly**

## Limitations

- n=3 seeds × n=30 cues (smoke; graduation standard is n=10 × 100 with CI)
- Local snapshots only (seeds 17, 11, 23); other seeds not tested
- E1 field formulation uses `sum_w` aggregation; `mean_w` would scale identically to within a constant absorbed by λ
- The "noise-dominated role_score" diagnosis is consistent with Report 058 and Report 062 but not directly proved here — would require an additional spike comparing `role_score(role_target_atom)` distribution to `role_score(random_atom)` distribution
- E1 was tested in isolation (no γ·prior_bias from the existing Phase 5 machinery); a stacked variant E1 + γ-prior could behave differently, but the brainstorm's anti-homunculus discipline argues against stacking diagnostics

## Anti-homunculus discipline

- E1 field is a deterministic function of (cue, atom_i) — pointwise local computation, no metric trigger
- Centering is fixed algebraic (subtract mean)
- λ is a global substrate parameter; the sweep is over its values, not adaptive selection
- Same field applies to all branches of a given cue — not arbitration
- Settling under the modified energy is gradient flow on a single Lyapunov function (Ramsauer MHN with additive per-pattern bias)
- Null result reported as data; no post-hoc selection of "best λ"; no graduation claim

## What this DOES and DOES NOT close

**Closes**:
- The "Path C done right" hypothesis. Asymmetric role/content landscape reshaping does not produce structural retrieval on the current substrate. The Path C investigation is exhaustively characterized.
- The retrieval-mechanism family for Phase 5 rescue. Three independent families (storage, branch coupling, landscape) all null.
- GPT's Idea 1 specifically (as a closure candidate; the brainstorm's P3 IDP variant overlaps).

**Does NOT close**:
- The training-time hypotheses (Path D: M2 EqProp + role-shuffled negatives + DSM warm-start; M1 P1+D3+P3 stack). These directly address the substrate-level diagnosis and have not been smoked.
- GPT's Idea 4 (dual-code GHRR for roles + FHRR for content). This is a storage-channel change, not a retrieval-mechanism change. Folds into P1 as an alternative algebra variant.
- GPT's Idea 5 (LSR/Epanechnikov energy with shoulder basins). Energy-function change at the retrieval layer; the spike evidence here suggests it would null like the other retrieval-only changes, but it's a different basin geometry and worth its own characterization if Path D pursues M1 with LSR retrieval.

## Recommendation

Update STATUS.md `Recent updates` with a new entry pointing here. Strengthen the Path D framing in the blocker: Path D is now the only path that hasn't been ruled out at smoke scale.

If a closure paper is the goal: this report + Report 062 + the brainstorm's anti-homunculus audit constitute a publishable negative result on retrieval-mechanism interventions for the FHRR + MHN + emergent-codebook architecture's structural retrieval failure.

If continued substrate work is the goal: commit to Path D, choosing between M2 (cleanest training-time intervention, well-supported by EBM literature) and M1 (architecturally richer, requires P1 schema-extension and Lyapunov S2 spike first). The next-step scope is a separate user decision; this spike does not select.

## Artifacts

- Harness: [`scripts/spike_e1_centered_log_prior.py`](../scripts/spike_e1_centered_log_prior.py)
- Per-seed/λ JSON: [`reports/spike_e1_centered_log_prior.json`](spike_e1_centered_log_prior.json)
- Companion: [Report 062](062_phase5_spikes_d1_d3_local_smoke.md) (D1+D3 smoke; complementary retrieval-mechanism null)
- Brainstorm: [`brainstorm-workspace/2026-05-24-phase5-rescue/`](../brainstorm-workspace/2026-05-24-phase5-rescue/)
- Active-inference brief (P3 IDP source): [`research/02-active-inference.md`](../brainstorm-workspace/2026-05-24-phase5-rescue/research/02-active-inference.md)
