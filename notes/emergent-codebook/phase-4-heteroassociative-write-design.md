# Phase 4 — Heteroassociative Consolidation Write (surgical design)

> Status: **BUILT + validated on real data** (2026-05-30). The surgical mechanism is
> **heteroassociative write + a cue-space decorrelator**. Modules:
> `src/energy_memory/phase4/hetero_write.py` (7/7 tests) +
> `src/energy_memory/phase4/decorrelator.py` (4/4 tests). The write **alone** collapses
> under real key correlation (Report 050 Part 1), but **+ the cue-space decorrelator it
> transfers** to real data: in the sparse-cue regime where store-as-is fails, the
> integrated pipeline recovers **0.59 vs 0.024** (Report 050 Part 2, reproduced by the
> production module). The decorrelator is **batch ZCA whitening** (`P = Σ^{-1/2}` over
> the cue subspace) — an **offline/batch statistic** of the cue distribution, so it is
> anti-homunculus-exempt (the same class as the project's existing batch `d_eff`/eigvalsh
> computations); it never reads a per-cue metric to gate or branch. **Finding:** the
> online single-phase anti-Hebbian (FEP `arxiv:2505.22749`) form is **impractically slow**
> here — the real cue covariance is ill-conditioned (cond ~1e5, the frequent-token
> direction dominates), so the gradient/anti-Hebbian iteration converges far too slowly;
> the closed-form batch whitening is the production form (same fixed point). Fork verdict
> unchanged: **NOT rebuild**. Original framing below.
>
> Grounded in Reports 048/049 + the research-grounded plan. Parent spec:
> [phase-3-consolidation-write-design.md](phase-3-consolidation-write-design.md).
>
> **Anti-homunculus review: PASS-WITH-CONDITIONS** (grounded domain-expert reviewer,
> 2026-05-30; every cite re-opened). The shape is clean (fixed offline local-geometry
> write + precommitted negative + `top_index_hits` read, fence-respecting). **Three
> conditions, all discharged below:** (1) name + remove the actual thermostat at
> `error_driven_learner.py:106` (the `sims.argmax()` negative) and its 103-115
> buffering path — NOT just the blend at 147-162; (2) demote MESH from "validated
> production form" to literature-motivated-pending-validation — only the dense `H`
> (Report 049) is *shown*; (3) the integration must *prove* batch-offline (a closed,
> seed-fixed buffer in a separate pass), not assert it. The AH pass does **not**
> license a graduation claim — the headline experiment still owes Recall@K vs
> store-as-is with CIs on real data.

## What the gates established (the warrant)

The fork is adjudicated **surgical-in-place** (no rebuild-trigger fired):
- **G-A** (Report 048): frozen-refit shows **no readout defect** → not a readout fix.
- **G-0/G-B/G-C** (Report 049): store-as-is bundle reproduces the 065/066 key-only
  null (→chance at N/D≥4), but an **error-correcting/contrastive heteroassociative
  write rescues key-only recovery ≥0.94 at N=4×D**; D↑ helps (the lever is N/D —
  "lower-D rescues" is backwards). → the "no key-only basin" wall is a **store-as-is
  wall, NOT a binding-algebra wall**.
- **Correlated-key stress** (Report 049 §Result 3): the write **shape matters** once
  keys correlate (cosine≥0.13: Hebbian collapses, delta/swap hold; swap>delta at high
  correlation). → the **margin/contrastive objective is warranted**, and the failure
  of phase3b was its **self-mined `sims.argmax` negative (a runtime thermostat)** +
  top-1 readout, not the contrastive idea.

## The mechanism

Replace the store-as-is structure (the atom-codebook pull/push at
`error_driven_learner.py:147-162`, and the bundle / MHN-over-bound-pairs read) with a
**heteroassociative role→target write**, computed **batch-offline** during
consolidation:

- **Key** `k` = the role/context cue (the slot-query: `normalize(unbind(scene/window,
  role))`, or the observed-context bundle) for an observation.
- **Value** `v` = the target atom (the masked/queried filler).
- **Write** = an error-correcting heteroassociative map `H` (the "structural binding
  field" R in PROJECT_PLAN), updated by the **delta rule** over the consolidation
  buffer: `H ← H + lr·(v − H k)⊗k†`, plus a **contrastive anti-Hebbian push** away
  from a **precommitted swap-negative** value `v⁻` (a different target from the same
  role-equivalence class, drawn by a seed-fixed permutation — **never** `sims.argmax`):
  `H ← H − lr_push·(v⁻)⊗k†`.
- **Read** = `cleanup(H k)` over the atom codebook via the **existing** Modern Hopfield
  retrieve → basin-membership `top_index_hits` (Report 066:19). No new readout family.

### Thermostat removal (AH condition 1 — the load-bearing fix)

The whole anti-homunculus warrant is "we replaced the **self-mined** negative with a
**precommitted** one." The self-mined negative is the `sims.argmax()` at
`error_driven_learner.py:106` (`predicted_local = int(sims.argmax()...)`), buffered at
:107-115 and consumed by the push at :143-144 — a runtime metric-reader (thermostat).
The heteroassociative write **does not call that path at all**: it populates its buffer
with **precommitted** observations `(k, v, v⁻)` where `v⁻` is a seed-fixed draw from the
role-equivalence class, and computes its negative offline. The integration must
**remove/bypass** the :103-115 `argmax`-mined buffering, not merely swap the :147-162
blend (which would leave the thermostat populating the buffer). The new module owns its
own closed buffer and never invokes `error_driven_learner`'s streaming `argmax` path.

### Production form (cost control): MESH-style fixed scaffold

A dense `H` is `D×D` = 16M complex at D=4096 (~128 MB). The dense form is the **only
primary-verified** rescue (Report 049). A **MESH-style** form (`pdf:mesh-2022`, in
corpus: "fixed scaffold + heteroassociation avoids the content-addressable-memory
cliff") — a fixed random scaffold + a sparse/low-rank heteroassociative component, so
the learned part scales with the number of associations, not `D²` — is the
**literature-motivated production target, NOT yet validated** (AH condition 2). Before
the production commitment it requires: (a) opening the MESH primary
(`/Users/dypatterson/Desktop/Neuro-AI/research/MESH.pdf`, currently card-routed only —
and the card cautions "fixed scaffold can conflict with emergent-codebook goals if
treated as the final architecture"), and (b) the corpus run showing the scaffold form
matches the dense rescue. Until then, only dense `H` is *shown*; MESH is *supports*.
**BTSP stays out of the write path** (`biorxiv:2025.05.15.654220` is link_only /
403-blocked, and an online one-shot variant would collide with the runtime-write ban).

## §Headline metric

Masked-token / role→target **Recall@K via `top_index_hits`** on a real corpus
(wikitext / repo-sample), **heteroassociative write vs the store-as-is baseline on the
same test set**, with the **vs-no-write anchor** `Δ = recall(write) − recall(store-as-is)`,
Wilson CI > 0, multi-seed. (The two-floor Selectivity-Δ from the parent spec applies
when a role-shuffle control is available.)

## §Required controls (same test set)

1. **store-as-is baseline** — the current write; the anchor the rescue is measured against.
2. **no-negatives ablation** — delta-only (drop the swap push); isolates the contrastive contribution.
3. **random-codebook** — must collapse recovery to chance.
4. **shuffled / swap-key selectivity** — cue the wrong role → chance.
5. **perfect-cue upper bound** — bounds the ceiling so a null is attributable.
6. **N/D capacity point(s)** — report the operating N/D so the result isn't read in the easy regime.

## Anti-homunculus check

- **What moves locally?** The heteroassociative weight `H` (or the MESH sparse
  component), via a **fixed batch residual rule** (delta) over the consolidation buffer.
- **Where does the apparent decision live?** In the geometry/energy: `H` shapes the
  heteroassociative landscape; role→target recovery is **cleanup-settling** in that
  landscape. No metric is read at runtime to gate, branch, or select.
- **Fixed dynamic replacing any `if metric then act`?** The delta residual `(v − H k)`
  is a fixed local quantity computed **offline/batch**; the swap-negative `v⁻` is a
  **precommitted seed-fixed** draw from the role-equivalence class.
- **Control that it is not hidden arbitration?** random-codebook collapses; no-write
  (store-as-is) fails; the swap-negative is precommitted — the **`sims.argmax`
  negative (the runtime thermostat) is REMOVED**, not retained.
- **Banned shapes avoided:** runtime error-driven updates stay **BANNED** (STATUS Live
  policies) — `H` is written only in the batch-offline consolidation pass; no
  recall-gated plasticity; no `min`/`argmin`-over-branches aggregator; the read path
  terminates in a `top_index_hits` count, never `mhn_energy`→ΔE (Phase-5′ fence).

**Verdict (self-assessed): PASS** — the offline/batch heteroassociative write with a
precommitted swap-negative is a fixed local-geometry dynamic, not arbitration. Routed
to `/anti-homunculus-reviewer` for an independent ruling before code lands.

## Non-negotiables check (PROJECT_PLAN)

No supervisor / no subsystem-arbitration (✓ — fixed energy/geometry); not a vector-DB
+ summaries (✓ — a learned associative landscape); LLM not the source of identity (✓ —
untouched); pure-Python reference backend retained (the module ships a reference path).

## Integration plan

> **STATUS (2026-05-31): INTEGRATED (step 2 DONE), DENSE H.** The validated mechanism is
> folded into the production consolidation-write orchestrator
> `src/energy_memory/phase34/online_codebook.py` (`OnlineCodebookUpdater`) as a **separate
> batch-offline pass** behind a default-off flag — `observe(..., cue=)` accumulates a CLOSED
> cue buffer (all observations, never quality-gated), `consolidate_hetero()` fits the L2
> decorrelator + freezes + writes a **dense `H`**, `recall_hetero()` reads via
> `cleanup(H·decorr(cue))` → `top_index`. The **key is the RAW masked cue** (not the
> post-unbind slot_query) — the graduated advantage is that `H` bypasses the scene-MHN+unbind
> that corrupts the slot_query at sparse cues. Reproducibility: `hetero_write_enabled=False`
> default → the streaming `_consolidate()` (pull/push, context-residual, C.2.x) is
> **byte-identical** (117 insertions / 0 deletions; the byte-identity test stays green). AH
> condition 1 discharged structurally (the `error_driven_learner:103-115` thermostat is never
> invoked). **Anti-homunculus reviewer: PASS** (2026-05-31, all 7 points + all 3 conditions).
> Tests: `tests/test_hetero_consolidation_integration.py` (7/7). Report 057.
>
> **DENSE H now; MESH-scaffold scaling RESOLVED → DEFER (Report 120, 2026-05-31).** The
> graduated `H` is intrinsically low-rank (effective rank ≈ value-space diversity: 8 on the
> topic-toy = L, ≈126 on real text ≪ D); a **rank-`r` factored / SVD-truncated `H` reproduces
> recall BYTE-IDENTICALLY** (measured ratio 1.000 at r ≈ D/16) at 4–130× less memory with ZERO
> validation risk (same linear map, stored as its dominant factors). So the cost fallback is
> **factored `H`, NOT MESH**; MESH's catastrophic-forgetting-cliff property is for an N ≫ D
> regime the project does not hit (it runs at N ≤ D). Trigger to implement factored `H`: dense
> `H`'s O(D²) memory (~134 MB at D=4096 — currently fine) becoming a bottleneck (D ≥ 8192, or
> many concurrent `H`'s). See §"Open scaling question" below + memory `neuro_ai_mesh_scaling_decision_open`.

1. Module `src/energy_memory/phase4/hetero_write.py` — the delta/swap heteroassociative
   write over a **closed, seed-fixed buffer** of precommitted `(k, v, v⁻)` tuples; the
   `top_index_hits` read. The module **owns its buffer** and never invokes the
   `error_driven_learner` streaming `argmax` path (AH condition 1). Unit-tested (rescue
   reproduces 049; a test asserts the buffer is frozen before the write fires —
   AH condition 3).
2. Wire into the masked-token contextual-completion path (the `error_driven_learner`
   domain) as a **separate batch-offline consolidation pass** — build the closed buffer
   first, freeze it, then write — **NOT** spliced into `train()`'s per-probe streaming
   loop (AH condition 3). Baseline = store-as-is.
3. Validate on a real corpus slice (wikitext/repo-sample): Recall@K via `top_index_hits`
   vs store-as-is + the controls above. Confirm the rescue survives integration. Flag
   Colab if the corpus run needs scale.

**Guard (AH review):** do not let `entropy`/`margin` drill-downs feed back into any
write-gating or branch-selection (that re-imports the thermostat in a new costume); do
not promote the adaptive/self-mined negative back in under the banner of "improvement."

## Open scaling question — RESOLVED (Report 120, 2026-05-31)

**Resolved: ship dense `H`; defer MESH; the cost fallback is factored low-rank `H`, not
the MESH scaffold.** [Report 120](../../reports/120_mesh_scaling_decision/report.md)
measured the load-bearing fact the docs only asserted: `H` is a delta-rule accumulation
(rank ≤ N) and is in practice **highly low-rank** — effective (participation-ratio) rank
= **8 on the topic-toy (= L, the value-codebook size)** and **≈126 on real text** (eff/D ≈
0.12), with 99% of spectral energy in the top ~8 / ~159 components. **Recall saturates at
rank ≈ D/16: an SVD-truncated `H_r` gives byte-identical recall (ratio 1.000) at r = D/16,
4–130× smaller than dense.**

This separates the two things line 87 conflated: (a) a **low-rank factored `H`** is the
cheap win — behaviorally identical, pure storage/compute, no validation, no
anti-homunculus surface; (b) the **MESH fixed scaffold** is a different mechanism for the
N ≫ D capacity-cliff, which the project (N ≤ D) does not hit, is unvalidated, primary-
unopened, and card-flagged as conflicting with emergent-codebook goals. So the cost
concern that motivated the MESH question never required MESH. Trigger to *implement*
factored `H`: dense cost biting (D ≥ 8192, or many concurrent `H`'s) — currently it does
not (~134 MB @ D=4096). `H`'s rank is bounded by **value-space diversity, not D or N**.
