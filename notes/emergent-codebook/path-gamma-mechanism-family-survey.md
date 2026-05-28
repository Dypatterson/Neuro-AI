# Path γ — Phase 3 mechanism-family survey

**Date:** 2026-05-27.
**Scope:** Design-and-precommit-only pass (no implementation authorized). Identify 3–5 candidate mechanism families whose shape could plausibly produce a robust corpus-specific learning signal that meets the **revised** Path C exit criterion at [phase-3-deep-dive.md:188-205](phase-3-deep-dive.md). Inputs: [literature-and-principles.md](literature-and-principles.md), 2026-05-24 unconsidered-paths brainstorm at [brainstorm-workspace/2026-05-24-unconsidered-paths/](../../brainstorm-workspace/2026-05-24-unconsidered-paths/), Path C closure entry in [2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](../notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md), and [Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md).

## What this survey does NOT do

- Does **not** commit Path γ to any single mechanism family.
- Does **not** authorize implementation. The next deliverable after this survey is a per-candidate precommit (one candidate, selected with the user) including an anti-homunculus-reviewer pass.
- Does **not** reopen Phase 5′. ΔE / bridge / M2 / matrix / headline / graduation work remains blocked per [STATUS.md](../../STATUS.md) until Phase 3 graduates a robust mechanism.

## Design constraints derived from the Path C walk-back

The four-run wikitext chain (Report 112) localized **what kind of mechanism shape is missing**, not just which hyperparameters were wrong. Three constraints fall out:

1. **The training-signal shape — not its strength — is the bottleneck.** v3 `wikitext_best` at lr_pull=1.0, n_events=3000 returned null. The Hebbian-pull + α_anti repulsion combination produces measurable structural change (regime tightening ~90% atoms `spread`→`tight`) without producing corpus-specific learning. *Any candidate that is "Hebbian-pull with a different multiplier" is in the same shape class and should be expected to hit the same noise floor.*

2. **Per-seed robustness, not pooled-CI-disjointness, is the bar.** σ_per-seed ≈ 0.13 vs μ ≈ +0.03 at the v3 operating point; σ/μ ≈ 3.3. A mechanism qualifies under the [revised C.3 criterion](phase-3-deep-dive.md) only if it shifts the **per-seed distribution mean** above the noise floor in ≥ 70% of seeds, not just pulls a pooled-CI margin.

3. **Substrate-capacity is not the dial.** D-curve reversed at D=1024, 2048, 16384; D=4096 vs D=8192 paired test showed only 40% of seeds improve. *Any candidate whose theoretical justification routes through "more D = more signal" is suspect by direct empirical falsification.*

A fourth constraint inherits from CLAUDE.md and the 2026-05-26 Path C precommit's H1–H7 discipline:

4. **Anti-homunculus compatibility is binding.** Mechanism shape must be expressible as local geometry / energy / settling / tension / consolidation. No supervisor that arbitrates between subsystems. No `if metric-X then act-Y` rule. The "apparent decision" must live in the dynamics, not in code that reads a metric and routes.

## Candidate table

Five families surveyed. The columns are: **(shape)** the mechanism's primitive, **(corpus-specificity)** the path by which corpus statistics enter the codebook update, **(anti-homunculus)** the local quantity the mechanism is expressible as, **(substrate touch)** what the existing FHRR + MHN + replay pipeline gains or loses, **(cost)** rough implementation effort to a first defensible smoke, **(rank)**: leader/alternate/heavy/declined.

| # | Family | Shape | Corpus-specificity | Anti-homunculus | Substrate touch | Cost | Rank |
|---|---|---|---|---|---|---|---|
| Γ1 | **Context-residual consolidation** (predictive-coding lineage: Dorrell/Whittington + Tang/Bogacz) | Additive vector update on the *unbind residual* `ε_j = atom_j − unbind(context-bundle, role_j)`; atoms move to reduce the residual their algebraic unbind from context leaves behind, not toward retrieval means | Direct: the residual is computed against the actual event-context bundle; corpus statistics enter through what each context's unbind produces | ✅ Local additive update on a substrate-pure vector quantity (the residual); no controller. Distinguished from existing "error-driven" pull/push at online_codebook.py:6-7 by being a vector-residual update rather than a discrete-classification-driven pull/push. | Adds a context-residual term to consolidation alongside or *replacing* Hebbian pull/push; existing FHRR substrate, MHN retrieval, and replay buffer all retained | ~2 weeks to defensible smoke | **Leader (Γ1)** |
| Γ2 | **Bundle-first scene-memory consolidation** (Report 067 lineage; Reports 075–099 evidence) | Codebook unit is a *scene bundle* in scene-MHN, not a per-atom slot; consolidation updates scene-level patterns; per-atom learning becomes a derived quantity from scene-decomposition | Indirect-but-strong: scene bundles inherit corpus structure from observed scenes; algebraic unbind exposes per-role atoms | ✅ MHN basin attraction is the dynamic; storage/cleanup happen as energy descent (already in production for scene-MHN reads, Reports 088–099) | Phase 3 is reframed: codebook items are scene patterns; the "atom" view becomes a readout from unbind, not the storage primitive | ~2–3 weeks (architecture reframe + Phase 3 driver rewrite) | **Alternate (Γ2)** |
| Γ3 | **SFA-head consolidation** (Franzius/Sprekeler/Wiskott 2007 lineage; brainstorm P3.B) | Slowness loss on the latent state across replay traces; atoms with slow trajectories become "role-like," fast-changing become "content-like" — the role/content distinction emerges from a local rate quantity | Direct: slow-vs-fast is computed from the actual replay trace, which is corpus-driven | ✅ Loss on a local quantity (state time-derivative); no supervisor | Adds an SFA loss term during consolidation; FHRR substrate retained; consolidation loop touched | ~2 weeks to smoke | **Alternate (Γ3)** |
| Γ4 | **Energy-based codebook training (EqProp / DSM warm-start)** (M2-adjacent per Report 067 §c.1 + Report 112 §"Why not M2") | Two-phase clamped/unclamped settling; codebook parameters update by contrastive divergence between phases | Direct: positive phase reflects the corpus, negative phase reflects the model | ✅ Energy descent + finite-difference recipe (no controller); EqProp's role-shuffled negatives need careful anti-homunculus review for "which contrast to apply" | Existing FHRR + MHN; consolidation loop replaced (not augmented) by EqProp recipe | ~2–3 weeks per Report 067 estimate | Heavy alternate (Γ4) |
| Γ5 | **Hyperseed-style content-addressable codebook growth** (Osipov et al.; literature-and-principles.md §"Emergent VSA codebooks") | Unsupervised competition over experiences; patterns that recur across context stabilize and become atoms; patterns that don't fade | Direct: which patterns recur is a corpus statistic | ✅ Hebbian-style stabilization + decay are local; the *growth* event (allocate-when-residual) is the part that needs anti-homunculus scrutiny | Replaces the fixed-size codebook with a growing one; non-trivial interaction with replay buffer & Phase 4 retrieval indexing | ~3 weeks (codebook-size becomes variable; lots of downstream code touched) | Heavy alternate (Γ5) |

### Self-Organizing Language (Eugenio) — *not* on the rank table

Per [literature-and-principles.md](literature-and-principles.md) it is the closest single-paper match to the project thesis. However it operates at a *higher* layer than Phase 3 in this project's stack: it builds tokens-and-grammar-and-reasoning from a sequential exposure stream via hierarchical Hopfield chains, where Phase 3 in *this* project is the codebook-growth layer beneath the static Phase 2 substrate. The Eugenio mechanism is a defensible candidate for Phase 5+/multi-scale work but is mis-scoped for Path γ-Phase-3-mechanism-redesign. Re-enters scope if a future Phase 5′ design needs a hierarchical-token primitive.

## Detail per candidate

### Γ1 — Context-residual consolidation *(leader)*

**Terminology note.** Distinct from the existing "error-driven contrastive updates" at [online_codebook.py:6-7](../../src/energy_memory/phase34/online_codebook.py), which drive Hebbian pull/push from a discrete classification error (`predicted_id != target_id`). Γ1's "error" is a *vector residual* in **atom-space**. The project name is **context-residual consolidation**; predictive coding (Tang/Bogacz NeurIPS 2023, Dorrell/Whittington ICLR 2025) is the theoretical lineage cited but not the project name.

**Substrate context (load-bearing).** At Phase 3, **roles are fixed substrate-position vectors** built by [phase2/encoding.py:18-26](../../src/energy_memory/phase2/encoding.py) (`positions[j] = bind(positions[j−1], step)` chained from a frozen `base`). The codebook holds **only filler atoms** (token → hypervector). There are no learned role atoms at this layer. Γ1 therefore updates **only filler atoms**; unbind operates on substrate-rooted quantities. (Phase-5 brainstorm candidates that assume role/filler symmetry — GHRR, modular resonator, phase-clock — are out of Path γ scope for this reason.)

**Mechanism — variant (c) post-cleanup atom-residual, asymmetric.** Replace the existing pull/push update at online_codebook.py:144-164 with: for each buffer entry where `predicted_id != target_id`, compute residual `ε = codebook[target_id] − codebook[predicted_id]` (atom-space, not query-space) and update `codebook[target_id] ← codebook[target_id] + lr_cr · ε`. Asymmetric: `codebook[predicted_id]` is *not* updated by Γ1 (it may still be moved by composed C.2.x dynamics; Γ1.c itself touches only the target atom). When `predicted_id == target_id`, ε = 0 by construction and no Γ1 update fires.

This is geometrically "push the correct atom away from the wrongly-retrieved atom in atom-space." Distinct from existing pull/push, which is atom-vs-cue (slot_query) geometry. The two forces compress different parts of codebook geometry: pull/push tightens basins around cues; Γ1.c separates confusable atoms from each other.

**Decision history during this design pass:**
- Terminology: "context-residual" chosen over "PE update" / "predictive-coding consolidation updates" to disambiguate from existing error-driven contrastive pull/push (which is also error-driven, but on discrete classification).
- Replace vs augment pull/push: **replace** (gated by config flag; pull/push retired but not code-deleted). Cleanest A/B against Path C empirical result.
- Residual definition: **variant (c)** post-cleanup atom-residual. Variants (a) leave-one-out and (b) pre-cleanup are both mathematically degenerate (collapse to identity update or to per-event Hebbian pull respectively).
- Symmetric vs asymmetric update: **asymmetric** (target atom only). Smaller blast radius for clean A/B; symmetric arm is a follow-up if asymmetric Γ1 partially-succeeds (60–69% per-seed paired Δ > 0).

**Why this shape avoids the C.2 noise floor.** Hebbian pull/push asks "where is the centroid of cues that retrieved this atom (correctly or incorrectly)?" — a quantity dominated by basin geometry, which the v3 → v6 chain showed produces structural-change-without-corpus-learning. The context-residual asks "where does this atom systematically *fail to be the unbind-target* of its own context?" — a quantity that requires corpus regularities (specifically, role-filler co-occurrence regularities) to compute. The two updates have different equilibria: pull/push equilibrates to within-basin centroid (geometric); context-residual equilibrates to a representation that minimizes role-conditional unbind residuals (statistical). The Path C empirical evidence is that the pull/push equilibrium is reachable but doesn't carry corpus signal. A context-residual equilibrium is a different fixed point, untested in this project.

**Anti-homunculus check.** The context-residual update is an additive substrate-pure vector force per atom. Every atom in the event gets its residual applied every update — there is no metric-triggered routing. The "decision" of how strongly to update atom `i` lives in the magnitude of `ε_i`, which is local geometry of the residual (substrate state, not C.1 diagnostic log — H6 preserved). ✅

**Concrete operationalization (precommit-shaped, not implementation).**

- Per consolidation event, compute the bundle `b = Σ_j bind(role_j, atom_j)` from the current event window.
- For each role `j` in the event: predict `q̂_j = unbind(b_minus_j, role_j)` where `b_minus_j` is the bundle without role `j` (i.e., the *context*).
- The PE for the atom in role `j` is `ε_j = atom_j_actual − q̂_j`.
- Update: `θ_{atom_j} ← θ_{atom_j} + lr_pe · ε_j` (FHRR-projected back to unit modulus per atom).
- The α_anti term remains in place but operates on the geometry that PE-equilibrium produces, not on Hebbian-pull geometry.

**Risks / pre-precommit grilling needed.**

- The "context" definition (b_minus_j vs b vs trace history) is a free design choice. Different definitions give different equilibria. mp-grill-with-docs should pin this down before precommit.
- PE updates can have circular interactions with the codebook the system is using to predict — careful staging required (e.g., predict-with-stale-θ-update-current-θ pattern).
- Computational cost: O(K_roles) more substrate operations per event vs Hebbian pull. At K=5, ~5× the consolidation cost.

**Falsifiable precommit shape.** A precommit can state: "We expect the PE-update mechanism, with α_anti at the same value Path C used, to produce per-seed Δ > 0 in ≥ 70% of seeds at the same wikitext-2 operating point (D=4096, β=10, vocab=1000, window=8, lr_pe sweep over {0.05, 0.1, 0.2}) with n ≥ 10 seeds. If this fails, the mechanism is rejected and a different family is considered." That's a concrete operating-point envelope plus the revised criterion plus a sweep dimension — a precommit can be written against this.

### Γ2 — Bundle-first scene-memory consolidation *(alternate)*

**Mechanism.** Promote `scene_bundle` (the multi-role bundle of a scene's atoms) from a *cue* into a *first-class memory primitive*. The codebook is no longer a per-atom table; it is a scene-MHN holding `N_scenes` bundles, with per-atom information derived from algebraic unbind. Consolidation operates at the scene level: similar scenes consolidate into shared bundle attractors; unique scenes form new attractors.

**Why this shape might work.** Reports 067 + 075–099 already established that multi-role bundle-first survives MQAR cleanup at K_roles ∈ {2, 4, 8} (Report 067), recovers context-completion at K=16 (Reports 075–081), and produces near-ceiling candidate recovery with clean controls on cleaned natural-source protocols (Reports 092–099). The structural memory primitive *itself* has been empirically validated; what hasn't been done is making it the Phase 3 codebook unit. Currently it lives on top of an atom-codebook trained by Phase 3 Hebbian dynamics. Path γ-Γ2 would *replace* atom-Hebbian-training with scene-MHN consolidation.

**Anti-homunculus check.** Storage is MHN energy descent; retrieval is energy descent; cleanup is energy descent. The "scene-vs-atom" decision isn't a decision — at any moment the system holds bundles, and per-atom queries are answered by unbind followed by cleanup, both of which are dynamics. ✅ (with the caveat that "when to form a new scene attractor vs. update an existing one" is the dynamics question to design; the current Phase 5′ work has solved it as a basin-attraction-with-threshold, which passes anti-homunculus if the threshold is `1/β`-equivalent rather than externally arbitrated).

**Risks / pre-precommit grilling.**

- This is the largest architectural reframe of the five candidates. Phase 3 driver, regime classifier, consolidation event interface, and Phase 4 candidate generation all change shape.
- The previous range-shaped replay downstream work (Reports 070–074, 110–111) closed *one specific* downstream lane — bundle-first scene memory has different downstream lanes that may or may not have the same failure mode.
- The currently paused Phase 5 ΔE bridge work has *already* localized two bridge failure modes (`raw_scene_energy_v0` saturation; `min_branch` consumer arbitration shape, per audit-phase5-2026-05-26.md §9). Γ2 Path γ work must not *re-open* the bridge — it stays in Phase 3 territory, focused on the codebook-as-scene-bundles question.

**Falsifiable precommit shape.** "The bundle-first Phase 3 mechanism, at the wikitext-2 operating point, will produce Recall@K (regime-stratified) per-seed Δ > 0 in ≥ 70% of seeds at n ≥ 10. The 'regime' axis stratifies scenes (not atoms) by their bundle-MHN basin tightness, computed via a scene-MHN analog of `consolidation-geometry-diagnostic.md`." The stratification axis is non-trivial design work and must be in the precommit, not invented during the experiment.

### Γ3 — SFA-head consolidation *(alternate)*

**Mechanism.** Add a slowness loss `L_SFA = Σ_i ||dθ_i/dt||²` during consolidation. The loss pushes atoms whose contexts genuinely change less per event to update less per event — these atoms become "slow features." Atoms with high-frequency context turnover update faster. The role/content distinction emerges from rate.

**Why this shape might work.** Franzius/Sprekeler/Wiskott 2007 showed slowness *alone* produces head-direction-like cells (= role-like cells in this project's vocabulary) from raw input streams. No supervision; the slow-vs-fast distinction is a corpus statistic about which features change at which rate. C.2's regime classifier already produces a `tight`/`spread` axis that may be reading the same underlying signal post-hoc; SFA bakes it into the update.

**Anti-homunculus check.** Slowness loss is a function of a local quantity (per-atom rate). No controller. ✅

**Risks.**

- SFA is genuinely white-space for VSA codebooks (no published group has done this). High variance in outcome.
- Interacts with α_anti in unclear ways — α_anti is a *fast* repulsion; SFA is a *slow* preservation. The two may fight.
- "Slow" requires a time-scale choice (rolling window over how many events?). Mis-set, SFA either dominates or vanishes.

**Falsifiable precommit shape.** "SFA-augmented consolidation at the wikitext-2 operating point, with `lr_pull` from Path C (0.1) and an SFA weight `λ_sfa` swept over {0.01, 0.1, 1.0}, will produce per-seed Δ > 0 in ≥ 70% of seeds at n ≥ 10." Sweep dimension is needed because SFA's effect-size is unknown a priori — it could need to dominate or barely whisper.

### Γ4 — Energy-based codebook training (EqProp / DSM warm-start) *(heavy alternate)*

**Mechanism.** Two-phase settling: positive phase clamps the input to a corpus example; negative phase frees the input; codebook updates by contrastive divergence between phase energies. DSM warm-start initializes the codebook from a denoising score-matching pretrain pass to avoid the cold-start instability EqProp is famous for.

**Why this shape might work.** EqProp's negative phase is essentially "sample from the model's current beliefs." Differences from the positive phase localize where the codebook is corpus-inconsistent. This is mathematically a different equilibrium from Hebbian pull (which is consistent-with-retrievals-only) and from PE (which is consistent-with-predictions-only). EqProp targets consistent-with-full-corpus-statistics.

**Why this is heavy (per Report 112 §"Why not M2").** Already estimated at ~2–3 weeks for a defensible smoke. Negative-phase tuning, role-shuffled-negatives anti-homunculus review, DSM warm-start design, and EqProp two-phase staging all need careful work.

**Why this is *alternate* not *declined*.** If Γ1 and Γ3 both hit the noise floor, the diagnosis becomes "all Hebb-and-PE-shaped updates fail, the substrate needs a fundamentally different training signal." EqProp is the canonical answer to that diagnosis. **Defensible to defer until Γ1/Γ3 close.**

### Γ5 — Hyperseed-style content-addressable updates *(heavy alternate)*

**Mechanism.** Codebook grows as the system observes. Each event computes residual prediction error in the current codebook; if the residual exceeds a threshold (allocation criterion), a new atom is allocated bound to the current context. Stable atoms (those that get reactivated by similar contexts repeatedly) survive; unstable atoms decay.

**Why this is heavy.** The codebook-size-becomes-variable change touches Phase 2 (substrate interface), Phase 3 (this), Phase 4 (replay buffer indexing), and Phase 5 (scene-MHN row count). Three weeks of refactoring before the first defensible smoke.

**Why this is *alternate* not *declined*.** Of all five candidates, this is the cleanest fit to literature-and-principles.md §"Growing codebooks" — the principle is "atoms enter the system through experience, not by declaration," which is exactly what Γ5 instantiates. **Defensible to defer until simpler candidates close.**

**Attribution caveat** (per [verification-report.md](../../brainstorm-workspace/2026-05-24-unconsidered-paths/verification-report.md)): the brainstorm doc's "Hyperseed" citations need re-verification against actual Osipov et al. papers before any precommit; the 2026-05-24 audit flagged citation drift in adjacent material.

## Decline list

Mechanisms considered and *not* surveyed in detail, with reasons:

- **Range-shaped replay (Dorrell-Whittington data-side intervention).** The downstream lane is closed by Report 111 (`not_viable_current_range_replay_downstream_novelty_without_retrieval`). The sampler algorithm is validated (Report 068); the downstream codebook+retrieval pipeline does not move under it. Re-attempting this as Path γ-Phase-3 requires a new mechanism precommit (per STATUS.md), not a scale-up of the closed lane. *Possibly* re-enters scope under a *new* shape (e.g., range-shaped *gradient* signals on consolidation, not just sampling) — but that's effectively Γ1 with a range-shape sampler in front, which is more cleanly proposed as a Γ1 sweep dimension than a separate candidate.
- **Residue HDC / GSBC / GHRR substrate swaps.** These are *substrate*-layer interventions; the Phase 3 mechanism question is orthogonal. Reserved for a separate phase-graduation decision (Phase 0/1 reopening), not Path γ.
- **PCN-above-MHN (Salvatori line).** This is a *Phase 5* mechanism (retrieval-side); Phase 5 is paused. Re-enters scope when Path γ produces a graduated Phase 3 mechanism that Phase 5 can be built atop.
- **Phase-clock slow variable, Modular resonator network, Theta-gamma binding.** All Phase 5 retrieval-side mechanisms per the 2026-05-24 brainstorm. Out of scope until Path γ closes.
- **C.2-style diagnostic-as-actuator variants with different specific dynamics.** Falsified at the *family* level by Path C. A new variant of "diagnostic measurement drives Hebbian-pull update" is in the same shape class as the closed C.2 mechanism and predicted to hit the same noise floor.
- **Eugenio Self-Organizing Language.** Mis-scoped for Phase 3 (see "not on rank table" above).

## Recommendation

**Lead with Γ1 (predictive-coding consolidation updates).** Three reasons:

1. **Diagnoses Path C cleanly.** Γ1 changes the *shape* of the consolidation update (Hebbian-pull → PE-update), keeping operating point, substrate, controls, and metrics constant. If Γ1 also fails, the diagnosis is much stronger than Path C left it: "neither Hebbian nor PE shapes carry corpus signal at this scale" — which justifies moving to heavier candidates (Γ4 EqProp, Γ5 Hyperseed). If Γ1 works, the project advances to Phase 5′ on a defensibly-different mechanism.
2. **Smallest blast radius among non-falsified candidates.** Substrate, MHN, replay buffer, Phase 4 candidate generation, Phase 5′ scene-MHN all retained. The change is a single update rule in [src/energy_memory/phase3/online_codebook.py](../../src/energy_memory/phase3/online_codebook.py) (or wherever `_consolidate()` lives) plus its precommit and tests.
3. **Externally endorsed.** Tang/Bogacz NeurIPS 2023, Salvatori et al. ECAI 2023, Dorrell/Whittington ICLR 2025 + Neuron 2025 cluster all converge on PE-style local updates in associative memory. The project has been carrying this citation cluster as "in scope" since the 2026-05-24 brainstorm without exercising it. Path γ-Γ1 *is* this work.

**Hold Γ2 (bundle-first scene memory) as the immediate fallback.** Γ2 is the most architecturally invasive of the alternates but has the strongest empirical support already accumulated (Reports 067 + 075–099). If Γ1 fails and the post-mortem says "the per-atom codebook is the wrong unit, not the update rule," Γ2 is the precommit that follows.

**Hold Γ3 (SFA-head) as the diagnostic alternate.** If Γ1 partially-works (e.g., per-seed Δ > 0 in 50–69% of seeds — *almost* meets the revised criterion), Γ3 is the cheapest follow-up to test whether adding a slowness term over Γ1's PE-update pushes per-seed robustness above 70%. SFA composes naturally with Γ1 (both are local loss terms during consolidation).

**Hold Γ4 (EqProp) and Γ5 (Hyperseed-growing-codebook) as heavy alternates.** Both have ~3-week implementation cost and broader architectural touch. Defer until Γ1+Γ2+Γ3 close.

## Next deliverable after this survey

**Γ1 precommit.** Single document under [notes/notes/](../notes/) (filename: `2026-05-2X-path-gamma-gamma1-pe-update-precommit.md`) covering:

- **(a)** Exact PE-update equations and context definition.
- **(b)** Mechanism diagnostic↔actuator identity (the H1–H7 discipline from the Path C precommit), explicitly listing what is *not* a controller.
- **(c)** Falsifiable graduation criterion: revised C.3 spec (per [phase-3-deep-dive.md:188-205](phase-3-deep-dive.md)) plus operating-point envelope plus sweep dimensions.
- **(d)** Anti-homunculus check rendered as a fillable section the reviewer agent can audit.
- **(e)** Implementation surface: which file changes, which tests must pass, which existing reports' results must be preserved as parity (no regression).

**Anti-homunculus-reviewer PASS on the precommit is binding before any code lands.** Per CLAUDE.md §"The anti-homunculus filter" and the Path C precommit's H1–H7 discipline.

## Open questions to surface before precommit

1. **Context definition for Γ1.** The PE-update needs a concrete `q̂_i(context)`. The cleanest minimal version is `q̂_i = unbind(b_event_minus_role_i, role_i)`. A richer version uses trace history (`unbind(EMA(b_traces), role_i)`). The user should pick one — they have different anti-homunculus profiles (the EMA version introduces a time-scale parameter whose justification needs scrutiny).
2. **Whether to retain α_anti during Γ1 smoke.** The Path C analysis identified α_anti as firing but corpus-independent. Retaining α_anti at Path C's value isolates the PE-vs-Hebb shape change; ablating α_anti isolates the corpus-specificity question. The cleanest precommit runs both as conditions.
3. **Whether to revise the operating point.** Path C's wikitext operating point (D=4096, β=10, vocab=1000, window=8) is well-characterized but isn't necessarily the operating point at which PE updates work. A 1-day Γ1 sensitivity probe could be cheap to add before the n=10 graduation gate.
4. **Whether to run Γ1 and Γ3 in parallel.** Γ3 (SFA) composes with Γ1 (PE); they can be precommitted as a single experiment with conditions {PE-only, SFA-only, PE+SFA, neither (= Path C baseline)}. This adds ~1 day of additional smoke time but produces a much stronger interaction matrix in the post-mortem.

---

*See also: [phase-3-deep-dive.md](phase-3-deep-dive.md) (revised C.3 criterion at lines 188-205), [literature-and-principles.md](literature-and-principles.md), [Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md), [2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](../notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md), [verification-report.md](../../brainstorm-workspace/2026-05-24-unconsidered-paths/verification-report.md) (citation caveats).*
