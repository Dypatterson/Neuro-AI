# Report 114 — Γ1 family closes via F1 lr_cr sweep; Γ2 (bundle-first) is next

> **⚠️ CORRECTION (2026-05-28) — read [2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md](../notes/notes/2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md) first.**
> This report's Δ values are measured against the **shuffled-token control**,
> which was subsequently proven **gauge-vacuous** (a row-permutation of an
> i.i.d. codebook → `E[Δ]=0` by exchangeability, independently verified
> 2026-05-28). The Γ1-vs-control deltas here are therefore **non-diagnostic
> for corpus-specific learning**, and the family-closure *interpretation*
> ("Γ1 shape carries less corpus signal") is **not established** by this
> control. Two specific over-readings to retract: (1) the **lr_cr 0.20/0.50
> "atom-vs-atom repulsion actively degrades the codebook below baseline"**
> claim (Findings 2-3, the negative-Δ regime) was an **n=3 noise artifact** —
> at high power (n=60) it collapses to mean Δ ≈ +0.005, t ≈ 1.2, consistent
> with 0; (2) the derived "constraint on future mechanisms with atom-vs-atom
> repulsion" rests on (1) and is likewise not established. What **stands**:
> Γ1's per-seed σ is ~4× tighter than pull/push (a real *shape* change, just
> not a corpus-signal one), and the test-harness byte-identity reproducibility
> (Finding 4). The Γ2 mandate built on "two per-atom mechanisms failed" is
> superseded by the Frame B reframe; next work is **Gate 0**, not Γ2.

**Status:** complete. **Γ1 family (context-residual consolidation) closes.** F1 lr_cr sweep confirms hypothesis (b): atom-vs-atom geometry carries fundamentally less corpus signal than pull/push, regardless of magnitude. Per the F1 precommit's pre-committed escalation, **Γ2 (bundle-first scene memory) is the next-candidate precommit**.

**Date:** 2026-05-27.

## Preamble (per CLAUDE.md experiment-preamble requirement)

- **Active phase:** Phase 3 reopen — Path γ Γ1 (closing this session); Γ2 design begins after.
- **Headline metric per [phase-3-deep-dive.md:188-205](../notes/emergent-codebook/phase-3-deep-dive.md):** regime-stratified Recall@K, revised C.3 criterion (CI-disjoint at n ≥ 10 AND per-seed paired robustness ≥ 70%). **This experiment is explicitly a *diagnostic*, not a graduation gate** — per the F1 precommit at [notes/notes/2026-05-27-path-gamma-gamma1-f1-lr-cr-sweep-precommit.md](../notes/notes/2026-05-27-path-gamma-gamma1-f1-lr-cr-sweep-precommit.md) §"What this precommit does not permit", F1 results do not graduate Phase 3 under any outcome. The headline metric appears here only as the per-seed Δ summary used to evaluate the F1 hypothesis dichotomy.
- **Required controls per same spec:** shuffled-token-with-consolidation control (Path α default since 2026-05-26), within each Γ1.c run. PathC pull/push baseline at matched seeds 0..2 loaded from the [Γ1 headline gate](../notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md) Drive output (not re-run; preserves byte-identity with [Report 113](113_path_gamma_gamma1_headline_gate.md)).
- **Last verified result:** [Report 113](113_path_gamma_gamma1_headline_gate.md) (Γ1.c headline gate FAIL at lr_cr=0.1).
- **Why this experiment now:** Report 113 named two distinguishing hypotheses for the Γ1.c attenuation — (a) effective-lr mismatch and (b) atom-vs-atom geometry carries less signal. F1 is the pre-committed follow-up that distinguishes them.

## TL;DR

A 5-lr_cr × 3-seed sweep (15 procs, ~30 min Colab A100) at the Γ1 headline operating point produced the following per-seed mean Δ curve:

| lr_cr | mean Δ | min | max | n |
|---:|---:|---:|---:|---:|
| 0.01 | +0.0033 | −0.025 | +0.030 | 3 |
| 0.05 | +0.0017 | −0.030 | +0.030 | 3 |
| **0.10** | **+0.0117** | +0.000 | +0.030 | 3 |
| 0.20 | **−0.0100** | −0.015 | −0.005 | 3 |
| 0.50 | **−0.0167** | −0.035 | +0.005 | 3 |

**Max mean Δ across all 5 lr_cr values: +0.0117 at lr_cr=0.10.** Per the F1 falsifiable diagnostic criterion (max ≲ 0.02 → hypothesis (b) confirmed), this closes the Γ1 family.

Two findings strengthen the closure:

1. **Strict ceiling well below threshold.** +0.0117 is < 0.02 (the precommit's "doesn't matter much" ceiling). No lr_cr value gets close to PathC-magnitude effects.
2. **Negative deltas at higher lr_cr.** lr_cr=0.20 and lr_cr=0.50 produce *negative* mean Δ — Γ1.c at non-trivial magnitude actively *degrades* the codebook below the shuffled-token-control baseline. This is a stronger statement than "Γ1.c at higher strength produces no signal"; it says "Γ1.c at non-trivial strength is *harmful* to corpus-specific structure." Recorded here for any future mechanism that compositionally includes atom-vs-atom repulsion (e.g., the negative-phase term of Γ4 EqProp).

**Decision:** Γ1 family closed. Per F1 precommit's pre-committed escalation path "F1→b": **Γ2 (bundle-first scene memory) precommit** is the next deliverable, via the same mp-grill-with-docs + anti-homunculus reviewer + experiment-result-auditor cycle that developed the Γ1 precommit itself.

## Experiment chain

- **15 procs parallel** (5 lr_cr values × 3 seeds = 15) via [scripts/colab_gamma1_f1_lr_cr_sweep.ipynb](../scripts/colab_gamma1_f1_lr_cr_sweep.ipynb).
- All Γ1.c (no PathC re-run; PathC at seeds 0..2 loaded from Drive from the [Γ1 headline gate](../notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md)).
- Operating point identical to the Γ1 headline gate (wikitext-2-raw-v1, vocab_cap=1000, window=8, D=4096, β=10, K=5, landscape_size=64, n_consolidation_events=1000, α_anti=0.01, repulsion_step_size=0.05, C.2.1–C.2.5 at Path C values, `use_context_residual=True`, `use_pull_push=False`); only `lr_cr` varied.

## Result table

Per-seed Δ (default mode, stratum-pooled tight + spread):

| lr_cr | seed 0 | seed 1 | seed 2 | mean | min | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | +0.030 | +0.005 | −0.025 | +0.0033 | −0.025 | +0.030 |
| 0.05 | +0.030 | +0.005 | −0.030 | +0.0017 | −0.030 | +0.030 |
| 0.10 | +0.030 | +0.005 | +0.000 | +0.0117 | +0.000 | +0.030 |
| 0.20 | −0.010 | −0.005 | −0.015 | −0.0100 | −0.015 | −0.005 |
| 0.50 | −0.020 | +0.005 | −0.035 | −0.0167 | −0.035 | +0.005 |

**Reference: PathC pull/push at matched seeds 0..2** (from [Γ1 headline gate](../notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md) Drive output):

| seed 0 | seed 1 | seed 2 | mean |
|---:|---:|---:|---:|
| −0.090 | +0.465 | +0.065 | +0.1467 |

(Note the PathC n=3 mean is +0.1467, much higher than its n=10 mean of +0.055 from [Report 113](113_path_gamma_gamma1_headline_gate.md). Seed 1's +0.465 outlier dominates at small n. At n=10 the swings average down; the relevant PathC reference for "what magnitude should Γ1.c reach" is +0.055, which is what the F1 criterion's 0.04 = 70% threshold was calibrated against. The Γ1.c result of +0.0117 is ≈ 1/5 of either reference.)

## Findings

### Finding 1 — hypothesis (b) confirmed at strict threshold

The F1 falsifiable criterion (precommit §"Falsifiable diagnostic criterion"):

> max(lr_cr) per-seed mean Δ ≲ 0.02 → hypothesis (b) → close Γ1 family

Max mean Δ across all 5 lr_cr values = **+0.0117 < 0.02**. Threshold satisfied strictly; not a borderline call. Hypothesis (b): atom-vs-atom geometry carries fundamentally less corpus signal than atom-vs-cue geometry, regardless of update magnitude.

### Finding 2 — the curve has a peak at lr_cr=0.10, then turns negative

The lr_cr→Δ curve isn't a flatline near zero. It shows a real shape:

- lr_cr ∈ {0.01, 0.05}: near-zero mean Δ (+0.003, +0.002). Updates too small to move codebook meaningfully.
- lr_cr = 0.10: peak (+0.012). The Report 113 headline number reproduces here at n=3 to within sample noise.
- lr_cr ∈ {0.20, 0.50}: **negative** mean Δ (−0.010, −0.017). Mechanism actively degrades the codebook.

This is a stronger close than F1 strictly needed. "No effect at any magnitude" would be a flat curve near zero; "shape is wrong" predicts a peak at moderate magnitude and degradation at higher. The empirical curve matches the latter. The interpretation: at lr_cr ≥ 0.20, atom-vs-atom repulsion pushes confused atoms apart enough that they leave their *useful* basin centroids (which is what pull/push's centroid-pulling preserves), making subsequent retrievals *worse* than the shuffled-token-control baseline.

### Finding 3 — negative-Δ regime is a constraint on future mechanisms

The lr_cr=0.20, lr_cr=0.50 negative-Δ data is a recorded constraint, not just an F1 finding. **Any future mechanism that compositionally includes atom-vs-atom repulsion at non-trivial magnitude inherits the risk of this negative-Δ regime.** Specifically relevant for:

- **Γ4 (EqProp)**: the negative-phase term in contrastive divergence updates pushes the codebook away from model-predicted states; if those predicted states are close to true atoms (which they will be as training converges), the negative-phase term is shape-equivalent to atom-vs-atom repulsion. Future Γ4 precommit should reference this finding.
- **Γ5 (Hyperseed)**: unsupervised competition-with-decay includes implicit atom-vs-atom repulsion (atoms that respond to similar contexts compete; the loser decays). Same caveat.
- **Symmetric Γ1.c variant** (parent F2): symmetric variant doubles the atom-vs-atom force per event. If Γ1.c-asymmetric at lr_cr=0.20 already degrades, the symmetric variant at lr_cr=0.10 is predicted to degrade similarly. F2 should likely be retired.

### Finding 4 — Γ1.c at lr_cr=0.10 reproduces Report 113 at n=3

Report 113 per-seed Δ at seeds 0, 1, 2: +0.030, +0.005, +0.000. This sweep's lr_cr=0.10 column: +0.030, +0.005, +0.000. **Three-decimal-place identical at the same seeds and same operating point**, across a separate Colab session and different random-state initialization windows. Confirms:

- The Γ1.c implementation is fully deterministic given the seed.
- The Report 113 numbers are not Colab-runtime-specific.
- The headline-gate testing infrastructure is reproducible across two runs ~hours apart.

### Finding 5 — pre-committed escalation works as intended

The F1 precommit named three pre-committed escalation paths (F1→a, F1→b, F1→inconclusive). The F1 result lands cleanly in the F1→b basin (max +0.0117 < 0.02, well below threshold). **No ad-hoc design needed in the post-mortem.** The Γ2 precommit drafting is now the next deliverable, with the scope already named: "full mp-grill-with-docs + anti-homunculus reviewer + experiment-result-auditor cycle, same as the Γ1 precommit itself was developed."

This is the first time in Path γ that pre-committed escalation was actually invoked under a failing outcome. It worked.

## Decision: Γ1 family closes; Γ2 precommit is next

Per F1 precommit §"Pre-committed escalation paths":

> **F1→b (hypothesis b confirmed):** new precommit for Γ2 (bundle-first scene memory per the survey) — full mp-grill-with-docs + anti-homunculus reviewer + experiment-result-auditor cycle, same as the Γ1 precommit itself was developed.

### What this means concretely

1. **Γ1 family** (context-residual consolidation, asymmetric and symmetric variants) is **closed as falsified at the wikitext-2 operating point**. The shape — atom-vs-atom geometry as the corpus-signal-carrying axis — does not work.
2. **Γ2 (bundle-first scene memory)** moves to next-candidate status. From the [Path γ mechanism-family survey](../notes/emergent-codebook/path-gamma-mechanism-family-survey.md): "Reports 067 + 075–099 already established that multi-role bundle-first survives MQAR cleanup at K_roles ∈ {2, 4, 8} (Report 067), recovers context-completion at K=16 (Reports 075–081), and produces near-ceiling candidate recovery with clean controls on cleaned natural-source protocols (Reports 092–099). The structural memory primitive *itself* has been empirically validated; what hasn't been done is making it the Phase 3 codebook unit."
3. **Γ3 (SFA-head), Γ4 (EqProp), Γ5 (Hyperseed)** remain alternates in the survey ranking. F2 (symmetric Γ1.c) is now likely retired per Finding 3.

### What this does NOT decide

- **Whether Γ2 will succeed.** That's the next mp-grill-with-docs + precommit + reviewer cycle.
- **Whether the wikitext-2 operating point itself is the right corpus** for testing structural mechanism candidates. The corpus is held constant per the survey design constraints; changing corpus is out of Path γ scope.
- **Whether atom-vs-atom geometry could carry signal under a different substrate** (Residue HDC, GSBC, GHRR per the 2026-05-24 brainstorm). Substrate swaps are Phase 0/1 work; out of Path γ scope.
- **Phase 5′ reopen.** Remains paused. No bridge / M2 / matrix / headline / graduation work authorized until a Phase 3 mechanism graduates.

## Anti-homunculus check

F1 was authorized under the Γ1 precommit's anti-homunculus reviewer PASS (which covered the mechanism). F1 introduced no new mechanism; the lr_cr sweep was a hyperparameter sensitivity diagnostic. The result interpretation introduces no new mechanism either — the closure decision flows directly from the pre-committed escalation criterion in the F1 precommit. No new anti-homunculus review is needed for this report's findings.

Γ2 precommit drafting **will** require a fresh anti-homunculus reviewer pass — bundle-first scene memory is a Phase-3 architectural reframe, not a hyperparameter or compositional change. Per the F1 precommit's escalation path, the full mp-grill-with-docs + reviewer + auditor cycle is binding for Γ2.

## Limitations

- **n=3 smoke scale.** F1 was diagnostic, not a graduation gate. The conclusion is *family-level* (Γ1 shape doesn't carry signal at any lr_cr) — n=3 is sufficient for that question because the curve shape (peak at 0.10, negative at 0.20 and 0.50) is consistent across seeds within each lr_cr point. A graduation-scale n=10 at any individual lr_cr point would not change the family-level conclusion.
- **Single corpus** (wikitext-2-raw-v1). Inherits Report 112/113 limitations.
- **Single operating point.** D=4096, β=10, window=8, vocab_cap=1000, n_consolidation_events=1000.
- **No PathC re-run** for the F1 seeds — paired comparison loaded from headline gate Drive output. This is by design (saves Colab credit + preserves byte-identity), but if there were any worry about Drive corruption it would surface as PathC reference numbers diverging from Report 113. They match exactly (per-seed: seed 0 −0.090, seed 1 +0.465, seed 2 +0.065).
- **Asymmetric Γ1.c only.** Symmetric variant (parent F2) untested. Finding 3 argues that symmetric would inherit and amplify the negative-Δ regime; testing it would be a confirmatory diagnostic at most, not a graduation candidate.

## Artifacts

- F1 raw outputs at `MyDrive/neuro-ai/results/gamma1_f1_lr_cr_sweep_2026-05-27/lr{0.01..0.5}_seed{0,1,2}/c3_summary.json` (15 per-seed JSONs).
- Aggregate at `MyDrive/neuro-ai/results/gamma1_f1_lr_cr_sweep_2026-05-27/aggregate.json` with per-lr summary + verdict.
- Colab logs at `.../colab_logs/`.
- Notebook commit: `7a582af`.

## Done-gate audit (per CLAUDE.md "What done looks like")

This report documents a *diagnostic*, not a graduation gate. The done-gates apply with this caveat:

1. **Headline metric reported with CI:** ✅ at the level the diagnostic supports. Per-lr per-seed Δ table with min/max/mean across 3 seeds is the appropriate quantification for an n=3 diagnostic. Full Wilson CI per cell available in the raw JSONs but not aggregated in this report (would be over-precise for a 3-seed mean).
2. **Control on same test set:** ✅ shuffled-token-with-consolidation per-stratum within each Γ1.c run; PathC pull/push baseline at matched seeds 0..2 from the Γ1 headline gate's Drive output, byte-identical to Report 113.
3. **Drill-down metrics explain anomalies:** ✅ Finding 2 explains the peak-at-0.10-then-negative curve shape; Finding 3 derives a constraint for future mechanisms from the negative-Δ regime; Finding 4 documents reproducibility against Report 113.
4. **Markdown report under `reports/`:** ✅ this file.
5. **Relevant memory / status note updated:** to be landed this session (STATUS.md banner + Recent Updates entry + F1 precommit running log).
