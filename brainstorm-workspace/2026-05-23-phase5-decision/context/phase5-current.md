# Phase 5 Current Context Summary — 2026-05-23
## Decision Point: Strategic Closure after Substrate-Saturation Finding

**Date:** 2026-05-23  
**Status:** Graduation-unattained; substrate-saturation finding confirmed at 6 independent instances; user at critical decision point before next major effort (Phase 5 redesign vs closure vs pivot).

---

## What Phase 5 Was Trying to Do

**Architecture headline:** Energy-guided structural branching wrapped around HAM. Deploy K=4+1 parallel settling trajectories seeded by competing schema priors (role-binding priors vs content priors) on the same cue; final states settle under energy bias; branches combine via energy-weighted bundling + unbiased re-settling. The mechanism aims to expose **structural role-filler retrieval** through energy margin: role-prior branches should find lower-energy final states than content-prior branches when retrieved at the same substrate.

**Mechanism in plain English:**
1. **Schema store** = post-death Phase 4 substrate (5–30 atoms per seed at W=2; more at W=3/W=4), ranked by `effective_strength` from the Benna-Fusi cascade.
2. **Branching** = for a cue, rank schema store by FHRR cosine similarity; pick top-K main schemas + 1 surprise branch (highest `u_1 / u_m` novelty ratio).
3. **Per-branch settling** = each branch runs HAM with energy biased by its schema prior, `E_k(q) = -logsumexp(β·Xq*) - γ·Re(⟨q, p_k⟩)`, settled via iterated Hopfield.
4. **Scoring** = final unbiased energy `E_k(q_k*)` (prior term omitted), softmax weights `w_k = softmax_k(-E_k / τ)`.
5. **Branch combination** = weighted FHRR sum `q_bundle = Σ w_k q_k*`, re-settled unbiased → `q*`.
6. **Atom-splitting diagnostic** = joint criterion (≥2 low-energy branches AND max pairwise state distance > δ_state) marks split-eligible atoms; measurement only, no split action.

**Graduation criterion (per spec):** ΔE = E_content-prior − E_role-prior, CI disjoint from zero at n_seeds ≥ 10, applied to held-out structural-retrieval cue set.

---

## What the Substrate-Saturation Finding Says

**Definition:** Six independent instances where the substrate's **clean-retrieval geometry (engineered through A+B+A1')** produces phenomena that foreclose advancing Phase 5 at D=4096.

**Instance 1: β smoke test (report 050, 2026-05-20)**  
All 1064 post-death atoms have role-fidelity `f_i = 0.9858 ±  0.0015` (variance σ² ~ 1e-6). Root cause: FHRR crosstalk noise floor `1/√D ≈ 0.0156` at D=4096 sets `f_i = mean(1 - |G_jk|) ≈ 0.984` independent of pattern content. The K-branch mechanism's ability to discriminate which atoms are most role-fidelity-relevant collapses to zero variance. **The substrate doesn't expose the signal the test looks for.**

**Instance 2: K-branch state divergence collapse (report 042, 2026-05-20)**  
Post-death: branches settle to final states with cosine distance 6.7e-6 to 2.7e-3 (equi-energetic to FP precision, softmax entropy exactly ln(K)); pre-death: 3×–4978× larger divergence. Root cause: **effective dimensionality collapse** (report 044). **K=4 branches, d_eff ≈ 5 post-death** → branches drain to degenerate point; no subspace span for distinct attractors.

**Instance 3: Softmax entropy saturates (reports 042, 050)**  
Cross-seed K-sweep at n=5: entropy = ln(K) to floating-point precision across K ∈ {1..8}. All branches are energetically equivalent. **Sharp-basin substrate produces equipotent attractors; the branches don't stratify by energy.**

**Instance 4: B2 control fires (reports 041, 042)**  
At n=5: K=1 ΔE ≥ K=4 ΔE across all 5 seeds (K=1 +1.3e-4, K=4 −1.5e-4). Branching is gratuitous; single branch outperforms. **The K-branching premise itself fails on post-death geometry.**

**Instance 5: Directional signal sub-noise (report 053, 2026-05-21)**  
n=10 headline: ΔE = +0.00130, CI [+0.00071, +0.00193], all 10 seeds positive. Magnitude floor (pre-committed at 5.5e-3 derived from substrate noise formula at D=4096, β=10, N≈1064) → **ΔE is 4.2× below floor.** Statistically detectable (CI excludes zero) but sub-substrate-noise in magnitude. **The signal is architecturally inert.**

**Instance 6: ΔE/basin-hit anti-correlation (report 058, 2026-05-23)**  
Cross-seed cue-regime sweep (24 cells, content_distortion × binding_noise_std grid):
- **Best ΔE cell** (cd=0.6, bns=0.05): ΔE +0.00248, **hit_role = 0.00**, rank_role = 462.1 (out of 1064)
- **Best basin-hit cell** (cd=0.0): hit_role = 0.02, **ΔE = −0.000109** (negative)
- **No cell carries both signals.** ΔE and basin-membership metrics move in opposite directions across the cue-regime axis. Whatever paired-ΔE measures in the cd ≥ 0.6 region is **not** role-target basin retrieval.

**Why these six are independent:**  
1 concerns per-atom encoding variance → substrate-side  
2 concerns branching geometry → substrate-side  
3 concerns settling landscape sharpness → substrate-side  
4 concerns branching utility → mechanism-side  
5 concerns magnitude-to-noise ratio → measurement-side  
6 concerns headline-metric misalignment → measurement-side  

All six point to the same root cause: **A+B+A1' substrate's clean-basin architecture produces sub-noise energy gaps between structurally-different priors.** The issue is not parameter tuning or mechanism breakdown; it's that the substrate was engineered for sharp self-retrieving basins, which makes the role-prior energy advantage vanishingly small relative to the substrate's energy noise floor.

---

## What's Foreclosed Empirically

**Option 2: Sub-floor advance to Phase 6**  
*Falsified by Instance 5 (report 053) + pre-committed magnitude-floor gate.*  
ΔE passes CI-disjoint-from-zero but fails magnitude floor by 4.2×. The pre-commit binds: "CI disjoint is necessary, not sufficient; magnitude ≥ 5.5e-3 required to be architecturally meaningful." Phase 5 does not graduate; advancing to Phase 6 on this result would vacuously claim "structural retrieval" on a signal below substrate noise.

**Option 3: Headline reformulation (basin membership / R@K metric)**  
*Falsified by Instance 6 (report 058), specifically the ΔE/basin-hit anti-correlation.*  
Report 057's suggestion ("different metric reads out same structure") was the leading direction. Empirical test: sweep 24 cells of cue construction; measure both ΔE and basin metrics simultaneously. **Result: they are anti-correlated.** Best ΔE cell (cd=0.6) has zero role-target hits. Best basin-hit cell (cd=0.0) has negative ΔE. No single cell can serve as a reformulated headline because the substrate carries no structure that looks like "role-target basin at high-energy state AND lower energy under role prior."

**Option 4: Basin-shape priors (design priors over basin geometry, not pattern identity)**  
*Weakened by Instance 6.* With basin hit ≈ 0 across the grid and role-target rank 264–504 out of 1064, there is no exploitable role-target basin geometry for a shaped prior to bias toward. Basin-membership is not a structured property of the retrieved states on this substrate; it's random-assortment geometry.

---

## What's Still Live

**Option 1: Lower-D redesign**  
The **only path with a plausible mechanism.** The magnitude-floor formula scales with D: `(1/β)·log(1 + (N−1)·exp(−β·(1−1/√D)))`. At D=512 the noise floor would be ~8× higher; at D=1024 it would be ~4× higher, making the current ΔE signal closer to ~1–2× floor rather than 4.2× below. Re-engineering Phase 4 substrate construction for lower D would require:
1. **D-sweep diagnostic** (Phase 5, ~2 days): measure ΔE vs floor scaling empirically across D ∈ {256, 512, 1024, 2048, 4096} on a single seed to characterize the dim-dependence of failure modes.
2. **Phase 4 retrain** (contingent, 1–2 weeks): if D-sweep shows a viable lower-D regime, retrain the consolidated substrate at that dimension and re-run Phase 5 headline.
3. **Design re-scoping** (contingent): whether a+B+A1' design transfers cleanly to lower D is an open empirical question; may need mechanism adaptations.

**D-sweep diagnostic:** Place 100–200 cue trials at K=1, β=10, γ=0.5 per dimension, compute ΔE and floor empirically, report scaling vs theory. **This answers the structural question: is lower-D a plausible path, or does the same anti-correlation problem recur?**

**Close + Pivot (lightweight contingency):**  
If D-sweep shows lower-D does not rescue ΔE (e.g., anti-correlation persists), or if the re-engineering cost is prohibitive:
- **Close Phase 5 as graduation-unattained**, documenting the substrate-saturation finding.
- **Pivot to pair #4 (metastability / replay-prioritization)** as Phase 5's actual target. The substrate already computes metastability per-retrieve call; adding per-atom EMA of metastability into ReplayStore priority is ~½ day plumbing. Headline becomes Δ meta-stable-rate at W=3 (substrate-pure metric Phase 4 already used) rather than ΔE on energy margin. **This closes one of the 2026-05-09 diagnostic-actuator pairs without requiring substrate reconstruction.**

---

## Subtle Constraints to Remember

**Audit constraint #10 (immovable):**  
No post-hoc retuning of κ, μ_obs, β, γ, K_main, formulation once a magnitude floor is set. The 5.5e-3 floor was derived *before* report 053 ran, using the substrate noise formula. No tuning in response to failure. This binds all options — lower-D requires explicitly re-deriving the floor for that dimension, not lowering the target.

**Anti-homunculus filter (design discipline):**  
Every mechanism in Phase 5 must be local geometric dynamic or measurement of one. No supervisor arbitrates. B2 (random-schema control) is *comparison condition only*, never in production. Schema-source robustness (section C of checklist) is diagnostic-only; system must NOT adaptively switch sources. Atom-splitting is measurement; the split action is deferred. Branch selection is energy-based only, never metric-based.

**Magnitude floor (5.5e-3 on D=4096):**  
Derived from `(1/β)·log(1 + (N−1)·exp(−β·(1−1/√D)))` at D=4096, β=10, N≈1064. **Any Phase 5 design claiming graduation must beat this floor numerically.** Not a soft target; it's the substrate's intrinsic energy noise relative to retrieval energies. Sub-noise signals are architecturally inert.

**K=4 vs K=1 dichotomy:**  
Decision-5 spike (2026-05-20) found per-pattern prior formulation wins at K=1: ΔE +2.6e-5 vs global-pull's −0.027. But post-death effective dimensionality (~5) cannot support K=4 distinct attractors (report 044). Single-branch (K=1) is empirically as good as K=4 on post-death substrates (report 041: K=1 ≥ K=4). The K-branching premise fails on small post-death geometry; the mechanism may recover on pre-death (d_eff ~40) or lower-D.

**ΔE/basin anti-correlation (instance 6):**  
Central finding: the substrate carries a directional ΔE signal (cd ≥ 0.6 regime, 9/10 seeds positive) at a regime where basin-membership is zero (role-target not in top-K). The two metrics measure different substrate properties. Whatever pair-energy measures is not role-target basin occupancy. **Any headline reformulation must address this decoupling explicitly.**

**Random-prior pathology (reports 053, 057, 058):**  
Random-prior condition outperforms both role and content priors in ~30–50% of cues across all cells. B1 control clean (random sits between content and role), but the prevalence of random-wins is consistent with sharp-basin substrate: random schemas happen to land near stored patterns; stored-pattern energy dominates. This is noise-floor mechanics, not a mechanism failure.

---

## What the 2026-05-20 Brainstorm Already Explored

**Brainstorm session:** 2026-05-20, post-research-review, pre-magnitude-floor gate. Generated 12 ideas organized into three tiers by cost-to-information ratio. By the time report 058 landed (2026-05-23), six instances of substrate saturation were already discovered, which reframes several ideas.

**Tier 1 (cheap, fast, high information):**

1. **Cue-regime sweep** (Idea 1)  
   Status: **COMPLETED** as report 058 (2026-05-23). Result: ΔE is pure function of content_distortion; binding_noise is invisible. Phase transition at cd ≈ 0.4–0.6. ΔE/basin anti-correlation confirmed. **Tier-1 diagnostic exhausted; answers the operating-point question empirically.**

2. **Settling-based f_i** (Idea 2, research B/C/E)  
   Status: **Deferred pending D-sweep outcome.** Idea: compute fidelity post-settling (`cos(q_settled_i, s_i)` or basin-membership flag) instead of pairwise unbind. Would have per-atom variance by construction if settling converges to discrete basins. Requires instance 6 (anti-correlation) to not be fundamental to the settling landscape itself. **Checkpoint: if D-sweep shows anti-correlation persists, this doesn't rescue.**

3. **Ganesan discriminability-gap fidelity** (Idea 3, research C)  
   Status: **Deferred.** Replace pairwise-distance with per-atom margin `Jp − Jn` from Ganesan 2021. Bounded per-atom variance independent of D. ~25 lines, ½ day. **Same checkpoint as Idea 2: only if instance 6 is measurement-artifact, not structural.**

**Tier 2 (real architectural moves, ~1 day each):**

4. **PAM-style predictor-distance** (Idea 4, research A)  
   Status: **Shelved pending D-sweep.** Train small MLP on substrate co-occurrence; define `f_i = exp(-||g_φ(cue_i) - s_i||²/σ²)`. Published PAM baseline: cosine AUC 0.789 → PAM 0.916 on paired-associate test. ~1 day, training loop. **Cost-benefit marginal on D=4096 if instance 6 is structural; worth revisiting if lower-D works.**

5. **Pair #4 pivot: metastability / replay-prioritization** (Idea 5, research D)  
   Status: **LIVE.** Add per-atom EMA `m_i` of metastability (already computed per retrieve() call, currently discarded); multiply into existing ReplayStore priority. Headline becomes Δ meta-stable-rate at W=3 (substrate-pure metric Phase 4 used). ~½ day plumbing. **This is the lightweight closure path if lower-D doesn't pan out — closes one of five diagnostic-actuator pairs without substrate reconstruction.**

6. **Permutation-binding as role operator** (Idea 6, research E)  
   Status: **Long-term option.** Replace FHRR convolution-binding with Kanerva/Plate permutation-binding. `u_1/D` crosstalk vs `(W−1)/D` for permutation means per-atom variance by construction. Recchia et al. 2015 show permutation > convolution at D=2048 (457 vs 381 m-correct, p=0.001). But requires Phase 4 retrain from scratch. **Phase 6 territory unless lower-D alone doesn't work.**

7. **Multi-scale D=512 ‖ D=4096** (Idea 7, research E)  
   Status: **Contingency option.** Keep D=4096 for capacity; add parallel D=512 for f_i measurement (3× higher noise floor = 3× larger fidelity variance). ~1 day. **Lower-cost version of lower-D redesign; worth revisiting if D-sweep shows lower-D is path forward.**

**Tier 3 (bigger reshapes, 2–5 days, Phase 6 territory):**

8. **Sparsemax / Hopfield-Fenchel-Young** (Idea 8, research B)  
   Status: **Deferred.** Replace softmax with sparsemax in settling; post-settling retrieved state has discrete zero support → downstream fidelity metrics have variance. Medium-confidence payoff. Honest uncertainty on whether it rescues at D=4096. ~1 day prototype to verify.

9. **Resonator Networks diagnostic** (Idea 9, research B)  
   Status: **Deferred.** Run role-binding decomposition through Frady & Sommer 2020 resonators; per-factor convergence trajectories heterogeneous by construction. Standalone diagnostic, not runtime. ~2 days.

10. **Drift / replay-pressure (pair #2)** (Idea 10, research D)  
    Status: **Deferred to pair #5 phase.** Per-atom drift accumulator coupled to replay weight. Same Saighi-EMA template as Idea 5. Follows pair #4 if it graduates. ~1 day.

11. **Capacity-constrained replay** (Idea 11, research A)  
    Status: **Phase 6+.** Reframe A+B+A1+A1' death dynamics as capacity-vs-corpus tradeoff. Dury 2026b: "capacity constraint as consolidation mechanism." Dissolves replay/consolidation split. Bigger rewrite than others.

12. **Cap-coverage as restructuring pressure (pair #5)** (Idea 12, research D)  
    Status: **Phase 6+.** Substrate-energy term penalizing low cap-coverage; restructuring pressure via continuous gradient flow. Deepest payoff (unifies splitting and restructuring). Requires θ′(β) calibration spike (still open), discovery-channel gating. **Phase 6 territory.**

**Tier-1 exhaustion:** Cue-regime sweep answered its question (operating-point vs substrate-side). Settling/Ganesan/PAM all hinge on whether instance 6 is structural (anti-correlation = fundamental to settling geometry) or measurement-artifact (would disappear under different metric). **Cannot proceed on these without D-sweep decision.**

**Tier-2 live option:** Pair #4 pivot is the **only path that doesn't require D-sweep to decide.** ~½ day to close one diagnostic-actuator pair; graduates on substrate-pure metric Phase 4 already validated. **This is the closure path if user chooses not to pursue lower-D.**

---

## Timeline and Decision Gates

**Immediate (next session):**  
User decision on two paths:
1. **Path A: Lower-D redesign** → D-sweep diagnostic (2 days) → Phase 4 retrain (1–2 weeks) → Phase 5 retry (contingent on D-sweep results)
2. **Path B: Close Phase 5 + pivot to pair #4** → Pair #4 implementation (~½ day) → Pair #4 graduation run (n=10, 2 days) → Document substrate-saturation finding

**Path A timeline (if chosen):**  
- **This week:** D-sweep diagnostic; place 100–200 cue trials per dimension D ∈ {256, 512, 1024, 2048, 4096}; measure ΔE and magnitude floor empirically; report scaling vs theory.
- **Decision gate:** if lower-D shows promise (ΔE within 1–2× floor at some D) AND anti-correlation doesn't worsen, commit to Phase 4 retrain. Otherwise close Phase 5.
- **Next 2–3 weeks (if gate passes):** Phase 4 retrain at lower D; Phase 5 retry with re-derived magnitude floor for that dimension.

**Path B timeline (if chosen):**  
- **This week:** Pair #4 (metastability/replay) plumbing (~½ day); commit falsification criteria (d_eff not regressed, meta-stable EMA integrates without destabilizing); run Colab n=10 headline (2 days).
- **Completion:** Pair #4 graduates or surfaces a different failure mode; Phase 5 documented as closure.

---

## Reading List for Brainstorm Session

1. **Design spec:** `/notes/emergent-codebook/phase-5-unified-design.md` (§Headline metric 256–281, §Required controls 285–293, §Decision 5 spike 501–534, §Anti-homunculus check 391–410)
2. **Exit checklist:** `/notes/emergent-codebook/phase-5-checklist.md` (§A Headline A1, §I Interpretation rule)
3. **Latest drill-down:** `/reports/058_phase5_cross_seed_cue_regime_sweep.md` (instance 6; Instance 6, anti-correlation, strategic implications)
4. **Beta closure:** `/reports/057_phase5_cross_seed_beta_sweep_K1.md` (confirms β=10 is optimum; "statistically significant but sub-magnitude-floor")
5. **Headline run:** `/reports/053_phase5_headline_n10_directional_subnoise.md` (instance 5; n=10 result; pre-committed gate logic)
6. **Geometry diagnostic:** `/reports/044_consolidation_geometry_diagnostic.md` (instance 2; d_eff collapse; mechanism for K-branch failure)
7. **Brainstorm-2026-05-20:** `/brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md` (12 ideas, tier organization, research briefs)
8. **Magnitude floor pre-commit:** `/notes/notes/2026-05-21-phase5-headline-magnitude-floor.md` (floor derivation; why 5.5e-3 binds)

---

## User Decision Question

> **At this decision point, is Phase 5 a lower-D redesign (Path A: 2–3 weeks, contingent on D-sweep), or a closure + pair-#4 pivot (Path B: 4 days total)? Or a different direction entirely?**

Path A is the "fight harder, build better" option; it directly addresses the root cause (substrate noise floor) by moving to a dimension where the floor is larger relative to the signal. Path B is the "honest closure + pivot" option; it documents the substrate-saturation finding and moves to a mechanism (replay-prioritization) that the substrate already has the machinery for, avoiding months of lower-D engineering.

Both are valid. The brainstorm should surface which one is more generative for the architecture's long-term vision, or whether a third direction emerges from the ideas not yet explored.

