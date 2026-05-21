# Phase 5 Reports — Empirical Context for Brainstorm

Date: 2026-05-20. Synthesis of `reports/038–050` + relevant earlier reports for the brainstorm on "what's next after the A+B → A1 → A1' → β chain falsified the K-branch state_divergence criterion."

## 1. One-paragraph state of the substrate

A 1064-atom W=4 substrate (seed 17, step 1800) produced by A+B+step3 + A1 + A1' continuous death dynamic. **d_eff = 35.23 / 4096** (pre-death ref ~40, post-binary-death ref ~5). Spread regime, d̄ = 0.7265. Continuous death works as designed: no atom deleted, `mean_strength` decays 5× (0.221 → 0.043), `r_ema_mean` rises 5× (0.017 → 0.090), and 1041 of 1064 atoms are "dead_ready" by step 1800. Step-3 sigmoidal retrieval weighting (ε=0.05, τ=0.02) exposes ~23 effective atoms at retrieval. **The substrate is what A+B+A1+A1' was designed to produce.** What fails on it is the K=4 schema-prior branching diagnostic.

## 2. The four-step debugging journey with numbers

### Step 0 — Pre-A+B baseline (binary mass death)
- W=4 step 1800: **12 atoms, d_eff = 5.36** (collapsed from ~40 by mass death between steps 1500 and 1800).
- 7×–5000× drop in K-branch state_divergence pre→post death across 5 seeds (report 043). Pre-death seed-17 reference: **role_K4 state_divergence = 0.0336** → target band [0.024, 0.044].
- At n=5, K4 ΔE CI includes zero [-5.2e-4, +2.1e-4]; K=1 outperforms K=4 (5/5 vs 3/5 positive). Branches collapse: softmax entropy = ln(K) to FP precision; state_divergence ~10⁻⁵–10⁻³ post-death (report 042). γ sweep ∈ {0.1, 0.25, 0.5, 1.0, 2.0} and K_main sweep ∈ {1, 2, 3, 4, 6, 8}: ΔE monotonically decreasing in K; K=8 produces ΔE=0 exactly.
- **Diagnosis (report 044):** d_eff/D drops 0.010 → 0.001 across all 5 seeds; with K=4 branches and d_eff ≈ 5, substrate's reachable subspace cannot support K distinct attractors. **MESH check: n_atoms=6 retrieves at recall=1.000** — capacity fine; subspace span is the issue.

### Step 1 — A+B (+step 3) substrate-construction failure
- A+B = continuous coverage-weighted reinforcement (A) + −α·log(d_eff) repulsion (B). Step 3 = E_i-sigmoidal retrieval weighting.
- **d_eff = 35.23 (with step 3: 35.20)**; passes ≥25 gate. n_atoms stays at 1064.
- **But K-branch state_divergence at n=1 = 0.000** across all conditions; all mean_n_branches = 1.00. FAIL.
- **Root cause:** top-8 by effective_strength are indices 1044–1063 — all Phase-4 discovery channel atoms — **FP-precision identical** (pairwise sim 1.0000). Discovery channel keeps adding near-copies; new atoms enter with `r_ema = 0`, accumulate unthrottled, out-compete original Phase-3 atoms whose `r_ema` ≈ 0.09.
- Baseline (no A+B) top-8: pairwise sim **0.361**, range [0.032, 0.842], across original Phase-3 indices.

### Step 2 — A1: substrate-aware r_ema init (failure mode 2: measurement)
- Initialize new atoms' `r_ema` from `_coverage_redundancy_instantaneous(P ∪ {new})` at add-time instead of 0.
- n=1 seed 17: top-8 pairwise sim **still 1.0000**; state_divergence **still 0.0**.
- **But A1 partially worked at strength:** discovery atoms' max eff. strength **12.115 → 9.646 (−20%)**; mean **0.5065 → 0.3385 (−33%)**; top-5: [12.11, 1.99, 1.19, 0.67, 0.50] → [9.65, 0.95, 0.29, 0.29, 0.20].
- **Root cause:** `_coverage_redundancy_instantaneous` uses Gram-row **RMS**. For a perfect duplicate of one atom + orthogonal to N-2 others, RMS gives r_inst ≈ √(1/1023) ≈ **0.031** (not 1.0). The init inherits the measurement's blind spot.

### Step 3 — A1': max-over-others reduction (failure mode 3: selector)
- One-line: `r_i = max_{j≠i} |G_ij|` instead of `sqrt(mean_{j≠i} |G_ij|²)`.
- **Substrate-level transformation:** discovery max eff. strength **9.646 → 0.0293 (−99.8%)**; mean **0.3385 → 0.0272 (−92%)**; original atoms' max also drops (0.0915 → 0.0586). **Top-1 atom is now idx 713 (original Phase-3) at 0.059** instead of idx 1044 (discovery) at 9.65.
- **Headline numbers at n=1 seed 17 (load-bearing anchors):**
  - d_eff = **35.23** (criterion #3 PASS)
  - top-8 pairwise sim mean = **0.9325**, range [0.7302, 1.0000] (criterion #1 FAIL, target [0.10, 0.60])
  - K-branch state_divergence = **0.000897** (criterion #2 FAIL, target [0.024, 0.044]; ~37× below band)
  - mean_n_branches = **2.0** (off 1.0 for the first time)
  - role_K4 ΔE vs content_K4 = 0
- **Root cause:** top-8 = `[713 (original) + 1054–1060 (7 discovery)]`. The 7 discovery atoms are **tied at effective_strength = 0.02935 to FP precision** (added in last few hundred cues with novelty_strength=1.0, A1' throttled accumulation to zero, bumped-then-decayed u-chain leaves them tied). Categorical top-K selector picks 7 of them because there are 7.

### Step 4 — β: continuous role-fidelity-weighted prior (failure mode 4: encoding)
- β replaces top-K selector with `prior = Σ_i (cue · s_i)^p · f_i^q · s_i` where `f_i = mean_{j≠k}(1 − |G_jk|)` over W=4 unbound fillers `{unbind(s_i, pos_r)}`.
- **β diagnostic:** `[β] full-substrate fidelities: N=1064, mean(f)=0.9858, std(f)=0.0000, min=0.9858, max=0.9858`. **Every atom has the same f_i value to FP precision.**
- **Theoretical anchor:** FHRR crosstalk at D=4096 ~1/√D ≈ 0.0156. Role-fidelity = `1 − |G_jk|` ≈ 1 − 0.0156 ≈ **0.984**, independent of pattern content. Observed 0.9858 matches.
- **Pre-committed criteria:**
  - #2 ΔE > 0 (q=1 vs q=0) → **FAIL** (ΔE = 0.0 every cue, fraction_positive = 0.10)
  - #3 q-sweep monotonicity → trivially monotonic at zero
  - #4 β q=1 lower E than role/content K4 → **FAIL** (E_unbiased: fid_K1_q1 = **-1.3562**; content_K1 = **-1.3568**; β slightly worse — continuous weighted sum dilutes the cue-aligned vector with ~1063 misaligned ones)
- **Read:** the substrate has zero variance in the feature β was designed to weight by. K-branch state_divergence criterion was looking for a property the substrate's encoding doesn't expose at D=4096.

## 3. What IS working (preserved through the chain)

| Item | Value | Source |
|---|---|---|
| Phase 4 D1 graduation Δms_w3 (C−A) | mean −0.7920, CI [-0.94, -0.65], 10/10 negative | report 038 |
| Phase 4 architectural contribution (C−B) | -0.720, CI [-0.871, -0.569], 10/10 negative | report 038 |
| ΔR@10 (C−A) replicated | +0.0089, CI [-0.002, +0.020], 7/10 positive | report 038 |
| d_eff preserved under continuous A+B | **35.23** vs binary-death 5.36 | reports 045/046/049 |
| d_eff trajectory flatness | 36.34 → 35.20 across step 500–1800 | report 045 |
| Discovery atoms throttled (A1') | max strength 12.11 → 0.029 (**99.8%**) | report 049 |
| Top-1 atom is original under A1' | idx 713 instead of 1044 | report 049 |
| K-branch separates at sufficient density | 3×–4978× higher state_div on pre-death (n=1024) vs post (n=6–12) | report 043 |
| Step 3 retrieval weighting | ~23 effective atoms vs ~1064 raw; substrate bit-identical to no-step-3 | report 046 |
| Anti-homunculus discipline | every fix passed audit; no parameter retuning | reports 045–050 |
| Garbage-collect no-op guard | binary death cannot run alongside A+B | report 045 |
| Substrate completion under drift | 1024 atoms hold ms_w3 = 0 by step 500 (replay-driven, full substrate) on most seeds | report 038 |
| MESH memory-cliff at all n_atoms ≥ 2 | recall = 1.000 even at n=6 | report 044 |

## 4. What ISN'T working

| Item | State | Source |
|---|---|---|
| K-branch state_divergence under role-prior | 0.000897, target [0.024, 0.044] (37× below) | report 049 |
| Top-8 pairwise schema similarity | 0.9325, target [0.10, 0.60] | report 049 |
| ΔE under β q=1 vs q=0 | identically 0.0 (uniform f_i) | report 050 |
| Per-atom role-fidelity f_i at D=4096 | uniform constant 0.9858 ± 0.0000 across 1064 atoms | report 050 |
| Phase 5 headline ΔE at n=5 K=4 | mean -1.52e-4, CI [-5.19e-4, +2.15e-4], 3/5 positive | report 041 |
| K=1 vs K=4 (B2 fires) | K1 5/5 positive CI excludes 0; K4 3/5 positive CI includes 0 | report 041 |
| Pre-death ΔE direction across seeds | bimodal — seeds 17, 23 strongly positive (~+4e-3 K1); seeds 11, 1, 2 negative | report 043 |
| Top1 regression under integration | -0.048, CI [-0.083, -0.013], 9/10 negative (structural Hebbian, not a Phase 4 gap) | report 038 |
| Δcap_t05 under binary death | variance-bound; seed-3 outlier -0.179 | reports 035/037 |

## 5. Open empirical questions the reports don't resolve

1. **Does f_i uniformity hold across seeds?** Theory says yes (1/√D is a property of D not content). Colab n=5 will confirm.
2. **Does β q=0 vs content_K1 ΔE distribution overlap zero at n=5/10?** Smoke test β q=0 slightly worse; cross-seed unknown.
3. **What does f_i look like at lower D?** At D=512, noise floor 1/√512 ≈ 0.044 (3× more favorable SNR) but capacity drops. Untested.
4. **Would alternative f_i formulation produce variance?** Report 050 §"β'" sketch: `f_i = corr(unbind(s_i, pos_r), codebook_filler_at_r)`. Codebook is in snapshot. Untested.
5. **Pre-death substrate bimodality.** Even at n_atoms=1024, ΔE direction is split (seeds 17/23 vs 11/1/2). K-branch capable on pre-death (state_div restored 3×–5000×) **but role-prior asymmetry isn't consistent.** Cue-design sensitivity (`binding_noise_std=0.05`, `content_distortion=0.6`) never swept.
6. **Is K-branch state_divergence itself the right target?** It was derived from binary-death pre-death geometry. Continuous A+B+A1' substrate has different structure and may need a different headline.
7. **What's the actual downstream consumer?** Path 2(b) (emit K branches as continuous FHRR superposition) was anti-homunculus blocked until a consumer spec exists. The consumer has never been written. Until one exists, the K-branch combiner question is under-constrained.
8. **W=2 and W=3 on the A+B+A1' substrate?** d_eff at step 1800: W=2: 20.52; W=3: 27.23; W=4: 35.23. Phase 5 diagnostic only run at W=4.
9. **Capacity vs effective dimensionality relationship under continuous death?** Uncharacterized.
10. **Role of cue regime?** Cue generator parameters never swept. Harder cues might require K-branch disambiguation that K=1 can't do.

## 6. Anomalies and surprises buried in earlier reports

### Report 040 (freq-weighted α sweep, the Phase 4→5 bridge)
- **λ=2.0 is supercritical**: strengths explode to 10⁸–10⁹, W=2 corr_u_m_retrieval_count inverts +0.92 → **-0.52**, Gini saturates at 0.997. **But Δms_w3 and ΔR@10 are unchanged** — mass death *absorbs* the cascade instability before it propagates to readouts. **Load-bearing for the brainstorm:** the architecture has a built-in "mass death absorbs upstream instabilities" property.
- The cascade-rate filter is functional at n_cues=300 (no mass death): corr 0.295 → 0.799 between λ=0 and λ=1.0. Redundant at production scale because mass death does the job in binary form.

### Report 036 (decay sweep + mass-death finding)
- **All meaningful Δcap_t05 happens AFTER mass death**: seed 17 cap 0.293, 0.276, 0.284, 0.284, then **0.316 at step 2000 when n_alive collapses 4127 → 32**.
- A_k decay is a 15× lever on A_max (0.93 → 13.6) but produces **bit-identical ΔR@10 and Δtop1** across the entire decay range.

### Report 037 (seed-3 collapse)
- Same mechanism, opposite outcomes: seed 17 ends n_alive=32 with Δcap=+0.036; seed 3 ends n_alive=5 with Δcap=-0.179. **6× fewer surviving atoms.** A_nz at step 1500 predicts survivor count at step 2000.
- First instance of "mechanism produces good OR bad outcome depending on seed-level dynamics trajectory" — real phenomenon, not sample-size artifact.

### Report 038 (D1 graduation)
- **Two distinct paths to D1 = 0**: seed 17 reaches floor by step 500 with 4125 atoms intact (replay+consolidation reshapes basins); seed 3 sits at ms_w3=0.43 for 1500 steps then drops to 0 when mass death fires. **Same endpoint, completely different mechanism.**

### Report 041 (n=5 Phase 5 partial)
- K=1 baseline (B2 control) outperforms K=4 (5/5 vs 3/5 positive; CI excludes 0 vs includes). The result H4 forbids promoting.

### Report 042 (branching collapse)
- **Softmax entropy = ln(K) to FP precision** across K ∈ {1, 2, 3, 4, 6, 8}. The K settled branch energies are *numerically identical*.
- K=8 produces ΔE = 0 exactly because K=8 covers the entire post-death schema store of size 8.

### Report 043 (substrate-scale)
- **Pre-death seeds 1 and 2 have the LOWEST state_divergence (0.018 and 0.008)** despite 1024 atoms. Seeds 11/17/23 sit at 0.044/0.034/0.093. Pre-death substrate quality is itself bimodal.

### Report 044 (consolidation geometry)
- Per-atom k=5 NN d_eff is ~3.7 pre-death and ~3.0 post-death — saturating near max=5 means most atoms' immediate neighborhoods are already low-dimensional even pre-death. The dimensionality story is **substrate-level, not per-atom**.
- θ′(β) calibration spike still open. At β=10, θ′=0.1, n_tight = 0 across all 10 snapshots.

### Report 050 (β smoke) — the deepest surprise
- **β prior at q=0 is already slightly worse than content_K1** (-1.3562 vs -1.3568). Even without using fidelity weighting, the continuous-weighted-sum *shape* is energetically inferior to picking the sharpest single cue-cosine atom. Independent of f_i uniformity. May indicate the β prior **shape** (vs weighting) is wrong on this substrate.

## 7. Anti-homunculus discipline status (binding on next moves)

Patterns rejected as wrong-shape:
- W1: gate at add-time in discovery channel
- W2: diversified top-k schema selection (MMR/DPP) — population-level chooser
- W3: scheduled strength rebalancing
- A2 (report 047): combined strength+diversity schema selector
- step-3 hysteresis on ε (controller-in-disguise caught in audit)

**No parameter retuning across the entire A+B → A1 → A1' → β chain.** All knobs set once: `alpha_anti=1.0`, `coverage_lambda=1.0`, `coverage_ema_rate=0.01`, `repulsion_step_size=100.0`, `retrieval_weight_epsilon=0.05`, `retrieval_weight_tau=0.02`. β added p=1, q=1 from the design note, not from tuning.

Binding on the brainstorm: proposals must be local geometric dynamics or measurements of them — never arbitrations over them.

## 8. Quick-reference anchors

| Anchor | Value |
|---|---|
| Current substrate d_eff | **35.23 / 4096** |
| Top-8 pairwise sim | **0.9325** (target [0.10, 0.60]) |
| K-branch state_divergence | **0.000897** (target [0.024, 0.044]) |
| Per-atom role-fidelity f_i | **0.9858 ± 0.0000** |
| FHRR crosstalk noise floor at D=4096 | **0.0156** |
| Discovery atom strength reduction | 12.11 → 0.029 (**99.8%**) |
| d_eff pre-death vs post-binary-death | 40 → 5 (**10× collapse**) |
| Phase 4 D1 Δms_w3 (C-A) | **-0.7920**, CI [-0.94, -0.65], 10/10 |
| Phase 5 n=5 K=4 ΔE | -1.52e-4, **CI includes 0** |
| K=1 (B2) n=5 ΔE | +1.28e-4, CI [+3.5e-5, +2.84e-4], **5/5** |

## 9. Brainstorm-relevant sketched framings (from report "next steps")

- **β' — codebook-correlation f_i**: `corr(unbind(s_i, pos_r), codebook_filler_at_r)`. Couples to codebook (substrate state).
- **γ — cue-regime distribution averaging**: across `(σ, δ) ~ Uniform`.
- **Lower-dimension substrate**: D=512 has noise floor 0.044 vs 0.016 — 3× more favorable SNR.
- **Re-scope Phase 5's headline**: K-branch state_divergence was derived from binary-death pre-death geometry; the new substrate may need a different headline.
- **Path 2(b) combiner**: emit K branches as continuous FHRR superposition. Requires downstream-consumer spec.
- **Cue-regime sensitivity sweep**: vary `binding_noise_std`, `content_distortion`, role-extraction depth.
- **Discovery-channel redesign in dynamic form**: don't call `add_pattern` when re-settled query lands in basin of existing atom — but in dynamic form, not the current resolve_threshold gate.
- **Continuous-rate death that produces under-capacity (not preservation)**: A+B preserves d_eff at 35; the design intent was an *under-capacity* slow store. Unexplored.

## 10. The deepest open question

**Has the substrate the architecture wants to build actually been built?**

The A1' substrate has 1064 atoms, d_eff = 35, throttled discovery atoms, top-1 = original Phase-3 atom, working continuous death dynamic. **By every substrate-level measure A+B+A1+A1' was designed to satisfy, it is the right substrate.** What fails is not the substrate but the downstream diagnostic — first the categorical top-K selector, then the role-fidelity weighting whose underlying signal is below the FHRR noise floor at this dimension.

Two reads:

1. **The architecture's "structural retrieval" claim, as operationalized, doesn't have an empirically detectable substrate signal at D=4096.** Falsification of the operationalization, not the claim. Need a new operationalization.
2. **The architecture has built the substrate; what it lacks is the downstream consumer.** Phase 5 was specified before such a consumer existed (per the path-2(b) anti-homunculus block). The brainstorm might usefully ask "what's the next thing past the substrate?" rather than "what's the next fix to make K-branch state_divergence pass?"
