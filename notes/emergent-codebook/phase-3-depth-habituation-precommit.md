# FROZEN PRE-COMMIT — Phase-3 depth × local-temporal-write oracle (`experiments/77`)

*Frozen 2026-06-02 BEFORE running. Branch `experiment/tem-local-reachability-oracle`. The
culmination of the 121-131 arc + the user's two reframes (not-flat → DEPTH; homunculus-dissolution →
TEMPORAL HABITUATION). SUPERSEDES the deflation step of
[phase-3-multilayer-deflation-precommit.md](phase-3-multilayer-deflation-precommit.md): keeps its
depth-stack/hold/read/gates, but SUBSTITUTES the GHA-projector deflation with a LOCAL HABITUATION
(divisive-activity) deflation. Grounded by the 2026-06-02 depth×temporal workflow (verdict:
build-with-conditions). Harness: reuse exp61/65/68/76/78 via importlib. Substrate-free oracle
IN-SCOPE; FHRR D=4096 build = rung-2/the user's fence.*

## 0. Preamble (CLAUDE.md experiment-preamble)

> **Active capability:** Codebook-growth (P3 structure), substrate-free oracle.
> **Headline:** within-paradigmatic-SET **label-shuffle B-KILL** on the STACKED codes, CONJOINED with
> `spec_ci_lo>0` (the 129 degenerate-code guard), CI-lo>0 in **≥8/10 seeds**. SOLE arbiter; g3 dead at n≤40.
> **Required controls (all mandatory):** depth-ablation (L=1 vs L>1); the **beats-inverse-frequency-stack**
> gate (the critical new one); CEILING reach-not-exceed (global NMF/SVD); Control C (no-deflation parallel);
> Control D (frozen-random deep stack); offline global k-means at matched depth (the 127 differentiator);
> g4 sparsity-matched rand; the +0.1092/0.222 anchor (INVALID otherwise).
> **Last verified:** 130 (single-projection channel-invariant), 131 (capacity necessary-not-sufficient),
> 129 (single-layer bound; adaptive ASSIGNMENT load-bearing). No multi-layer local writer has been run.
> **Why now:** both branches (channel, capacity) are closed; depth + adaptive-nonlinear-capture +
> local-temporal-deflation is the one direction left that is not single-projection-anything.

## 1. The mechanism (the only new code)

Operator `M₀ = build_S` (primary; `M_trans` secondary). For layer ℓ = 1..L:
- **CAPTURE (nonlinear — the live arm):** `Yℓ = exp68.kwta_features(M_{ℓ-1}, kℓ, cap=kℓ//4)` — assembly
  k-WTA = a local PARTITION/assignment (the 127 winner; 129's load-bearing operation; SQHN's one-hot
  capture ported as CAPTURE ONLY, never threaded through a downward weight). Also wire the **linear
  `exp76.tem_frozen_write` arm** as the PRE-REGISTERED PREDICTED-NULL (linear depth = ordered PCA =
  the 123 collocational stall, deeper — a banked sub-result, not a design failure).
- **HABITUATION DEFLATE (NEW, replaces the GHA projector):** a per-token activity trace
  `āℓ[i] = ‖Yℓ[i,:]‖₁` (how strongly token i participated in layer ℓ's captured mode); deflate the
  operator by **divisive normalization** `Mℓ[i,j] = M_{ℓ-1}[i,j] / ((1+κ·āℓ[i])·(1+κ·āℓ[j]))`, κ FIXED.
  Constantly-/strongly-active (dominant-mode) tokens FADE; the residual `Mℓ` (next mode now dominant)
  is HELD as a distinct V×V matrix and passed UP. (Carandini-Heeger divisive normalization = a fixed
  local statistic, anti-homunculus-exempt; the streaming running-trace `āℓ←(1−ρ)āℓ+ρ‖Yℓ‖₁` version is
  rung-2.) Emit `‖Mℓ‖` per layer (signal-exhaustion diagnostic).
- **HOLD:** each `Yℓ` is a DISTINCT `V×kℓ` block — **concatenate `[Y₁;…;Y_L]`, never sum, never iterate
  on one code** (this distinct-holding is the whole reason depth escapes single-projection; collapse it
  and it falls back into the channel-invariant wall, 130).
- **READ:** static `exp68.read_specificity` on the stacked `[Y₁;…;Y_L]`. **NEVER grow_G.**

## 2. Why this escapes single-projection (and the two disguises that would collapse it)

ESCAPE: each layer operates on a DIFFERENT operator `Mℓ` (the held residual), so layer ℓ+1's dominant
mode is M₀'s next-subdominant mode — local deflation reaching non-dominant modes with purely local
rules (Sanger/GHA / Pehlevan-Chklovskii), realized WITHOUT a top-down target (so NOT PC=backprop). The
single sharpest separator the diagnostics must show: **per-layer `para_centroid_rank` RISING** across
layers (paradigmatic structure moving toward dominant in the residual as layers deflate). Flat centroid
= NULL(a) = the bound survives genuine depth. **DISGUISES (pre-registered):** (i) if depth+habituation
summed ≡ the global SVD-by-stages → still the HONEST CEILING (a local route to the global code; a real
first escape of the single-LAYER bound, but NOT "beats SVD"; exceeding the NMF ceiling = artifact);
(ii) if L=1 already passes or Control C reproduces L>1 → depth/concat is the lever, deflation is inert
(the 127/PM-9 overclaim shape) — g5_depth + Controls C/D separate these.

## 3. The gate (frozen) — PASS / NULL

Per (arm × operator × L), n=10, **PASS = ALL of:** g1 `spec_ci_lo > +0.04`; g2 beats grow_G floor ≥+0.02;
g4 beats sparsity-matched `rand_nonneg` ≥+0.02; **g5_depth** (L>1 strictly beats L=1 by ≥+0.02 on B-KILL
**AND** per-layer `para_centroid_rank` rises); **B-KILL ≥8/10 ∧ spec_ci_lo>0**; **beats-inverse-frequency-
stack ≥+0.02** (THE critical gate — see §4); CEILING reach-but-NOT-exceed (global NMF/SVD; exceeding =
artifact); beats offline global k-means via online-locality.

**NULL sub-cases (pre-registered):** (a) **no-composition** — L>1≈L=1, centroid flat ⇒ depth doesn't help
⇒ pivot to Codebook-growth⇄Replay or substrate/data change. (b) **noise-accumulation** — spec FALLS as L
grows / `‖Mℓ‖`→0 ⇒ crude deflation, residual is noise. (c) **dim-inflation** — B-KILL up but Control C/D
reproduces it ⇒ depth/concat is the lever, not deflation. (**deflates-to-PMI**) — activity-trace stack ≈
inverse-frequency stack ⇒ the **5th narrowing of the year**, banked-not-won. **INVALID** = anchor misses
+0.109/0.222, OR L=1 already PASSES (a leaked top-down/global signal — the apparatus check).

## 4. THE critical condition (the #1 risk, grounding-flagged)

`build_sppmi` ALREADY divides by marginals (`uni[i]·uni[j]`), so the **L=1 habituation deflation is
structurally a second inverse-frequency pass** — and inverse-frequency/PMI is the ghost that deflated
4 ideas this year (126/127/exp70/128 §8). ⇒ **MANDATORY gate:** run the IDENTICAL stack with `āℓ`
replaced by a codes-INDEPENDENT marginal trace `gfreq[i]=1/(uni[i]+1)` (the exp78 A_freq pattern). A
PASS **must beat this inverse-frequency stack by ≥+0.02** on B-KILL. If activity-trace ≈ inverse-freq →
NULL(deflates-to-PMI). **The escape MUST come from L≥2** (where the trace is over residual, non-marginal
activity), or it is the 5th narrowing.

## 5. Anti-homunculus (two code-review flags — make-or-break)

(a) what moves locally = the per-token activity trace `āℓ` + the codes `Yℓ`; (b) "which mode survives"
lives in the habituation dynamics + fixed deflation geometry, not a metric-read branch; (c) divisive
normalization + the fixed k-WTA cap + the fixed depth L = fixed problem-generic scaffolds (CONTEXT.md §3);
(d) Control D (frozen-random stack) must NULL **and** L=1 must reproduce the floor — else a top-down/global
signal leaked in. **FLAG 1 (PC=BP guard):** signal flows UP ONLY — NO downward weight, NO target/label
tensor, NO two-phase free/nudge, NO autoencoding objective; the deflation is a deterministic readout-then-
reweight, NOT a decoder fit to minimize input reconstruction. **FLAG 2:** ρ/κ/cap are FIXED schedules,
NEVER routed on a downstream metric. Both code-reviewed BEFORE the WikiText run.

## 6. Honest ceiling

A PASS = "local layer-by-layer composition REACHES the paradigmatic structure that was previously only
GLOBALLY reachable — locally, without backprop." The FIRST local route past the single-LAYER bound. NOT
"new structure," NOT "beats SVD." Even a clean PASS is a Phase-3-structure result, not the Phase-5
"more-than-memory" deliverable. **Skeptical prior (honest): the linear arm nulls; the nonlinear arm faces
a real fight at L≥2 against the PMI ghost. A genuine open wager, not a likely win.**

## 7. Build checklist

- [ ] Reuse via importlib: `exp61.{build_S, build_cooccurrence, build_sppmi, pick_k_by_density, load_corpus}`,
      `exp63.{select_pairs, raw_sppmi_svd_anchor}`, `exp65.nmf_slots`, `exp68.{kwta_features,
      read_specificity, derange_partners, random_nonneg}`, `exp76.{tem_frozen_write, local_kwta,
      spectral_diagnosis}`, `exp78.frustrated_xy`(N/A) — and the exp78 `A_freq` inverse-freq pattern.
- [ ] New code only: `deflate_by_habituation(M, Y, kappa)`, `multilayer_write(M0, capture, L, kappa, ...)`
      (capture→habituate-deflate→stack), the inverse-frequency stack, Controls C/D, the g5_depth gate
      (centroid-rise via `exp76.spectral_diagnosis` per layer).
- [ ] CODE-REVIEW the §5 admissibility guard (no downward weight/target/two-phase) BEFORE the WikiText run.
- [ ] Anchor every run; L=1-reproduces-the-floor apparatus check; planted smoke before WikiText.
- [ ] Differential reporting ONLY (B-KILL, A−B, beats-inverse-freq, beats-offline-k-means); never absolute
      cosine; spectral reference rank-300-matched, never raw top-k (the 130 inflation caveat).
- [ ] Freeze: kℓ-grid, L∈{1,2,4}, κ, cap=kℓ//4, n=10, B-KILL≥8/10 ∧ spec_ci_lo>0, the 4 PASS/NULL
      sub-cases, beats-inverse-freq ≥+0.02, build_S primary / M_trans secondary, SimLex≥5 sha.
- [ ] Adversarial 3-lens verification BEFORE banking ANY positive (the 129 pattern).

## 8. Post-smoke note (2026-06-02, before the WikiText headline run)

- **Trace correction (first-principles, not a result):** the precommit §1 trace `āℓ=‖Yℓ‖₁` is computed on
  ℓ2-NORMALIZED k-WTA codes → ~uniform across tokens → a uniform divisive deflation is a NO-OP
  (k-WTA is scale-invariant). Corrected to the **current operator's residual row-mass**
  `āℓ[i]=‖Mℓ[i,:]‖₁` (non-uniform; at L=1 ~marginal-mass, at L≥2 the un-captured residual mass — the
  depth-dependent signal). The inverse-frequency CONTROL is the **static marginal** `uni` (codes-
  independent, same fade-frequent direction) so A−control tests whether depth-dependent residual-mass
  deflation beats static frequency. κ applied to the max-normalized trace.
- **Planted smoke is INSUFFICIENT to discriminate the arms** (n_para=1 → the single-pair read is
  dominated by the identical layer-1 code; hab/freq/nodeflate coincide on planted as a resolution
  artifact, NOT evidence of inert deflation). It validates only: pipeline runs, k-WTA recovers the
  planted lift, linear arm negative (predicted-null direction), INVALID/anchor logic fires.
  **Arm-differentiation + the depth-ablation are validated on the WikiText run (n_para=40), which is
  the actual test.**
