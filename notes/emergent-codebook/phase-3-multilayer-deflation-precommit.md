# FROZEN PRE-COMMIT — Phase-3 multi-layer local deflation writer (the depth frontier)

*Frozen 2026-06-02 BEFORE running. Branch `experiment/tem-local-reachability-oracle`.
Harness: `experiments/77_multilayer_deflation_oracle.py` (reuses exp61/62/63/65/68/76 via
importlib). Follows Report 129's signpost (#1: the 121-129 arc is a SINGLE-LAYER bound) and the
2026-06-02 grounding workflow (tPC implicit-whitening = deflation; FEP "learn only the unexplained
variance"; EqProp + feature-space-AM REJECTED as backprop-in-disguise). The user lifted the
build-gate and chose the multi-layer frontier directly.*

## 0. Preamble (CLAUDE.md experiment-preamble)

> **Active capability:** Codebook-growth (P3 structure) — the **depth frontier**. The 121-129 arc
> closed every SINGLE-LAYER local writer; this is the first genuine MULTI-LAYER local writer.
> **Headline metric:** within-paradigmatic-SET **label-shuffle B-KILL** pair-specific residual,
> static read on the stacked codes — NOT `grow_G`. (Per Report 129 gate-vocabulary: g3 is
> DIAGNOSTIC-ONLY at n≤40; B-KILL is the sole arbiter.)
> **Required controls:** raw-SPPMI-SVD anchor (+0.109/0.222, INVALID otherwise); flat-SPPMI
> `grow_G` floor (= the L=1 apparatus check, must reproduce ~+0.0002); the global-NMF/SVD CEILING
> on the same operator (a PASS must REACH-BUT-NOT-EXCEED it); offline global k-means at matched
> depth/k (the Report-127 guard); depth-and-sparsity-matched `rand_nonneg` (g4); within-set
> label-shuffle (B-KILL); per-layer spectral diagnosis.
> **Last verified:** Report 129 (single-layer TEM-local NULL (b); the bound is single-layer).
> **Why now:** the brain is not single-layer; depth is the named, untested escape. User chose it.

## 1. The honest ceiling (what a PASS can and cannot mean) — pre-registered framing

A legal upward-only local deflation stack **converges to the global code (Sanger/GHA)**. So the
MOST a PASS can show is: **local layer-by-layer composition REACHES the paradigmatic structure that
was previously only GLOBALLY reachable — locally, without backprop.** That vindicates the bet
(local dynamics do what global computation does) and is the FIRST local route past the single-layer
bound. It is **NOT** "new structure" and **NOT** "beats SVD." A PASS that EXCEEDS the global-NMF/SVD
ceiling on the same operator is an ARTIFACT (inflation/hubness), not a win.

## 2. The admissibility line (the make-or-break; both disguises guarded)

This direction is INADMISSIBLE if it collapses into the banned global computation. Two routes:
- **PC=BP regime** (Millidge 2020): if any layer's objective is to reconstruct its INPUT with a
  TOP-DOWN error/target, it is provably backprop. **GUARD (structural, code-reviewed): signal flows
  UP ONLY — NO downward weight, NO target/label tensor, NO two-phase free/nudge, NO autoencoding
  objective.** The per-layer reconstruction is a DETERMINISTIC readout of the codes (not an
  optimized decoder fit to minimize input-reconstruction error). L=1 MUST reproduce the pure local
  floor (apparatus check) — if it doesn't, a top-down signal leaked in.
- **Sanger/GHA convergence** (NOT a homunculus, but must be named): even the legal stack ≈ online
  SVD. **GUARD: reach-but-not-exceed the global ceiling; frame as "local route to the global
  code," never "beats SVD / new structure."**
The depth L is a FIXED problem-generic scaffold (CONTEXT.md §3 — a legal evolution-style prior,
identical across problems); no controller adds layers based on a metric.

## 3. The mechanism (the only new code)

Operator M₀ = `build_S` (primary; `M_trans` secondary). For layer ℓ = 1..L:
- **Capture:** `Yℓ = local_writer(M_{ℓ-1}, kℓ, seed)` — configurable: `kwta` (the Report-127
  nonlinear winner; the false-negative analysis says a nonlinear recode is likely needed) or
  `tem_frozen` (the linear 129 writer; the "linear whitening alone may null" comparison).
- **Deterministic local deflation (GHA-style, upward-only):** orthonormalize the captured codes
  `Qℓ = orth(Yℓ)`; deflate the captured subspace symmetrically `Mℓ = (I − Qℓ Qℓᵀ) M_{ℓ-1} (I − Qℓ Qℓᵀ)`.
  This removes exactly what layer ℓ captured; the residual Mℓ (where the next mode is now dominant)
  passes UP. No target, no backprop, deterministic — a batch statistic (anti-homunculus-exempt,
  same class as the existing operators/decorrelator).
- **Read:** static slot-vector cosine (`exp68.read_specificity`) on the **stacked** codes
  `[Y₁;…;Y_L]` (and per-layer, for the diagnosis). NEVER `grow_G`.
- **Signal-exhaustion diagnostic:** report `‖Mℓ‖` per layer (if it → 0, depth is spent).

## 4. The gate (frozen) — PASS / NULL sub-cases

Per (writer × operator × L × kℓ-rule), n=10 seeds, **PASS = ALL of**:
- **g1** stacked-read para-vs-random spec bootstrap **CI-lo > +0.04**.
- **g2** beats the flat-SPPMI `grow_G` floor by **≥ +0.02**.
- **g4** beats the depth-AND-sparsity-matched `rand_nonneg` stack by **≥ +0.02** (inflation guard).
- **g5_depth (NEW, the multi-layer proof):** L>1 **strictly beats L=1** by **≥ +0.02** on B-KILL-bearing
  spec AND the per-layer `para_centroid_rank` **rises** across layers (deflation working). This
  separates "depth helps" from "127 re-run at a deeper input" / dimension-inflation.
- **B-KILL** within-set label-shuffle CI-lo > 0 in **≥ 8/10** seeds (the SOLE pair-specificity arbiter).
- **CEILING guard:** spec **does NOT exceed** the global-NMF ceiling on the same operator (exceeding ⇒ artifact, NOT a PASS).

**NULL** = no cell clears all of the above. **INVALID** = anchor misses +0.109/0.222.

Pre-registered NULL sub-cases (read off the diagnostics):
- **(a) no composition** — L>1 ≈ L=1, `para_centroid_rank` flat ⇒ stacking is redundant; depth does
  not help ⇒ the bound survives genuine local depth ⇒ Codebook-growth⇄Replay combination or
  substrate/data change (NOT more layers).
- **(b) noise accumulation** — spec FALLS as L grows / `‖Mℓ‖`→0 ⇒ deterministic reconstruction too
  crude; the residual is noise ⇒ try a different (still-local, still-target-free) deflation, OR bank
  that crude local deflation is insufficient.
- **(c) dimension-inflation false-positive** — B-KILL up but Control-C (parallel, no-deflation) or
  Control-D (frozen-random deep stack) reproduces it ⇒ the depth/concat is the lever, not the
  deflation composition ⇒ NOT a multi-layer result (the 127/PM-9 overclaim shape).

## 5. Required controls (all pre-registered)

- **Control A — depth-ablation:** L ∈ {1,2,4} (8 if signal still rising). L=1 = the 129 apparatus check.
- **Control C — no-deflation parallel:** all L layers run on M₀ (no residual). PASS must beat this.
- **Control D — frozen-random deep stack:** same depth/kℓ, all-random codes, no local writer. Noise floor.
- **Control (Report-127):** offline global k-means at matched depth/k — depth must beat it with
  ONLINE-LOCALITY as the named differentiator (else it reduces to a global partition).
- **g4 control:** depth-AND-sparsity-matched `rand_nonneg`. **B-KILL control:** within-set label-shuffle.
- **Ceiling:** global-NMF (`exp65.nmf_slots`) + SVD anchor on the same operator.

## 6. Build checklist

- [ ] Reuse via importlib: `exp61.{build_cooccurrence,build_sppmi,build_S,pick_k_by_density,
      d_eff,cooc_counts_for_pairs,corr_bootstrap_ci,load_corpus}`, `exp62.{_cos_real,_boot_diff}`,
      `exp63.{select_pairs,raw_sppmi_svd_anchor}`, `exp65.{build_transition_operator,nmf_slots,
      faithful_read}`, `exp68.{read_specificity,derange_partners,random_nonneg,kwta_features}`,
      `exp76.{tem_frozen_write,local_kwta,spectral_diagnosis}`.
- [ ] New code only: `deflate_subspace` (GHA-style symmetric projection), `multilayer_write`
      (the capture→deflate→stack loop), Controls C/D, the g5_depth gate.
- [ ] CODE-REVIEW the admissibility guard: NO downward weight / target tensor / free-nudge phase /
      autoencoding loss anywhere. Signal up-only.
- [ ] Calibration anchor every run; INVALID if it misses +0.109/0.222. L=1 reproduces the floor.
- [ ] Per-layer spectral diagnosis + `‖Mℓ‖` emitted every run.
- [ ] Planted-corpus smoke before WikiText.
- [ ] Freeze: γ=0.9, W=6, max_vocab=2000, k∈{8,16,32}, L∈{1,2,4}, thresholds +0.04/+0.02,
      B-KILL≥8/10, n=10, SimLex≥5 sha, build_S primary.
- [ ] Adversarial 3-lens verification BEFORE banking any positive (the 129 pattern).
- [ ] Note: tPC/FEP/EqProp sources are link_only — a full source card from the primary is required
      before a substrate BUILD (this oracle is authorized; a build is not).

## 7. Post-freeze grounding addendum (2026-06-02 — architecture survey + SQHN card + Dury)

After freezing this precommit, a 3-workflow grounding pass (architecture survey → SQHN card from
the primary → decision) and the user's [Dury PAM note](2026-05-11-pam-dury-papers.md) converged on
the following, which REFINES (does not replace) the frozen spec:

- **SQHN ([2026-06-02-sqhn-card.md](../../docs/ground-truth/source_cards/2026-06-02-sqhn-card.md)) IS
  the `kwta` arm of this oracle**, not a new architecture — external survey + this precommit
  independently converged. Nothing in the 2025 AM survey or the missing primaries dominates it on
  the decisive axis (substrate-fit + local-non-backprop-write + per-layer-nonlinear). HAM lacks a
  local write; PC-AM/Rao-Ballard/Olshausen/RICA are PC=BP; Bio-SFA/SR are spectral (Report-125
  NULL); NSM is the Report-127 partition primitive.
- **ADMISSIBILITY CORRECTION (load-bearing):** SQHN's NATIVE write (Eq. 4) is the PCN free-energy
  gradient (its own Eq. 9) — reconstructive against the PARENT's top-down prediction = the §2
  inadmissibility shape (escapes FULL backprop only by being single-edge). ⇒ **port SQHN's kwta
  one-hot as the CAPTURE step ONLY; the residual pass stays this precommit's deterministic
  upward-only GHA deflation (§3). Do NOT thread Eq. 4 through a downward weight.** The §2/§6
  code-review guard (no downward weight/target/two-phase) is mandatory before any build.
- **Optional capture variant:** add SQHN's node-LOCAL neurogenesis (Eq. 5, FIXED Dirichlet decay
  `ε=α/(t+α)`) as a `kwta+neurogenesis` capture arm — anti-homunculus-clean as a fixed schedule;
  the legitimate SQHN contribution over plain kwta.
- **Dury compression caveat (new variable, motivating-only — Dury is uncarded/future-dated, and its
  MLP+InfoNCE mechanism = the banned Report-119 shape):** paradigmatic structure that TRANSFERS may
  emerge under a CAPACITY BOTTLENECK (under-fit → compress to recurrent regularities), not depth per
  se. This is in TENSION with neurogenesis (which GROWS capacity → toward memorization = the floor).
  ⇒ if neurogenesis is used it must be BOTTLENECKED, and add an **optional compression/capacity
  sweep** (vary slot count / cap; measure paradigmatic B-KILL vs compression ratio) as an orthogonal
  cheap probe of whether COMPRESSION — not depth — is the operative lever. Report 126 already saw an
  over-capacity regime FAKE a positive, so the capacity regime is a known confound.
- **Unchanged + reaffirmed:** the honest ceiling (§1: a PASS = local route to the global code, never
  beats-SVD; exceeding the global-NMF ceiling = artifact); the B-KILL headline + g5_depth gate; the
  offline-global-k-means control (the Report-127 differentiator = locality+online+depth); the
  build-gate fence (this authorizes the substrate-free oracle ONLY).
