# FROZEN PRE-COMMIT (hardened via /grill-with-docs — freeze before the run) — Phase-3 nonlinear-competition kill-test

*Drafted 2026-06-01, hardened by a /grill-with-docs interview (6 questions, Q1-Q6 resolved inline; freeze
before running). After the
flat-code growth family was exhausted (Reports 121/123/124/125, route-invariant across 7 LINEAR
single-projection operators) and BOTH the behavioral reframe (Report 126, NULL) and the inverse-recall
static check (experiments/67, the Idea-1 collapse-check) confirmed the bound is CAPABILITY-level not a
metric artifact, a grounded 9-agent ladder-check (this session) isolated the SINGLE remaining place a
linear-bound escape could live: a NONLINEARITY INSIDE A LOCAL RECURRENT FIXED POINT (NSM / assembly
k-WTA) — the brain-legal LOCAL cousin of the one GLOBAL positive in the whole arc, Report 125 Oracle E
(NMF, +0.124/+0.176/+0.188, controls real). Harness: `experiments/68_nonlinear_competition_kill_test.py`
(reuses exp61/62/63/65 via importlib). Floor (055-058) untouched.*

## 0. Preamble — DRILL-DOWN feasibility oracle (NOT graduation)

> **Active phase:** Phase 3 (FOUNDATION; floor 055-058 read-only).
> **Headline metric (frozen below):** the para-vs-random gauge-free specificity of a LOCAL
> nonlinear-competitive writer (NSM / k-WTA) on the second-order / transition operator, vs (a) the
> LINEAR `grow_G` anchor (+0.021 local), (b) the GLOBAL NMF ceiling (Oracle E, +0.19), with the
> MANDATORY within-set LABEL-SHUFFLE pair-specificity gate (Report 126 lesson). NOT the phase-graduation
> headline → explicitly a DRILL-DOWN per CLAUDE.md.
> **The ONE question:** does a nonlinearity INSIDE the recurrent fixed point (not a per-cue softmax that
> is then linearly summed) let a LOCAL writer recover most of Oracle E's +0.19, or null toward the
> LINEAR +0.021 like Oracles B/C/D?
> **Required controls (frozen):** the raw-SPPMI-SVD calibration anchor (+0.1092 CI[0.082,0.137], kq
> 0.222 — INVALID otherwise); the LINEAR flat-SPPMI `grow_G` baseline (+0.021); the GLOBAL NMF ceiling
> (Oracle E, reproduced in-harness); within-set LABEL-SHUFFLE (same tokens, broken pairing); random-
> nonneg control (Report 125 §3); d_eff collapse floor; 5 seeds; SimLex≥5 n=40 frozen pairs.
> **Last verified result:** Report 126 (behavioral NULL) + experiments/67 (inverse-recall static NULL).
> Oracle E is the GLOBAL positive whose LOCAL writer is untested (Report 125 §5).
> **Why now:** every LINEAR-local route + the behavioral reframe are closed; the grounded ladder-check
> names hard local competition as the one live escape — and grill Q1 upgraded its prior: nonneg-SM
> provably escapes PCA/dominant (the 2018 manifold-tiling result), so the open question is narrowed to
> paradigmatic-ALIGNMENT (the gate measures it), not whether it escapes at all. Killable substrate-free
> in one afternoon.

**LITERATURE STATUS (binding on the framing; corrected via grill Q1, 2026-06-01).** Distinguish two
objectives: (i) **LINEAR** similarity matching → PCA → the DOMINANT subspace (the locality trap;
Oja/Sanger likewise reach the top eigenvector(s)); (ii) **NONNEGATIVE** similarity matching — the
rectifying `[W x − M y]₊` network — → **manifold-tiling LOCALIZED receptive fields**, a part-based /
clustering structure that is NOT dominant-subspace projection (Sengupta-Tepper-Pehlevan-Genkin-Chklovskii,
NeurIPS 2018, "Manifold-tiling Localized Receptive Fields are Optimal in Similarity-preserving Neural
Networks"). **The rectification inside the fixed point is precisely the ingredient that breaks the
PCA/dominant result** — and the part-based regime it produces is the SAME representational class NMF
(Oracle E) used to reach +0.19 globally. So READ 3 (NSM) is the documented LOCAL/online cousin of NMF,
NOT a dominant-subspace method, and the literature SUPPORTS that it escapes the dominant-subspace trap.
**The remaining open question is ALIGNMENT, not escape:** does the part-based structure align with the
PARADIGMATIC subdominant modes (NMF/Oracle-E is direct evidence it does, globally) or with
frequency/collocational structure (the LABEL-SHUFFLE gate is there to catch exactly that). HARD-RULE-3
HEDGE: the manifold-tiling + similarity-matching + sparse-Hopfield primaries (arxiv:1703.07914, the 2018
NeurIPS manifold-tiling paper, arxiv:2411.08590, arxiv:2309.12673, pdf:sqhn-2024) are link_only/uncarded
→ they MOTIVATE this oracle and the upgraded prior but are NOT load-bearing for a build until carded. A
PASS licenses opening + carding them before any build; a NULL kills the k-WTA/NSM/sparse family
route-invariantly. The genuine uncertainty is paradigmatic-alignment, which the gate measures — NOT
whether nonneg-SM escapes PCA (it does, by the 2018 result).

## 1. Anti-homunculus / discipline

The competitive writer must be a LOCAL geometric/energy dynamic, never an arbiter:
- **NSM** `y ← [W x − M y]₊` iterated to a recurrent fixed point: (a) what moves locally = the rectified
  hidden activations y under FIXED W (feedforward Hebbian) and M (lateral anti-Hebbian); (b) the
  "decision" lives in the recurrent fixed point of the competitive dynamics (settling/energy descent of
  the similarity-matching objective), NOT a metric read; (c) the fixed dynamic replacing if/then = the
  rectification `[·]₊` applied EVERY step, never a runtime "if-mode-is-dominant-then-subtract"; (d)
  control = label-shuffle + random-nonneg. **PASS.**
- **k-WTA / assembly cap** keeps the top-k activations per step: this is a FIXED rank-threshold cap
  (the Assembly-Calculus primitive), not a metric-gated branch — admissible IF k is PRECOMMITTED and
  applied uniformly every step (NOT chosen by reading the output). A **SOM BMU writer is BANNED** (the
  argmax-unit-select-and-update is the canonical if-metric-then-act thermostat; SOM is read-only-exempt).
- **β / k / cap are FIXED and PRECOMMITTED**, never swept-until-the-gate-passes. Tuning-to-pass is the
  metric-fishing INVARIANT-VIOLATION this project's g3/n=40 history shows it is prone to.
- **Batch-offline compliance (grill Q4, VERIFIED not assumed):** NSM's objective `min‖XᵀX − YᵀY‖`
  superficially looks like a reconstruction-ERROR objective (which would trip the "no online
  error-driven writes" ban). It is NOT error-driven in the banned sense: Pehlevan-Chklovskii's result is
  that this objective is optimized by PURELY LOCAL Hebbian (W ← +y xᵀ) / anti-Hebbian (M ← +y yᵀ)
  updates — no global error signal propagated, no target−output gradient, no reward. Same class as the
  project's existing `H_anti` and `CueDecorrelator` (themselves similarity/whitening statistics). AND we
  stream a FROZEN operator (3a) / frozen window buffer (3b) in OFFLINE passes (sleep-phase replay), not
  runtime — batch-offline by construction. Doubly compliant. k-WTA likewise updates by Hebbian
  expansion + a fixed cap, no error signal. FHRR-native read. 055-058 floor untouched.

## 2. The three reads (frozen — everything else held fixed)

Same WikiText-2 frozen windows (V=2002, W=6, γ=0.9), same n=40 SimLex≥5 non-cooc pairs, same gauge-free
means-based para-vs-random specificity, same calibration anchor. Operator = the symmetrized transition
M_trans = (T+Tᵀ)/2 (the EXACT operator Oracle E factorized — so the NSM/E comparison is apples-to-apples)
AND/OR the second-order SPPMI `exp61.build_S` (the +0.021 linear anchor's operator).

- **READ 1 — LINEAR control** (the +0.021 anchor): `exp61.grow_G(row_center(M))` over η_sep grid. The
  collapse-free best reproduces Report 123/125's +0.021. The floor any escape must beat.
- **READ 2 — SOFTMAX-REWEIGHT, FAITHFUL Ideas 2/7** (predicted-null; grill Q6 = keep faithful):
  build the Report-66 graduated memory H + CueDecorrelator (reuse `experiments/66`), the per-cue
  completion distribution `p(·|cue) = softmax(β · (H·decorr(cue)) · value_cbᵀ)` at **β=10 PRECOMMITTED**
  (the replay β per the live-operational-policy, NOT swept), then the coactivation operator
  `S = Σ_cue p(·|cue) ⊗ p(·|cue)` over the frozen cue buffer, read by the SAME LINEAR `grow_G` on
  row_center(S). This tests Ideas 2/7 DIRECTLY in-harness (not only by the `experiments/67` reduction) and
  makes EXPLICIT that a per-cue softmax LINEARLY SUMMED then power-iterated does NOT escape — the
  nonlinearity is OUTSIDE the recurrent fixed point. The load-bearing distinction from READ 3 (where the
  rectification is INSIDE the fixed point). `experiments/67` already closed the static-cosine cousin
  (Idea 1) at the 126 NULL; READ-2 closes the soft-coactivation cousin (Ideas 2/7) here.
- **READ 3 — HARD LOCAL COMPETITION** (the genuine bet): the nonlinearity INSIDE the fixed point.
  **TWO precommitted competitive writers (grill Q3, "2-arm, test everything, no assumptions"):**
  - **W1 = NSM:** `y ← [W x − M y]₊` recurrent fixed point + online Hebbian (W) / anti-Hebbian (M)
    nonnegative-similarity-matching rule. The manifold-tiling LOCAL cousin of NMF/Oracle-E;
    anti-homunculus-cleanest (soft rectification, no hard argmax).
  - **W2 = assembly k-WTA:** an expand-then-cap rectified recurrent map keeping the top-k activations per
    step (Papadimitriou-Vempala Assembly Calculus). **Anti-homunculus:** the cap is a FIXED PRECOMMITTED
    rank-threshold applied UNIFORMLY every step — a fixed dynamic, NOT a metric-read-then-branch arbiter,
    so admissible (grill Q3); a **SOM-BMU writer remains BANNED** (argmax-unit-select-and-update = the
    thermostat). The cap k is FROZEN, never tuned-to-pass.
  Each writer is run in TWO input modes (grill Q2; "do both" — they bracket the question, core shared):
  - **3a — operator-level, E-MATCHED (primary):** symmetric nonnegative similarity-matching of `M_trans`
    (the EXACT operator Oracle E factorized) into k∈{8,16,32} nonneg token-codes Y (YᵀY ≈ M_trans),
    streamed one token-row per step over a few offline passes. Read static para-vs-random on the Y rows
    (= Oracle E's `static_read`); report **as a fraction of E's in-harness NMF specificity** at matched k.
    Asks: does the LOCAL recurrent rule recover E's +0.19, or is the global batch optimization load-bearing?
  - **3b — fully-online from raw windows (faithfulness):** stream the actual windowed observations (no
    precomputed operator); a token's code = the NSM output aggregated over its windows. More
    biologically faithful (the mechanism a build would use). **REDUCTION CAVEAT (grill Q2):** the
    per-window output aggregated per token is structurally close to the context-centroid object nulled in
    Idea 1 / Report 126 — so 3b carries a real risk of collapsing back onto the 126 null. Therefore 3b
    requires, ON TOP of the label-shuffle, an explicit **"3b beats the Idea-1/126 centroid baseline by
    ≥+0.02"** check, or a pass could be the centroid in disguise.
  - **Interpretation matrix (the payoff):** 3a-pass∧3b-pass = strong (local writer recovers E AND survives
    raw stream); 3a-pass∧3b-null = the operator pre-computation was load-bearing; 3a-null∧3b-null = the
    family is dead route-invariantly; 3a-null∧3b-pass = surprising, scrutinize.
  - **3a CORRECTION (build-time finding, 2026-06-01 — supersedes the draft):** symmetric-NSM matching
    `M_trans` DIRECTLY requires inputs whose Gram = M_trans, i.e. `sqrt(M_trans)` = a global
    EIGENDECOMPOSITION — the banned SVD-homunculus shape INSIDE the mechanism. So the genuinely-LOCAL
    writer is **ROW-STREAMING** (inputs = operator rows → preserves the 2nd-order Gram, NO eig).
    **Row-streaming is the 3a PRIMARY**, and the operator is UNIFIED to a 2nd-order similarity so the
    linear floor (`grow_G`), the NMF ceiling (Oracle E, re-anchored IN-HARNESS on the same operator), and
    the NSM/kWTA candidates all factorize the SAME operator (apples-to-apples). Primary operator =
    `exp61.build_S` (SPPMI 2nd-order = the +0.021 grow_G floor operator); secondary = `M_trans` (E's +0.19
    operator), with E re-anchored in-harness on each. The 2nd-order Gram is a SANCTIONED batch-offline
    operator (same shape as `build_S = SPPMI@SPPMIᵀ`); only the eig/sqrt is banned.
- **CEILING — GLOBAL NMF** (Oracle E, reproduced in-harness at matched k): +0.19. READ 3's recovery is
  reported AS A FRACTION of this (the "does the LOCAL writer recover most of E" question).

## 3. Frozen gate (PASS = ALL; the label-shuffle is the headline + the multiple-comparisons guard)

**The grid (test everything, grill Q3):** {W1 NSM, W2 k-WTA} × {3a operator-level, 3b fully-online} ×
k∈{8,16,32} × 5 seeds — ALL cells run and reported (no cherry-picking). The family of headline
candidates = the **4 mechanism-cells** {NSM·3a, NSM·3b, kWTA·3a, kWTA·3b}, each read at its best
collapse-free cell over the k-sweep (the k-sweep is MATCHED to Oracle E, precommitted — the matched
comparison, not fishing). A cell PASSES iff ALL of:
- **g1** para-vs-random specificity bootstrap **CI-lo > +0.05** (tightened from +0.04 — the family-wise
  margin for the 2-writer × 2-mode grid; clears the +0.021 linear ceiling decisively).
- **g2** beats READ-1 (linear `grow_G`) by **≥ +0.02** AND beats READ-2 (softmax-reweight) by ≥ +0.02
  (isolates the recurrent nonlinearity-INSIDE-the-fixed-point as the active ingredient).
- **g4** d_eff_ratio ≥ 0.5 (no collapse).
- **B-KILL (the headline + the MULTIPLE-COMPARISONS GUARD, Report 126 lesson):** the within-set
  LABEL-SHUFFLE pair-specific residual (real − shuffled) bootstrap CI-lo > 0 on BOTH read variants, in
  **≥ 4/5 seeds**. This is what makes "test everything" honest: a para-set-hubness false positive FAILS
  the label-shuffle no matter how many cells we run, and a chance pass surviving 4/5 independent seeds
  across the grid is ~4e-4 — so the grid does not inflate the false-positive rate.
- **3b additional control:** a 3b cell must ALSO beat the Idea-1/126 context-centroid baseline (the
  `experiments/67` static read) by **≥ +0.02** (else 3b's per-token aggregate is the nulled centroid).
- **random-nonneg control:** a random-nonnegative writer of the same shape scores ≈0 (Report 125 §3).
- **frequency-matched rand-pairs (secondary, reported, grill Q4):** in addition to the cooc-matched
  `exp63.select_pairs` rand pairs, draw a rand-pairs set FREQUENCY-matched to the para token set, to
  fully close the g1/g2 para>rand interpretation (the headline B-KILL is already frequency-immune — its
  shuffle draws from the same para pool — so this sharpens only the secondary para>rand arm).
- **NOT g3** (corr<0.15 is dead at n=40 for static reads — even the +0.109 SVD anchor fails it).
**PASS verdict** = at least one of the 4 mechanism-cells clears the FULL gate consistently (≥4/5 seeds);
the full grid is reported either way. **NULL** = no mechanism-cell clears it. **INVALID** = the
raw-SPPMI-SVD anchor misses +0.109/0.222. All hyperparameters (NSM iters / learning rates, k-WTA cap,
the k-sweep, β for READ-2, the +0.05/+0.02 thresholds, the ≥4/5-seed bar, the SimLex sha) are FROZEN
before the run — NONE swept-to-pass (the metric-fishing INVARIANT-VIOLATION).

## 4. Disposition (frozen)

- **PASS** (READ 3 clears g1∧g2∧g4∧B-KILL, anchor valid, random-nonneg ≈0): hard local competition
  breaks the linear bound → the escape is specifically the recurrent rectifying nonlinearity → open +
  CARD the NSM / sparse-Hopfield primaries (arxiv:1703.07914, arxiv:2411.08590, arxiv:2309.12673,
  pdf:sqhn-2024) BEFORE any build, then a competitive-writer grounding + precommit. **A BUILD remains the
  Phase-5 fence (the user's to lift).**
- **NULL** (READ 3 nulls near +0.021, or para>rand but FAILS the label-shuffle): the linear bound holds
  even under read-side competition → Ideas 2/7/9 + the whole k-WTA/NSM/sparse family are dead
  route-invariantly → do NOT build a competitive writer; AND we learn the load-bearing thing — Oracle E's
  positive was carried by GLOBAL deflation/optimization, not the nonlinearity per se (the same warning
  Report 125 gave for the TEM factorization). → the one remaining live route is the TEM local-reachability
  oracle (the slot-binding LOCAL writer for E), Step 4 of the ladder.
- **Convergence note:** READ 3 (NSM, the competitive local writer for E's NMF) and the TEM oracle (the
  slot-binding local writer for E's factorization) are TWO local-writer candidates for the SAME global
  positive (Oracle E). A NULL on both = the global optimization is irreducibly load-bearing → escalate to
  the replay-interleaving (CLS) reframe or the grounding reframe.

## 5. Build checklist (`experiments/68_nonlinear_competition_kill_test.py`)
- [ ] Reuse via importlib: `exp61.{build_cooccurrence,build_sppmi,build_S,pick_k_by_density,row_center,
      grow_G,d_eff,corr_bootstrap_ci}`, `exp62.{_cos_real,_fcos,_boot_diff}`, `exp63.{select_pairs,
      raw_sppmi_svd_anchor,build_directional_cooccurrence}`, `exp65.{build_transition_operator,nmf_slots,
      static_read,gate}` (Oracle-E ceiling + the gate), `exp66.{encode_cue_true,boot_mean_ci}` +
      `exp66`'s memory-build (for READ-2's H and the 3b Idea-1/126 centroid baseline).
- [ ] New code: `nsm_features(S, k, iters, ...)` (symmetric NSM `[Wx−My]₊` recurrent rule + offline
      Hebbian(W)/anti-Hebbian(M) update; 3a operator-stream + 3b window-stream input pipelines),
      `kwta_features(S, k, cap, iters)` (expand-then-cap recurrent map), `softmax_coactivation_operator(H,
      cues, beta)` (READ-2's faithful `S=Σ p⊗p` at β=10), `freq_matched_rand_pairs(...)`.
- [ ] Calibration anchor EVERY run; INVALID if it misses +0.109/0.222.
- [ ] Oracle-E NMF reproduced in-harness as the CEILING (per-k); READ-1 linear `grow_G` as the +0.021
      FLOOR; READ-2 faithful Ideas-2/7 (H-coupled) as the outside-fixed-point control.
- [ ] The 4 mechanism-cells {NSM,kWTA}×{3a,3b} × k∈{8,16,32}, 5 seeds; report the FULL grid.
- [ ] Controls wired: within-set label-shuffle (B-KILL, ≥4/5 seeds), random-nonneg, freq-matched
      rand-pairs, the 3b Idea-1/126 centroid baseline (reuse `experiments/67`).
- [ ] Planted/small SMOKE validates NSM + k-WTA convergence + the gate before the WikiText run.
- [ ] FREEZE before the run: NSM iters + learning rates, the k-WTA cap, k∈{8,16,32} (matched to E),
      β=10 (READ-2), the **+0.05**/+0.02 thresholds, the ≥4/5-seed B-KILL bar, the SimLex sha — all
      precommitted, NONE swept-to-pass. Substrate-free verdict; FHRR-port is the deferred Stage-1 (fence).

## 6. Grill resolutions (all resolved 2026-06-01 via /grill-with-docs)
1. ~~Does NSM/similarity-matching provably recover only the DOMINANT subspace...?~~ **RESOLVED (grill Q1):
   LINEAR SM → PCA/dominant; NONNEGATIVE SM (rectifying network) → manifold-tiling LOCALIZED/part-based
   receptive fields (Sengupta et al. NeurIPS 2018), the same class NMF/Oracle-E used.** The rectification
   breaks the PCA result. The open question is paradigmatic-ALIGNMENT (the gate measures it), not escape.
2. ~~M_trans vs SPPMI-2nd operator?~~ **RESOLVED (grill Q2): M_trans PRIMARY (the exact operator E
   factorized → apples-to-apples E-recovery), SPPMI-2nd as a secondary robustness arm (the +0.021 linear
   anchor's operator). Two input modes 3a (operator-level, E-matched) + 3b (fully-online, faithfulness).**
3. ~~k-WTA distinct from NSM, or three-faces-of-one?~~ **RESOLVED (grill Q3): run BOTH as a precommitted
   2-arm {NSM, k-WTA} with a tightened gate (g1 CI-lo>+0.05) + the ≥4/5-seed label-shuffle MC guard —
   "test everything, no assumptions," but the multiple-comparisons are controlled.**
4. ~~random-nonneg vs frequency-matched control?~~ **RESOLVED (grill Q4): keep random-nonneg + the
   label-shuffle (the headline, frequency-immune by construction) + ADD frequency-matched rand-pairs as a
   cheap secondary for the para>rand arm. NSM batch-offline compliance VERIFIED (§1).**
5. ~~substrate-free vs FHRR-port?~~ **RESOLVED (grill Q5): SUBSTRATE-FREE is the complete capability
   verdict (reads Euclidean Y rows exactly as Oracle E was read); the FHRR-port is a DEFERRED Stage-1
   BUILD (the nonneg-code→FHRR encoding is a design decision = the Phase-5 fence), NOT a read bolted on
   here. Keeping them separate is what let exp63 kill the flat-R3 build cheaply.**
