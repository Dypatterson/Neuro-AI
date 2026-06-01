# FROZEN PRE-COMMIT — R3 Directional-Successor Spectral-Reachability Oracle

*Frozen 2026-06-01 BEFORE the run. Synthesized by a 4-agent grounding workflow
(domain-expert operator-design verification + gate authoring + adversarial
pre-mortem → synthesis). Tests the Report-123 §5 R3 conjecture — does directional
asymmetry move the paradigmatic signal into the locally-reachable dominant modes?
— as a cheap, substrate-free, gauge-free DRILL-DOWN oracle, the directional analog
of the trusted §10 oracles in `experiments/62_exp61_oracles.py`. Harness:
`experiments/63_directional_successor_oracle.py`.*

## 0. Experiment preamble — DRILL-DOWN / FEASIBILITY ORACLE (NOT a graduation experiment)

This is a **DRILL-DOWN feasibility oracle**, the directional analog of the trusted §10 oracles in `experiments/62_exp61_oracles.py`. It is **NOT** a Phase-3 graduation experiment and **cannot license an R3 build by itself** even on PASS (see §9 corpus-gap fence).

> **Active phase:** Phase 3 (emergent-codebook growth — the FOUNDATION; floor 055–058 untouched).
> **Headline metric per** `notes/emergent-codebook/phase-3-second-order-growth-precommit.md:40` (the demeaned para-vs-random specificity + `corr<0.15` decorrelation gate) **+** `experiments/62_exp61_oracles.py:54,74-85` (means-based `_boot_diff` = `mean_cos(para) − mean_cos(random)`): **para-vs-random specificity read as a function of retained spectral rank r — `spec(r)`** — compared DIFFERENTIALLY directional-minus-symmetric.
> **Required controls per** `phase-3-second-order-growth-precommit.md:288-291` (powered SimLex≥5, abort rule), `:40` (corr<0.15), `reports/123_growth_redesign_sweep_null/report.md:24-26` (the symmetric-SPPMI calibration anchor +0.1092 CI [0.082,0.137], king/queen 0.222): the **symmetric SPPMI operator** (`build_S`, exp61:298) as the known-subdominant calibration baseline; means-based random arm as the contraction control; `corr(log cooc, drift)` as the collocational discriminator.
> **Last verified result:** Report 123 — local B′+H_anti NULL 0/60; global SVD +0.1092 CI [0.082,0.137]; local best collapse-free +0.021 (~1/5 of global); king/queen global cos 0.222. The decisive §4 finding: paradigmatic ≡ the *subdominant* modes of the symmetric SPPMI operator, unreachable by local `S'@G` iteration.
> **Why this experiment now:** Report 123 §5 pre-registered R3 (a directional/predictive/successor local growth) as the next fork, on the *hope* that directional asymmetry reshapes the operator spectrum so the paradigmatic (substitutability) signal lands in the DOMINANT modes a local power-iteration can reach. This oracle tests EXACTLY that hope — cheaply, substrate-free, gauge-free — BEFORE any R3 build. A NULL kills R3-on-flat-codes before a build (routes to the latent layer); a PASS supplies the pre-frozen reachability target an R3 build must hit.

**Calibrated prior (domain expert):** ~15–20% PASS. The corpus's own card-level verdict (Appendix A sol. 3, `precommit:413`: *"Raw SR cosine = frequency-dominated collocational"*) and the decisive symmetric bound (Report 123 §4) both predict the directional split stays collocationally dominated → **likely NULL → latent/hierarchical layer.** This low prior is *why the oracle is worth running*: it is the cheapest substrate-free way to convert an open conjecture (predicted-null-by-card, no opened directional primary) into an empirical bound, and a clean NULL cleanly licenses the latent-layer escalation **without sinking an R3 build.** Running it is decision-relevant regardless of outcome.

**Anti-homunculus statement.** Every operator (F, B, Mf, Mb, Sig, S_dir, S_fwd, S_sym) is a fixed batch-offline statistic — same AH class as SPPMI's `−log k` shift. The SVD is a read-only diagnostic flashlight on a fixed operator (exactly `experiments/62`); the gates only halt/label, never feed back into the operator. **Online TD is BANNED** — F is one offline pass over the frozen stream with a fixed γ-discount window weight (the sleep/wake batch-offline split; Appendix A sol. 3). **There is no learned next-token predictor and no reconstruction-error term** — Reports 017/018 killed that first-order online class (error_driven/reconstruction, both below random Recall@1); this is a batch second-order directional co-occurrence count + PMI, cited as the killed baseline in the harness header so a reader cannot mistake it for the dead mechanism. LOCAL `S_dir@G` is the (future) mechanism; this oracle does not build it.

---

## 1. Operator family (≤3 variants; symmetric SPPMI baseline included as control)

All three operators are **V×V symmetric-PSD Gram matrices** in the same family, read via the identical Levy-Goldberg embedding `W_r = U[:,1:r+1]·Σ[1:r+1]^0.5` (remove mode 1). Directionality lives in the *signature*; symmetry in the *Gram*. Reuse `exp61.build_cooccurrence/build_sppmi/build_S/load_simlex_pairs/random_matched_pairs/corr_bootstrap_ci/hierarchical_bootstrap_ci` verbatim via importlib (no drift), exactly as exp-62 does.

1. **V1 — S_dir (directional concat) [PRIMARY].** Build directional cooc `F[i,j] += γ^(d−1)` for j at distance d∈{1..W} *after* i; `B = Fᵀ`. Directional SPPMI `Mf` = build_sppmi-analog on F (PMI-correct with directional marginals: row-marginal = ΣⱼF[i,j], col-marginal = ΣᵢF[i,j]); `Mb = Mfᵀ` (since B=Fᵀ with swapped marginals). **ℓ2-row-normalize each arm before concat** (the equal-weight guard — NOT exp61's sum-`rownorm`): `Sig = [ℓ2norm(Mf) ‖ ℓ2norm(Mfᵀ)]`, then ℓ2-normalize Sig rows. `S_dir = Sig @ Sigᵀ` (symmetric PSD Gram).
2. **V2 — S_fwd (forward-only) [ASYMMETRY DISCRIMINATOR, nearly free].** `S_fwd = ℓ2norm(Mf) @ ℓ2norm(Mf)ᵀ`. The cheapest test of the *actual* hypothesis: if directionality matters, fwd-only must produce a different reachability curve than symmetric. (`S_bwd` from `ℓ2norm(Mfᵀ)` is computed and REPORTED as a diagnostic but is NOT a separate gated variant — keeps the count at 3.)
3. **V3 — S_sym (symmetric SPPMI) [KNOWN-SUBDOMINANT CALIBRATION BASELINE].** `S_sym = build_S(sppmi) = rownorm(SPPMI@SPPMIᵀ)` (exp61:298) — the exact operator whose local iteration nulled in Report 123. Same vocab, same windows, **k_dir pinned by `pick_k_by_density` on the symmetrized F+B** so the directional arms share the 0.3–0.5 density band; **k_sym pinned on the symmetric presence-cooc C** so S_sym stays byte-comparable to the Report-123 calibration anchor.

**HELD IN RESERVE (not run unless V1/V2 ambiguous):** the proper successor matrix `Ψ=(I−γT)⁻¹`, `T=rownorm(F)`. Flagged **flashlight-only** (global inverse, no local analog) — a PASS there would NOT license a local build, only restate signal existence. Run only as a labeled contingency if V1/V2 give a non-null-but-ambiguous curve.

---

## 2. Frozen spectral-reachability gate (concrete numbers)

**Read.** For each operator S: `U,Σ,_ = svd(S)`; assert symmetry (`(S−Sᵀ).abs().max()<1e-5`) and `Σ≥0` (PSD); **remove mode 1**; embedding `W_r = U[:,1:r+1]·Σ[1:r+1]^0.5` (drop-column convention, applied IDENTICALLY to all operators and both pair arms). `spec(r) = mean_cos(para,W_r) − mean_cos(random_matched,W_r)` via `_cos_real` + `_boot_diff` (means-based; **NO glob double-subtraction**, the `dc642a0` fix).

**Probe r-grid (frozen):** `r ∈ {2, 3, 5, 10, 20, full}`, full = min(svd_rank=300, V).
**R_lo = 5** (post-common-mode modes 2–6). Anchored to Report 123: local B′+H_anti ran ~20 power-iteration steps and reached +0.021 ≈ 1/5 of the global +0.109, i.e. the locally-reachable regime is r ≤ 5 after removing mode 1. Frozen BEFORE the run; the full curve is reported so a reader sees whether R_lo caught a real low-r concentration or a bump.

**Bootstrap.** Flat means-based `_boot_diff` (exp-62, the trusted oracle) is the **gating CI**; also report the hierarchical seed×pair `hierarchical_bootstrap_ci` (exp61:414). n_boot=4000, seed=7 (exp62:74).

**FROZEN PASS — directional concentrates the signal in low r (ALL of):**
1. **Reachable signal is real:** `spec_dir(R_lo=5)` means-based bootstrap **CI-lo > 0**.
2. **Captures most of its own achievable signal in low r:** `spec_dir(5) ≥ f·spec_dir(full)`, **f = 0.50** (a genuine spectral re-concentration, not the symmetric tail-spread).
3. **Materially beats the known-subdominant baseline at the same r (DIFFERENTIAL — the load-bearing condition):** `spec_dir(5) − spec_sym(5) ≥ margin`, **margin = +0.04** (~2× the local-iteration ceiling of +0.021; clear of the ≈0 symmetric-at-r=5 floor whose CI half-width ≈0.027).
4. **Not directional-collocational (HARD gate, wired into the conjunction):** `corr(log cooc, drift) CI-hi < 0.15` (exp61 `corr_bootstrap_ci`), computed on the SAME object the headline reads (cosine-in-W_r at r=R_lo for the spectral arm; port_drift for the FHRR-port arm). This is the gate that killed all 11 collapse-free positives in Report 123.

`directional_pass = (spec_dir(5)_ci_lo > 0) AND (spec_dir(5) ≥ 0.50·spec_dir(full)) AND (spec_dir(5) − spec_sym(5) ≥ 0.04) AND (corr_ci_hi < 0.15)`.

---

## 3. Frozen PASS / NULL disposition

**PASS → green-light R3 grounding + pre-commit.** Directional asymmetry moves the paradigmatic signal into the locally-reachable dominant modes → the R3 γ-discounted directional growth is licensed to enter *grounding + its own pre-commit* (NOT directly to build — see §9 fence), carrying `spec_dir(5)` as the pre-frozen reachability target and the §2 collapse-free + `corr<0.15` gates forward.

**NULL → escalate to the latent/hierarchical layer.** NULL = `spec_dir(5)` CI overlaps 0 **OR** `spec_dir(5) − spec_sym(5) < 0.04` (the directional operator hides the para signal in the *same subdominant modes* as the symmetric one; `spec_dir(full)` may still lift while `spec_dir(5) ≈ 0` — the diagnostic that directionality did NOT re-concentrate). Pre-registered disposition: the local-vs-global bound (Report 123 §4) holds for **flat codes regardless of directional asymmetry** → directional reshaping is not the missing DOF → **escalate to the latent/hierarchical layer** (PCN/SFA, precommit §10 "Option B", Report 123 §5 contingency), **NOT** more S'@G / γ / window knobs. A flat-code directional growth is no longer licensed. (Phase fence: this points at Option B but does NOT authorize Phase-5 work — that fence is the user's to lift. §9.)

---

## 4. Build checklist (all adversarial-pre-mortem guards folded in)

- [ ] **Operator = symmetric PSD Gram**, read via `U·Σ^0.5` from its eigendecomposition. Assert `(S−Sᵀ).abs().max()<1e-5` and `Σ≥0` before reading. Do NOT SVD raw asymmetric `Mf` and read its U as a token embedding.
- [ ] **Per-arm ℓ2-row-normalize Mf and Mfᵀ BEFORE concat**; ℓ2-normalize Sig rows before the Gram. Report each arm's mean row-norm and `mean cos(ℓ2norm(Mf)_i, ℓ2norm(Mfᵀ)_i)` — if arms are near-identical per token the concat carries no directional DOF and a NULL is uninformative (it's the symmetric test again).
- [ ] **Arm-ablation panel:** `spec(r)` for Mf-only (V2), Mfᵀ-only (diagnostic), and [Mf‖Mfᵀ] (V1). If V1 doesn't beat the better single arm, "directional" reduces to single-direction frequency-leak.
- [ ] **Means-based metric only, NO glob double-subtraction** (`dc642a0`); random arm = `random_matched_pairs(..., max_cooc=2, seed=123)` is the contraction control.
- [ ] **DIFFERENTIAL PASS:** `spec_dir(R_lo) > spec_sym(R_lo)` by ≥ margin — contraction cancels because both operators contract identically as r→1. The single highest-leverage guard.
- [ ] **Report per-arm absolute cosines `mean_cos(para,W_r)`, `mean_cos(random,W_r)` AND an effective-rank panel at every r.** A spec rise riding random-cosines toward 0.9 (everything collapsed onto few axes) is the contraction artifact; a rise with random-cosines low is real concentration.
- [ ] **`corr(log cooc, drift) CI-hi < 0.15` as a HARD gate in the conjunction**, on the SAME object the headline reads. Collocational-pairs SANITY arm is the positive control for the discriminator, NOT a PASS condition.
- [ ] **king/queen reported as a single labeled probe for continuity with Report 123, explicitly NOT a gate.** Verdict is the powered SimLex≥5 set + hierarchical seed×pair bootstrap.
- [ ] **Freeze + hash the pair list** (`load_simlex_pairs` sha into the report output). Min-survivor abort rule frozen (§7).
- [ ] **γ, W, max_vocab, paradigmatic_max_cooc, R_lo, mode-removal convention, and the NULL disposition all FROZEN and hashed before the run.** Any γ/W sensitivity sweep is an explicitly-labeled DRILL-DOWN reported AFTER and SEPARATE from the frozen-γ headline.
- [ ] **Mode-removal = drop-column** `U[:,1:r+1]·Σ[1:r+1]^0.5`, applied identically to all operators/arms. Additionally report `cos(u_1, mean_direction)` so a reader can see whether "remove mode 1" actually corresponds to the common mode (`row_center` = S − rowmean, the offline analog).
- [ ] **FHRR single-shot port arm** (exp62:158-172, swapping `row_center(S_dir)` for the operator): `centroid = (M@G0.real)+1j(M@G0.imag)`, `Gp = sub.normalize(centroid)`, `_fcos` para-vs-random with the same means-based bootstrap. A spectral PASS that dies in the port = SVD artifact the local dynamic can't use. The `corr<0.15` read for §2.4 is computed on `port_drift`.
- [ ] **Batch-offline assert in the harness preamble:** the successor statistic is computed once offline over fixed windows, never updated from a running estimate; no gradient, no error term. Cite 017/018 as the killed baseline in the header.

---

## 5. Sanity gates (corpus-scale validity, run BEFORE interpreting the headline)

- **Collocational floor (every operator, FULL rank):** `spec_collo(full) = mean_cos(collo) − mean_cos(random) > 0` CI-lo > 0, for ALL of S_dir, S_fwd, S_sym, where collo = SimLex≥5 *co-occurring* pairs (cooc > 2). If flat at full rank → the corpus is too small for ANY distributional signal → **SCALE verdict, NOT a mechanism verdict** (exp62:23-25); do not read the para headline.
- **king/queen in vocab** (exp62:140-143); report S_dir / S_fwd / S_sym king/queen full-rank cosine vs the Report-123 anchor 0.222.

---

## 6. INVALID vs NULL bright line (run is INVALID / re-run, not a true NULL, if ANY of)

- **Symmetric baseline does not reproduce Report 123:** `spec_sym(full)` outside +0.109 CI [0.082,0.137] by more than the bootstrap half-width, OR S_sym king/queen full-rank cosine not ≈ 0.222 (±0.03). S_sym is the *calibration control* — if it doesn't land on the known anchor, the harness/vocab/k/density differs from Report 123 and the directional comparison is uncalibrated → fix the build, re-run.
- **Density mismatch between arms** > 1.5× → directionality confounded with sparsity → re-pin k so both arms share the 0.3–0.5 band.
- **Sanity collocational floor fails** (§5) → scale verdict, para read uninterpretable → INVALID for the mechanism question.
- **Realized n_para < 30** and the §7 frozen fallback not yet applied → apply fallback, then it is a valid run.

A true NULL (§3) requires: symmetric arm calibrated to the anchor, both sanity floors PASS, n_para ≥ 30 — and *then* `spec_dir(5)` fails §2. Only that is a real local-vs-global verdict.

---

## 7. Pre-registered constants (frozen, hashed into the report before the run)

| Constant | Frozen value | Justification |
|---|---|---|
| **γ (discount)** | **0.9** | γ=0.9 over W=6 gives a real directional gradient (weights 1.0→0.59), not a flat box; matches the exp-61/exp-62 window regime. Single value, NOT tuned. |
| **W (window)** | **6** | exp-61/exp-62 default (Report 121/123 config); directional test comparable to the symmetric baseline at the SAME window. |
| **max_vocab** | **2000** | exp-62 oracle scale; CPU-runnable offline. |
| **paradigmatic_max_cooc** | **2** | exp-62 default; defines paradigmatic (non-co-occurring) vs collocational split. |
| **simlex_min_sim** | **5.0** | precommit §5 powered set (n≈40); hashed pair list. |
| **R_lo** | **5** (post-mode-1) | Report 123 ~1/5-of-global anchor (§2). |
| **f (low-r fraction)** | **0.50** | half of full-rank specificity in first 5 modes = genuine re-concentration. |
| **margin (vs S_sym at r=5)** | **+0.04** | ~2× local-iteration ceiling (+0.021); clear of the ≈0 symmetric floor. |
| **corr gate** | **CI-hi < 0.15** | Report 123 / precommit:40 collocational discriminator. |
| **random seed** | **123** (pairs), **7** (bootstrap) | exp-62 verbatim. |
| **Min-survivor abort (frozen order)** | n_para < 30 → SimLex≥4.0 → add WordSim-353 as *separately-reported* secondary → max_vocab 4000 | precommit:288-291; each frozen before observing outcomes. |

---

## 8. Anti-homunculus check (PASS)

Who builds F/B: a fixed γ=0.9 directional count, no metric read. Who builds Mf/Mb: PMI's fixed marginal division + `−log k`. Who picks r: the grid {2,3,5,10,20,full} and R_lo=5 are frozen BEFORE the run by the Report-123 anchor. Where the "decision" lives: nowhere at runtime — the SVD is a read-only flashlight on a fixed operator; the gates only halt/label, never feed back. Online TD: absent — F is one batch-offline pass over the frozen stream. No error/reconstruction term (017/018 killed that class). **PASS.**

---

## 9. Corpus-gap + phase fences (flagged, not overrides)

1. **A PASS cannot be load-bearing for an R3 BUILD by itself.** The R3 directional operator's literature backing is **card-only/absent**: the canonical SR primary (Stachenfeld-Botvinick-Gershman 2017) is NOT in `docs/ground-truth/source_manifest.jsonl`; there is no TCM source; the one directional primary (tPC, arxiv:2305.11982) is abstract-only. Per Hard Rule 3 this is correctly scoped a DRILL-DOWN. A PASS green-lights R3 *grounding + pre-commit* (which must open + card a directional primary), NOT a direct build.
2. **The NULL contingency (Option B latent/hierarchical layer) points at Phase-5 architecture** (precommit:170-174). A NULL here ROUTES to Option B but does NOT authorize Phase-5 work — that fence is the user's to lift.
3. **No TCM variant** — zero corpus grounding.
