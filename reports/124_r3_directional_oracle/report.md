# Report 124 — R3 directional-successor reachability oracle → NULL → escalate to the latent/hierarchical layer

**Status:** Phase-3 second-order growth re-scope. A **DRILL-DOWN feasibility oracle** (NOT a
graduation experiment), the directional analog of the §10 oracles in `experiments/62`. Frozen
pre-commit: [phase-3-r3-directional-oracle-precommit.md](../../notes/emergent-codebook/phase-3-r3-directional-oracle-precommit.md).
Harness: `experiments/63_directional_successor_oracle.py`. Floor (055–058) untouched.

**Verdict: NULL.** A **directional / successor-context** operator does **NOT** move the
paradigmatic (king/queen substitutability) signal into the locally-reachable dominant modes.
The Report-123 §4 local-vs-global bound — *paradigmatic structure lives in the subdominant
modes that local iteration cannot reach* — **holds for flat codes regardless of directional
asymmetry.** Per the pre-registered §3 disposition, this **kills the flat-code directional R3
build before it is built** (the oracle's entire purpose) and routes the project toward the
**latent/hierarchical layer (Option B)**. **Phase fence (precommit §9): routing toward Option B
does NOT authorize Phase-5 work — that fence is the user's to lift.**

---

## 1. The question + why an oracle

Report 123 §5 pre-registered **R3** (a predictive/successor-context *local* growth) on the
*hope* that **directional asymmetry** reshapes the operator spectrum so the paradigmatic signal
— which exists in the corpus but sits in the **subdominant** modes of the **symmetric** SPPMI
operator (global SVD reaches it: +0.109; local iteration cannot: +0.021) — lands in the
**dominant** modes a local power-iteration *can* reach. The domain-expert calibrated prior was
**~80–85% NULL** ("raw SR cosine = frequency-dominated collocational", precommit Appendix A
sol. 3). This oracle tests that hope **substrate-free, gauge-free, before any R3 build**.

## 2. What ran (WikiText-2, V=2002, 292 788 windows, γ=0.9, W=6, n_para=40 SimLex≥5)

Three symmetric-PSD operators read via the Levy-Goldberg embedding `W_r = U[:,1:r+1]·Σ^½`
(drop mode 1), plus the actual local dynamic:
- **S_dir** (directional concat: γ-discounted fwd/bwd directional-SPPMI signature `[ℓ2(Mf)‖ℓ2(Mfᵀ)]`) — PRIMARY.
- **S_fwd** (forward-only) — asymmetry discriminator; **S_bwd** reported as a diagnostic.
- **S_sym** (symmetric ℓ2-SPPMI cosine Gram) — apples-to-apples differential baseline; **S_2nd = build_S(SPPMI)** = the exact Report-123 operator (diagnostic).
- **Faithful local-iteration read:** `exp61.grow_G` (the real S′@G dynamic, α_anti=1, η_sep∈{0,0.05,0.1,0.2}) on `row_center(·)` of each operator — the verdict-bearing measure.

**Calibration ✓ (the INVALID bright line):** the raw-SPPMI-SVD anchor reproduces Report 123
**exactly** — para-vs-random **+0.1092, CI [0.082, 0.137]**, king/queen **0.222**. Collocational
sanity floor passes for every operator (CI-lo > 0) → this is a **mechanism** NULL, not a
corpus-scale verdict.

## 3. The verdict-bearing read — the FAITHFUL local dynamic (`grow_G`)

Best **collapse-free** (`d_eff_end/init ≥ 0.5`) FHRR para-vs-random specificity under the
*actual* local growth — directly comparable to Report-123's **+0.021** symmetric best and the
global **+0.109**:

| operator | best collapse-free spec | corr(cooc,drift) CI-hi | gate |
|---|---|---|---|
| **S_dir** (directional) | **+0.0067** [0.002, 0.012] | +0.259 | corr **FAIL** |
| S_fwd (forward-only) | +0.0104 | +0.389 | corr **FAIL** |
| S_sym (ℓ2 symmetric) | +0.0007 | +0.259 | — |
| S_2nd (exact Report-123 op) | −0.0003 | +0.205 | — |

- **Directional − symmetric = +0.006 — within noise**, ~**1/15** of the global-achievable +0.109,
  and **collocational** (every collapse-free cell fails `corr < 0.15`). Directionality buys
  essentially nothing the symmetric operator didn't already have.
- **Identical collapse signature to Report 123:** at η_sep=0 (pull-only) spec *looks* high
  (+0.34) but `d_eff_ratio = 0.00` — total collapse; H_anti prevents collapse but spec falls to
  ~0. The "high" number is the collapse, not paradigmatic structure.
- **The asymmetry discriminator strengthens the NULL:** forward-only is the **weakest** arm at
  full rank (+0.105 < S_sym +0.168) → directionality is not the missing degree of freedom.

*(Independently reproduced: a verification agent re-ran `grow_G` from scratch and obtained the
same dir +0.0067 / sym +0.0007.)*

## 4. The contraction artifact (a third metric trap, caught before banking)

The seductive spectral headline `spec_dir(r=5) = +0.531` vs `spec_sym(r=5) = +0.451` (diff
+0.080, which *passes* the frozen c1–c3) is a **low-rank-prominence CONTRACTION ARTIFACT**,
confirmed three independent ways:
1. The **symmetric** operator — which Report 123 already proved nulls under local iteration —
   scores the *same* ~+0.45 at r=5. A real directional effect cannot be shared by the control.
2. The para-cosine curve **drops +0.5 → +0.17** toward full rank (the fingerprint of shared
   word-prominence, not paradigmatic concentration — which would *stay* high); random-pair
   cosine decays +0.2 → 0.
3. The artifact inflates both arms (eff-rank 138 vs 29), so the +0.080 differential is two
   contractions subtracting — it **vanishes to +0.008 at full rank** (S_dir +0.175 ≈ S_sym +0.168).

**The frozen gate's c1–c3 "pass" is spurious:** its +0.04 margin was pre-registered on the
premise `spec_sym(5) ≈ 0` (precommit §2), which the run **falsified** (`spec_sym(5) = +0.451`).
Only **c4 (the corr discriminator) is premise-valid — and it FAILS** (CI-hi 0.457). The
SVD-spectral arm is therefore **demoted to a contraction-contaminated upper bound** that carries
no verdict.

## 5. Validity — the spectral-differential arm is INVALID (irreducibly), the NULL is not

The run is flagged INVALID on a **directional-vs-symmetric density mismatch (1.93 > 1.5)** that
is **structurally irreducible**: both arms are already at k=1 (the densest SPPMI shift) and a
directional successor co-occurrence is **intrinsically sparser** than a symmetric presence
co-occurrence, so the frozen §6 remedy ("re-pin k") is unavailable. **No re-run can fix it.** Per
the verification's adjudication, this INVALIDates **only the cross-arm spectral differential**
(already demoted in §4) — **not** the run. The NULL rests entirely on the three
**density-confound-immune, on-mechanism** reads, none of which cross-arm SVD-project:
- the **corr(log cooc, drift)** gate on S_dir's own r=5 readout — **+0.206, CI-hi 0.457, FAIL**;
- the **FHRR single-shot port** of `row_center(S_dir)` — **+0.154** but corr **CI-hi 0.472, FAIL**
  (a positive port spec that is collocational; cf. Report-122's symmetric port +0.101);
- decisively, the **faithful `grow_G` local-iteration** read (§3).

## 6. Decisive finding + disposition (pre-registered §3)

**The Report-123 §4 local-vs-global bound holds for flat codes *regardless of directional
asymmetry*.** A directional/successor operator inherits the same frequency dominance; the
paradigmatic (substitutability) structure remains in the subdominant modes the global SVD
isolates but local iteration — symmetric *or* directional — cannot reach. The forward/backward
split is a real DOF (per-token signature overlap 0.172) but it does not promote the
substitutability axis locally.

**→ Per the frozen §3 NULL disposition: escalate to the latent/hierarchical layer (Option B —
PCN/SFA, a predictive-coding hierarchy that puts paradigmatic structure in a latent layer
*above* the flat code). NOT more `S'@G` / γ / window knobs — and NOT a flat-code directional R3
build, which is now killed for ~zero build cost (exactly the oracle's job).**

**Phase fence (precommit §9, NOT an override):** Option B is tied to Phase-5 architecture. This
oracle *routes toward* it but does **not** authorize Phase-5 work — **lifting that fence is the
user's decision.**

## 7. Artifacts

- `experiments/63_directional_successor_oracle.py` (the new harness; reuses exp61/exp62 via
  importlib; the one new statistic is `build_directional_cooccurrence`).
- `reports/124_r3_directional_oracle/_oracle_wikitext_full.json` (+ `_localiter.json` copy,
  `.stderr`) — the full read incl. the faithful `grow_G` panel + S_2nd calibration.
- Frozen pre-commit: `notes/emergent-codebook/phase-3-r3-directional-oracle-precommit.md`
  (4-agent grounding workflow); verified by a 5-agent adversarial workflow (artifact /
  verdict-robustness / gate-integrity / faithfulness lenses → NULL, high confidence,
  proxy-demotion ruled legitimate, all flipping-dissents raised-and-refuted).
- Banked side-finding: the SVD-rank "reachability" proxy is **contraction-contaminated** — the
  faithful read is the local-dynamic `grow_G`, not an idealized low-rank SVD projection.
