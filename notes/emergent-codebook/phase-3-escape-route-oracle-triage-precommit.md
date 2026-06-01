# FROZEN PRE-COMMIT — Phase-3 escape-route oracle triage (4 substrate-free oracles)

*Frozen 2026-06-01 BEFORE running. After the flat-code growth family was exhausted
(Reports 121/123/124), two workflows (Option-B grounding + non-flat brainstorm) + a
completeness critic converged on a small set of bound-escape routes. This batch triages the
FOUR distinct structural escape-routes with cheap substrate-free oracles BEFORE any substrate
build — the exp63 pattern. Synthesis: [brainstorm-workspace/2026-06-01-nonflat-phase3/SYNTHESIS.md].
Harness: `experiments/65_escape_route_triage.py` (reuses exp61/62/63 via importlib).*

## 0. Preamble — DRILL-DOWN feasibility oracles (NOT graduation experiments)

> **Active phase:** Phase 3 (FOUNDATION; floor 055-058 untouched).
> **Headline metric:** the FAITHFUL `grow_G` local-dynamic gauge-free para-vs-random
> specificity (means-based, exp62/63), NOT the SVD low-rank spectral read (Report 124 §4
> proved that proxy is a contraction artifact — retained only as a contaminated upper bound).
> **Required controls:** the raw-SPPMI-SVD calibration anchor (must reproduce +0.1092
> CI[0.082,0.137], king/queen 0.222 — INVALID otherwise); the flat-SPPMI `grow_G` baseline
> (matched schedule); `corr(log cooc, drift) < 0.15` collocational discriminator; d_eff
> collapse floor; SimLex≥5 frozen pair list (n≈40).
> **Anchors:** global SVD +0.109; flat local (Report 123) +0.021; directional local (Report
> 124) +0.0067. The "decisively past the bound" bar is **+0.04** (~2× the flat-local ceiling).
> **Why now:** the flat-code family is exhausted; these triage the four escape-routes for
> ~zero build cost before any substrate is built.

**Anti-homunculus / discipline (all four).** Every operator is a fixed batch-offline statistic
(co-occurrence / transition counts + a fixed transform — same class as SPPMI's `−log k`).
The SVD is a read-only diagnostic flashlight. The verdict-bearing `grow_G` read uses the
substrate's `normalize` + `repulsion_force` (H_anti) — the H_anti is the LOCAL
whitening/decorrelation analog (load-bearing for Oracle B per the critic). Online TD is BANNED;
all transition/SR statistics are batch-offline. The 055-058 floor is untouched by construction
(separate codebooks, never read at recall).

## Shared gate (all four oracles; frozen)

A candidate operator `M` is read by `exp61.grow_G(row_center(M))` over η_sep∈{0,0.05,0.1,0.2}
(α_anti=1), epochs=20, α0=0.3, decay=0.9, D=4096, on the FROZEN WikiText-2 windows (V≈2002,
W=6, γ=0.9 — identical to exp63 so the anchor calibrates). **PASS = ALL of:**
- **g1** best COLLAPSE-FREE (d_eff_ratio ≥ 0.5) `grow_G` para-vs-random specificity, means-based
  bootstrap **CI-lo > +0.04** (clears the +0.021 flat ceiling and the +0.0067 directional ceiling).
- **g2** beats the matched flat-SPPMI `grow_G` baseline (same schedule) by **≥ +0.02**.
- **g3** `corr(log cooc, drift)` bootstrap **CI-hi < 0.15** (the discriminator that killed every
  flat positive in 123/124).
- **g4** floor-guard: best cell d_eff_ratio ≥ 0.5 (no WS-InfoNCE-style collapse).
**NULL** = no cell clears g1∧g2∧g3∧g4. **INVALID** = the raw-SPPMI-SVD anchor misses +0.109/0.222.
SVD spectral read reported ONLY as a contraction-contaminated upper-bound diagnostic.

---

## Oracle B — SFA / SR slowness (Route I/III) — THE LEAD

**Bio:** Stachenfeld-Botvinick-Gershman 2017 (grid=SR eigenvectors); Sprekeler (SFA≡SR);
Lipshutz-Pehlevan-Chklovskii Bio-SFA (local Hebbian/anti-Hebbian whitening); 2026 "Word classes
emerge from SR." **Operators (all VxV, dominant modes = the candidate paradigmatic axis):**
- `M_trans` = symmetrized one-step transition: `T = rownorm(F)` (F = γ-discounted directional
  cooc), `M_trans = (T + Tᵀ)/2`. Its top eigenvectors (after mode-1 removal) are the slow / SR
  low-frequency modes (the substitutability axis per the lineage).
- `M_SR` = symmetrized truncated SR: `Ψ = Σ_{t=0}^{K} γ^t Tᵗ` (γ=0.9, K=10, batch-offline — the
  γ-reweight INFLATES the slow modes, the critic's subdominant→dominant mechanism), `M_SR=(Ψ+Ψᵀ)/2`.
- CONTROL: `M_cooc` = the exp61 SPPMI second-order operator (known-subdominant).

**THE CRITIC'S LOCALITY TRAP (load-bearing):** power iteration converges to the DOMINANT mode
(the collocational common mode). Reaching the slow paradigmatic modes requires the common mode
removed each step — which `row_center` + H_anti do LOCALLY (not a global deflation). The verdict
is whether `grow_G(row_center(M_trans/M_SR))` + H_anti reaches king/queen collapse-free. A PASS
proves the subdominant→dominant transform is LOCALLY reachable; a NULL proves the bound is
operator-invariant for single-projection latents → the escape needs a genuine multi-LAYER
hierarchy. **Prior ~60-65% NULL.** PASS → greenlight an SFA-latent grounding+precommit (NOT a
build; FHRR-port Stage-1 + phase fence remain). Primary to open before any build: pmc:PMC1963505.

## Oracle C — eligibility × surprise two-timescale gate (the missing family)

**Bio:** Gerstner et al. 2018 (eligibility traces / three-factor); Kozachkov-Krotov 2024
(neuron-astrocyte slow field); Edelman (neural Darwinism). **Operator:** `M_elig[i,j] =
cooc[i,j] · novelty[i,j]`, novelty MULTIPLICATIVE and frequency-suppressing
(`novelty[i,j] = 1/(1+min(freq_i,freq_j))^β`, β swept {0.5,1.0}) — distinct from SPPMI's marginal
DIVISION. **Read via the shared grow_G gate.** **NON-NEGOTIABLE DISCRIMINATOR (g2 sharpened, the
critic's caveat):** PMI-surprise is already inside SPPMI, so `M_elig` must beat the **plain
SPPMI baseline** (M_cooc) by ≥ +0.02 AND drop `corr(cooc,drift)` below 0.15 where SPPMI did not —
else it collapses to 123 and the family is killed. **Honest scope flag (report):** this is the
CHEAP frequency-novelty surrogate of the full two-timescale eligibility×surprise mechanism (whose
faithful form needs the substrate's MHN settling-residual as the local surprise signal); a PASS
greenlights building the faithful version, a NULL kills the cheap surrogate. **Prior ~70% NULL.**

## Oracle D — order-vs-context channel split (Route IV)

**Bio:** Howard-Kahana TCM; BEAGLE (Jones-Mewhort) — order yields paradigmatic where context
yields syntagmatic. **NOT a Report-009 re-run** (009 tested order for ordered-vs-shuffled
RECALL@1, never the paradigmatic gate — verified). **Operator:** two FHRR vectors per token from
one pass — `ctx[i]` = window co-occurrence bundle (the existing channel) and `ord[i]` = the
position-bound BEAGLE order vector (`Σ` over occurrences of bound permuted neighbors
`Π^{+k}(nbr)`), using the substrate's permutation. Read paradigmatic specificity per channel via
the **static** gauge-free para-vs-random (cosine on the bundled vectors) AND the grow_G gate on
the order-channel Gram. **PASS** (g1-g4 on the ORDER channel) = order carries paradigmatic where
context does not. **Flagged prior:** the directional precommit's theory says order-binding is
syntagmatic (collocational) → likely g3 FAIL; the empirical BEAGLE result says otherwise → worth
the ~30-line test. Guard: do NOT collapse the position spectrum to k=±1 (that degenerates to the
Report-124 successor operator). **Prior ~65% NULL.**

## Oracle E — TEM transition-slot factorization (Route II)

**Bio:** Whittington et al. 2020 (TEM); Behrens 2018. **Operator:** factor the symmetrized
transition operator `M_trans` into k≈8 nonnegative relational slots via NMF (GLOBAL = flashlight
for the oracle; the LOCAL Hebbian-path-integration version is the build). Token i's slot-vector
= its NMF row; paradigmatic = shared slot-distribution (same structural role, swappable content).
**Read:** static gauge-free para-vs-random on the slot-vectors (cosine) — the slot factorization
IS the embedding, no grow_G needed (the structure factor is read directly). **PASS:** slot-vector
para-vs-random CI-lo > +0.04 AND beats the flat-SPPMI baseline by +0.02 AND corr(cooc,drift)<0.15
AND king/queen share slots (cooc≈0). **Sweep k∈{8,16,32}.** A PASS greenlights grounding the
Hebbian-path-integration TEM (mandatory: grow the structure factor by Hebbian path-integration,
NOT backprop — TEM-as-published is backprop, which violates the bet + the online-error ban).
Strongest FHRR-fit (`g⊛x` = the existing bind); natively delivers analogy. **Prior ~45% NULL**
(the highest-ceiling bet; the open risk is whether Hebbian path-integration alone factorizes,
which the BUILD, not this oracle, tests — the oracle only tests whether the factorization carries
paradigmatic structure at all).

---

## Disposition (frozen)

- **Any oracle PASS** → greenlight that route's grounding + precommit (NOT a build; the FHRR-port
  Stage-1 go/no-go + the phase fence remain the user's gates). Open the route's primary (Hard
  Rule 3) before any build.
- **All four NULL** → the bound is route-invariant at the single-projection/substrate-free level
  → the escape (if any) needs a genuine multi-LAYER hierarchy or a substrate change that this
  triage cannot cheaply probe → re-scope with the user. NOT more operator knobs.
- **Reinvent guards:** Oracle C must beat plain SPPMI (else = 123); Oracle B must use the
  transformed operator + LOCAL whitening (else = 124); Oracle D must not collapse to k=±1 (else
  = 124); Oracle E must (at build) grow g by Hebbian path-integration not backprop.
- **Phase fence:** all four are substrate-free and in-scope; a substrate BUILD on any PASS is a
  separate gate (the user's to lift).

## Build checklist

- [ ] Reuse via importlib: `exp61.{build_cooccurrence,build_sppmi,build_S,pick_k_by_density,
      row_center,grow_G,d_eff,load_simlex_pairs,random_matched_pairs,cooc_counts_for_pairs,
      corr_bootstrap_ci}`, `exp62.{_cos_real,_fcos,_boot_diff}`, `exp63.{build_directional_
      cooccurrence,select_pairs,raw_sppmi_svd_anchor,local_iteration_read}`.
- [ ] New code only: `build_transition_operator` (B), `build_SR_operator` (B),
      `build_eligibility_operator` (C), `build_order_vectors` (D), `nmf_slots` (E).
- [ ] Calibration anchor in EVERY run; INVALID if it misses +0.109/0.222.
- [ ] grow_G faithful read is the verdict; SVD spectral read is a labeled contaminated diagnostic.
- [ ] Planted-corpus smoke per operator before WikiText.
- [ ] Freeze γ=0.9, W=6, max_vocab=2000, K_SR=10, β∈{0.5,1.0}, k_slots∈{8,16,32}, the +0.04/+0.02
      thresholds, the SimLex sha — all before the run.
