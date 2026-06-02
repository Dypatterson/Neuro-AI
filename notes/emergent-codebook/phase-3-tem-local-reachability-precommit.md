# FROZEN PRE-COMMIT — Phase-3 TEM local-reachability oracle (Fork B)

*Frozen 2026-06-02 BEFORE running. Branch `experiment/tem-local-reachability-oracle`.
Harness: `experiments/76_tem_local_reachability_oracle.py` (reuses exp61/62/63/65/68 via
importlib). This is the LOCAL-WRITER counterpart of Oracle-E from
[phase-3-escape-route-oracle-triage-precommit.md](phase-3-escape-route-oracle-triage-precommit.md):86b96
(lines 99-114), selected as the decisive-and-faithful-soonest next move by the 2026-06-02
faithfulness pre-reg triage of forks A/B/C. The build-gate was lifted by the user 2026-06-02.*

## 0. Preamble — the experiment-preamble (CLAUDE.md), stated before the run

> **Active capability:** Codebook-growth (P3 structure) — specifically the **local-writer test of
> Oracle-E**. Oracle-E's GLOBAL NMF of the symmetrized transition operator is a confirmed
> *flashlight positive* (Report 125 §3: +0.124/+0.176/+0.188; the paradigmatic target is REAL),
> with its **local writer UNTESTED** (Report 125 §5:85 — "exactly what the SVD oracle was for the
> flat code"). This oracle tests whether a **fixed-random-slot, online-Hebbian, no-global-
> optimization** writer reaches that target.
> **Headline metric per [escape-route precommit:35-44 + :99-114] + 2026-06-02 synthesis:** the
> **within-paradigmatic-SET label-shuffle B-KILL** pair-specific residual (the Report-126 hubness
> discriminator), static slot-vector cosine read — NOT `grow_G`.
> **Required controls:** (1) raw-SPPMI-SVD calibration anchor (must reproduce +0.109 CI[0.082,0.137]
> / king-queen 0.222 — INVALID otherwise); (2) flat-SPPMI `grow_G` linear floor (g2 baseline);
> (3) the global-NMF Oracle-E ceiling (recover-fraction); (4) `rand_nonneg` random-codes control
> (g4 anti-inflation — accumulating real neighbours must beat random codes); (5) within-set
> label-shuffle (B-KILL); (6) **the mandatory spectral diagnosis** (see §3).
> **Last verified result:** Report 125 (Oracle-E global flashlight positive; local writer untested);
> Report 127/128 (the bound is linear-projection-vs-nonlinear-partition; competition reduces to
> global k-means; replay deflates to inverse-PMI).
> **Why this experiment now:** the 2026-06-02 A/B/C faithfulness triage ranked this Fork-B oracle
> #1 — the ONLY fork that is faithful AND un-run AND substrate-free/in-scope AND the one
> local-writer route route-invariance does not foreclose (it is NOT a single projection of one
> operator; it is a frozen-random projection + static multi-mode read). Fork A is answered (bank
> 128's inverse-PMI deflation); Fork C has no honest cheap version.

## 1. The contested load-bearing claim (what a faithful test must preserve)

Oracle-E's +0.19 came from NMF **jointly and globally optimising BOTH the slot basis W AND the
token→slot assignment H** via Lee-Seung multiplicative updates (exp65:134-145 `nmf_slots`). The
*single most important* faithfulness fact (2026-06-02 audit): **the global optimisation of the
slot ASSIGNMENT is the suspected load-bearing step.** A faithful local writer must therefore
**FREEZE the slot basis to fixed-random BEFORE seeing king/queen** and accumulate token→slot
occupancy by **local online Hebbian binding only** (no NMF iterations coupling all entries, no
SVD/eig, no backprop). If recovering E's positive *requires* the global slot optimisation, this
oracle NULLs — and that null is itself the decisive verdict (TEM does not escape the bound for any
non-backprop local writer; TEM-as-published is backprop, which the bet bans).

## 2. The writer (the only new mechanism) — fixed-random-slot online-Hebbian path-integration

`S_rand` = a **frozen** V×k nonnegative slot-signature matrix (ℓ2-rows), drawn once per seed
BEFORE any pairs are looked at — the problem-generic structural scaffold (CONTEXT.md §3: a fixed
problem-generic scaffold is a legal evolution-style prior; a scaffold *hand-shaped to the answer*
is the banned design-time homunculus — `S_rand` is random, so it is the former).

Token i's slot-occupancy is accumulated by **local Hebbian binding**: every time i co-occurs with
a neighbour j, i's occupancy gains j's frozen signature. Over the corpus this is
`W = OP @ S_rand` (nonneg), where `OP` is a **fixed batch-offline co-occurrence statistic** (same
class as SPPMI's `−log k`; the BANNED thing is global *optimisation*, not forming a count
statistic). The matmul is the batched form of the per-window accumulation (sanctioned
"batch-offline = sleep-phase" Hebbian, per STATUS live policy + exp68:64). **NO** eig, NMF, SVD,
or backprop touches `W`. Read = ℓ2-normalise rows, **static slot-vector cosine** (exp65:311 /
exp68.read_specificity) — **NOT** `grow_G`, which is the damped power iteration (exp61:348-377)
that converges to the dominant mode and would re-impose the very 121-125 bound this targets.

Writers (all static-read, all frozen-slot), swept k∈{8,16,32}, over OP∈{`M_trans` (E's operator),
`build_S` (the SPPMI 2nd-order floor operator)}:
- **`tem_frozen`** — `l2rows(OP⁺ @ S_rand)`. The headline Fork-B writer (immediate-neighbour).
- **`tem_frozen_sr`** — `l2rows(M_SR⁺ @ S_rand)`, `M_SR = Σγ^t Tᵗ` (exp65.build_SR_operator,
  γ=0.9, K=10): the **path-integration** best-honest-shot (γ-reweight inflates the slow/SR modes
  — Oracle-B's subdominant→dominant mechanism — projected onto frozen slots, read across modes by
  static cosine, which `grow_G` could not do because it collapses to mode-1).
- **`tem_frozen_comp`** — `tem_frozen` + a **local per-token** k-WTA (each token competes among
  ITS OWN k slots; cap=k//4). LOCAL competition, explicitly NOT the global k-means over tokens
  that Report 127 reduced to. Guards against falsely crediting/penalising local sparsification.

References (reuse): `linear_floor` = flat-SPPMI `grow_G` (exp65.faithful_read); `E_nmf` = global
NMF Oracle-E ceiling (exp65.nmf_slots); `rand_nonneg` = pure random V×k codes, no co-occurrence
(exp68.random_nonneg).

## 3. THE MANDATORY SPECTRAL DIAGNOSIS (converts an ambiguous null into a decisive one)

A frozen-random-slot null is, alone, **ambiguous**: it cannot separate "the substrate can't help a
local writer" from "we froze the global slot-assignment that did E's work." We therefore **never
run the writer alone** — every run carries a spectral diagnosis of *where paradigmatic structure
lives in the slot-occupancy*. For each writer's `W`: column-center, SVD `W_c = U diag(s) Vᵀ`; the
static-cosine Gram is `W Wᵀ = Σ_m s_m² u_m u_mᵀ`, so mode m's contribution to a pair (a,b)'s
similarity is `s_m² · u_m[a]·u_m[b]`. Define per-mode para-discrimination
`align[m] = mean_para(u_m[a]u_m[b]) − mean_rand(u_m[a]u_m[b])`, mass `w[m] = s_m²·max(align[m],0)`.
Report: **`para_centroid_rank`** = `Σ m·w[m] / Σ w[m]` normalised by k (low ⇒ para in DOMINANT/
reachable modes; high ⇒ SUBDOMINANT), and **`dominant_frac`** = mass in the top-⌈k/4⌉ modes. Run
the identical diagnosis on `E_nmf` slots and on the raw operator's SVD `U` for reference.

## 4. The gate (frozen) — PASS / NULL sub-cases

Per writer×operator×k, aggregated over seeds (default n=10), **PASS = ALL of**:
- **g1** static para-vs-random specificity bootstrap **CI-lo > +0.04** (clears the +0.021 flat /
  +0.0067 directional ceilings; matches escape-precommit:39).
- **g2** beats the matched flat-SPPMI `grow_G` linear floor by **≥ +0.02**.
- **g3** `corr(log cooc, para static-cos)` bootstrap **CI-hi < 0.15** (the collocational
  discriminator that killed every flat positive in 123/124; escape-precommit:41/107).
- **g4** beats `rand_nonneg` by **≥ +0.02** (anti-inflation: real-neighbour accumulation must beat
  random codes; nets out the low-k cosine inflation Report 125 §3 caveat-1 flagged). *Refinement
  caught in the 2026-06-02 planted smoke (before the headline run): `tem_frozen_comp` is k-WTA-sparse,
  and sparse codes inflate cosines vs DENSE random — so its g4 control is the **sparsity-matched**
  `local_kwta(rand_nonneg)`, not dense `rand_nonneg`. The dense control stays g4 for `tem_frozen`.*
- **B-KILL** within-set label-shuffle pair-specific residual CI-lo > 0 in **≥ 8/10 seeds**
  (the Report-126 hubness discriminator — the headline; para-set hubness alone fails this).

**NULL** = no writer×op×k clears all five. **INVALID** = anchor misses +0.109/0.222.

**The two distinguishable null sub-cases (pre-registered; read off the §3 diagnosis):**
- **(a) basis-change-not-spectrum** — `tem_*` NULL AND its `para_centroid_rank` is HIGH /
  `dominant_frac` LOW (para still subdominant in slot-occupancy) ⇒ the factorisation changed only
  the BASIS, not the spectrum's shape ⇒ **the flat-code bound GENERALISES to nonnegative-factorised
  single-layer codes; substrate-invariant for single-layer local writers** ⇒ escalate to a genuine
  MULTI-LAYER hierarchy or the Codebook-growth⇄Replay combination. (NOT more operator knobs.)
- **(b) frozen-assignment-was-load-bearing** — `tem_*` NULL BUT `E_nmf`'s slots put para in
  DOMINANT modes (low centroid_rank, high recover-fraction for E) while the frozen projection does
  not ⇒ **the global slot-ASSIGNMENT optimisation was the load-bearing step** ⇒ TEM inherits the
  bound for any non-backprop local writer (TEM's escape needs the banned backprop). **Flag this as
  an ambiguous-for-substrate null; do NOT over-bank it as "substrate can't help."**

## 5. Disposition (frozen)

- **PASS** (a frozen-slot local writer clears g1∧g2∧g3∧g4∧B-KILL≥8/10) ⇒ the **first local writer
  to break the locality trap** — escalate to the **rung-2 FHRR D=4096 port** (the exp74 pattern,
  n=3-5 multi-seed CI) BEFORE banking; only then does the TEM substrate BUILD become licensed.
  Magnitude reads as "local writer reaches the factorisation target," **NOT "beats SVD/NMF"**
  (Report 125 §3 caveat-1: 8-32-slot cosines are dimensionality-inflated; the §5:70 "E>SVD" claim
  is forbidden — different operators).
- **NULL** ⇒ record which sub-case (a)/(b) the §3 diagnosis selects; both are decision-grade. Bank
  the verdict; the next move (multi-layer / combination / re-scope) is the user's.
- **Phase/​build fence:** this rung-1 oracle is substrate-free and IN-SCOPE. The rung-2 FHRR port is
  the escalation; the TEM substrate BUILD itself remains the user's Abstraction-node gate (now
  lifted, but a PASS is its precondition, not its trigger).

## 6. Build checklist

- [ ] Reuse via importlib: `exp61.{load_corpus,build_cooccurrence,build_sppmi,build_S,
      pick_k_by_density,row_center,grow_G,d_eff,cooc_counts_for_pairs,corr_bootstrap_ci}`,
      `exp62.{_cos_real,_boot_diff}`, `exp63.{select_pairs,raw_sppmi_svd_anchor}`,
      `exp65.{build_distance_coocs,build_transition_operator,build_SR_operator,nmf_slots,
      faithful_read}`, `exp68.{read_specificity,derange_partners,random_nonneg}`.
- [ ] New code only: `tem_frozen_write`, `local_kwta`, `spectral_diagnosis`.
- [ ] `S_rand` drawn per-seed BEFORE pair selection is read (frozen-random precondition).
- [ ] Static slot-cosine read (exp68.read_specificity); **`grow_G` used ONLY for the linear floor.**
- [ ] Calibration anchor in EVERY run; INVALID if it misses +0.109/0.222.
- [ ] §3 spectral diagnosis emitted for `tem_*`, `E_nmf`, and the raw operator — EVERY run.
- [ ] Planted-corpus smoke before WikiText (apparatus must recover a planted king/queen lift).
- [ ] Freeze BEFORE the run: γ=0.9, W=6, max_vocab=2000, K_SR=10, k∈{8,16,32}, cap=k//4,
      η-grid/epochs/α for the floor (exp65 defaults), thresholds +0.04/+0.02/0.15, B-KILL≥8/10,
      n=10 seeds, SimLex≥5 sha.

## 7. Post-run deviations & corrections (2026-06-02, recorded after the run + verification)

- **Result:** [Report 129](../../reports/129_tem_local_reachability/report.md) — **NULL (b) frozen-assignment-was-load-bearing** (anchor valid; 3-lens adversarial verify, high-conf).
- **Verdict-logic bug (load-bearing, FIXED in `experiments/76:298`):** the auto `(b)`-detector gated on
  `any(E_nmf['PASS'])`, which requires `g3` (corr<0.15). g3 is **dead at n=40** (the +0.109 SVD anchor
  fails it too) → the `(b)` branch was dead code → falsely emitted `(a)`. Fixed to gate on E-beats-floor
  + the §4 spectral diagnosis (decoupled from g3); verdict re-derived offline from the unchanged n=10
  arrays → `(b)`. **2nd occurrence** of mechanically applying the dead g3 (1st: exp65 / Report 125 §4).
- **g3 deviation from this frozen spec:** §4 froze g3 as a hard PASS-gate; Report 125 §6 had already
  recommended removing it for the TEM precommit (missed). g3 is retained as COMPUTED but is now
  **diagnostic-only**; the within-set label-shuffle **B-KILL is the powered arbiter**. (No TEM writer
  passes under either reading — they fail g1/B-KILL regardless — so the NULL is robust to this.)
- **Minor checklist items not emitted (verdict-impact none, per all 3 lenses):** separate spectral
  diagnosis for `tem_frozen_comp` (reused `diag_tem`); the raw-operator SVD reference centroid
  (`diag_op`). The `(b)` call rests on E-vs-tem centroid + recover-fraction + B-KILL.
- **g4 sparsity-matched control** added during the planted smoke (pre-headline) — see §4 note.
