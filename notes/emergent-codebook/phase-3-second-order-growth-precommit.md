# Phase-3 Second-Order Codebook-Growth Re-scope — Pre-commit

*Status: DRAFT pre-registration. Created 2026-05-31; **substantially revised
2026-06-01 after an 8-agent red-team grill** (a toy ran the core mechanism on the
real substrate and reshaped the gate — see Appendix B). The re-scope target opened
by [Report 121](../../reports/121_phase3_structure_gate_3b/report.md). Tests a NEW
growth mechanism against the 3b paradigmatic gate. Suggested harness:
`experiments/61_phase3_second_order_growth.py` (a fork of
`experiments/60_phase3_structure_gate_3b.py`). **RAN on WikiText → NULL → §10 oracles
(signal EXISTS) → pressure-test (basin-read rescue KILLED) → the GROWTH REDESIGN (§GR
below) is the current plan. Full chain:
[Report 122](../../reports/122_phase3_second_order_growth_oracles/report.md).***

> **Naming (set by the grill, G8).** `G` is the **PARADIGMATIC (second-order)
> codebook**. We deliberately avoid calling it "structural" — that word is bound
> to Phase-5 *structural retrieval* (`phase-5-unified-design.md:12,282`) and
> Phase-4's *structural binding map* `H` (`phase-4-heteroassociative-write-design.md:60`).
> A 3b-on-`G` PASS proves paradigmatic **geometry** forms; it does NOT prove
> Phase-5 structural **retrieval** works (§0 caveat).

---

## GR — GROWTH REDESIGN (post-pressure-test, 2026-06-01): B′ + energy-native anti-collapse

*Supersedes the forward-looking parts below. The WikiText run RAN → NULL; the §10 oracles
showed the paradigmatic signal EXISTS (SVD king/queen 0.222) and survives a single FHRR
port with contraction; an adversarial pressure-test KILLED the basin-read rescue (R1) —
the 3b null was no-signal-written, not a smush — and redirected here. Full chain:
[Report 122](../../reports/122_phase3_second_order_growth_oracles/report.md).*

**THE MECHANISM (converged across 3 research workflows + the pressure-test):** compose the
**row-centered SPPMI pull-similar** (B′) with a **local energy-native anti-collapse
"keep-apart"** — the substrate's `H_anti = −α·log(d_eff)` repulsion
(`torch_fhrr.py:147-182`, already built, currently inert at `α_anti=0`):

```
G_pull = normalize(α·normalize(S'@G) + (1−α)·G)               # B′ pull-similar (S' = rownorm(SPPMI@SPPMIᵀ) − rowmean)
G      = normalize(G_pull + η_sep · repulsion_force(G_pull))   # H_anti keep-apart (zeros if α_anti=0 → byte-identical)
```

*Why this is the fix (diagnosis, Report 122 §3):* the smush is **operator-spectrum
collapse** — `S'` is near-rank-1 even after row-centering (top eigenvalue 0.150 vs next
~0.005), so `S'@G` is one step of power-iteration onto the leading shared mode (random
pairs → 0.58). `H_anti` descends `−log(d_eff)` on the **centered** Gram
(`torch_fhrr.py:141`), pushing each atom off the population-mean direction — removing
exactly the common-mode the bundle injects, while leaving the small idiosyncratic
king/queen block (which sits *below* the dominant mode) intact. The brain's
lateral-inhibition / homeostatic-decorrelation solution (the *local* analog of the global
SVD that provably doesn't contract); it also **retires the threshold-`_apply_repulsion`
homunculus**.

- **ANTI-HOMUNCULUS (PASS):** `α_anti` is fixed at construction, never read back from
  observed d_eff (`torch_fhrr.py:48-55`) — the gradient *is* the actuator.
- **LOCAL-NOT-GLOBAL (PASS):** `repulsion_force` depends only on pairwise overlaps via the
  d_eff identity `(tr G)²/tr(G²)` — NOT an eigensolve/PCA. REJECTED: full
  Földiák/Pehlevan-Chklovskii whitening of G (fixed point = PCA/ZCA = the forbidden global
  word2vec shortcut).

**THE GATE (gauge-free — the stream-shuffle gauge is INVALID for 2nd-order operators).**
The WikiText run proved the SPPMI gauge LEAKS (`corr(S_real,S_shuffle)=0.79`:
`SPPMI@SPPMIᵀ` is frequency-dominated, frequency survives the shuffle). Headline becomes
**para-vs-random specificity** (the oracle's gauge-free test): paradigmatic
(non-co-occurring SimLex) drift > matched-random drift, hierarchical-bootstrap CI > 0; AND
`corr(log cooc, drift) CI-hi < 0.15` (decorrelation) + the PMI/collocational floor; AND
the collapse floor `d_eff_end/init ≥ 0.5 ∧ off-max < 0.99`. *(Retiring the stream-shuffle
gauge is justified by the measured leak — a control-validity correction, not a
drop-to-pass; ratify into the 3b spec.)*

**ARMS (pre-registered):** A0 = B′ alone (`α_anti=0`); **A1 = B′ + H_anti** (`α_anti>0`,
`η_sep` swept) — primary; CTRL-collapse = uncentered B; CTRL-global = the owned
CueDecorrelator full-ZCA on G (the FORBIDDEN-line control — measured to locate the
rank-1-vs-global boundary, NOT adopted). Per-arm fresh-random G; k pinned by density.

*η_sep SCALE (smoke-measured, `experiments/61` H_anti arm built 2026-06-01):* the d_eff
gradient is ~1e-4 per coordinate for unit-modulus phasors, so **`η_sep` lives at ~1e3–5e4,
not ~0.1** (`α_anti=0` OR `η_sep=0` → byte-identical to B′, verified: planted
`drift_king_queen=0.12719…` unchanged). The smoke shows a clear trade-off on a collapsing
schedule (B′ alone → d_eff_ratio 0.05): too-high `η_sep` decorrelates king/queen *with*
everything (specificity lost, e.g. 1e3 → kq +0.19 ≈ random +0.19); a specificity-preserving
anti-collapse regime exists (1e4 → d_eff_ratio 0.05→0.26, kq +0.40 > random −0.06,
distractor −0.28). **Sweep `α_anti × η_sep` on WikiText.**

**PRE-REGISTERED NULL DISPOSITION:** if no (`α_anti`, `η_sep`, k, α) point clears the
para-vs-random gate AND the collapse floor for B′+H_anti, then *local common-mode removal
does not write paradigmatic structure on real text* → the second-order `S'@G` growth is
the wrong shape → escalate to a true predictive/successor-context growth target ("R3"),
NOT more knobs. (The signal EXISTS in the SPPMI statistics per the §10 SVD oracle, so a
null here is about the local GROWTH dynamic, not the corpus.)

**β-decoupling note (banked, pressure-test):** high static cosine ≠ merged Hopfield basin
— at β=30 a smushed codebook (mean cos ≤ ~0.65, d_eff ≥ ~6.6) still retrieves selectively;
collapse only past ~0.67 (recoverable at β≥100). So the collapse floor's `d_eff ≥ 0.5`
guard is *conservative* relative to actual basin integrity — a future dynamical-readout
drill-down (NC1-class, on the floor's settling) could tighten it, but is NOT this gate.

---

## Experiment preamble (mandatory, per `CLAUDE.md` §Experiment preamble)

- **Active phase:** Phase 3 — the FOUNDATION (codebook growth). The growth-mechanism
  redesign the Report-121 null re-scoped to; **NOT** a Phase-5 build, **NOT** a
  larger re-run of the 121 null.
- **Headline metric — REVISED post-grill (G1), pending ratification into the spec.**
  The 3b design doc's headline (`phase-3-structure-gate-3b-design.md` §"Headline
  metric") is raw paradigmatic real−shuffle drift CI > 0. The grill proved that read
  **passes on global collapse** (a contracted blob lifts every pair), so this
  experiment's headline is the **demeaned matched specificity** read (§3.1).
  *This is a drift from the design spec and must be ratified into it (with user
  agreement) before the run, or the experiment is by definition a drill-down — see
  the §"Open: spec ratification" note.*
- **Required controls per `phase-3-structure-gate-3b-design.md` §"Required controls":**
  gauge-safe stream-shuffle, paradigmatic/collocational split, random-pair
  specificity, init-as-baseline DiD, PMI-collocation floor, d_eff/max-sim collapse
  guard — **plus the revised gate + guards** below.
- **Last verified result:** [Report 121](../../reports/121_phase3_structure_gate_3b/report.md)
  — the incumbent first-order learner is paradigmatic-NULL (WikiText −0.0084,
  CI [−0.013,−0.004]; `corr(log cooc, drift) = +0.41`/`+0.82` → collocation-driven).
- **Why now:** tests whether a **second-order context-profile** growth dynamic clears
  the paradigmatic gate the first-order centroid failed.
- **Pre-registered null disposition (committed BEFORE the run):** a NULL across all
  variants = the **flat-codebook re-scope is FALSIFIED at this scale** → escalate to
  a **latent/predictor layer** (Option B; PCN/SFA put paradigmatic structure in a
  latent layer above the code). **NOT** widen the readout, **NOT** scale vocab/epochs.
  *Crucially (grill G1): "no `(k, α)` point gives specificity CI > 0 AND survives the
  collapse floor simultaneously" is itself a pre-registered NULL of this disposition
  — signal exists in `S` but the FHRR growth cannot read it without collapse.*

---

## 0. Architecture: a SEPARATE paradigmatic codebook G

**The second-order operator grows a SEPARATE paradigmatic codebook `G`, read
directly by the 3b harness — it does NOT overwrite the validated value/content
codebook.** Mixing `G` into recall is a later, separately-gated step, not part of
this experiment.

*Why (endorsed by the 10-agent menu vetting — Appendix A):*
- **Empirical:** Report 119 shaped a structure objective *directly on the value
  codebook* → `d_eff` collapsed 341→~165, held-out margin ≤ 0. A separate `G`
  forecloses that overwrite-collapse class permanently.
- **Floor protection becomes architectural, not promissory.** `experiments/60` is
  already codebook-agnostic (builds its own init table), so `experiments/61` grows a
  fresh `G` and runs the unchanged 3b gate on `cos(Gᵢ, Gⱼ)` — the 055–058 floor is
  untouched *by construction* (cf. §9).
- **It isolates the question.** 3b-on-`G` answers *"does paradigmatic geometry
  form?"* cleanly, decoupled from *"can recall use it?"* (the harder, deferred
  `bind(G,X)+H` question).
- **End-state (TEM-shaped — but the split is hygiene, NOT the mechanism):**
  `Xᵢ` = stable content atom; `Gᵢ` = paradigmatic atom; `Pᵢ = bind(Gᵢ, Xᵢ)`;
  `H` = the validated recall floor. The structure is carried by the **operator**
  (PMI's marginal division), *never* by the split — a separate `G` fed first-order
  co-occurrence reproduces the 121 null exactly ("empty-carrier illusion").

**LOAD-BEARING CAVEAT:** a 3b-on-`G` PASS proves paradigmatic geometry *forms*, NOT
that it is *usable* by the floor. The hard, unbuilt step is the later `bind(G,X)+H`
mixing, and Report 119 warns shaping anything near the recall codebook can collapse
`d_eff`. Do not declare victory at the cheap 3b-on-`G` PASS — §10 adds a mixing-
feasibility pre-check before any mixing build is licensed.

**PHASE-ORDER ACKNOWLEDGMENT (grill G7; required by `CONTEXT.md` §3).**
1. *Experiment 61 is Phase-3-legal:* it grows `G` and reads the king/queen test on
   `cos(Gᵢ, Gⱼ)` — the Phase-3 emergent-structure criterion. It adds **no** runtime
   router between `G` and `X`; recall never touches `G`.
2. *ACKNOWLEDGED PHASE SKIP:* the deferred end-state `Pᵢ = bind(Gᵢ, Xᵢ)` mixed into
   `H` recall **is Phase-5 architecture** (bind-vs-bundle / structural composition
   read by recall, `experimental-progression.md` §Phase-5; `phase-5-unified-design.md:12-43`).
   It is **NOT authorized by this precommit** and must not be built until (a)
   3b-on-`G` passes AND (b) it is opened as its own pre-committed experiment.
3. *Binding shape-constraint on that future step:* the mixing MUST be a local
   energy/binding/settling dynamic, **never** an `if-metric-then` router that selects
   `G`-vs-`X` at runtime (`PROJECT_PLAN.md` anti-homunculus rule).

---

## 1. The mechanism being replaced (the 121 incumbent)

`src/energy_memory/phase2/codebook_learner.py` — `centroid = C @ codebook`, `C =
rownorm(inv_freq-weighted FIRST-ORDER co-occurrence)` (`build_cooccurrence` ≈ :52–77;
`train` ≈ :97–131; `inv_freq = 1/√count` ≈ :42–50). Verdict: **collocational PASS,
paradigmatic NULL** (Report 121). It pulls each token toward tokens it **co-occurs
with** (syntagmatic), not toward tokens with **similar neighborhoods**
(paradigmatic). The incumbent already applies a *weak* frequency discount (`1/√count`),
so the contrast is "√-inv-freq first-order" vs "SPPMI second-order," not
"no-correction vs correction."

## 2. Variants (offline, batch — the sleep/wake split; no runtime error-driven write)

All variants grow a **separate paradigmatic codebook `G`** (fresh random init; value
codebook untouched — §0) via `G ← normalize(α·centroid + (1−α)·G)`; only the operator
producing `centroid` differs. The anti-collapse repulsion (≈ :137–172) and
`substrate.normalize` as the final step are kept fixed.

**Build counts once:** windowed co-occurrence `#(i,j)`, marginals `#(i)`, total `|D|`,
then `SPPMI[i,j] = max( log(#(i,j)·|D| / (#(i)·#(j))) − log k, 0 )` (Levy-Goldberg
Eq 12). **`k` is pinned by SPPMI *density*, not set to 5** (grill G1: `k=5` zeroed a
small-vocab matrix to 26/1681 nnz → dead). Pre-register a density target (nnz
fraction ≈ 0.3–0.5 of `V²`) and choose `k` to hit it; sweep `k` ±1 order around it.

- **Variant B′ — ROW-CENTERED SPPMI second-order (PRIMARY; the grill's rescue, G2).**
  `S = rownorm(SPPMI @ SPPMIᵀ)`; **`S' = S − rowmean(S)`** (subtract each row's mean —
  removes the near-uniform leading eigenvector that causes global contraction);
  `centroid = S' @ G`. *Why primary:* the raw iteration `S@G` is power-iteration toward
  `S`'s dominant (≈ uniform) eigenvector → collapse; row-centering makes the
  **paradigmatic block-structure the actual fixed point** (G2 measured: contrast 0.90
  at mean-off-diag ≈ 0.000, α-schedule-robust). Row-centering is a fixed offline
  transform on `S` (same AH class as the `−log k` shift), so it stays anti-homunculus
  clean. **Run first.**
- **Variant B — UNCENTERED SPPMI (CONTROL; demonstrates the collapse B′ fixes).**
  `centroid = S @ G`. Expected (grill G1/G2) to either collapse (dense `k`,
  `d_eff`→2) or be inert (sparse `k`, drift ≈ 0) — failing the §3.2 collapse floor.
  Its job is to show row-centering earns its keep, not to pass.
- **Variant A — naive `P@Pᵀ` (FIRST-ORDER CONTROL).** `P = rownorm(C)`;
  `S = rownorm(P @ Pᵀ)`; `centroid = S' @ G` (row-centered). **Attribution claim
  DEMOTED (grill G1):** on a frequency-controlled toy A ≈ B (SPEC 0.89 vs 0.97), so
  A only discriminates on **real, frequency-imbalanced** text. A-vs-B separation must
  be *demonstrated on WikiText*, not assumed; an A-pass does NOT invalidate B′.
- **Variant C — sparse top-k S (ROBUSTNESS; only if B′ passes).** Top-k per row with a
  **fixed, pre-registered, uniform k**. AH-clean *only* as fixed sparsification — any
  data-gated threshold or collapse-guard feedback into the update is a violation.

## 3. Pass gate (REVISED post-grill — decisive and unfakeable)

A variant PASSES only if **ALL FOUR** hold (G1, G2, G4):

1. **HEADLINE — demeaned matched specificity, real − shuffle, hierarchical-bootstrap
   CI > 0.** Per-pair paradigmatic drift **minus the per-token global-mean drift**
   (demeaned, so a global contraction cancels), **minus the matched random-pair arm**
   (random pairs drawn at the *same* low-cooc + frequency regime), real minus
   stream-shuffle. *Raw paradigmatic drift is NON-DIAGNOSTIC* (grill G1: B at `k=2`
   gave raw real−shuffle = +0.76 driven entirely by `d_eff`→2) — report it only
   alongside `d_eff`/off-diag-max so a collapse-driven raw lift is visibly disqualified.
2. **COLLAPSE FLOOR — HARD gate, co-equal with the CI (NOT a drill-down).**
   `d_eff_end / d_eff_init ≥ 0.5` AND `max-off-diag cosine < 0.99` AND global
   off-diag mean-cosine drift below a frozen ceiling (G2 promoted this from a report
   to a gate). Failing any → NOT a pass, regardless of the CI.
3. **DECORRELATION — wired + numbered (grill G4: this was decorative).**
   `corr(log cooc, paradigmatic-subset drift)` bootstrap 95% CI **upper bound <
   +0.15** (> 60 % drop from the +0.41 incumbent). Add to the harness conjunction:
   `headline_pass = specificity_ci_lo > 0 AND collapse_floor_ok AND corr_ci_hi < 0.15`
   (the forked line from `experiments/60:278`).
4. **GAUGE VALIDITY — Guard 2, now numbered (grill G4: the control LEAKS at W=6/k=5).**
   `corr(S_real_offdiag, S_shuffle_offdiag) < 0.40`. Above it the stream-shuffle
   control is leaking marginal structure → the run is **INVALID (not a null)** →
   re-run at a `(W, k)` that separates. *(Simulated 121 config W=6/k=5 → 0.79: FAILS.
   Choose `(W, k)` empirically to clear 0.40 before the headline run.)*

**Pre-registered NULL (grill G1):** if **no** `(k, α)` grid point satisfies
conditions 1 AND 2 *simultaneously*, that is a NULL → §10 latent-layer fork (the
signal is in `S` but the FHRR growth cannot read it without collapse). Run a `k × α`
grid, not a single point.

## 4. Guards (folded into the gate above + per-arm S)

- **Matched random-paradigmatic arm** — now **part of the headline** (§3.1), not a
  side guard: paradigmatic-similar must beat paradigmatic-random *under identical
  cooc + frequency regime*. A global contraction lifts both equally → cannot fake it.
- **Per-arm S construction** — build `SPPMI`/`S`/`S'` **separately** for the real and
  stream-shuffle arms (a shared `S` leaks). **Corrected rationale (grill G4):** the
  old "shuffle SPPMI ≈ 0" justification is *empirically false* (shuffle SPPMI density
  ≈ real at W=6/k=5). The real guard is the §3.4 numeric bound
  `corr(S_real_offdiag, S_shuffle_offdiag) < 0.40`.

## 5. A-priori power fix (BEFORE any run — pre-registered, frozen, hashed)

The 121 paradigmatic subset is **n = 11** — underpowered to license the "re-scope
falsified" disposition. Fix power a priori (grill G5 verified feasibility on-disk):

- **Source + threshold (pinned):** **SimLex-999** (Hill et al. 2015), genuine
  *similarity/substitutability*, **filtered to SimLex ≥ 5.0** (genuine-similarity, not
  any-sim — the any-sim "88 survivors" include labeled antonyms like old/new that
  would pollute the matched-random arm). **Realized powered n at this checkout = 40**
  (report band counts 40/26/21 at ≥5/≥6/≥7). **Do NOT use relatedness sets
  (WordSim-353) as primary** — relatedness ≈ collocation; allowed only as a
  separately-reported secondary arm.
- **Deterministic filter:** both tokens in-vocab under `max_vocab`; within-window
  cooc ≤ `paradigmatic_max_cooc`; exclude specials. **Freeze + hash the list into the
  report.**
- **Inference — hierarchical bootstrap (grill G5).** Replace the flat pair-bootstrap
  (`experiments/60:230-233` collapses seeds first → anticonservative, CI ~27 % too
  narrow) with a **seed×pair cluster bootstrap** (resample seeds *and* pairs); make
  the hierarchical CI the gating one; report both.
- **Minimum-survivor abort rule (frozen fallback order):** if realized SimLex ≥ 5
  survivors < 30 at run time, ABORT and fall back **in this frozen order**: drop to
  SimLex ≥ 4.0 → add WordSim-353/MEN as a *separately-reported* secondary → raise
  `max_vocab` to 4000. Each frozen before observing outcomes; never loop on outcomes.
- **BRIGHT LINE:** drawing more pairs from the pre-specified distribution *before*
  outcomes = legitimate; loosening source/threshold/cutoff *after* a result = the
  forbidden "widen the readout to rescue a null."

## 6. Reused from the 3b harness (mechanism-agnostic — free)

gauge-safe stream-shuffle; paradigmatic/collocational split; init-as-baseline DiD;
held-out PMI-collocation floor; bootstrap CIs. **Re-instrumented (not free):** the
demeaned-specificity headline (§3.1), the collapse-floor gate (§3.2), the wired corr
gate (§3.3), the gauge-validity gate (§3.4), and the hierarchical bootstrap (§5).

## 7. Anti-homunculus check

`S`, `S'` (row-centered), `SPPMI`, and the top-k sparsification are all **fixed
offline batch statistics** (same AH class as SPPMI's `−log k` shift and `max(·,0)`).
The update `G ← normalize(α·(S'@G) + (1−α)·G)` is a **local geometric write** — no
runtime metric-reader, no `if-metric-then` branch, no supervisor selecting a variant
(that is a precommitted offline choice). The collapse/corr/gauge gates only
**halt/label**; they never feed back into the update. Variant C's top-k stays a fixed
uniform sparsification. **PASS.**

## 8. Grounding dependency

Levy-Goldberg 2014 (SPPMI) + Mikolov 2013 (SGNS / negative sampling); card at
[`docs/ground-truth/source_cards/2026-05-31-levy-goldberg-sppmi-card.md`](../../docs/ground-truth/source_cards/2026-05-31-levy-goldberg-sppmi-card.md).
**PENDING (Hard Rule 3):** upgrade `source_manifest.jsonl` (primary-opened +
checksummed PDFs). **FHRR port — now TOY-TESTED, not merely "untested" (grill G1/G2):**
the paradigmatic signal is genuinely present and gauge-clean in `S`
(`S[king,queen]=0.70`, `S[king,distractor]=0.000`), but the *uncentered* `S@G` growth
cannot read it without collapse; **row-centering `S` (Variant B′) is the rescue** that
makes the structure the fixed point. The headline run tests whether this holds on real
WikiText at scale — the §3 gate is the guard against a cosmetic port.

## 9. Floor protection (the no-regression boundary)

Floor protection here is **architectural** (§0): the operator grows a separate
paradigmatic codebook `G`; the validated value/content codebook is never written.
`experiments/61` does not touch the consolidation-write FLOOR
(`phase34/online_codebook.py` ≈ :419–481; `phase4/hetero_write.py` +
`phase4/decorrelator.py`; Reports 055–058). Keep `substrate.normalize` as the final
step of every update (the floor needs `G`'s rows to be valid unit-modulus FHRR atoms
*only if* `G` is ever mixed in). **If a passing variant is ever integrated into the
production codebook path, RE-RUN the 055/056/058 floor + wiring checks** (binding).

## 10. Decision after the run

- **B′ passes (collapse floor + specificity + decorrelation, on real text)** → first
  paradigmatic signal in the project. **This proves paradigmatic geometry FORMS in
  `G` — NOT yet that recall can use it** (§0 caveat). Next gates, in order:
  1. **Mixing-feasibility pre-check (cheap, ~1h offline; grill G3)** — for the passing
     `G`, compute `d_eff` and max-sim of `P = bind(G, X)` and the cos-drift retention
     (corr of pre-bind vs post-bind pair cosines). **Bright line:** if `d_eff(P)` drops
     into the Report-119 danger zone (require `d_eff(P) ≥ ~0.7·d_eff(random codebook)`
     AND paradigmatic-cos retention ≥ 50 %), the mixing route is presumed a
     third-memory and is **NOT licensed** — do not sink the V×D table + bind-surface
     build until the anisotropy-vs-separability tension is solved.
  2. wire the NC1 codebook-group `d_eff`/ETF drill-down to *localize* the clustering.
  3. only then open `bind(G,X)+H` mixing as its own pre-committed Phase-5 experiment
     (§0 phase-order acknowledgment) with a full 055/056/058 floor re-run.
- **All variants null** → run **TWO oracles (grill G6 — the NMF oracle was circular
  with B and is REMOVED):**
  1. **Substrate-free signal oracle = truncated-SVD of SPPMI**, read 3b directly on the
     real-valued SPPMI rows (L&G's *own* factorization; zero FHRR, zero nonnegativity).
     This tests whether the paradigmatic signal EXISTS in the corpus's SPPMI statistics
     at this scale.
  2. **FHRR-port oracle** = port the *same* real SPPMI rows once into the substrate
     (`S'@G` + `substrate.normalize`, single-shot) and read 3b. Isolates FHRR-port damage.
  - **SVD nulls** → signal genuinely insufficient at this scale → **Option B**
    (latent/predictor layer above the code) is licensed.
  - **SVD passes, FHRR-port oracle nulls** → the signal exists but the FHRR `S'@G`
    re-bundle/normalize destroyed it → fix the *port* (e.g. diagonal-zeroing, spectral
    control), not the source.
  - **SVD passes, FHRR-port passes, B′ nulls** → the *growth dynamics* are the problem
    → iterate the operator (the γ-discounted *directional* successor window, vetted
    solution 3, is the one genuinely-additive DOF — offline, PMI-corrected).
  - **In no case** scale vocab/epochs or widen the readout to rescue a null.

---

## Open: spec ratification (do BEFORE the run, needs user agreement)

The revised gate (§3) **drifts from** `phase-3-structure-gate-3b-design.md`'s headline
(raw drift) — the grill proved raw drift passes on collapse. Per `CLAUDE.md`, a
spec/experiment contradiction is a *finding to surface*, not paper over. **Action:**
ratify the demeaned-specificity headline + the collapse/decorrelation/gauge gates +
the SimLex-999 source into `phase-3-structure-gate-3b-design.md` §Headline/§Required-
controls (with user agreement), OR explicitly label experiment 61 a drill-down. Until
ratified, this precommit's gate is the binding one and the divergence is flagged here.

## Implementation checklist (when the build is greenlit)

- [ ] **Ratify the revised gate into the 3b spec** (above) — user agreement.
- [ ] Land the Levy-Goldberg/Mikolov card + manifest upgrade (§8).
- [ ] Build + freeze + hash the SimLex ≥ 5.0 pair list (n≈40); write the filter into
      the spec (§5).
- [ ] Fork `experiments/60` → `experiments/61`; grow a fresh paradigmatic `G`; read 3b
      on `cos(Gᵢ,Gⱼ)`. Implement `build_sppmi` (k pinned by density), `S' = S − rowmean`
      (B′ primary), uncentered `S` (B control), `P@Pᵀ` (A control), top-k (C) (§2).
- [ ] Re-instrument the gate: demeaned-specificity headline, collapse floor
      (`d_eff_end/d_eff_init ≥ 0.5`, off-diag-max `< 0.99`, mean-off ceiling), wired
      corr gate (`corr_ci_hi < 0.15`), gauge-validity gate
      (`corr(S_real,S_shuffle) < 0.40`), hierarchical seed×pair bootstrap (§3–§5).
- [ ] Run B′ + B + A on a `k × α` grid (repo_sample local go/no-go, then WikiText);
      report all four gate conditions + the raw-drift/`d_eff`/off-diag panel.
- [ ] If B′ passes: run the mixing-feasibility pre-check (§10.1) BEFORE any mixing build.
- [ ] If null: run the SVD-of-SPPMI + FHRR-port oracles (§10) to route the fork; do
      not scale.

---

## Appendix A — Vetted alternatives (9-solution menu, 2026-05-31)

10-agent vetting (corpus / anti-reinvention / anti-homunculus / FHRR / floor).
**Convergent finding: every alternative collapses to first-order/collocational (the
121 null) unless it carries PMI's marginal-division — the second-order-ness lives in
the OPERATOR, not the carrier.** Nothing in the menu dominates the SPPMI operator.

| # | Solution | Verdict | Disposition |
|---|---|---|---|
| 1 | Random-indexed FHRR context signatures | reinvention-of-killed | `cos(Gᵢ,Gⱼ) ≈ raw co-occurrence cosine` = variant A in a random basis. **Rejected.** |
| 2 | BCPNN / PPMI Bayesian-Hebbian | reinvention-of-killed | Single PMI-hop = variant A; BCPNN weight is pairwise/first-order. **Folded into B's inner matrix.** |
| 3 | Successor-representation / predictive map | wrong-shape | Raw SR cosine = frequency-dominated collocational; online TD = banned. γ-discounted *directional* window = the one B-conditional DOF (§10). |
| 4 | TEM structural/entity split (G+X) | reinvention + architectural-improvement | `bind(G,X)` already exists; faithful-TEM engine banned + Dorrell-dead. Split is hygiene → §0. |
| 5 | SQHN sparse quantized attractors | wrong-shape | VQ = first-order input-similarity; `if activity<threshold then allocate` = homunculus violation. **Rejected.** |
| 6 | Holographic ordered-context / BEAGLE | reinvention-of-killed | Already built (Report 009); order-binding = position-typed co-occurrence = syntagmatic. **Rejected.** |
| 7 | Predictive-coding over G (not C) | reinvention-of-killed | "predict G_context from G_center" = first-order by definition; = killed `error_driven_learner`. Separate-G adopted → §0. |
| 8 | Sense-splitting | reinvention-of-killed | Phase-5-fenced; "else birth a sense" = metric-gated thermostat; wrong quantifier. **Rejected.** |
| 9 | Nonnegative / part-based factorization (NMF) | superseded | **REMOVED as the null-oracle (grill G6: circular with B).** Replaced by SVD-of-SPPMI + FHRR-port oracles (§10). |
| 10 | Separate-G architectural discipline | architectural-improvement | **Adopted → §0.** |

## Appendix B — Red-team grill findings (2026-06-01, 8 agents)

The grill ran the core mechanism on the real substrate (G1) and adversarially
attacked each load-bearing bet. **Net: the mechanism is salvageable but the original
precommit would have banked a spurious collapse-driven PASS or nulled the only working
regime; the fixes are folded above.**

| # | Bet attacked | Verdict | Fix folded in |
|---|---|---|---|
| G1 | `S@G` on FHRR recovers paradigmatic structure (TOY, ran code) | needs-revision (high) | Signal is in `S` but uncentered growth collapses (dense `k`) or is dead (sparse `k`). → demeaned-specificity headline (§3.1), collapse floor (§3.2), `k` by density (§2), A≈B attribution demoted (§2). |
| G2 | `S@G` converges to paradigmatic clustering (theory) | needs-revision (med) | Fixed point = `S`'s near-uniform leading eigenvector (collapse); the "pass" was an α-transient. → **row-centered `S'` = Variant B′ primary** (§2); mean-off-diag promoted to a gate (§3.2). |
| G3 | a 3b-on-G PASS is actionable | needs-revision (med) | Risk of a cheap PASS that mixing destroys. → mixing-feasibility pre-check (§10.1). |
| G4 | the pass gate is decisive & unfakeable | needs-revision (high) | corr gate was decorative + unwired; gauge-safe control LEAKS at W=6/k=5. → wired corr gate `< 0.15` (§3.3), numeric gauge-validity gate `< 0.40` (§3.4), Guard-2 rationale corrected (§4). |
| G5 | SimLex filter yields ≥30 powered pairs | needs-revision (low) | Feasible (n=40 at SimLex ≥ 5), but flat bootstrap anticonservative. → pin SimLex ≥ 5.0, hierarchical bootstrap, abort rule (§5). |
| G6 | the NMF oracle disambiguates rule-vs-signal | **cracked** (med) | Circular with B (signal from `M`'s PMI, not the nonneg). → **removed**; replaced by SVD-of-SPPMI + FHRR-port oracles (§10). |
| G7 | no unacknowledged phase-skip | needs-revision (low) | `bind(G,X)+H` is Phase-5. → phase-order acknowledgment block (§0). |
| G8 | consistent with binding docs & language | needs-revision (med) | "structural" collides with Phase-5 term. → renamed to **paradigmatic codebook G**; revised gate flagged for spec ratification (Open note). |
