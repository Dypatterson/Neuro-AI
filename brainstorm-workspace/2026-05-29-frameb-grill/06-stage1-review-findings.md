# Stage-1 full review — verified findings (workflow wb57o8hnc)

**Date:** 2026-05-29 · 8 streams × verify→refute + (synthesis agent hit token
cap, returned null — synthesis below is the orchestrator's, built only from the
verified streams). Every stream's citations were re-opened by an independent
refute agent; fabrication_check = clean across all 8 (one MS-ratio slip and one
unsupported chi-sq CI flagged inside stats-sufficiency, noted below).

## HEADLINE: my Stage-1 read was an OVER-READ (verify + refute agree)

"window-draw dominant ⇒ slope reframe won't help ⇒ reconsider" **skips a
logical step.** The crux (now verified against code + spec):

- The slope β is OLS, **invariant to ANY per-seed additive constant** — codebook
  OR window-draw (FB design:66-68, verbatim).
- Stage-1's `run_variance_decomposition` decomposes the **endpoint lift**
  L = Recall_A − Recall_C (gate0:692-698, verbatim). That is **level/intercept
  variance**, and it CANNOT distinguish variance that loads on the intercept
  (slope removes it) from variance that loads on the slope/shape (slope keeps it).
- Therefore "window-draw dominates the endpoint variance" does **not** imply
  "the slope won't shrink σ." If window-draw acts as a per-seed additive level
  shift, the slope removes it and the reframe still works. Only the
  shape-changing fraction defeats the slope — and Stage-1 didn't measure that.

**What Stage-1 DID establish (robust):** the codebook/atom draw is small
(point est. 3.8%) and window-draw ≫ atom-draw ≫ binomial (ratio ~24×). This
**confirms the reframe's *mechanism*** (codebook luck is a small additive
intercept, exactly as design:30-32 says) while **refuting the reframe's
*sales emphasis*** (that codebook variance is the dominant problem — it isn't;
corpus-window geometry is).

## STATS SUFFICIENCY: one 10×4 run is NOT enough for the point estimate
- Internal arithmetic consistent: atom 0.000502 + window 0.012162 + binom
  0.000553 ≈ var_total 0.013179 ✓; var_fraction_atom = 0.000502/(0.000502+
  0.012715) = 3.8% ✓.
- BUT var_atom is `max(0, (ms_between−ms_within)/W)` on **9 df** — point estimate
  is fragile, near the floor; the **3.8% number could plausibly be anywhere from
  ~0 to high** (the refute agent flagged the verify agent's exact CI [0,97%] as
  an *unsupported* chi-sq calc, and corrected MS-ratio 1.04→1.16). What IS robust
  is the **qualitative ranking** window ≫ atom (24× margin survives the noise).
- Minimal added confidence (highest ROI first): **(1) Stage-1B** `--variance-report`
  on the existing Gate-0 summary → `corr_AC_BD` (zero compute); **(2)** rerun
  Stage-1 at **W=8** (same D, ~2.8× tighter SE); **(3)** a **2nd operating point**
  (D=2048 or 8192) to test op-point-specificity.

## WINDOW-AVERAGING vs SLOPE: different, COMPLEMENTARY variance (not substitutes)
Verified projected σ (hold atom=0.000502, shrink within by 1/K):
- K=1: sd_total 0.115 · K=2: 0.083 · K=4: 0.061 (~47%↓) · K=9: 0.044 (~62%↓)
- **Floor as K→∞: sd 0.0224** (the atom-draw component — window-averaging cannot
  touch it; only slope / more atom-seeds / CUPED can).
- Window-averaging attacks the **96% within-seed corpus-window** variance; the
  slope removes **per-seed additive level shifts**. Orthogonal.
- Window-averaging is **compatible with the existing Frame A endpoint DiD** (keeps
  the LEVEL headline), needs **no substrate code** (harness exists via
  `window_seed_override`, c3:858), and **passes anti-homunculus** (pure
  measurement change, like running more test windows).
- Refute caveat: √K assumes **independent** window draws; if the val+test pool is
  small vs n_test_windows×K, reduction saturates sub-√K — **check pool size
  before committing to K=9.**

⇒ The decision is NOT "window-averaging vs slope." It's "window-averaging FIRST
(cheap, dominant lever, keeps Frame A), THEN see if the residual ~0.022 atom
floor needs the slope/CUPED."

## CITATION INTEGRITY across drafts 01-05 + notes (the audit you asked for)
- **02 (defects):** verdict **holds** — floor math (ln(1000/16)=4.135, 0.041 not
  0.06-0.07; fix-b 0.0145/e-fold) all re-derived correct; item-9 axis + item-14a
  Γ language verbatim-accurate. **No fabrication.**
- **03 (precommit H1-H5):** **holds** — all FB line cites verbatim; H1/H2/H3
  problems real. No errors.
- **04/05:** **holds** — the "_build_substrate/_make_world don't exist" self-catch
  is verified true; corr>0.3 = gate0:652 verbatim (minor: borrowed from
  corr(A−C,B−D) context, not corr(intercept,slope) — reasonable transfer, noted).
- **notes (code-claim, q2, q3):** **holds** — item-11 byte-identity still holds;
  q2 active-drift-absent-from-PROJECT_PLAN holds; Report 115 absence confirmed;
  reliability warnings present; one non-load-bearing q3 line-range drift.
- **01 (smoke spec):** **partially-holds** — 3 editorial fixes (none factual):
  (a) item-7 "E[β_shuffle]>0 non-vacuity" is *my inference* from the branch
  table, not a verbatim spec claim — relabel as inference; (b) the 0.05 σ bound
  is a spec **PROPOSED placeholder** (design:256/263), not a derived value — my
  "mirror" framing overstated it; (c) state explicitly graduation uses seeds 0..9
  so pilot holdout 1000..1009 stays disjoint.

**Net:** no NEW fabrications in the drafts. The one substantive reasoning error
this session was the **Stage-1 over-read itself** (now corrected above).

## SYNTHESIS (orchestrator, from verified streams only)
1. **Stage-1 read = over-read.** It falsified the reframe's *motivation*
   (codebook ≠ the variance), not the reframe's *viability* (slope may still
   remove additive window-draw level shifts — untested).
2. **Q1 conditional was never actually tested.** Q1 approval was conditional on
   the smoke showing within-seed pairing shrinks σ. Stage-1 (endpoint
   decomposition) does **not** measure σ(β) or σ(Δβ) — so the conditional is
   still **open**, not failed. My "condition not met" framing was wrong.
3. **Recommended path: window-averaging FIRST, keep the slope open.** Pursue the
   √K window-averaging win on the existing Frame A endpoint DiD (cheap, dominant
   lever, keeps the LEVEL headline, no new mechanism) — and run the zero-cost
   Stage-1B `corr_AC_BD` + (if pursuing slope) the Stage-2 hook to settle
   whether the slope adds anything below the ~0.022 atom floor.
4. This is **option (e) "window-averaging-plus-keep-slope-open"** from the menu,
   NOT the clean "withdraw" I implied last turn.

All of the above is decision-support; nothing pinned to a binding doc. The
reframe call and the window-averaging-first call are the user's.
