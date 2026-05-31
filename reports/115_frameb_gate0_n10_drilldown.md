# Report 115 — Frame B Gate-0 n=10 drill-down + variance baseline

**Status:** done-gate #4 discharged (on-repo endpoint level-DiD + gauge control E)
· DRILL-DOWN / VARIANCE-BASELINE, not a graduation claim
· Date: 2026-05-29

> **Active phase:** 3 (Growing Codebook / Frame B)
> **Headline metric per notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md:57-62 (TL;DR) + :64-101 (estimand):** within-seed exposure-recall slope-DiD Δβ(s)=β_real−β_shuffle (PROPOSED, pending sign-off)
> **Required controls per same spec :117-125 (§Control):** global token-stream shuffle (primary) + gauge arm E (FB→confound check only); proof of control validity at [2026-05-29-stream-shuffle-control-validity.md](../notes/notes/2026-05-29-stream-shuffle-control-validity.md)
> **Last verified result:** Gate-0 n=10 wikitext, DiD +0.019 CI [-0.121,+0.159], 6/10 positive, σ≈0.19, G0->weak (STATUS.md:18)
> **Why this experiment now:** discharge the standing Report-115 gap (done-gate #4 unmet; no numbered report exists for the Gate-0 n=10 result)

**THIS IS A DRILL-DOWN / VARIANCE-BASELINE REPORT, NOT A GRADUATION CLAIM.**
No phase graduates here. The PROPOSED slope-DiD headline (`Δβ(s)=β_real−β_shuffle`)
was **not** measured by this run — Gate 0 ran only the endpoint level-DiD. This
report discharges done-gate #4 ("the result is written up as a markdown report
under `reports/`") for the *Gate-0 n=10 endpoint* result that previously had no
numbered home; it does not measure, confirm, or sign off the slope headline.

> *Citation note (line numbers verified against the current file 2026-05-29).* The
> headline-design spec gained ~24 lines this session from two walk-back boxes
> (§Why-a-slope, §Variance-handling), so the spec is now 297 lines. The preamble
> citations above point at the **current** ground truth: §TL;DR Headline at :57-62,
> §"Headline metric — the estimand" at :64-101, and §Control at :117-125. The
> control spec is corroborated by the anchor note
> `2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md`
> (Part 4–5). The numbers below are unaffected.

---

## 1. Headline — on-repo endpoint level-DiD (discharges done-gate #4)

Gate 0 ran at n=10 wikitext seeds. The headline read is the per-seed
difference-in-differences `[(A)−(C)] − [(B)−(D)]` on stratum-pooled Recall@K
(consolidation's benefit on the real world minus on the global-stream-shuffle
world, each world self-consistent, landscape differenced out via the frozen-codebook
arms C/D).

- **Endpoint level-DiD:** **+0.019**, Student-t 95% CI **[-0.121, +0.159]**.
- **Per-seed positive fraction:** **6/10** positive.
- **Per-seed σ(DiD):** **≈0.19** (~5× the binomial floor → structural, not
  measurement noise).
- **Verdict:** **`G0->weak`** — DiD mean > 0 but the CI overlaps zero and the
  per-seed fraction is < 70%. Per the pre-committed branch table (anchor note
  Part 5), `G0->weak` is **underpowered, not a redesign trigger**; the route is
  escalate n, not reframe-or-abandon.

**Control on the SAME test set — gauge arm E (confirmed):** the retired
codebook-row-permutation control was run as Gate 0's arm E as a confound check.
- **4a (identity permutation):** **byte-identical** per-stratum outcomes — the
  pipeline threads the permuted codebook consistently (C1 equivariance holds).
- **4b (across perm seeds):** Δ-CI **contains 0** — consistent with the
  `E[Δ]=0` exchangeability prediction; no consistent offset, sign flips present.
- Arm E therefore **did not fire `G0->confound`**; the gauge finding is
  empirically confirmed as a side-effect of Gate 0.

**Done-gate #4 is discharged by THIS on-repo number** — the Gate-0 n=10 endpoint
level-DiD (+0.019, CI [-0.121,+0.159], 6/10, σ≈0.19) together with its on-the-same-
test-set gauge control (arm E: 4a byte-identical, Δ-CI contains 0) now has a
numbered report. This is the result that was missing a written-up home.

## 2. Variance drill-down (PROVENANCE-FLAGGED — does NOT discharge done-gate #4)

A Stage-1 variance decomposition ranks the within-seed sources of the endpoint
σ. The **qualitative ranking** is: within-seed **corpus-window draw dominates the
codebook/atom draw ~24×** (window ≫ atom ≫ binomial), i.e. a point split of
**≈96/4** between window-draw and atom-draw.

> **Provenance flag — this is a ranking, not a reproducible percentage.**
> The split is sourced from an **off-repo** artifact (`variance_decomp.json`,
> read from Google Drive by the colab notebook). It was **never git-committed**
> and has **not been regenerated** on-repo (the only JSON files on disk are
> unrelated; none is the gate0/variance/summary artifact). The point split rests
> on a **single 10×4 run at df=9**, so the exact 96/4 figure is fragile — the
> atom fraction could plausibly range widely; only the **24× ordinal margin**
> (window ≫ atom) survives the noise. Cite this **only** as a provenance-flagged
> *qualitative ranking*, **never** as a figure that discharges a reproducibility
> gate.

**done-gate #4 is NOT discharged by this decomposition.** The reproducibility
done-gate is discharged solely by the on-repo endpoint number in §1; the §2
decomposition is an off-repo, single-run, unreproduced ranking.

Note on what the ranking *does* establish: it **confirms the reframe's mechanism**
(per-seed codebook luck is a small additive intercept, so an OLS slope — invariant
to any per-seed additive shift — differences it out) while **refuting the reframe's
sales emphasis** (codebook variance is NOT the dominant problem; corpus-window
geometry is). The slope remains viable on *additivity* grounds (window-draw enters
as a per-seed additive level shift the slope removes), but σ(β)/σ(Δβ) was not
measured — the Q1 conditional is **open, not failed**.

## 3. Dispositive off-zero arithmetic (drill-down: why more windows won't clear the bar)

> **⚠️ CORRECTION 2026-05-30 (see [Report 117](117_frameb_leveldid_feasibility_consolidation_variance.md)).**
> Two numbers in this section are **wrong**, though the conclusion survives:
> (1) **`K=9 ⇒ σ=0.044` is wrong** — under the same model K=9 ⇒ σ=**0.0685**;
> reaching σ=0.044 needs **K≈26**. (2) **"WikiText val+test is only thousands of
> windows / saturates at K=9" is wrong** — the test pool is **~48,900** windows;
> window-averaging is **compute-limited, not pool-limited**. The conclusion
> ("√K window-averaging cannot clear the DiD off zero at n=10") **holds a
> fortiori**: the correct K=9 CI is **[−0.030, +0.068]** (wider, still spans
> zero). Crucially, window-averaging via `window_seed_override` is **K×
> re-consolidation, not cheap eval**, so "n≈27" is NOT a cheap path — total cells
> to clear zero is monotonic in K (min at K=1, n≈424 ≈ 42× the n=10 run). See
> Report 117 §1–§2.

The endpoint level-DiD point is +0.019 with per-seed σ≈0.19; at n=10 the
Student-t critical value is t(df=9)=2.262, so the CI half-width is
`t·σ/√n = 2.262·σ/√10` around +0.019. A √K window-average shrinks the
within-seed (corpus-window) component but **cannot touch the atom-draw floor**.

- **K=9** (σ=0.044 — the best the held-out pool supports; WikiText val+test is
  only thousands of windows, so window-averaging saturates here): CI
  **[-0.013, +0.051]** — **still spans zero**.
- **Atom floor σ=0.0224** (physically unreachable — window-averaging cannot
  reduce the atom component): CI **[+0.003, +0.035]** — barely clears zero.

**Conclusion of the arithmetic:** a √K window-average **cannot clear the
endpoint level-DiD off zero at n=10 by construction**. More windows is not the
lever. **n≈27 seeds** (not more windows) is the lever that clears zero at the
+0.019 effect. This explains why the `G0->weak` verdict routes to n-escalation,
not to a window-averaging fix of the *endpoint*.

## 4. What this report does NOT license

- **No graduation.** Neither Frame A (endpoint level-DiD) nor Frame B (slope-DiD)
  graduates. The endpoint DiD is `G0->weak` (underpowered); the slope-DiD headline
  was not measured at all.
- **No Phase 5′ reopen.** Phase 5′ remains paused; nothing here touches ΔE / bridge
  / M2 / matrix / headline / graduation work.
- **No threshold sign-off.** The PROPOSED slope-DiD thresholds (G1 σ-bound, G2 corr,
  G3 fanning, G4 β_shuffle) remain unsigned. The G2 0.4 corr and G1 0.05 σ-bound are
  flagged provisional/disputed in the smoke drafts; this report does not approve them.
- **No reproducibility claim on the variance split.** The ≈96/4 / ~24× decomposition
  is an off-repo, single-run, df=9 ranking; it is not a reproduced figure and does
  not discharge any reproducibility gate.

---

### Done-gate ledger for this report

| done-gate | status here |
|---|---|
| 1. Headline + CI | endpoint level-DiD +0.019, CI [-0.121,+0.159] (NOT the slope headline) |
| 2. Control on same test set | gauge arm E (4a byte-identical, 4b Δ-CI contains 0) |
| 3. Drill-downs explain anomalies | variance ranking (§2) + off-zero arithmetic (§3) |
| 4. Written up under reports/ | **this file** — discharged for the Gate-0 n=10 endpoint result |
| 5. Status note updated | STATUS.md:18 already records the Gate-0 n=10 result and `G0->weak` verdict |
