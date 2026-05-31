# Report 116 — Frame B branch decision: `corr_AC_BD` drops the slope-DiD

**Status:** BRANCH-DECISION / DRILL-DOWN — resolves the window-vs-slope fork
· NOT a graduation claim · artifact-recovery + verdict-reconciliation
· Date: 2026-05-30

> **Active phase:** 3 (Growing Codebook / Frame B)
> **Headline metric per [2026-05-28-frame-b-exposure-slope-headline-design.md:66-73 (TL;DR) + :75-112 (estimand)](../notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md):** within-seed exposure-recall slope-DiD `Δβ(s)=β_real−β_shuffle` (PROPOSED, pending sign-off). **This report retires that PROPOSED headline** via the pre-committed `corr_AC_BD` branch rule; the active headline reverts to the endpoint **level-DiD**. *(Spec line numbers re-verified 2026-05-30 after this session prepended a SHELVED banner to that spec, shifting them +9/+11; the pre-banner cites :57-62/:64-101/:117-125 now point ~10 lines high.)*
> **Required controls per same spec :128-136 (§Control):** global token-stream shuffle (primary) + gauge arm E (`FB→confound` check only).
> **Last verified result:** Gate-0 n=10 wikitext, DiD +0.019 CI [-0.121,+0.159], 6/10 positive, σ≈0.196, `G0->weak` ([Report 115](115_frameb_gate0_n10_drilldown.md), STATUS:18).
> **Why this experiment now:** discharge the one gating input for the window-vs-slope branch (`corr_AC_BD`), which was blocked because the Gate-0 n=10 summary JSON was off-repo (Drive-only). Recover it, compute `corr_AC_BD`, apply the pre-committed rule.

**THIS IS A BRANCH-DECISION / DRILL-DOWN REPORT, NOT A GRADUATION CLAIM.**
No phase graduates here. `corr_AC_BD` is a variance-decomposition diagnostic
(`gate0_frame_a.variance_report_from_summary`), **not** the spec headline. This
report (a) recovers a missing artifact, (b) computes `corr_AC_BD`, (c) applies
the user's pre-committed branch rule, and (d) reconciles a verdict-label
discrepancy. It measures, confirms, or signs off **no** graduation metric.

---

## 1. Headline of this report — `corr_AC_BD = +0.10` → drop the slope (pre-committed)

`gate0_frame_a.py --variance-report` on the recovered Gate-0 n=10 summary
(read-only, zero new compute):

```
corr_AC_BD                         = +0.10076   (worlds' lifts ~uncorrelated)
contrast_sd.A_minus_C (real lift)  =  0.14850
contrast_sd.B_minus_D (shuf lift)  =  0.14303
contrast_sd.DiD                    =  0.19553   ( = sqrt(.149^2+.143^2-2*.10*.149*.143) ✓ )
did_mean                           = +0.01914
did_sd_over_binomial_floor         =  10.9      (structural, NOT sampling noise)
single_arm_binomial_floor          =  0.01796
n_for_80pct_power_at_effect(0.02)  =  750
diagnosis: "...corr(A−C, B−D)=+0.10 — the worlds' lifts are weakly/anti
            correlated, so the DiD does NOT cancel seed variance — pairing on
            the codebook is not buying much. σ(DiD) ≫ binomial floor → the
            variance is structural (codebook×corpus draw)..."
```

**Pre-committed branch rule** (the user's ruling; restated post-walk-back in
HANDOFF-2026-05-29:36-37; read-protocol
[05:21,30](../brainstorm-workspace/2026-05-29-frameb-grill/05-stage1-read-protocol.md)
— the `0.3` threshold is `gate0._variance_diagnosis`'s own `if c > 0.3` cutoff,
reused for consistency, not invented):

> *drop the slope if `corr_AC_BD < ~0.3`, default to window-averaging on ties;
> "apply verbatim, no rationalizing."*

`corr_AC_BD = +0.10 < 0.3` → **the slope-DiD reframe is dropped. The active
headline reverts to the endpoint level-DiD (Frame A).**

## 2. What `corr_AC_BD` does and does NOT falsify (the load-bearing nuance)

`corr_AC_BD = corr(A−C, B−D)` is the across-seed correlation of the
consolidation **lift** in the two worlds. It bears on the slope-DiD's **outer**
variance kill only:

- **Outer (real−shuffle) kill — FALSIFIED.** `Var(Δβ) = Var(β_real) +
  Var(β_shuffle) − 2·Cov`. The outer differencing reduces variance only if the
  worlds co-move (`Cov > 0`). At `corr_AC_BD = 0.10` the level lifts barely
  co-move — a strong prior that the slope lifts won't either — so the outer
  differencing cancels ~nothing. The variance algebra is exact at the endpoint:
  `σ(DiD)=0.196 ≈ √(0.149² + 0.143² − 2·0.10·0.149·0.143)`. The
  codebook-pairing premise ("pairing on the codebook cancels seed luck") is
  **empirically falsified**.
- **Inner-slope kill — UNTESTED (Q1 open).** The slope's *primary* claimed
  advantage per spec §"Why a slope" is the **within-seed slope removing the
  per-seed additive intercept** on the exposure curve. `corr_AC_BD` does **not**
  measure this. The 2026-05-29 walk-back (read-protocol
  [05:9,23](../brainstorm-workspace/2026-05-29-frameb-grill/05-stage1-read-protocol.md))
  is explicit: window-draw dominance (the dominant per-seed component per
  [Report 115 §2](115_frameb_gate0_n10_drilldown.md)) does **not** auto-kill the
  OLS slope, which is invariant to per-seed additive shifts — so **σ(β)/σ(Δβ)
  remains untested; the Q1 conditional is open, not failed.**

**Therefore the drop is a pre-commit-honoring decision, not a measured kill of
the slope.** It is well-founded: rather than build a ~1-day slope rig on an
*untested* inner kill whose only *testable* sibling (the outer kill) just
failed, escalate n on the **proven** level-DiD. But the report does not claim
σ(β) was measured. (Per read-protocol guardrail 05:27-29: Stage-1B "can FALSIFY
the reframe but CANNOT confirm it.")

### Step-6 reconciliation (walk-back-first, applied to the read)

The grilled plan's Step 6 mandates reconciling the 2026-05-29 over-read into the
stale pre-06 binary **before** acting on any Stage-1 read: route window-draw
dominance to **keep-slope-open**, *never* to the stale "slope not the lever"
binary. Honored here: the slope is **not** dropped on the window-draw read
(Q1 stays open); it is dropped **only** on the user's explicit `corr_AC_BD<0.3`
pre-commit, with the outer-vs-inner distinction recorded so the drop is not
mistaken for a σ(β) measurement.

## 3. Forward path — n≈27 level-DiD (pre-committed escalation)

> **⚠️ SUPERSEDED 2026-05-30 (see [Report 117](117_frameb_leveldid_feasibility_consolidation_variance.md)).**
> The "n≈27 / window-averaging" path below rests on Report 115 §3's numbers,
> which are **wrong** (corrected in Report 115 §3 + Report 117 §2). Window-averaging
> via `window_seed_override` is **K× re-consolidation**, and the cheapest
> clear-zero config is K=1 at **n≈424 ≈ 42× the n=10 run** (n=40 reaches only
> t≈0.61). The endpoint level-DiD is **not economically rescuable** at this op
> point; the active deliverable is now a strategic decision (cheap landscape
> diagnostic vs mechanism reconsideration). The §2 outer-vs-inner nuance and the
> corr_AC_BD slope-drop (§1) are **unaffected** and stand.

The pre-committed branch table (anchor Part 5 / [Report 115](115_frameb_gate0_n10_drilldown.md))
routes `G0->weak` to **escalate n, not reframe**. Combined with the dispositive
off-zero arithmetic in [Report 115 §3](115_frameb_gate0_n10_drilldown.md):

- A √K window-average shrinks σ 0.196→~0.044 but **saturates at K=9** (the
  held-out WikiText pool is only thousands of windows) ⇒ n=10 CI ≈ [−0.013,
  +0.051], still spans zero.
- **n≈27 seeds — not more windows — is the lever** that clears the +0.019 DiD
  off zero. (`n_for_80pct_power_at_effect=750` at the raw σ; window-averaging to
  σ≈0.044 brings honest-n to ~37 for 80% power, ~27 to clear zero at the point.)

**Next deliverable:** window-averaged endpoint level-DiD at **n≈27 seeds**, with
the Step-3a codebook-only eval-isolation guard and the Step-3c verdict firewall
landing first (the firewall is now *also* relevant: no slope classifier should
ever serialize a level-DiD verdict, and vice-versa).

## 4. Artifact recovery + verdict reconciliation (provenance)

The Gate-0 n=10 summary was **absent on-repo** (firsthand-confirmed: no
gate0/variance/summary JSON on disk; git never added one; the colab notebook
reads it from a Google Drive path). Recovered this session:

- **Source:** `MyDrive/neuro-ai/results/gate0_2026-05-28/gate0_summary.json`
  (Drive file id `1qJxZr_zYmZLLk79u3R71P-ppREeNMdOZ`, 151 465 bytes, created
  2026-05-29), via the Google Drive connector.
- **Persisted on-repo** under `reports/gate0_2026-05-28/`:
  - `gate0_summary.as-recovered-from-drive.json` — exact recovered bytes
    (forensic; stored verdict `G0->dead`).
  - `gate0_summary.json` — reclassified by current code; verdict
    **`G0->weak`** with a `verdict_reclassified_from: "G0->dead"` breadcrumb.
  - `gate0_summary.md` — regenerated, carries `G0->weak`.

**Verdict-label discrepancy — resolved, a checked fact, not an assertion.** The
Drive artifacts (`.json` + `.md`) both store **`G0->dead`**, but STATUS:18 and
Report 115 record **`G0->weak`**. The per-seed DiD stats are byte-identical
across all three (same run). Running the *current* (post-fix) classifier
`reclassify_summary(...)` on the recovered stats yields `G0->dead → G0->weak` at
the default `meaningful_effect_floor=0.02`. So the Drive copy is the
**pre-fix** artifact (the "verdict-classifier bug" STATUS's 2026-05-28 entry
says was "caught/fixed"); the on-repo code and STATUS/Report 115 are correct.
No correction to STATUS is needed; the recovered artifact's stale label is
fixed in place with the breadcrumb.

## 5. What this report does NOT license

- **No graduation.** Neither the level-DiD (`G0->weak`, underpowered) nor the
  slope-DiD (dropped) graduates anything.
- **No σ(β) claim.** The inner-slope variance kill is **untested**; the slope is
  dropped by pre-commit, not because σ(β) was measured. The Q1 conditional is
  open. (If a future session ever wants to revisit the slope, the honest gate is
  the Stage-2 σ(β) holdout pilot, not `corr_AC_BD`.)
- **No Phase 5′ reopen.** Phase 5′ stays paused; nothing here touches
  ΔE / bridge / M2 / matrix.
- **No reproducibility claim on the ≈96/4 / ~24× variance split.** That remains a
  provenance-flagged single-run ranking ([Report 115 §2](115_frameb_gate0_n10_drilldown.md));
  it is corroborated *directionally* here only insofar as `corr_AC_BD=0.10`
  confirms the per-seed variance is structural and not codebook-pairing-cancelable.
- **No headline pin.** The slope-DiD spec is shelved (never pinned); the level-DiD
  becoming the active headline by elimination is recorded in STATUS, not pinned
  into `phase-3-deep-dive.md §Headline` as a new graduation spec without sign-off.

---

### Done-gate ledger for this report

| done-gate | status here |
|---|---|
| 1. Headline + CI | `corr_AC_BD=+0.10` (point diagnostic, no CI by construction); underlying DiD +0.019 CI [-0.121,+0.159] carried from Report 115 |
| 2. Control on same test set | gauge arm E in the recovered summary (4a byte-identical, 4b Δ-CI contains 0); branch rule is a re-analysis, no new arms |
| 3. Drill-downs explain anomalies | §2 outer-vs-inner kill decomposition; §4 verdict-reclassification; variance-report fields quoted verbatim §1 |
| 4. Written up under reports/ | **this file** |
| 5. Status note updated | STATUS Current-state (Headline + Active-deliverable) walked back; Recent-updates entry added; size-guard passes |
