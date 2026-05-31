# Report 117 — Frame B level-DiD feasibility: window-averaging is K× re-consolidation; the level-DiD is not economically rescuable

**Status:** FEASIBILITY / COST analysis (no new run; re-analysis of the recovered
n=10 + code audit, 9-agent verification workflow) · NOT a graduation claim
· Date: 2026-05-30

> **Active phase:** 3 (Growing Codebook / Frame B)
> **Headline metric per [phase-3-deep-dive.md §Headline](../notes/emergent-codebook/phase-3-deep-dive.md) (level-DiD; slope-DiD SHELVED [Report 116](116_frameb_corr_ac_bd_slope_drop.md)):** endpoint level-DiD on stratum-pooled Recall@K, real vs global-stream-shuffle.
> **Required controls per [2026-05-28-frame-b-exposure-slope-headline-design.md:128-136](../notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md):** global stream-shuffle (the B/D arms) + gauge arm E.
> **Last verified result:** Gate-0 n=10, DiD +0.019 CI [-0.121,+0.159], 6/10, σ=0.196, `G0->weak` ([Report 115](115_frameb_gate0_n10_drilldown.md)/[116](116_frameb_corr_ac_bd_slope_drop.md)).
> **Why this experiment now:** before committing a Colab budget to the pre-committed `G0->weak`→escalate-n route (n≈40 was proposed), cost the run honestly. The audit found the run's cost basis (cheap window-averaging) was wrong.

**THIS IS A FEASIBILITY / COST RE-ANALYSIS, NOT A GRADUATION CLAIM.** No phase
graduates. No new run was launched. All numbers are reproduced directly from
`reports/gate0_2026-05-28/gate0_summary.json` (recovered 2026-05-30) and from a
code audit; the load-bearing ones were re-verified by the report author and by
an independent 9-agent workflow.

---

## 1. The dispositive finding — window-averaging is K× re-consolidation, and it makes the bill *worse*

**Window-averaging via `window_seed_override` re-runs consolidation each draw.**
The reseed chain (`experiments/c3_phase3_exit_criterion.py`): `c3:869` `ws = window_seed_override or seed` → `c3:870-874` reseeds the **train** subsample
(2048 of ~219 591 windows) → `c3:882` `landscape_windows = train_windows[:64]` →
`c3:932-933` `cons_train = train_windows[64:]` then `_consolidate_codebook(...)`
runs whenever the arm is consolidated. So a new window seed yields a different
landscape + different consolidation corpus → **a different codebook and recall**
for the 3 consolidating arms (A, B, E). It is **not** the "K× eval, no extra
consolidation" the prior plan assumed (digest line 65). (The override isn't even
wired into the 5-arm gate today — it lives only in `run_variance_decomposition`;
using it in the gate needs new orchestrator code.)

**Total compute to clear zero is monotonically *increasing* in the window-average
count K** (cells = ⌈n⌉·K·5; the per-K variance floor is the atom draw, so adding
window draws multiplies cells without lowering the required n-for-fixed-σ). The
minimum is at **K=1 (no window-averaging)**. With σ_K = √(σ_atom² +
(σ_total²−σ_atom²)/K), σ_total=0.1955, σ_atom=0.0224, effect=+0.019, t≈2.0:

| config | σ_eff | n to clear zero | cells | × the n=10 run (50 cells) |
|---|---|---|---|---|
| **K=1 (no window-avg)** | 0.196 | **424** | **2118** | **42×** |
| K=9 window-avg | 0.069 | 53 | 2385 | 48× |
| K=26 window-avg | 0.044 | 22 | 2860 | 57× |

For **80% power** the K=1 figure is n≈856 → ~4280 cells ≈ **86×** the n=10 run.

**There is no affordable n≈40 clear-zero run.** n=40 plain (K=1) = 200 cells (4×
the n=10 run) but reaches only **t ≈ 0.61** — an order of magnitude short of
clearing zero. The "n≈40" / "n≈27" figures from the prior session were wrong:
they implicitly assumed window-averaging cheaply delivers σ≈0.044, which is false
(see §2). **The endpoint level-DiD is not economically rescuable at this operating
point** (~42× the n=10 run to clear zero; ~86× for power).

## 2. Corrections to Report 115 §3 (numbers were wrong, conclusion survives)

[Report 115 §3](115_frameb_gate0_n10_drilldown.md) (and the framing it seeded
into STATUS + Report 116) contained three errors, all of which made
window-averaging look cheaper/better than it is:

1. **"K=9 ⇒ σ=0.044" is wrong.** Under the report's own model, K=9 ⇒ σ_K =
   **0.0685**; reaching σ=0.044 needs **K≈26**. (Verified: `σ_K(9)=0.0685`,
   `σ_K(26)=0.0442`.) The downstream CI arithmetic was self-consistent *given*
   σ=0.044; the error is the K→σ mapping.
2. **"WikiText val+test is only thousands of windows, so window-averaging
   saturates at K=9" is wrong.** Test pool = 391 205 / 8 = **48 900** windows
   (~95 disjoint 512-draws); train pool ≈ 219 591. Window-averaging is **not
   pool-limited; it is compute-limited.**
3. **"n≈27 is the cheap clearing path" is misleading.** n≈27 silently assumes
   σ≈0.044 ⇒ K≈26 re-consolidations/seed ⇒ ~2860 cells (57×) — *worse* than
   K=1, n≈424 (2118 cells, 42×).

The **conclusion of Report 115 §3 survives a fortiori**: the correct K=9 CI is
**[−0.030, +0.068]** (wider than the reported [−0.013, +0.051]) and still spans
zero, so "√K window-averaging cannot clear the endpoint DiD off zero at n=10"
holds — it just can't be rescued cheaply by *any* K.

## 3. The root cause — consolidation *injects* the variance (new diagnostic)

A per-arm decomposition not in Report 115/116 (reproduced from the recovered
per-seed recalls):

| arm | σ | mean | consolidates? |
|---|---|---|---|
| A (real, consolidated) | **0.157** | 0.223 | yes |
| C (real, frozen) | **0.069** | 0.191 | no |
| B (shuffle, consolidated) | 0.124 | 0.217 | yes |
| D (shuffle, frozen) | 0.057 | 0.204 | no |

**Consolidation more than doubles per-seed variance** (σ_A/σ_C = 2.3×). The
variance lives in *how much consolidation helps on a given 64-window landscape
draw*, and `corr(A,C)=+0.34` is too weak for the frozen-arm differencing to
cancel that landscape luck. **The DiD makes it worse, not better:** `corr(A−C,
B−D)=+0.10 ≈ 0`, so `σ(DiD)=0.196 > σ(A−C)=0.149` — the real−shuffle differencing
*adds* the shuffle world's variance and *halves the mean* (+0.032 → +0.019). The
estimand is working against itself.

## 4. Levers evaluated (what does and doesn't help)

| lever | σ reduction | cost | sign-off | verdict |
|---|---|---|---|---|
| **Full-pool eval** (`n_test_windows=len(all_test_windows)`, drop-in at c3:875) | 0.1955→0.1922 (~1.7%) | **free** | none | **Take it** — free, removes the ~3% binomial confound. Does NOT clear zero. |
| CUPED on frozen arm C | ρ²=0.9% (corr(DiD,C)=−0.09) | free | none | **Immaterial.** (D looks correlated but is a constituent of the DiD → mechanical/overfit.) |
| Window-averaging (K↑) | √K on the window component | **monotonically worse** in cells | new code | **Counterproductive** (§1). |
| Switch headline to A−C alone | σ 0.149, mean +0.032 (cheaper) | none | yes — drops the corpus-specificity control | Cheaper, but abandons the shuffle control (the scientific point). |
| **Bigger landscape** (L 64→256→512) | *maybe* σ_consolidated→σ_frozen≈0.07 ⇒ honest-n 750→~105 | retrieve is linear in L; ~same GPU-s as window-avg; **cheap to TEST** | **required** (op-point change; precommit:184 forbids sweeps) | **Unverified hypothesis** — but the only lever that attacks the root cause (consolidation-injected variance). Cheap to test (§5). |

**On the slope-DiD (shelved [Report 116](116_frameb_corr_ac_bd_slope_drop.md)):**
the slope removes a per-seed *additive intercept*. But this diagnostic shows the
dominant variance is in the consolidation *lift* (σ injected *by* consolidation),
which is what the slope itself measures — so there is now positive reason to
expect σ(β) is **also** large. The slope is not a demonstrated rescue; σ(β)
remains untested (Q1 open) and the evidence tilts against it.

## 5. The reframed options (the decision is the user's)

1. **Cheap landscape-size diagnostic (recommended next test).** L ∈ {64,256,512},
   n=10, 5 arms, K=1 = **150 cells ≈ 13× the n=10 run (~1.8 GPU-hr)**. Tests
   whether a denser memorized landscape collapses σ_consolidated toward the
   frozen floor (~0.07). If yes → honest-n falls ~7× and *both* the level-DiD and
   the slope become affordable. If no → the variance is irreducible at this
   architecture and Frame B needs a mechanism rethink. **Needs op-point-change
   sign-off (precommit:184); label it a drill-down, not graduation.**
2. **Reconsider the mechanism / operating point.** The variance is structural and
   consolidation-injected; this may be the signal that corpus-specificity isn't
   detectable at D=4096 / landscape=64 / β=10 without an architectural change.
3. **Pay for the clear-zero level-DiD run.** ~42× the n=10 run (n≈424, K=1) to
   clear zero, ~86× for power. **Not recommended** — buys confirmation of a
   +0.019 effect whose estimand is structurally unable to detect it cheaply.

**Free wins to bank regardless of the choice:** full-pool eval as the default;
the §2 corrections to Report 115 §3; this per-arm decomposition.

## 6. What this report does NOT license

- No graduation; no run launched.
- No op-point change executed — the landscape sweep (§5.1) is a *proposal* needing
  sign-off, not a decision.
- No re-opening of the slope (it stays shelved; §4 only notes the evidence now
  tilts further against it).
- No Phase 5′ reopen.

---

### Done-gate ledger

| done-gate | status here |
|---|---|
| 1. Headline + CI | feasibility analysis; underlying DiD +0.019 CI [-0.121,+0.159] carried from Report 115; cost table §1 |
| 2. Control on same test set | re-analysis of the recovered n=10 (gauge arm E present); no new arms |
| 3. Drill-downs explain anomalies | §3 per-arm decomposition (consolidation-injected variance) is the root-cause explanation |
| 4. Written up under reports/ | **this file** |
| 5. Status note updated | STATUS Current-state walked back (the n≈27/window-averaging framing was wrong); Recent-updates entry added |
