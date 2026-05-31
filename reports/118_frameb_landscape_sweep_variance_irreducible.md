# Report 118 — Landscape-size diagnostic: the consolidation-injected variance is NOT reducible by landscape size

**Status:** DIAGNOSTIC / DRILL-DOWN result (parallel fan-out, n=10 × L∈{64,256,512})
· NOT a graduation claim · closes the "bigger landscape" lever
· Date: 2026-05-30

> **Active phase:** 3 (Growing Codebook / Frame B)
> **Diagnostic metric per [2026-05-30-landscape-sweep-diagnostic-precommit.md](../notes/notes/2026-05-30-landscape-sweep-diagnostic-precommit.md):** σ_A(L) (consolidated real-arm per-seed SD) vs the frozen floor σ_C(L) and the lift mean(A−C)(L), read against the pre-committed thresholds.
> **Last verified result:** [Report 117](117_frameb_leveldid_feasibility_consolidation_variance.md) — consolidation injects the variance (σ_A=0.157 vs σ_C=0.069 at L=64); level-DiD not economically rescuable.
> **Why this experiment now:** the user-authorized cheap test of whether a larger Hopfield landscape collapses the consolidation-injected variance toward the frozen floor (the one lever that would make any downstream estimand affordable).

**DIAGNOSTIC, NOT A GRADUATION CLAIM.** No phase graduates. The run measured only
arms **A** (consolidated, real) and **C** (frozen, real) — all the pre-committed σ
read needs — via the parallel per-(L,seed) fan-out (`landscape_sweep_parallel.py`),
**byte-identical** to a serial run (cells are order-independent; the σ_A(64)=0.157
anchor reproduced the recovered run exactly).

---

## 1. Result

| L | σ_A (consolidated) | σ_C (frozen) | mean(A−C) | n |
|---|---|---|---|---|
| 64 | **0.157** | 0.069 | +0.032 | 10 |
| 256 | 0.150 | 0.033 | +0.050 | 10 |
| 512 | 0.139 | 0.023 | −0.006 | 10 |

**Reproducibility anchor PASSED:** σ_A(64)=0.157 reproduces the recovered Gate-0
n=10 run byte-for-byte (the harness warns if it leaves 0.13–0.18). The run is trustworthy.

**Verdict (pre-committed): `CAPACITY-WALL`** — mean(A−C)(512) = −0.006 < 0.015, so
the L=512 landscape exceeded Hopfield capacity at β=10 and the consolidation signal
collapsed. The capacity guard overrides the σ read. Independently, the σ read is
also **`VARIANCE-IRREDUCIBLE`**: σ_A(512)=0.139 ≥ 0.13 (only an ~11% drop from 0.157
across an 8× landscape increase). **Both routes close the "bigger landscape" lever.**

## 2. The decisive structure — consolidation re-injects variance independent of L

The hypothesis ([Report 117 §3](117_frameb_leveldid_feasibility_consolidation_variance.md))
was that the per-seed variance is a *landscape-sampling* artifact (each seed memorizes
a different 64-of-220k-window landscape) that a larger landscape averages out. The
data **falsifies** that:

- **The frozen floor σ_C DOES drop with L** (0.069 → 0.033 → 0.023) — so landscape
  *sampling* variance is real and reducible: a bigger memorized landscape stabilizes
  the pre-consolidation recall.
- **But σ_A (consolidated) does NOT follow** (0.157 → 0.150 → 0.139). It stays high
  while the floor it was "supposed to" collapse toward keeps dropping. The
  **σ_A/σ_C gap widens from 2.3× to 6×.**

**Conclusion:** consolidation re-injects per-seed variance *on top of* the (now-tamed)
landscape-sampling floor, and that re-injection is **independent of landscape size**.
The variance is a property of the consolidation dynamics — different seeds' codebooks
adapt differently regardless of how stable the landscape is — not a sampling artifact
landscape size can average away.

## 3. Drill-down — the lift is non-monotonic (one genuine positive nugget)

mean(A−C): **+0.032 (L=64) → +0.050 (L=256) → −0.006 (L=512)**. Consolidation's
benefit on the real corpus *peaks at a moderate landscape* (L=256 gives a ~56%
larger lift than L=64) before the capacity wall kills it at L=512. So the mechanism
**works** and is landscape-tunable up to capacity — the problem is purely that the
per-seed *variance* (σ_A ≈ 0.15) doesn't shrink, so even the larger L=256 lift is
swamped (per-seed SNR 0.050/0.150 ≈ 0.33, still ≪ 1).

**Caveat (binding):** this run measured only the real-world lift A−C, **not the DiD**
(no shuffle arms B/D). The L=256 lift bump is generic consolidation benefit on real,
**not** corpus-specificity. Whether the DiD (real−shuffle) tracks it is unmeasured.

## 4. What this closes, and the standing picture

This is the **third lever closed this session** on the Frame B level-DiD:
1. **Slope-DiD** — dropped (corr_AC_BD=+0.10, [Report 116](116_frameb_corr_ac_bd_slope_drop.md)).
2. **n-escalation / window-averaging on the level-DiD** — not economically rescuable
   (~42× the n=10 run; window-averaging is K× re-consolidation, [Report 117](117_frameb_leveldid_feasibility_consolidation_variance.md)).
3. **Landscape size** — does not tame the consolidation-injected variance (this report).

Per the precommit, `CAPACITY-WALL` / `VARIANCE-IRREDUCIBLE` routes to a **Frame B
mechanism / operating-point rethink** — not more compute. The variance is structural
and consolidation-intrinsic. The corpus-specificity signal the DiD chases is not
cheaply detectable at this substrate/operating point.

## 5. Threads for the rethink (not decisions — input for the user)

- **The capacity wall implicates β, not just L.** L=512 collapsed *at β=10*. Sharper
  basins (higher β, or larger D) might support a bigger landscape without collapse —
  a β/D × L exploration could reveal whether a different regime tames σ_A. (Op-point
  change; needs sign-off.)
- **Consolidation as a variance amplifier** is arguably the system *correctly* adapting
  the codebook to each landscape — which may mean the real−shuffle DiD is the wrong
  contrast (it pairs worlds on the codebook, not on the landscape; corr_AC_BD=0.10
  shows that pairing buys nothing). A contrast that controls the landscape draw might
  be needed.
- **Or:** accept that Phase 3 Frame B's corpus-specificity effect is structurally too
  weak/variable to graduate at this scale, and pivot.

---

### Done-gate ledger

| done-gate | status here |
|---|---|
| 1. Headline + CI | σ_A(L) table (point SDs over n=10; per-seed artifact on Drive for CIs) |
| 2. Control on same test set | frozen arm C is the per-seed within-cell baseline; σ_A(64) anchor reproduces the recovered run |
| 3. Drill-downs explain anomalies | §2 σ_C-drops-but-σ_A-doesn't; §3 non-monotonic lift / capacity wall |
| 4. Written up under reports/ | **this file** |
| 5. Status note updated | STATUS Current-state walked forward; Recent-updates entry added |
