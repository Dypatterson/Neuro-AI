# Report 132 — Depth × habituation-deflation oracle: NULL — depth does not compose and the divisive-habituation peel is empirically INERT; the only signal is the known single-layer k-WTA (Report-127, ceiling-exceeding = inflation, not a local-route win)

**Status:** NULL (depth doesn't help; habituation deflation inert). Banked, with a load-bearing scope caveat. The discipline held — no false positive banked (the tempting +0.24 is the 127 inflation trap, correctly flagged).
**Date:** 2026-06-02. **Branch:** `experiment/tem-local-reachability-oracle`.
**Harness:** `experiments/77_depth_habituation_oracle.py`. **Precommit:** [phase-3-depth-habituation-precommit.md](../../notes/emergent-codebook/phase-3-depth-habituation-precommit.md) (§8 post-smoke note).
**Origin:** the culmination of the 121-131 arc + the user's two reframes (not-flat → depth holds the residual; homunculus-dissolution → temporal habituation does the peeling). Grounding verdict was `build-with-conditions` (a genuine open wager, not a likely win).

## Result (WikiText-2, anchor VALID +0.1092/0.222; n=10; k=16, L∈{1,2,4}, κ=1)

| arm | L=1 | L=2 | L=4 | B-KILL |
|---|---|---|---|---|
| kwta + **habituation** (candidate) | +0.241 | +0.231 | +0.228 | 10/10 |
| kwta + **inverse-freq** (control) | +0.241 | +0.230 | +0.225 | 10/10 |
| kwta + **no-deflation** (Control C) | +0.241 | +0.231 | +0.228 | 10/10 |
| **linear** + habituation (predicted-null) | +0.000 | +0.000 | +0.000 | 10/10* |

Ceiling: NMF +0.167; floor (grow_G) +0.000; rand-stack −0.004. *(linear B-KILL 10/10 at spec 0 = the degenerate-zero-code artifact, Report 129 — not a pass; the headline conjoins B-KILL with spec_ci_lo>0.)*

## Verdict — NULL (corrected from an auto-verdict bug)

1. **Depth does NOT compose** — it mildly *hurts* (L=1 +0.241 ≥ L=2 +0.231 ≥ L=4 +0.228). `g5_depth` fails. Stacking layers added nothing.
2. **The habituation deflation is empirically INERT** — habituation ≈ inverse-freq ≈ no-deflation at every L (Δ<0.01). The divisive residual-row-mass peel barely changes the k-WTA partition; it did not peel.
3. **The only signal is the single-layer k-WTA** (+0.241) = the Report-127 result, which **EXCEEDS the NMF ceiling (+0.167)** → low-dim-inflation / k-means-reproducible = an artifact (ceiling-guard `g_ceiling_reach_not_exceed` FALSE), **not** a genuine local-route win. The linear arm = 0 (predicted-null confirmed).

**Verdict-logic bug (2nd today, after 129's (b)-detector):** the harness auto-emitted INVALID via "L=1 already PASSES → leaked top-down signal." That apparatus check is valid only for the LINEAR arm; for the NONLINEAR k-WTA arm, L=1 passing is EXPECTED (it is Report-127's single-layer result, not a leak). The check must be arm-conditional. Corrected verdict by reading the data: **NULL.** *(Gate-vocabulary carry-forward: auto-verdicts have now mis-fired twice — gate `g3` (129) and the L=1-leak check (132) — any future harness must not gate the headline/verdict on a condition that the known single-layer/global result also satisfies.)*

## Load-bearing scope caveat (do NOT over-claim)

**The deflation step did not actually deflate** (inert). So this NULL kills **"depth + divisive-row-mass habituation,"** NOT "depth + *any effective* local peel." The peel we know *works* — the GHA orthogonal-projection from the original [multilayer-deflation precommit](../../notes/emergent-codebook/phase-3-multilayer-deflation-precommit.md) — is **un-run**; but by the grounding's own honest ceiling that route is **SVD-by-stages** (a local route to the global code, never "beats SVD"). So the clean remaining test of the depth hypothesis is the GHA-peel version, and even a PASS there is the honest-ceiling result.

Also banked-as-sub-result: the **linear depth arm = 0** (linear deflation stack = ordered PCA = collocational, deeper — exactly as predicted by the grounding).

## What this banks — the sharpened spec clause

Three branches closed with banked nulls this session — **channel** (130, single-projection is channel-invariant), **capacity** (131, tightness is not the lever), and **depth-via-cheap-local-peel** (132, the local peel is inert or converges to SVD-by-stages). Clause 9 sharpens:

> *The escape is not a single projection in any channel, not capacity-tightness, and not a cheap local-deflation peel (which either does nothing — habituation/inert — or, done effectively, IS global SVD-by-stages — GHA). What reaches the subdominant paradigmatic mode is, on all evidence so far, genuinely outside the local-single/cheap-deflation family.*

## Disposition (the path the walls now describe)

1. **Cleanest remaining depth test:** re-run `experiments/77` with the **GHA-projector peel** (`deflate_subspace`, the original precommit) — the peel that demonstrably peels — to settle whether depth+*effective*-peeling composes (honest ceiling: SVD-by-stages local route, not "beats SVD"). Cheap, in-scope.
2. **Or** accept that local-deflation peels don't escape and the frontier needs a different primitive — **ideation round-2** (the invent harness recorded which walls each dead idea hit; round-2 targets non-single-projection, non-cheap-deflation primitives) or a **substrate/data change** (the Abstraction-node build, the user's fence).
3. The discipline held: **no false positive banked.** The +0.24 single-layer inflation was caught by the ceiling-guard, not promoted as a win.
