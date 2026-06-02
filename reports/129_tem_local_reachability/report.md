# Report 129 — TEM local-reachability oracle (Fork B): the global slot-ASSIGNMENT is load-bearing; a fixed-random-slot single-layer local writer does NOT reach pair-specific paradigmatic structure

**Status:** NULL (b) frozen-assignment-was-load-bearing — AMBIGUOUS-for-substrate (do NOT over-bank as "substrate can't help"). Adversarially verified (3 lenses, high confidence). Banked.
**Date:** 2026-06-02. **Branch:** `experiment/tem-local-reachability-oracle`.
**Harness:** `experiments/76_tem_local_reachability_oracle.py`. **Precommit:** [phase-3-tem-local-reachability-precommit.md](../../notes/emergent-codebook/phase-3-tem-local-reachability-precommit.md).
**Artifacts:** `result.json` (n=10), `run.log`.

## Preamble (CLAUDE.md experiment-preamble)

- **Active capability:** Codebook-growth (P3 structure) — the **local-writer test of Oracle-E**. Oracle-E's GLOBAL NMF of the symmetrized transition operator is a confirmed flashlight positive (Report 125 §3: +0.124/+0.176/+0.188); its **local writer was untested**.
- **Headline metric:** within-paradigmatic-SET **label-shuffle B-KILL** pair-specific residual (Report-126 hubness discriminator), static slot-vector cosine read — NOT `grow_G`.
- **Required controls:** raw-SPPMI-SVD calibration anchor (+0.109/0.222); flat-SPPMI `grow_G` linear floor; global-NMF Oracle-E ceiling (recover-fraction); `rand_nonneg` (dense + sparsity-matched) anti-inflation; the §3 spectral diagnosis.
- **Last verified:** Report 125 (Oracle-E global positive; local writer untested); 127/128 (bound = linear-vs-nonlinear-partition; replay deflates to inverse-PMI).
- **Why now:** the 2026-06-02 A/B/C faithfulness triage ranked this Fork-B oracle #1 (faithful, un-run, substrate-free/in-scope, the one local-writer route route-invariance does not foreclose). The user lifted the build-gate.

## Result (WikiText-2, D=4096, n=10 seeds; anchor VALID +0.1092 / king-queen 0.222; linear floor +0.0002)

| operator | writer | spec | recover-of-E | **B-KILL** | centroid (low=dominant) |
|---|---|---|---|---|---|
| M_trans | `tem_frozen` (frozen slots, linear) | +0.002 | 1% | 2/10 | 0.237 |
| M_trans | `tem_frozen_comp` (local k-WTA) | +0.099 | 49% | **4/10** | 0.237 |
| M_trans | `E_nmf` (GLOBAL ceiling) | +0.201 | 100% | **10/10** | 0.148 |
| build_S | `tem_frozen` | +0.000 | 0% | 8/10 *(spurious at spec≈0; fails g1/g2)* | 0.362 |
| build_S | `tem_frozen_comp` | +0.046 | 28% | 1/10 | 0.165 |
| build_S | `E_nmf` | +0.164 | 100% | **10/10** | 0.168 |
| M_SR (path-integration) | `tem_frozen` | +0.001 | 0% | 0/10 | 0.434 |
| M_SR | `tem_frozen_comp` | +0.020 | 15% | 1/10 | 0.434 |
| M_SR | `E_nmf` | +0.135 | 100% | 4/10 | 0.247 |

Pooled (n=90 each): **tem centroid 0.338 (subdominant) vs E centroid 0.233 (dominant), separation +0.104 > the pre-registered +0.10**; E recover-fraction 1.00.

## Verdict — NULL (b), corrected from an auto-verdict bug

**The NULL is genuine and honest** (not a degraded false-null): the apparatus is calibrated (anchor valid, floor reproduces +0.0002), the mechanism code is correct (slots frozen-before-pairs-read, `tem_frozen_write` = `l2rows(OP⁺@S_rand)`, `local_kwta` per-token), and the headline B-KILL is wired right. The fixed-random-slot LOCAL writer genuinely fails the pair-specific test (best 4/10, per-seed residuals straddle zero with negative median) while global NMF reaches it (10/10) — the exact Report-126 "para-set hubness, not pair-specific" signature for the local writer's partial gain.

**The verdict label is `(b) frozen-assignment-was-load-bearing`** — the global slot-ASSIGNMENT optimization is the load-bearing step; the frozen-random local writer cannot do it. This is **AMBIGUOUS-for-substrate**: it does NOT prove "the substrate can't help a local writer," only that *this* (single-layer, fixed-random-slot, Hebbian, linear-static-read) writer cannot, and the part that worked is the global optimization TEM performs **via backprop** (Whittington 2020, confirmed from the primary this session — TEM's generalizing structural code is backprop-trained; its Hebbian component does not generalize). So a fixed-random+Hebbian writer is, by TEM's own construction, **TEM's non-generalizing half** — its null is *expected and honest*.

**Harness bug (corrected, 2nd occurrence).** The harness auto-emitted `(a) basis-change-not-spectrum`. This was a logic bug: the `(b)`-detector gated on `any(E_nmf['PASS'])`, which requires `g3` (corr<0.15) — and **g3 is dead at n=40** (the accepted +0.109 SVD anchor fails g3 too; Report 125 §6 explicitly recommended removing g3 from this precommit — missed). With the detector decoupled from g3 (matching precommit §4's literal text — gate on E-beats-floor + spectral diagnosis), the registered `(b)` signature fires (sep +0.104, E dominant + recover 1.0). Fixed in `experiments/76:298`; verdict re-derived offline from the unchanged n=10 arrays → `(b)`. This is the **second** time a g1–g4 auto-verdict mechanically applied the dead g3 (first: exp65 / Report 125 §4) — see the gate-vocabulary note below.

## Adversarial verification (3 lenses, high confidence)

- **Lens 1 (code):** mechanism correct, slots frozen, B-KILL wired right, NULL not a fake. (Endorsed the buggy `(a)` at face value — *missed* the detector bug.)
- **Lens 2 (stats + a/b):** found the detector bug; corrected label = `(b)`; confirmed g3 unachievable at n=40; `tem_frozen_comp` +0.099 is hubness (kill_lo straddles zero, negative median) not an under-counted positive.
- **Lens 3 (false-negative trap):** the null is honest, NOT degraded — opened the TEM primary and confirmed the generalizing factor is backprop-trained, so the fixed-random+Hebbian writer is TEM's non-generalizing half; flagged that the auto-`(a)` "substrate-invariant / basis-not-spectrum" phrasing **over-reaches** (E's factorization *did* move para into dominant modes — it changed the spectrum's discriminability).

## What is banked (narrow, honest) — and what is NOT

**BANKED:** A single-layer, fixed-random-slot, online-Hebbian, **linear-static-read** local writer does **not** reach the pair-specific (B-KILL) paradigmatic target that global-NMF slot-ASSIGNMENT reaches (B-KILL ≤4/10 vs 10/10; recover ≤49% vs 100%). The **global slot-assignment optimization is load-bearing**, and TEM performs it via backprop (banned). Local per-token competition recovers magnitude but **not pair-specificity** (hubness). Path-integration (M_SR) was the *weakest* lever — slow-mode inflation did not help a frozen-slot writer.

**NOT banked (the auto-`(a)` over-claims):** "substrate-invariant" / "the flat-code bound generalizes to all nonneg-factorized single-layer codes" — contradicted by E's own dominant-mode recovery in this run. Also NOT "beats SVD/NMF" (magnitudes dimension-inflated; Report 125 §3 caveat-1). And `(b)` is qualified: **dominant-in-slot-space ≠ pair-specific** (build_S `tem_frozen_comp` centroid 0.165 dominant yet B-KILL 1/10) — the B-KILL is the true arbiter, not the centroid.

## Where this lands — the single-layer bound, and the depth signpost

Every bound in the 121–129 arc — including this oracle — is a **single-layer / single-projection** bound (Lens 3: `W = OP⁺@S_rand` then `l2rows` IS one linear projection). The brain is not single-layer. The honest next move is therefore **not another single-layer oracle**; it is one of:

1. **A genuine LOCAL multi-layer writer** (the depth direction): layer-by-layer local deflation (predictive-coding / local Hebbian stacks / equilibrium-propagation — NOT backprop) is the principled, brain-faithful way to peel the dominant collocational mode and expose the subdominant paradigmatic one. Whether a *local* deep rule reaches it is genuinely untested — the real frontier.
2. **The Codebook-growth ⇄ Replay combination** (CONTEXT.md §4 LIVE DIRECTION).
3. **Cheapest omitted single-layer lever (a bridge probe):** an **iterated/recurrent slot-update** writer (`W ← l2rows(OP@S)`, `S ← l2rows(OPᵀ@W)`, slots co-adapted by LOCAL Hebbian, random-init, NOT NMF's coupled global update, NOT backprop) — the cheapest non-backprop approximation of E's joint optimization. Directly tests the `(b)` hypothesis that ASSIGNMENT (not basis) is load-bearing. Prior: likely still NULL on B-KILL (still a single-operator dynamic), but it is the one genuinely-omitted faithful lever.

Build-gate (multi-layer substrate) remains the user's; a PASS on any of the above is its precondition.

## Gate-vocabulary note (carry forward)

Any future oracle reusing the g1–g4 gate **must** (i) NOT gate sub-case logic on full PASS, and (ii) treat **g3 (corr<0.15) as diagnostic-only at n≤40** — it is dead even for the known-positive (+0.109 SVD anchor, E_nmf). The within-set label-shuffle **B-KILL is the powered collocational/pair-specificity arbiter** that g3 was meant to be. Minor open items from the precommit checklist not emitted this run: separate spectral diagnosis for `tem_frozen_comp` (reused `diag_tem`; verdict-impact none) and the raw-operator SVD reference centroid (`diag_op`) — the `(b)` call rests on the E-vs-tem comparison + recover-fraction + B-KILL, which all 3 lenses found sufficient for the decision.
