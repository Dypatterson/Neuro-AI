# Report 152 — meta-ICL Stage-1: the in-context mechanism does NOT reproduce at our scale (direction check, n=2)

*Bet B. **Type: DIRECTION CHECK (n=2) — NOT the n≥5 graduation.** The cheap kill-or-validate before
committing to the full run. Charter: [notes/betb-metaICL-consolidation-door-precommit.md](../../notes/betb-metaICL-consolidation-door-precommit.md).
Code: [experiments/100_betb_metaicl_stage1.py](../../experiments/100_betb_metaicl_stage1.py); data:
`experiments/exp100_dir2.json` (gitignored). Date: 2026-06-29.*

## Preamble (CLAUDE.md)

> **Active capability:** Bet-B meta-ICL Stage-1 reproduction (the regime-permeability gate for the
> grow-in-context → consolidate-into-weights door).
> **Headline per [precommit §5](../../notes/betb-metaICL-consolidation-door-precommit.md):** Δ =
> EM(meta-ICL, k=M−1 support) − EM(vanilla, k=0), CI-disjoint > 0 on MCD2 AND MCD3, **n≥5**, bootstrap CI.
> Reproduction target (paper): MCD1 21.8→~60–71, MCD2 25.6→~53–75.
> **Controls:** vanilla = same Transformer architecture (isolates the regime, not transformer-vs-GRU);
> label-shuffle arm; the k=0 collapse check.
> **This run is a DIRECTION CHECK at n=2** — explicitly below the n≥5 headline bar; reports a trend +
> the qualitative mechanism signal, not a graduation verdict.
> **Anti-homunculus:** PASS (reviewer-confirmed; the regime is a learning dynamic + a static data knob +
> a boundary measurement).

## Config

Small from-scratch causal Transformer: d=256, 4 layers, 4 heads, ff=512, **2.14M params** (paper: 8-layer
/ 512-d / **25.2M** — we are ~12× smaller). M=10, 4000 steps, batch 32, lr 3e-4, `max_eval=192`, **2 seeds**,
splits mcd1/2/3, arms {vanilla, meta-ICL, meta-ICL+label-shuffle}, k=0 collapse check on. MPS.

## Results

| split | vanilla (k=0) | meta-ICL (k=M−1) | meta-ICL (k=0) | meta-ICL+shuffle (k=M−1) |
|---|---|---|---|---|
| mcd1 | 0.023 | 0.060 | **0.109** | 0.005 |
| mcd2 | 0.010 | 0.047 | **0.138** | 0.003 |
| mcd3 | 0.026 | 0.039 | **0.091** | 0.000 |
| *paper meta-ICL* | *(base 21.8)* | *60.4 / 53.3 / 50.7* | — | *71.2 / 74.8 / 38.7* |

Δ(meta-ICL_{k=M−1} − vanilla): mcd1 +0.036, mcd2 +0.036, mcd3 +0.013.

## Interpretation

**1. The auto "HEADLINE = True" flag is a 2-SEED ARTIFACT — NOT a pass.** The bootstrap CI resamples from
only 2 points, so the deltas read as "CI-disjoint positive" with degenerate-tight intervals (e.g.
[0.036, 0.036]). The deltas themselves are tiny (+0.013 to +0.036) and the absolutes are **~10× below the
paper** (vanilla 0.01–0.03 vs 0.218; meta-ICL 0.04–0.06 vs 0.51–0.60). Treat the flag as a false positive
(the same few-seed false-PASS failure mode caught in Reports 126/127); the real bar is n≥5.

**2. The robust, load-bearing finding: the in-context mechanism is INVERTED.** Across **all 3 splits × 2
seeds**, meta-ICL with support (k=M−1) does **worse** than with no support (k=0): 0.060<0.109, 0.047<0.138,
0.039<0.091 (~2–3× worse *with* support). This is the **opposite** of the paper's mechanism and the opposite
of the signal the reframe needs (composition living in the context → k>0 helps, k=0 collapses). At this
scale the model maps query→output directly and treats the support set as noise; the **in-context
composition circuit did not emerge**.

**3. What DID happen: a small weight-resident regime lift.** meta-ICL evaluated at k=0 beats vanilla at k=0
(0.09–0.14 vs 0.01–0.03, ~4–6×) — the meta-ICL *training distribution* (each command seen in many contexts)
buys better atom-level robustness in the weights, but **not** in-context composition. Note this k=0 lift is
on the atom axis, which [Report 151](../151_betb_mcd_covariate_shift/report.md) shows is saturated /
0-headroom on the compound axis — consistent with "regime helps a little, compound axis untouched."

**4. Label-shuffle collapsed** (~0 everywhere) — the harder context-dependent regime is unlearnable at this
size/budget.

## Verdict

**Pre-registered "didn't reproduce AT THIS SCALE", not "the regime fails."** In-context learning is
emergent with scale, and we are ~12× below the paper. The MPS run is a valid "does it port small?" probe and
the answer is: **the meta-ICL *regime* ports (small weight-resident lift); the in-context *mechanism* — the
load-bearing part for the grow-in-context→consolidate reframe — does NOT port at 2.1M params.** Do NOT bank
"regime fails"; bank "mechanism absent at small scale; faithful-scale test pending."

**Next (decision is the user's):** (a) **faithful-scale reproduction on Colab/CUDA** (8-layer / ~25M, M∈{10,25,50},
n≥5) — required to test the real in-context mechanism and therefore the reframe; OR (b) **bank the
small-scale null** and stop. The Stage-2 consolidation experiment is only meaningful once Stage-1's
in-context composer exists (k=M−1 ≫ k=0), which it does not at this scale.

## Done-gate honesty

This is a **direction check (n=2)**, not a graduated result: no n≥5, so no headline-with-CI claim is made
(the auto-flag is explicitly disavowed above). Gates met: control run (vanilla, same architecture) ✓;
drill-down explains the headline (the k=0>k=M−1 inversion) ✓; report exists ✓; STATUS updated ✓.
