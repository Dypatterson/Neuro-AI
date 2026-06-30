# PRE-COMMIT — Bet B: pretraining RELOCATES the structure-injection wall; the open door is meta-ICL (grow composition in working memory) → CONSOLIDATE into weights

> *Status: DESIGN + Stage-1 build (2026-06-29). Synthesis of this session's multi-workflow investigation
> (walls verified; EMBER/spiking ruled out; the pretraining-as-foundation question). Charter:
> [CONTEXT-B.md](../CONTEXT-B.md). Arena: the GECA-resistant SCAN-MCD ([Report 146](../reports/146_betb_scan_mcd_stage0/report.md)).
> Anti-homunculus review: see §6 (run BEFORE Stage-2 consolidation code lands).*

## 0. Provenance

Three workflows this session (`wf_f8f9906e`, `wf_ac82a9e1`, `wf_e1196790`) deep-read the primary
literature (full papers, not abstracts), verified the project's recurring walls, and tested the user's
question: *can pretraining a foundational structure that continual learning builds from overcome the
walls?* This note banks the verdict + the one door it surfaced.

## 1. Verdict — pretraining-a-foundation RELOCATES the wall, it does not dissolve it

The decisive seam is **(a) atoms/representations vs (b) the composition RULE**, and the wall lives
entirely on (b):

- **(a) A foundation of atoms** = the project's own `clause-aug` writ large. [Report 151](../reports/151_betb_mcd_covariate_shift/report.md)
  proves it has **zero headroom** on the discriminating axis (atom TV≈0.10 / 0-novel; whole-command
  compound TV=1.000, 326/326 novel). Saturated — can't cross.
- **(b) A foundation that carries the recombination rule** crosses the wall but **IS structure-injection**
  (the thesis-weakening winning side) — the crossing just happens at pre-training time.
- Every real pretraining objective (MLM / LM / span-corruption) produces (a), never (b). Empirics
  (full-text-verified): **Furrer 2020** T5-11B pretrained **40.9%** vs from-scratch **21.4%** on CFQ-MCD
  (~2× vanilla, then ~50 pts below the injected ~90% ceiling; *hurts* −8.5% on SCAN-length); **Qiu 2022**
  scaling is **flat-to-negative** (MCD1 61%@Base → 55.5%@11B); **MLC** (Lake-Baroni) solves COGS-*lexical*
  only and gets its bias from a hand-designed meta-grammar; the "scaling→composition" result
  (Redhardt/Hupkes 2025) is **synthetic-only** and *requires* training-distribution coverage of the task
  space — the explicit negation of compound divergence.
- **The brain version doesn't dodge the charge:** Zador's developmental scaffold is the (b)-injected /
  (a)-learned split verbatim; the genomic-bottleneck algorithm *distills* the innate prior from a trained
  net (a transfer of an outer optimizer's solution, not a local rule that grows the bias); and the
  cataract / rod-monochromacy dissociations show foundation-*gating* is empirically soft.

**The unifying wall (verified this session):** *no local / no-global-backprop / emergent mechanism
manufactures a compositional inductive bias.* The field has the local-learning half (EBM / predictive
coding / equilibrium-prop = local, backprop-grade credit assignment); nobody has a local rule that
**grows** the compositional bias.

## 2. The new finding — the compound wall is REGIME-PERMEABLE, not fundamental

**meta-in-context learning** (Towards Understanding ICL & Compositional Generalization, arXiv:2403.11834)
lifts SCAN-MCD with a **same-size, from-scratch model by a training *regime* alone — no grammar injected:**

| split | baseline C-Transformer | meta-ICL | meta-ICL + label-shuffle |
|---|---|---|---|
| MCD1 | 21.8 ± 3 | 60.4 ± 13 | **71.2 ± 7** |
| MCD2 | 25.6 ± 2 | 53.3 ± 2 | **74.8 ± 9** |
| MCD3 | 19.7 ± 2 | 50.7 ± 7 | 38.7 ± 8 *(label-shuffle HURTS MCD3)* |

By the project's own grown-vs-injected arbiter this is on the **GROWN side** of the wall, on the
**compound axis** where `clause-aug` provably can't reach — the strongest evidence in the whole sweep that
the wall is **not absolute**. **The catch (the whole game):** the composition lives in the **context
window (working memory)**, *not consolidated into weights*. Russin-Pavlick-Frank frame this as
**complementary learning systems** — ICL = fast hippocampal composition; in-weight = slow neocortical
consolidation whose gradient bias does not compose.

Method (for reproduction): episodes τ = [x¹;y¹;…;xᴹ;yᴹ], M∈{10,25,50}; **8-layer causal Transformer**
(d=512, 25.2M params, abs-pos), from scratch, next-token loss on **output positions only**; label-shuffle
= per-variant remap of the output vocabulary so >1 output maps to each input across training (forces
context-use); eval with **k=M−1 support** examples sampled from train at test time.

## 3. The reframe (the real answer to the user's intuition — *inverted*)

The structure **cannot be pre-injected into** the foundation (that's injection / relabel). But it **can be
grown in working memory by a training regime**, and the open question becomes **consolidating that into the
foundation**. *The foundation is the OUTPUT of consolidation, not the input to it.* That — fast in-context
composer → local consolidation into weights — **IS the Bet-B thesis** (CLS replay/consolidation), now with
the first concrete-arena evidence the fast composer is reachable without injection.

## 4. The 2-stage oracle

- **STAGE 1 (this build — cheap reproduction, kills the door if it fails):** a small causal Transformer on
  the existing `data/scan/mcd_split`. Arms (n≥5): **vanilla** (M=1, k=0, reproduce ~21.8), **meta-ICL**
  (M=10, eval k=M−1, target ~60), **meta-ICL+label-shuffle** (target ~71 on MCD1/2). If meta-ICL does NOT
  give a CI-disjoint lift over vanilla on **MCD2/MCD3** at our scale → bank (regime-impermeable at this
  scale; document the scale reduction, do not over-claim a mechanism null).
- **STAGE 2 (the thesis crux — future):** freeze the support and test whether a **local consolidation step
  banks the in-context composer into weights** — **k=0 must not collapse.** That step clears the exact
  consolidation wall meta-ICL leaves open, and it is the Bet-B thesis in one experiment.

## 5. Headline metric (CLAUDE.md preamble cites THIS line)

**Stage-1 headline:** `Δ = mean_exact_match(meta-ICL, k=M−1 support) − mean_exact_match(vanilla, k=0)`,
**CI-disjoint > 0 on MCD2 AND MCD3** (the splits where atoms saturate / compound TV=1.000), ≥5 seeds,
bootstrap CI. Reproduction target (paper): MCD1 21.8→~60–71, MCD2 25.6→~53–75.
**Required controls:** (i) **vanilla = same Transformer architecture** (isolates the *regime*, not
transformer-vs-GRU capacity — do NOT compare to the GRU 0.158 floor); (ii) the **label-shuffle** arm
(its MCD3 *regression* is a reproduction fidelity check); (iii) the **k=0 collapse check** = the Stage-2
crux (meta-ICL with no support must drop — proves the composition is in the context, setting up the
consolidation question). **Last verified result:** [Report 151](../reports/151_betb_mcd_covariate_shift/report.md)
(compound covariate-shift; atom/clause headroom = 0). **Why now:** the only grown-not-injected,
compound-axis-positive datapoint in the whole walls sweep; load-bearing unknown = consolidation = the
project's core thesis.

## 6. Anti-homunculus check

The meta-ICL **regime** is a training-data/episode construction + a **standard causal-LM next-token loss**
+ an output-vocab permutation. There is **no supervisor, no `if-X-then-Y`, no module that arbitrates** —
"composition" emerges because ordinary SGD on the episode distribution makes attending-to-support the
loss-minimizing solution. The apparent decision (use the support to map+compose) lives entirely in the
attention dynamics learned by the gradient, not in an arbiter. **PASS** for Stage 1 (a learning objective,
not an arbitration). **Stage-2 BUILD CONDITION:** the consolidation step that banks ICL→weights must be a
**local dynamic** (replay / weight-anchoring / a measured-gate), NOT a supervisor that reads accuracy and
copies the support in — else it smuggles a homunculus. (Run the `anti-homunculus-reviewer` before Stage-2.)

## 7. Honest prior

**~0.10–0.12** that this genuinely overcomes the wall (crosses the compound axis without re-injecting (b)
AND banks it in weights). Stage-1 reproduction is *itself* an open empirical question at our reduced scale —
the 2403.11834 numbers are an 8-layer/25.2M result; a small-Transformer null is "didn't reproduce at this
scale," not "regime fails." Go in expecting to learn whether the wall is regime-permeable *at our scale*,
which gates everything downstream.

## 8. Build

`experiments/<NNN>_betb_metaicl_stage1.py` — a small from-scratch causal Transformer (reuses
`exp87.load_pairs` + the `mcd_split` data) + the meta-ICL episode regime + vanilla & label-shuffle arms +
MCD1/2/3 exact-match with k=M−1 support; `--smoke` first. Report → `reports/152_...`; STATUS walk-back first.
