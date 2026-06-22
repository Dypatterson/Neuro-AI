# Verification round 2 — all items addressed + overlooked seams (2026-06-22)

*Goal-driven re-verification of every load-bearing + second-tier uncertainty I flagged, via 5 external
research agents + the benchmark-framing sub-agent + a direct repo audit. Verdict legend: UPHELD /
HARDENED / SOFTENED / CORRECTED / CONFIRMED-WORRY. Caveat carried by ALL agents: WebFetch returned HTTP
403 on arxiv/ar5iv/aclanthology/openreview/paperswithcode this session — only `raw.githubusercontent.com`
worked — so most external numbers remain convergent-snippet-level, with ONE exception (the CFQ
leaderboard, now table-verified).*

## Headline of the verification

The core empirical claim **hardened**, but the *recommended mechanism* and *two of my own supporting
claims* did not survive cleanly. In order of importance: (1) a **confound in our own bar** — COGS-structural
is partly a decoder artifact (ReCOGS) and MCD is covariate-shift biased toward injected winners; (2) the
**splitting→hierarchy leap is unsupported** as a *compositional* claim and confounded with plain
optimization; (3) the **grokking gate is undefinable on MCD** (fix in hand); (4) the **pretrain-atoms arm
is largely a re-run of Report 149**; (5) the neuro **"stabilizer-not-manufacturer" claim is softened** —
the brain does manufacture-by-recombination; (6) **GRU buildability** is a real build, not a drop-in. The
bank itself survived 8 additional corners with 2 new disconfirmers.

## The 8 items, each addressed

### Load-bearing

**1. Numbers were snippet-level, not primary — ADDRESSED, with corrections.**
The **CFQ-MCD leaderboard is now table-verified** (raw GitHub, google-research/cfq README): **UT 18.9±1.4 ≈
vanilla Transformer 17.9±0.9** — the load-bearing disconfirmer — is rock-solid, with per-split (Transformer
MCD1/2/3 = 34.9/8.2/10.6; UT = 37.4/8.1/11.3), HPD 67.3, T5-11B 40.9, T5-11B-mod 42.1 all confirmed.
Corrections to apply: **CSL CFQ-MCD "~90–91%" → ~89.9%** (likely a web misquote; one snippet had conflated
CSL's 99.5 *COGS* number onto CFQ — so CSL ~89.9 is now *below* LeAR's confirmed 90.9); **LeAR 90.9 = the
MCD-mean** (confirmed, per-split 91.7/89.2/91.7); **COGS "35%" → 16–35% range**; **citation Yao & Koller
2022** (not "Weißenhorn/Yao/Koller"); **Furrer SCAN "−8.5%" UNTRACED** (drop or flag). None changes the
structural conclusion (injected ~90 vs flat ~18).

**2. "Splitting = grow a compositional level" — CONFIRMED WORRY; the leap is UNSUPPORTED.**
The splitting/Firefly papers optimize *training-loss reduction + parameter-efficiency* (CIFAR/ImageNet
top-1, MACs, continual-learning average accuracy) — **never an OOD/systematic split**. The growing-network
literature and the compositional-generalization literature are **disjoint**; the one paper getting
composition from "modular growth" (ICLR 2024) did it by *recovering ground-truth modules*, not by growth.
A NAACL 2024 depth study corroborates our own **Report 132** (depth's compositional benefit saturates
fast). The splitting literature itself says it just "turns sub-optimal minima into saddles" — so the
**parsimonious explanation for any gain is better optimization, not a new abstraction level** (the deepest
confound, primary-sourced). Open *only* in the saddle-to-saddle direction (plateau escapes add new
features of increasing complexity — right *shape*, but "new feature ≠ new compositional level"). **Fix:**
the headline must be the **systematic-minus-iid gap**, never raw accuracy, plus three discriminators
(below) that separate "grew a hierarchy" from "grew capacity."

**3. Grokking gate undefinable on MCD — CONFIRMED; fix in hand.**
Restricted/excluded-loss measures are defined by ablating the *known* key Fourier frequencies — they
*require* the closed-form circuit MCD doesn't have. **Drop the grokking measure.** Replace with: **primary
gate = descent-stationarity** (the Splitting native trigger; zero-calibration, deterministic, the cleanest
anti-homunculus fit — but it gates "parametric local optimum," necessary-not-sufficient for
generalization); **confirmatory = the Local Learning Coefficient** (singular-learning-theory /
developmental-interpretability; purpose-built to detect stagewise structure formation, validated beyond
toy tasks, but SGLD-calibration-fragile); **label-free fallback = HTSR-α or weight intrinsic-dimension**
if LLC won't calibrate on the small model.

**4. Never grepped reports/ for redundancy — ADDRESSED (repo audit).**
- **The pretrain-atoms arm is largely Report 149 `exp96` relabeled:** `--clause-aug` (102 single-clause
  SCAN commands, 10×) **already lifted vanilla MCD 0.158→0.297**. So arm B's result is *predicted* ~0.297,
  not novel. The genuinely open question collapses to **does gated-growth ON TOP of the 0.297 atomic floor
  beat 0.297?** (C-vs-B, not C-vs-D).
- **Report 131 already closed "just grow capacity"** (NULL-MONOTONE, necessary-not-sufficient).
- **Reports 122/123 nulled Bet-A growth rules** — different mechanism class (local, no-backprop codebook
  growth, the 121–132 bound), but the finding "the paradigmatic signal lives in subdominant modes the
  *local* iterative rule can't reach, only global SVD isolates" is a warning if the split criterion is local.
- The **specific Bet-B mechanism is genuinely new** (`Firefly`/`neuron split`/`stationarity`/`LLC`/`progress
  measure` appear nowhere in the repo except the precommit).

### Second-tier

**5. Verification was LLM-on-LLM — ADDRESSED, and the risk MATERIALIZED.** Agent V1 fetched the actual
cfq leaderboard table and **caught the CSL ~90–91% misquote that the first verification pass had
propagated** from shared snippets — a live instance of the shared-blind-spot risk. Residual numbers remain
snippet-only; treat decimals as ±0.5.

**6. Buildability was asserted — CORRECTED; my tag was optimistic.** The harness (`exp87.Seq2Seq`) is a
**single-layer GRU encoder + single-layer attention-GRU decoder**. Splitting/Firefly were built for
feedforward/conv nets; **neuron-splitting a GRU mid-training is a substantial build** (consistent resizing
of all gate matrices; the second-order split criterion for recurrent weights isn't standard). The gate
signal and scheduled control are buildable; the growth *operator* on a GRU is the real risk. **Mitigations:**
Firefly's gradient-grown fresh-neuron variant (simpler than the eig-split, also fence-cleaner); depth-stacking
instead of width-splitting; or swap to a small Transformer backbone where splitting is well-defined.

**7. Neuro details soft — SOFTENED (a real correction to findings.md).**
- **"Consolidation stabilizes, does NOT manufacture" is overstated.** An adversarial search found genuine
  manufacture evidence: **Wagner 2004** (Nature, behavioral-*causal*: sleep "by *restructuring* memory
  representations" extracts a hidden rule, >2×); **Lewis & Durrant iOtA** ("progressively *builds*
  schematic representations… the basis of cognitive abstraction," incl. false memories); **Wittkuhn 2025**
  (PNAS: replay linked to the *formation* of successor representations). Honest reframing: consolidation
  does **both** — stabilizes existing traces AND **manufactures new abstraction by overlap-driven
  recombination** (not de-novo creation of structure absent from inputs).
- **Inter-areal gating SOFTENED** — published hypothesis on staggered-timing correlation; the only causal
  critical-period evidence is *within-area* (Hensch E/I threshold), a competing explanation. Gating stays
  **unestablished**.
- **Cataract disconfirmer UPHELD** (congenital timing + DNN reproduction + cascading-control rule out the
  explain-aways), with a face-selectivity exception → strong-but-not-total decoupling.

**8. Search coverage not exhaustive — ADDRESSED; bank survives 8 more corners, with the round's biggest find.**
No method-level counterexample. Seven method corners hardened the bank, **two with new disconfirmers**:
**Csordás 2021** (emergent modularity *exists* on SCAN yet the net fails to *reuse* it compositionally —
direct evidence against the emergent-modularity reopen-door) and the **DEQ↔UT equivalence** (DEQ iterates
the same tied block → predicts DEQ ≈ UT ≈ 18%, closes DEQ without a run). Empty cells remain: SSM/Mamba on
CFQ-MCD, diffusion-LM on COGS/CFQ, TTT-on-COGS. The closest *new* grown contender, **Redhardt 2025**
("Scaling can lead to compositional generalization," NeurIPS), is synthetic-hyperteacher only — a future
threat if ported to real MCD, not a current counterexample.

## ★ The overlooked seam that matters most: our own bar is partly confounded

The benchmark-framing corner — which neither prior sweep questioned — found a genuine, two-layer seam:
1. **COGS-structural's "0–12% floor" is partly a decoder string-format/length ARTIFACT.** **ReCOGS**
   (Wu/Manning/Potts, TACL 2023) shows semantically-equivalent logical-form reformulation removes spurious
   string-edit/length sensitivities. So a grown-vs-injected verdict on *raw* COGS-structural is confounded
   — a grown method "failing" it may be failing the artifact, not composition.
2. **CFQ-MCD is worst-case compound COVARIATE SHIFT, not pure composition** (Keysers 2020 design) — a
   framing for which an *injected structural invariance is the cheapest possible fix*, **structurally
   biasing the benchmark toward injected winners**. Per Hupkes et al. (Nature MI 2023), "the hard split"
   conflates ≥5 distinct generalization types.

**Honest scope narrowing:** not "no grown mechanism solves compositionality," but **"no grown mechanism yet
solves systematicity-under-compound-shift (MCD) or recursion/productivity (COGS-struct/SLOG)."** This does
*not* overturn the bank — but it's the one place the search was not coverage-complete, and it's exactly
where the program's existing **training-distribution-entropy reopen-door (Wold 2025)** — a *soft* data-prior
that injects neither tree nor grammar — is most likely to bite.

## Mechanisms worth testing, to close the gaps found (ranked)

1. **★ Re-ground the bar before running any grown mechanism: ReCOGS + SLOG, MCD-as-covariate-shift.** The
   highest-value move. If a chunk of the floor is a decoder artifact, *every* grown-vs-injected verdict —
   including the program's banked ones — is partly confounded. Cheap, and it's about benchmark validity,
   not a new architecture.
2. **★ Test the entropy soft-prior (Wold 2025) on the artifact-cleaned bar.** The cleanest possible test of
   the program's thesis: a soft data-distributional prior (high training-entropy / atom-substructure
   augmentation, *no* held-out compounds) that could reduce compound divergence *without* a tree/grammar.
   Connects two previously-separate findings (the entropy reopen-door + the framing seam) into one decisive
   experiment. Leakage-audited (atoms shared, compounds disjoint).
3. **Multi-scale successor-representation replay (manufacture-by-recombination).** The neuro now supports
   *manufacture-by-overlap-recombination* better than pure stabilization; SR-Dyna/Wittkuhn build a
   multi-resolution predictive hierarchy by *reactivation* at multiple γ — buildable on the existing replay
   substrate. **Caveat (from our own repo): Bet A banked SR-nulls (Reports 124/125)** — so this is not a
   free lunch; the multi-scale/offline-replay form is the part not yet exercised.
4. **If still running the growth experiment — the 3 discriminators that make it decisive** (separate "grew a
   hierarchy" from "grew capacity"): (a) **tree-projection score** (Murty et al., ICLR 2023 — training-free,
   predicts compositional generalization) should *jump at the split event* and correlate with the
   systematic-iid-gap drop; (b) **grow-then-freeze + ablate-the-grown-level**, scoring systematic vs iid
   separately (a real level shows a large *selective* systematic drop); (c) **capacity-matched + split-timing
   control** (split-at-plateau vs split-at-random-step vs born-wide) — the leap predicts a *systematic*
   advantage specific to plateau-timed growth. Headline = **systematic-minus-iid gap**, never raw accuracy.
5. **SSM/Mamba on CFQ-MCD — a cheap, decisive empty-cell run.** The one grown architecture that could
   plausibly differ from UT (input-dependent selective state vs tied block); never run; theory (Merrill:
   SSMs lack permutation-composition state) makes it *predictive either way*. Laptop-sized on the harness.

## What this does to the precommit

The neuron-splitting grown-hierarchy precommit is **demoted and must be revised, not run as-is**:
- **Demoted** below (1)/(2) above — re-grounding the bar and testing the soft-prior are higher-value and
  cheaper, and the bar-confound means a splitting result on raw MCD would be partly uninterpretable.
- **Gate signal:** drop grokking measures → descent-stationarity primary + LLC/HTSR-α confirmatory.
- **Headline metric:** systematic-minus-iid gap + the tree-projection-jump discriminator + capacity/timing
  controls (else a positive is indistinguishable from "growth aided optimization").
- **Pretrain-atoms arm:** reframe as "gated-growth on top of the Report-149 0.297 atomic floor — beat 0.297?"
- **Buildability:** specify the Firefly gradient-grown variant or depth-stacking on the GRU, or a Transformer
  backbone; flag the build as non-trivial.
- **Bar:** run on ReCOGS/SLOG, not raw COGS-structural; treat MCD as covariate-shift.

These revisions are applied as a marked R2 block at the top of the precommit.
