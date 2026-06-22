# Verification & extensions (2026-06-22, 4-agent workflow)

*Follow-up to [findings.md](findings.md), run as a 4-agent orchestration: (1) adversarial
verification of the report's 9 load-bearing claims; (2) gap-hunt — library-learning / iterated
learning; (3) gap-hunt — other unconsidered mechanisms; (4) the user's pre-training-to-initialize
idea, developed into a design. Same discipline: claim → strongest disconfirmer → verdict; grown-vs-
injected is the arbiter; leakage + anti-homunculus checks.*

## 1. Verification verdict — the report holds

**All 9 load-bearing claims UPHELD. None refuted, none overclaimed at the verdict level.** The
headline recommendation (run measured stabilization-gated growth via neuron-splitting on the MCD/COGS
harness, against an injected-hierarchy ceiling + a scheduled-growth control) **survives at ~90%
confidence.** Spot-checks that held exactly: UT **18.9% ≈ vanilla 17.9%** on CFQ-MCD-mean (Keysers
2020 + the google-research/cfq leaderboard; per-split MCD1/2/3 = 37.4/8.1/11.3); LeAR **COGS 97.7 /
CFQ 90.9** riding an injected syntax↔semantics *homomorphism*; seq2seq **~0%** on COGS structural
splits; Firefly's own quotable caveat that splitting-at-stationarity *alone* stalls; the neuro
premise (stabilizer-not-manufacturer upheld; stabilize-N-gates-N+1 unestablished — cascade-of-critical-
periods is an explicit *hypothesis*, the 2025 cataract dissociation and Tse-2007-as-assimilation both
confirmed).

**4 precision fixes (applied inline to findings.md):**
1. **NDR (most material):** not "silent on CFQ" — it reports the *easy* CFQ **output-length** split
   (81%) and skips MCD. This *strengthens* the metric-shopping thesis and the NDR-on-MCD sub-experiment.
2. **Firefly** grows **width AND depth** (not width-only) — *understated* its relevance as a hierarchy
   grower; corrected.
3. **CSL** wording: the paper's claim is "T5+CSL-augmentation > T5–CSL ensemble," not "CSL-alone < ensemble."
4. **Lewis & Durrant "indistinguishable from Hebbian"** is labeled as *our reading*, not the primary's claim.

*Caveat (already flagged in findings.md):* most publisher PDFs / arXiv / ACL returned HTTP 403 to
direct fetch this session, so a few exact decimals (CSL's 90–91% MCD digit) are snippet-level, not
table-fetched. Corroborated across ≥2 independent results; confidence high.

## 2. Extensions — new candidates from the gap-hunts

| Candidate | grown/injected | role | buildable on harness | hard-split evidence | verdict / disposition |
|---|---|---|---|---|---|
| **DreamCoder wake-sleep library learning** (Ellis 2021) | grown | **hierarchy GROWER (real, level-by-level)** | partly — symbolic DSL search, **not** seq2seq | **untested** on SCAN/COGS/MCD | **grown-ceiling reference + loop blueprint** — see §3 |
| **Grokking progress-measures** (Nanda 2023) | — | **the GATE SIGNAL** | yes (instrumentation) | mod-arithmetic only | **PROMOTE — sharpens the headline's gate** |
| **Compositional energy-based composition** (Du/Liu) | injected operator | **the COMPOSE operator** (native to Hopfield/FHRR) | **yes, high fit** (energy summation ≈ 1 line) | vision-only, untested on hard splits | **ADD as the substrate-native compose op** |
| **Sparse autoencoders / dictionary learning** | grown | flat features | yes | **evidenced NULL** (2026 "Stop Probing, Start Coding": fails OOD compositional shift) | **CLOSED — door shut, not just untested** |
| **CIBP / cascading-IBP** (Adams 2010) | grown | genuine depth-grower | low (variational, off-harness) | untested | grown but **data-fit-gated, not stabilization-gated**; off-harness |
| **Neural cellular automata / DGCA** | grown | grows topology by local rules | low (grids/morphogenesis) | none symbolic | on-thesis-for-"grown" but zero symbolic-reasoning evidence |
| **Iterated learning / emergent comm** (Ren 2020; Vani 2021) | grown | **flat, NOT a grower** | yes (tiny) | toy referential games; VQA-SyGeT only; **no SCAN/COGS/MCD** | **DEMOTE** — Chaabouni 2020 null: emergent compositionality ⊥ generalization |
| **MLC** (Lake & Baroni 2023) | grown (meta-distribution) | flat | yes | **COGS *lexical* only, never MCD** | **metric-shopped like CSL** → offline-meta-distribution ceiling bucket |
| **Resonator networks / VSA** | injected (codebooks given) | flat factorizer | yes (native FHRR) | vision-scene only | the project's own substrate — a parser, not a grower |
| **Modular meta-learning** (Alet) | grown | composes flat modules | yes | relational/graph; no MCD | structure-search = borderline homunculus; demote |

**The two genuine upgrades to the headline experiment:**

- **★ Grokking progress-measures as the gate trigger.** This is the most valuable single addition. The
  headline needs a *measured, anti-homunculus* signal for "level N has stabilized → grow N+1." Splitting-
  at-stationarity (descent plateau) is one such signal; the grokking literature supplies a sharper,
  literature-grounded alternative — information-theoretic / restricted-loss "progress measures" that
  provably *rise before* a generalizing circuit crystallizes (Nanda 2023). Use it as the gate trigger
  (or as a second, convergent gate). It complements — does not replace — the neuron-splitting headline.
- **Compositional energy-based composition as the compose operator.** Du/Liu compose concepts by
  *summing energies* (product-of-experts) — and a Modern Hopfield network *is* an energy model, so this
  is a ~1-line, substrate-native way to *compose* the levels a grower produces. Not a grower itself
  (composition operator over given factors, vision-evidenced only), but the cleanest way to combine
  grown levels on the FHRR/Hopfield substrate.

**The big "missed candidate" — DreamCoder — and why it stays a reference, not a replacement.** It is
the only mechanism in either sweep that *demonstrably* grows reusable abstractions level-by-level (the
sourced filter→max→nth-largest→sort bootstrapping chain), and the original findings never named it.
But: (a) its "sleep" phase **manufactures** abstractions (mints new named functions via compression) —
the *opposite* of the project's well-supported "consolidation stabilizes, does not manufacture" half,
and its growth gate is "compress when enough tasks solved," not a measured-stability gate; (b) its
substrate is **symbolic DSL program-search**, not gradient seq2seq, and porting to MCD means supplying
a primitive DSL = re-injecting a grammar prior. So file DreamCoder exactly as CSL was filed — a
**grown ceiling + a blueprint for the wake-sleep loop *shape*** — not as the bet. Iterated learning /
emergent communication are flat (not hierarchical), have no hard-split evidence, and carry the
Chaabouni 2020 disconfirmer; they demote alongside latent-tree induction.

## 3. The pre-training-to-initialize direction (user's idea, developed)

**Verified leverage.** Pre-training (T5-11B) lifts CFQ-MCD from **17.9% → 40.9%** — doubles vanilla,
but stays far below injected methods (HPD 67%, LeAR 90%). Crucially, pre-training's *large* wins are
on **COGS-lexical / SCAN-primitive** (atom-substitution) splits, **not** the compound-divergence
splits that define the bind (Furrer 2020 is the canonical "pre-training helps but doesn't solve, and
*hurts* ~8.5% on SCAN-length"). MLC is a *meta-distribution*, not a pretrained init (and SCAN/COGS
only, never MCD); MAML/Reptile give fast adaptation, not systematicity; "Break It Down" (Lepori 2023)
shows pretraining makes emergent modularity *more reliable* but still fragile.

**The load-bearing distinction (this is what keeps it out of the bind).** A prior can inject two
separable things: **(a) atoms/representations** or **(b) the composition rule** (tree / grammar /
homomorphism). Every method that wins CFQ-MCD injects **(b)** — that is *the* result. **Pre-training on
atoms injects (a) but leaves (b) to be grown.** So "pretrain-atoms → grow-composition" sits as *grown
with an injected atomic floor* — a strictly weaker injection than every method in the injected tier,
and exactly the §4½ picture (stabilize level-0 atoms; grow the composition over them). It re-enters
the bind **only** if you pretrain on parse-structured intermediate representations (that injects (b)).
**Knife-edge: pretrain representations, not rules.** Bonus: a pre-stabilized level-0 is a non-degenerate
plateau to split from — plausibly curing the **Firefly cold-start stall**.

**Disposition: a COMPONENT of the headline experiment, not a standalone reopen-door.** Two honest
risks: at the toy scale the 40.9% is an 11B-param result that may not transfer; and "pretrain atoms
then compose" may *be* the project's already-banked atomic-injection ~2× result (Report 149b)
relabeled — a redundancy risk.

## 4. Integrated decisive experiment (headline, now enriched)

The headline survives and gets three concrete upgrades folded in: the **grokking progress-measure** as
a sharper gate trigger, **energy-summation** as the substrate-native compose operator, and
**pre-trained atoms** as an optional level-0 seed. The pre-training agent's 4-arm design is the clean
way to test the seed's contribution. On the SCAN-MCD / COGS-structural harness (`experiments/87/94/99`),
≥8 seeds, log-FTSR-style bootstrap CIs, pre-registered:

- **A. From-scratch flat baseline** — the ~18% MCD / ~0–12% COGS-structural floor.
- **B. Pretrained-atoms init → flat training** — isolates how much the atomic floor alone buys.
- **C. Pretrained-atoms init → stabilization-gated neuron-splitting growth** (gate = descent-stationarity
  and/or a grokking progress-measure; compose via energy-summation) — the experimental arm.
- **D. From-scratch → stabilization-gated growth** — the findings.md headline as-is (cold start);
  C-vs-D is the registered test of whether the seed is load-bearing.
- **Ceiling: injected hierarchy** (Tree-LSTM / LeAR) ~90% CFQ-MCD / ~97% COGS — the gap marker.

**Pre-registered reads.** *C > D and C > B (CI-disjoint)* → pretraining-on-atoms is a load-bearing
foundation growth can't bootstrap alone (vindicates "init as level-0"; Firefly cold-start is the
mechanism). *C ≈ D* → the seed is redundant with what growth discovers — fold in for efficiency, don't
claim a reopen (and beware it's just 149b relabeled). *all of A–D far below ceiling* → the whole grown
family stalls on the discriminating arena and only injected-(b) clears the bar — a **clean earned
negative** that closes the grown door and hardens the bank. *B lifts but C/D don't exceed B* →
pretraining helps but the growth gate adds nothing — demotes the headline mechanism, not the seed.

**Anti-homunculus seam to watch:** the "level-0 done → start growing" handoff must NOT be a supervisor
reading an accuracy threshold — it must be the *same measured stationarity/progress gate*, or the
design has smuggled in a homunculus. **Leakage guard:** audit compound-disjointness mechanically (MCD
shares atoms by construction, so atom-overlap is fine; any test *compound* — even as a subtree — in
pretraining is disqualifying), plus a shuffled-atom control that must NOT help.

## 5. Net

The original findings are **verified and strengthened, not overturned.** The headline bet (measured
stabilization-gated growth, the one empty cell in the grown × tested-on-hard-splits matrix) stands,
now equipped with a sharper gate (grokking progress-measures), a substrate-native compose operator
(energy summation), and an optional pre-trained-atom level-0 seed with a registered test of whether
that seed is load-bearing or redundant. DreamCoder is the most important *conceptual* addition (the
wake-sleep loop shape) but is filed as a grown-ceiling reference, not the bet, because it manufactures
rather than stabilizes and lives off the seq2seq harness. The single biggest remaining risk is
unchanged and honest: every disconfirmer points the same way — pure-emergent/grown hierarchy keeps
failing the hard splits — so the integrated experiment is designed to be **decisive either way**, with
a clean earned negative as a fully acceptable, bank-hardening outcome.
