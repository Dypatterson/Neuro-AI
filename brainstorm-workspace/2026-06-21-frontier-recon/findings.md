# Frontier recon — did we bank too early? (2026-06-21, deep-research team: 110 agents, ~4.7M tokens)

*Adversarial sweep of the 2024–2026 frontier against the program's banked conclusion ("every method
clearing SCAN-MCD/COGS/CFQ injects structure; no emergent/brain-shaped mechanism wins"). Job: hunt for the
counterexample, verify hype against primaries. The per-claim 3-vote verification logs follow below; this is
the synthesis.*

## Bottom line: the bank SURVIVES the frontier — with one genuine reopen-door

No clean emergent, non-structure-injecting mechanism solves the hard held-out compositional splits
(SCAN-MCD / COGS / CFQ) as of 2024–2026. Every candidate that improves on them either injects structure
(architecturally or via a meta-training/curriculum distribution), is bounded to easier benchmarks
(SCAN-only / synthetic CFGs), or is a per-task test-time fit on a *different* benchmark (ARC-AGI) that does
not compound transfer. **We did not bank too early.**

## Verdict table

| Candidate | Verdict | Strongest disconfirmer |
|---|---|---|
| **MLC** (Lake & Baroni, Nature 2023) — the headline "emergent" hope | **STRUCTURE-INJECTION-IN-DISGUISE + bounded** | ~100k engineered grammars; only SCAN/COGS, *never MCD/CFQ*; collapses 86.7→21.0% when the compositional meta-distribution is removed (Compositional-ARC, arXiv 2504.01445) |
| **Scale / in-context learning** on COGS/CFQ/MCD | **REDUNDANT / UNVERIFIED** | no contamination-clean emergent win on the hard splits; Xu et al. COLM 2024: parameter-scale (no-CoT ICL) gives *no* gain on compose-by-steps |
| **Test-time training** (ARChitects 2024, NVARC 2025) | **GENUINE emergent in-weight adaptation, but low ceiling + no compounding** | best 2025 system = **24%** on ARC-AGI-2; per-task ARC fit, not SCAN-MCD/COGS/CFQ, no cross-task transfer |
| **CompressARC** (MDL, no pretraining, Dec 2025) | **ORTHOGONAL / supports the bank** | per-task fit + hand-engineered equivariances; the *opposite* of compounding-transfer |
| **Neuroscience: consolidation manufactures abstraction?** | **UNESTABLISHED in this recon** | no 2023–2026 primary surfaced that shows replay/consolidation *manufactures* (vs stabilizes) abstraction — neither confirmed nor refuted |

## The honest nuance (the recon corrected ME, not just the field)

- **"TTT = structure injection" is REFUTED.** TTT's core is *genuine emergent in-weight adaptation*: with
  ZERO geometric augmentation, per-task LoRA still lifts 5→13 ARC tasks (Akyürek 2024, Fig 3); the
  augmentations are a data-efficiency booster, not a symbolic-structure injection. The honest read is "TTT
  is a real emergent adaptation with a low hard-abstraction ceiling," not "TTT cheats via injected structure."

## The two reopen-doors (emergent, non-structure-injecting, but UNTESTED on the hard splits)

1. **★ Training-distribution ENTROPY** (Wold, Charpentier, Simon — ACL Findings 2025, arXiv 2505.13089). A
   **vanilla seq2seq generalizes systematically when training-distribution entropy is high — no architectural
   prior, no structure injection.** This is the single most "emergent, non-structure-injecting" candidate in
   the whole sweep. Caveat: demonstrated **only on a custom synthetic SCAN-CFG**, never COGS/CFQ/MCD. **It has
   never been run on the hard splits — and we have the exact harness to do it.** A real test-or-falsify lever.
2. **ICL-as-inductive-bias from data-distributional properties** (Chan et al. 2022, arXiv 2205.05055): ICL
   emerges from data burstiness / many-rare-classes, not a hand-built curriculum. Emergent, but the
   compositional-generalization version (Han & Padó 2024) is so far an *engineered* meta-training regime on
   small models — not yet shown emergent-at-scale on MCD/COGS/CFQ.

## The most promising + most honest reopen-door

**Training-distribution entropy**, run on *our* MCD/COGS harness with a vanilla seq2seq. The non-leaky
version: raise entropy over atoms/sub-structures **without exposing the held-out compounds** (else it's
leakage). Outcome is decisive either way — if a vanilla high-entropy model cracks MCD, the bank reframes to
"structure-injection OR sufficient data-entropy" (a genuine emergent escape); if it nulls, the bank hardens.
Laptop-sized, reuses `experiments/87`/`94`/`99`.

## Second thread worth a deeper dive

The **neuroscience premise gap**: this recon found no primary backing "consolidation *manufactures*
abstraction" — the biological foundation the whole program leaned on. Caveat: *absence in this recon ≠ absence
in the literature* (the sweep may have missed Tse et al. schema-consolidation, sleep-and-insight/generalization
work, Lewis & Durrant, the Tononi-Cirelli synaptic-homeostasis line). A dedicated neuro deep-dive could
either shore up or further undercut the premise — and is the kind of thing that should have been pinned down
*before* the build, not after.

---

## Voter 2/3 (adversarial) — claim verification: MLC evaluation scope (CFQ/MCD)

**Claim:** MLC (Lake & Baroni 2023) was evaluated on SCAN + COGS + a few-shot grammar task — NOT on CFQ or MCD held-out splits. Therefore the MLC escape is bounded: it has not been shown to clear the MCD/CFQ splits the banked conclusion centers on.

**Verdict: NOT REFUTED (claim stands).** Tried to refute on three axes; all failed.

- Quote-support: Multiple independent sources (Nature paper summaries, brendenlake/MLC repo, the 2024 arxiv 2404.13074 semantic-parsing survey, NeurIPS-era SCAN/COGS literature) consistently report MLC's ML-benchmark evaluation as **SCAN + COGS only** (plus the human few-shot grammar task). The companion code repo explicitly scopes itself to "SCAN and COGS." No source places MLC results on CFQ or MCD splits.
- Contradiction search: Found none. No 2023–2026 source reports an MLC number on a CFQ or SCAN-MCD split, neither by Lake/Baroni nor by a follow-up replicating MLC on those splits. (Keysers 2020 owns MCD/CFQ; MCD mean accuracy <20% for standard seq2seq architectures — the very regime the bank centers on, and MLC is absent from it.)
- Source quality: Sufficient. The claim is a *negative scoping* claim (no eval on X), which the primary paper + official code repo + an independent survey jointly support. Negative claims of this kind are appropriately evidenced by absence across the primary + secondary corpus.

**Caveat (does not refute, but qualifies):** the claim is correct that MLC's *published* evals exclude CFQ/MCD, but it should not be read as "MLC provably fails MCD" — it's "untested on MCD," which is a weaker, honest bound. The claim as written says exactly that ("has not been shown to clear"), so it is accurate, not an overreach. SCAN-MCD ≠ the open-vocab SCAN split MLC targets, so MLC's SCAN success does not transfer credit to MCD.

Strongest source: Lake & Baroni, *Nature* 623:115–121 (2023) + brendenlake/MLC repo (SCAN/COGS scope). Strongest (attempted) disconfirmer: none found — search for MLC-on-MCD/CFQ returned zero results.

---

## Voter 3/3 (adversarial) — claim verification: Wold et al. 2025 (entropy paper) evaluation scope

**Claim under review:** The result in arxiv 2505.13089 is demonstrated ONLY on a modified synthetic SCAN grammar; it is NOT evaluated on COGS, CFQ, or MCD splits, so it does not constitute a counterexample to the bank's claim about the harder MCD/COGS/CFQ leaderboards.

**Paper:** "Systematic Generalization in Language Models Scales with Information Entropy" — Wold, Charpentier, Simon (Univ. of Oslo), ACL Findings 2025 (also arxiv 2505.13089 v1/v2).

**Verdict: NOT REFUTED (claim stands).** Tried to refute on all five checklist axes; all failed.

- **Quote-support (check 1):** The claim is fully supported, not an overreach. The paper's own text: "we define a context-free grammar (CFG) based on the SCAN grammar (Lake and Baroni, 2018)... we set |V| to 8, as opposed to the original four verbs." It builds a *custom* synthetic CFG with an *entropy-based* train/test split (train support excludes v₁; test support includes v₁) — NOT the standard SCAN add-jump/length split and NOT COGS/CFQ/MCD. Two independent WebFetch passes over the full HTML both returned "No evaluation on COGS, CFQ, or MCD anywhere in the paper."
- **Contradiction search (check 2):** None found. The paper *references* COGS and CFQ in related work as sequence-prediction compositional benchmarks, but never reports a number on them or on MCD. No source claims otherwise.
- **Source quality (check 3):** Sufficient and strong — the disconfirmer here is the PRIMARY source itself (the paper's experimental section), plus the ACL Anthology published version (aclanthology.org/2025.findings-acl.90). A negative-scope claim ("not evaluated on X") is exactly what the primary text settles.
- **Recency (check 4):** Not outdated — paper is May 2025 / ACL 2025, current.
- **Marketing/cherry-pick (check 5):** N/A — peer-reviewed ACL Findings paper; the claim is a sober scoping statement, not hype.

**Qualifier (does not refute):** The entropy paper is an interesting *data-centric* result (systematicity scales with training-data entropy, no architectural prior needed), and is arguably the most "emergent, non-structure-injecting" candidate in this recon. But the claim under review is narrowly about *evaluation scope*, and on that narrow point it is correct: a custom low-entropy SCAN-CFG win is NOT a result on the MCD/COGS/CFQ leaderboards the bank centers on. It neither overturns nor reopens the bank on its own evidence; it would need a follow-up run on real MCD/COGS/CFQ to count.

Strongest source: arxiv 2505.13089 v2 experimental section + ACL Findings 2025 published version (own text, custom SCAN-CFG + entropy split). Strongest (attempted) disconfirmer: none found — no version of the paper and no third party reports a COGS/CFQ/MCD number for it.

---

## Adversarial verification note — voter 1/3 — Han & Padó 2024 (arXiv:2403.11834)

**Claim reviewed:** "Causal Transformers do NOT achieve compositional generalization by ordinary supervised learning; they only improve when training is deliberately structured to force in-context learning (shuffled labels + reordered instances)... ICL-as-inductive-bias is an engineered training regime, not an emergent property of scale or standard training."

**Verdict: REFUTED (overreach beyond the source).** The primary is real and the quote is verbatim, but the claim over-generalizes the paper in two load-bearing ways:

1. **"do NOT achieve ... at all" is too strong.** The paper says Transformers *"struggle"* — the standard supervised baseline scores **21.8% on SCAN-MCD1** (nonzero) vs **60.4%** for the meta-trained model. A relative gap, not a categorical failure.
2. **"NOT an emergent property of scale or standard training" is contradicted** by (a) the paper's OWN acknowledgment that ICL emerges at scale ("only some in-context-learning LLMs can compositionally generalize and only as they scale up"; it cites "improvement in compositional generalization for large Transformer-based LLMs"), and (b) the established literature: GPT-3 (Brown 2020) shows ICL emerging from scale with no explicit ICL objective, and Chan et al. 2022 (arXiv:2205.05055) shows ICL emerges from data-distributional properties (burstiness, many rare classes). The paper positions its engineered meta-training as a **controlled study mechanism**, NOT as the only route to ICL.

**Strongest disconfirmer:** Chan et al. 2022 (arXiv:2205.05055) — ICL is an emergent property of training-data distribution, not a hand-engineered regime. Plus the paper itself frames scale-induced ICL as the real-world analog.

**Caveats relevant to the bank:** benchmarks are SCAN/COGS/**GeoQuery** (NOT CFQ as the question framed); models are **small from-scratch** Transformers (plus a GPT-2 follow-up), so the result does not speak to large-scale emergence. The engineered ICL regime is itself a **structure-injecting curriculum** (all-few-shot-problems meta-training), which is consistent with — not a counterexample to — the bank's "no emergent escape without structure injection" thesis.

---

## Voter 3/3 (adversarial) — claim verification: Han & Padó 2024 (arxiv 2403.11834)

**Claim under review:** "Causal Transformers do NOT achieve compositional generalization by ordinary
supervised learning; they only improve when training is deliberately structured to force in-context
learning (shuffled labels + reordered instances). ICL-as-inductive-bias is an engineered training
regime, not an emergent property of scale or standard training."

**VERDICT: NOT REFUTED (claim stands).** Confidence: HIGH.

### Why it survives the refutation attempt
- **Quote→claim fidelity:** the supporting quote is verbatim from the abstract and directly describes the
  engineered regime (shuffle labels + reorder instances = "all possible few-shot learning problems"). No misread.
- **Ordinary training fails (verified numbers):** baseline causal Transformer SCAN-MCD1 21.8%, MCD2 25.6%,
  COGS 51.9%. Meta-ICL regime: 71.2% / 74.8% / 75.7%. The improvement is gated on the engineered regime.
- **Engineered, not emergent (verified in conclusion):** paper says it "explicitly incentivize[s] in-context
  learning" via a deliberately constructed meta-training distribution. It does NOT observe ICL emerging from
  scale or standard training — it has to force it. This is exactly the claim's load-bearing assertion.
- **Source quality:** primary, peer-reviewed (LREC-COLING 2024), recent (2024). Strength matches claim.
- **Independent corroboration (not just self-citation):** Lake & Baroni MLC (Nature 2023) — same finding,
  meta-learning curriculum is the active ingredient, standard training does not get there. "When can
  transformers compositionally generalize in-context?" (arxiv 2407.12275, 2024) — ICL compositional
  generalization needs an engineered bottleneck separating task-inference from task-execution; does not
  arise from vanilla training. COGS original (Kim & Linzen) — standard Transformers: 96-99% in-dist,
  16-35% gen. The "ordinary training fails" half is the field consensus, not a contested claim.

### Honest caveats (recorded; do NOT flip the verdict)
- The claim's word "only" is marginally stronger than the paper. The engineered regime IMPROVES but does
  not SOLVE: MCD splits "remain difficult," COGS was *matched* not beaten, GeoQuery gain was marginal
  (40.8 vs 37.4). The claim correctly asserts the regime is engineered — it does NOT claim the benchmark is
  solved — so this caveat does not contradict the claim as written.
- Scope: this is one architecture (causal Transformer) on synthetic benchmarks, empirical (no theory). The
  claim does not over-generalize beyond that.

### Relevance to the bank
This is a SUPPORTING data point for the program's banked conclusion, not a reopen-door. ICL-as-compositional-
generalization is an *engineered training regime* (meta-learning curriculum injecting the few-shot/in-context
inductive bias) — i.e., a structure-injection-in-disguise at the training-distribution level, NOT an emergent
escape from scale or standard learning. It does not constitute a brain-inspired emergent mechanism that beats
known methods; it IS a known-method-class (Lake-Baroni meta-learning) result.

**Strongest source:** Han & Padó 2024, arxiv 2403.11834 (LREC-COLING), abstract + results + conclusion.
**Strongest disconfirmer found:** none that overturns; only the "only/solved" nuance above, which the claim
does not actually assert.

---

## Voter 2/3 (adversarial) — claim verification: Xu/Shi/Liang 2024 (arXiv:2407.15720) "scaling gives NO improvement on hard composite tasks"

**Claim under review:** "On complex composite tasks requiring multi-step reasoning (each step = one sub-task), LLMs underperform and scaling up model size provides NO improvement — directly contradicting the 'scale alone solves compositionality' hypothesis for the hard, systematic case."

**Paper:** "Do Large Language Models Have Compositional Ability? An Investigation into Limitations and Scalability" — Zhuoyan Xu, Zhenmei Shi, Yingyu Liang. COLM 2024 / arXiv:2407.15720 (Jul 2024). Quote verbatim-confirmed in primary HTML.

**VERDICT: REFUTED (partial overreach — narrow finding real, strong framing not).** Confidence: MEDIUM. The quote is accurate, but the claim inflates a *task-type-conditional, no-CoT, in-context-learning* result into a general anti-scaling refutation, and contradicts a strawman.

- **Quote-support (check 1) — OVERREACH, the decisive failure.** The claim suppresses the paper's other headline half. The paper splits composite tasks into **"compose by parts"** (f(x),g(y) on disjoint input segments) — where scaling **DOES help** ("the models demonstrate decent compositional ability, while scaling up the model enhances this ability") — and **"compose by steps"** (f(g(x)) multi-step) — where scaling does not. The claim quotes only the negative half and presents "scaling provides NO improvement" as the result. Its hedge "for the hard, systematic case" partially rescues this, but the headline phrasing reads as a general anti-scaling claim the source does not make.
- **Setup is a narrow elicitation regime.** Protocol = in-context learning with only SIMPLE-task demonstrations and **no chain-of-thought / no fine-tuning**. The "no scaling benefit" finding is conditional on excluding CoT and test-time compute — the mechanisms the field's actual scaling story relies on. CoT literature + theory (each CoT token can encode an FSA state transition) shows the same multi-step composition becomes tractable with CoT. So the result is "parameter-scale, no-CoT, ICL gives no benefit," NOT "scale gives no benefit."
- **Strawman contradiction (checks 2–3).** Almost no serious 2024–2026 position holds raw parameter count with no CoT solves systematic multi-step composition; the live scaling axis the field invokes is **test-time-compute / CoT scaling**, which this paper never tests. Contradicting a hypothesis nobody defends, with a setup that omits the relevant scaling variable, is not a "direct contradiction" of the real scale story.
- **Theory scope is narrow.** Supporting theory uses a **simplified linear self-attention** model assuming inputs decompose into **disjoint subspaces with confined support** — not general transformers. Single COLM-2024 paper; fine for the narrow empirical finding, thin for the sweeping framing.
- **Recency (4):** current (2024). **Marketing (5):** N/A — legitimate paper; overreach is in the claim's restatement.

**What survives:** the narrow form — "in a no-CoT ICL setting, parameter-scaling alone does not improve compose-by-steps multi-step tasks" — IS supported (would be NOT-REFUTED). As written, with "NO improvement" generalized and "directly contradicting scale-alone for the hard case" asserted, the claim overreaches on two axes (drops the compose-by-parts-scales half; omits the CoT/test-time-compute scaling axis).

**Relevance to the bank:** this paper actually *supports* the bank's thesis (parameter-scale alone is not an emergent escape) — but it is NOT evidence against the CoT/test-time-compute scaling story, and never touches SCAN-MCD/COGS/CFQ leaderboards. Cannot be used to bank "scale is dead for compositionality" in general.

**Strongest source:** arXiv:2407.15720 (COLM 2024) own text — compose-by-parts (scales) vs compose-by-steps (doesn't) split + no-CoT ICL protocol. **Strongest disconfirmer:** the paper's own compose-by-parts result ("scaling up the model enhances this ability") + CoT-scaling literature showing multi-step composition becomes tractable once CoT/test-time compute is allowed.

---

## Adversarial claim check (voter 2/3) — 2026-06-21

**Claim under review:** "LLMs only show 'decent' compositional ability when the composite task can be
decomposed by applying distinct mappings to different input SEGMENTS (positionally separable), not when
genuine sequential recombination is needed — a structure-injection-in-disguise signal, since the
'composition' that works is the trivially separable kind."

**Source:** arXiv:2407.15720, *Do Large Language Models Have Compositional Ability? An Investigation into
Limitations and Scalability* (Xu et al.). ICLR 2024 Workshop ME-FoMo poster; arXiv v1 2024-07-22, v2
2024-08-11. ICL few-shot, no-CoT regime; logical/linguistic composite tasks (Cap+Swap, Cap+TwoSum,
Past+PlusOne, Phrase Recombination, etc.).

**VERDICT: REFUTED (partial — the empirical core is real, but the claim's load-bearing INTERPRETATION
overreaches the source on three counts).**

### What IS supported
The paper's headline finding genuinely is the separable-vs-sequential dichotomy: models show "decent"
composition on tasks that apply distinct mappings to distinct input segments and improve with scale;
they underperform on multi-step/sequential tasks and scale does not help. The theory grounds this in
"confined support" — each subtask's input embedding lives in a disjoint feature subspace with low
cross-covariance (Assumption 1). So "positionally separable composition works, sequential does not" is
faithful to the abstract.

### Why the claim is still REFUTED — three overreaches

1. **Misreads the paper's own framing of "trivial."** The claim says the separable success is "the
   trivially separable kind," implying the paper dismisses separable composition as a non-result. The
   paper says the OPPOSITE about a key sub-case: *"This basic generalization seems trivial, yet we
   observe that LLMs fail to generalize in this way."* The paper treats even simple separable
   generalization as a real, non-trivial capability that LLMs often LACK — not as a freebie they always
   get. "Decent" ≠ robust; the paper documents failures even on separable tasks. The claim's "the
   composition that works is the trivially separable kind" is an editorialization the paper does not make.

2. **"Structure-injection-in-disguise" is the claim-author's gloss, not in the source.** The paper never
   uses "structure injection" and offers no argument that separable-segment success is a disguised form
   of injected structure. Its mechanism ("confined support" / disjoint feature subspaces) is an *emergent*
   property of self-attention over separable inputs, not externally injected symbolic/grammatical
   structure. Bolting the project's "structure-injection" framing onto this paper is unsupported.

3. **Scope/recency overreach.** The result is specifically the **ICL few-shot, no-CoT** regime; the paper
   explicitly does NOT test chain-of-thought, and 2024–2026 literature (CoT as an emergent ability; RL-
   trained reasoners o1/o3/R1; latent multi-hop work arXiv:2402.16837) shows the sequential/multi-step
   ceiling is method-dependent, not a fixed property of "LLMs." So "LLMs only show composition on
   positionally-separable tasks" is too strong as a general statement; it holds for plain frozen-weight ICL.

### Strongest source / strongest disconfircer
- **Strongest source (for the empirical kernel):** arXiv:2407.15720 abstract + theory section.
- **Strongest disconfirmer (against the claim's interpretation):** the paper's own line
  *"This basic generalization seems trivial, yet we observe that LLMs fail to generalize in this way"* —
  directly contradicting "the composition that works is the trivially separable kind," plus the absence
  of any "structure injection" framing in the source.

### Bearing on the bank
Does NOT, by itself, reopen the structure-injection bank — if anything it is *consistent* with the bank's
spirit (emergent ICL composition is brittle and breaks on entangled/sequential recombination). But it is
NOT clean evidence FOR "structure-injection-in-disguise," because the paper's mechanism is emergent
(confined support), not injected structure, and the regime is narrow (no-CoT ICL). Treat as
REDUNDANT-WITH-KNOWN at best, not a positive signal for the claim as written.

---

## Voter 1/3 (adversarial) — claim verification: CompressARC is pure per-task / test-time learning (no cross-task compounding)

**Claim under review:** "CompressARC trains on only the single target inference puzzle, with no transfer from a training corpus — making it a pure per-task / test-time-learning method rather than one that compounds structure across tasks."

**Source:** arXiv:2512.06104, *ARC-AGI Without Pretraining* (Isaac Liao, CMU). Supporting quote: "CompressARC is the only deep learning method for ARC-AGI where training happens only on a single sample: the target inference puzzle itself."

**VERDICT: NOT REFUTED (claim stands).** Confidence: HIGH. Tried all five checklist axes; none flips it.

### Why it survives
1. **Quote→claim fidelity (check 1).** The quote directly supports the claim, no overreach. The paper's own three-bullet characterization is even more explicit: **"No pretraining"** (randomly initialized, trained only at test time), **"No dataset"** (one model trains on a single target task and produces one answer), **"No branching search"** (gradient descent only). The abstract separately states *"CompressARC does not train on the pre-provided ARC-AGI 'training set'."* Each puzzle gets its own independent network weights optimized at inference from the target puzzle (solution removed). That is exactly "pure per-task / test-time-learning, does not compound structure across tasks." Verified verbatim against both the arxiv HTML (arxiv.org/html/2512.06104v1) and the abstract.
2. **Contradiction search (check 2).** Found none. No source disputes the single-sample / no-training-set characterization — it is the paper's central selling point and is reported identically by the ARC Prize 2025 technical report (arxiv 2601.10904) and the author's blog. 76K params, ~20% eval / 34.75% train, ~20 min/puzzle on an RTX 4070.
3. **Source quality (check 3).** Primary (the arxiv paper itself + author blog + independent ARC Prize technical report). Matches the claim's strength — the claim is a *narrow methodological* claim, not an extraordinary capability claim.
4. **Recency (check 4).** Current (Dec 2025 arxiv; ARC Prize 2025 cycle). Not outdated.
5. **Marketing/cherry-pick (check 5).** N/A to the claim under review. The 20% headline could be debated, but the claim is only about *training regime* (single-sample, no corpus), which is a factual architectural property, not a benchmark boast.

### Honest caveat (recorded; does NOT flip the verdict)
The *architecture itself* is heavily hand-engineered domain knowledge that IS shared across all puzzles — custom ops (cummax, directional shifts, multitensors), hardcoded equivariances (rotations/flips/color-and-example permutations), authors' own word "heavily engineered." The paper even notes the architecture is amortized across puzzles in the description-length template ("the architecture definition only appears once in Algorithm 1 while seeds appear repeatedly"). So "no transfer of ANY kind" would be too strong — there is human-injected structural prior baked into the net. BUT the claim under review does not assert that: it asserts no transfer *from a training corpus* and no *compounding of learned structure across tasks*, and on that narrow point the claim is exactly correct (no learned-weight transfer; per-task weights are independent). The "no branching search" bullet is also mildly misleading (gradient descent is directed search through weight space), but again is not part of the claim under review.

### Bearing on the bank
This claim, if anything, **supports** the program's banked conclusion rather than reopening it. CompressARC is a *per-task test-time-learning* method (MDL/VAE compression of a single puzzle) — it is **NOT** a continual / compounding-transfer mechanism, so it cannot serve as a counterexample to "no emergent mechanism compounds structure across tasks." It is the opposite of compounding-transfer: it explicitly refuses any cross-task learning. It is also not a SCAN-MCD/COGS/CFQ result (different benchmark, ARC-AGI). So as a frontier-recon candidate it is TOO-NEW/ORTHOGONAL — interesting as a pure test-time-compression-as-learning datapoint, but not a reopen-door for the compositional-generalization or compounding-transfer bank, and it injects hand-designed equivariance structure rather than being a clean emergent escape.

**Strongest source:** arXiv:2512.06104 own text (abstract + "No pretraining / No dataset / No branching search" + Algorithm 1 template discussion). **Strongest (attempted) disconfirmer:** the hand-engineered, equivariance-baked architecture shared across puzzles — which qualifies "no transfer" in general but does NOT contradict the narrow per-task / no-corpus claim as written.

---

## Adversarial verification note — voter 1/3 — Dong et al. 2024 (arXiv:2402.15938)

**Claim reviewed:** "When an LLM is exposed to leaked benchmark data, performance keeps rising on the leaked
items but stagnates or degrades on held-out similar items — a direct memorization-vs-generalization
dissociation that explains how compositional benchmark scores can be inflated."

**Paper:** Dong, Jiang, Liu, Jin, Li — "Generalization or Memorization: Data Contamination and Trustworthy
Evaluation for Large Language Models" — ACL 2024 Findings (arXiv:2402.15938). Proposes CDD (contamination
detection via output-distribution peakedness) + TED (trustworthy eval via distribution correction).

**Verdict: NOT REFUTED (core dissociation stands); one scoping caveat on the "compositional" clause.**
Confidence: HIGH on the core claim, MEDIUM on the compositional extension.

- **Quote-support (check 1):** Verbatim and accurate. Paper intro: "their performance keeps improving on
  leaked data but stagnates and even degrades on similar data." Backed by a *controlled* experiment
  (Figure 1: CodeLlama fine-tuned on HumanEval-leaked + 50K StarCoder, swept 0-20 epochs) — a genuine
  memorization-vs-generalization dissociation, not mere observation. The claim's first two clauses are an
  exact paraphrase.
- **Contradiction search (check 2):** None found that overturns. The phenomenon is corroborated by the
  broader contamination literature (LiveCodeBench, LLMSanitize/arXiv:2404.00699, contamination surveys
  arXiv:2406.04244 / 2502.14425): contamination inflates scores ~10-30%, controlled-for LLMs lean on
  memorization. Field consensus, not a contested claim.
- **Source quality (check 3):** Strong — primary, peer-reviewed (ACL 2024 Findings), controlled experiment
  across 24 settings x 21 contamination degrees, plus real-world ChatGPT/HumanEval evidence. Matches the
  claim's strength.
- **Recency (check 4):** Current (2024). Not outdated.
- **Marketing/cherry-pick (check 5):** No — sober peer-reviewed methods paper.

**Scoping caveat (qualifies, does not refute):** the paper's evidence is on HumanEval (code) and GSM8K
(math reasoning) — NOT on compositional-generalization splits (SCAN/COGS/CFQ/MCD). The claim's last clause
("explains how *compositional* benchmark scores can be inflated") is an EXTRAPOLATION, not a demonstrated
result. The hedge "can be inflated" (modal, not "are") keeps it honest, and the mechanism is plausibly
transferable, but no source verifies leakage-driven inflation specifically on MCD/COGS/CFQ. Treat the
compositional application as UNVERIFIED-BUT-PLAUSIBLE, not established.

**Relevance to the bank:** SUPPORTING, with a methodological warning. This is exactly the dissociation the
program must guard against when a literature "escape" claims a held-out compositional win — the bank's
discipline (distinguish a real held-out result from a leak/contamination artifact) is *vindicated*, not
threatened, by this paper. It is not itself a reopen-door.

**Strongest source:** Dong et al. 2024, arXiv:2402.15938 (ACL Findings) — intro statement + Figure 1
controlled contamination experiment. **Strongest disconfirmer found:** none on the core dissociation; the
only weakness is that the "compositional" clause is extrapolated beyond the paper's HumanEval/GSM8K evidence
(no MCD/COGS/CFQ test exists).

---
## Adversarial verification log (voter 3/3) — CompressARC MDL claim

**Claim reviewed:** "The mechanism is a generic MDL (compression) objective, not a symbolic DSL or program-synthesis search, and the authors attribute its generalization to the MDL objective itself." (Source: arXiv:2512.06104, Liao & Gu, "ARC-AGI Without Pretraining", Dec 2025.)

**VERDICT: refuted=true (partial overreach), confidence=medium.**

Part 1 (mechanism = MDL, not symbolic DSL/program search): TRUE. Paper minimizes description length via a VAE-style loss + gradient descent at inference; "no search ... just gradient descent." No discrete DSL / combinatorial program synthesis. Solid.

Part 2 (authors attribute generalization to the MDL objective ITSELF): OVERREACH. The supporting quote is from the ABSTRACT ("The MDL endows CompressARC with extreme generalization abilities typically unheard of in deep learning"). But the paper BODY attributes generalization heavily to a hand-engineered EQUIVARIANT architecture: "The most important feature of our architecture is its equivariances" (example-permutation, color-permutation, rotation, flip); plus directional cummax/shift/multitensor layers "designed specifically for the purpose of conferring those abilities." NO ablation isolates MDL from the architecture. So MDL-alone-on-a-generic-net is not demonstrated to generalize.

**Strongest disconfirmer:** the paper's own Section-4 framing — generalization is engineered by "trading off architecture description length to allow for shorter seeds," i.e. the equivariances ARE a structural prior doing load-bearing work. Equivariance to symmetry groups is exactly structure-injection (a baked-in symbolic-geometric prior), just not a DSL.

**Relevance to the bank:** Even if accepted, this is ARC-AGI (20% eval), NOT SCAN-MCD/COGS/CFQ. No evidence it touches the compositional-generalization benchmarks at issue. And "no search/no pretraining" still rides a per-puzzle gradient-descent fit + symmetry-equivariant architecture = structure-injection-in-disguise for the question this program cares about. Does NOT reopen the door as an "emergent, non-structure-injecting" escape.

---

## Claim verification (voter 2/3, adversarial) — 2026-06-21

**Claim under review:** "The 2025 1st-place winner (NVARC) achieves only 24.03% on the
private ARC-AGI-2 evaluation set, far below the ~97-98% human baseline, showing that
test-time training + synthetic data still falls dramatically short of human-level extreme
generalization — i.e., TTT is not a solved escape to systematic compositional reasoning."

**Verdict: NOT REFUTED (claim well-supported, minor factual nit on the human-baseline number).**

### What checks out (primary-corroborated)
- NVARC = 1st place, **24.03%** on ARC-AGI-2 private eval. Confirmed by primary
  (arxiv 2601.10904v1, ARC Prize 2025 Technical Report) AND ARC Prize blog AND NVIDIA dev blog.
- NVARC explicitly **builds on the 2024 ARChitects entry (test-time training) + heavy
  synthetic-data generation** — quote in the claim is verbatim from the source. So "TTT +
  synthetic data" is an accurate characterization of the winner.
- 24.03% is the **new competition SOTA** on the ARC-AGI-2 private set (top Kaggle score). So
  this is the *best* TTT+synth result, not a weak straggler — strengthens the "falls short" point.
- 2nd (ARChitects, 16.53%) and 3rd (MindsAI, 12.64%) are ALSO TTT-family — whole podium is
  TTT/diffusion-LM, all <25%. Reinforces "TTT is not a solved escape."

### The one inaccuracy (does NOT flip the verdict)
- The "~97-98% human baseline" figure is **not the number ARC Prize actually reports**. The
  reported figures are: **100%** panel pass-rate (every task solved by ≥2 independent
  non-expert humans) and **~60-66%** average *individual* human accuracy. There is no
  "97-98%" anywhere in the ARC-AGI-2 documentation. This looks like a half-remembered
  conflation (possibly with ARC-AGI-1, where strong humans hit higher).
- BUT: under EITHER honest human number (100% panel or ~60-66% individual), 24.03% is still
  dramatically below human level. The directional claim — "far below human, TTT not solved" —
  holds on any reading. The error is in a non-load-bearing decoration, not the conclusion.

### Adversarial attempts to break it (all failed)
1. *"Is 24.03% cherry-picked / not really the winner?"* — No; it's the documented 1st-place
   private-set score across 3 independent sources. Refutation fails.
2. *"Is the source marketing/forum hype?"* — No; primary is the official ARC Prize 2025
   Technical Report (arxiv), corroborated by the org blog and NVIDIA's own writeup. Refutation fails.
3. *"Is it outdated?"* — No; this is the Nov-2025 competition result, current as of 2026-06.
   Refutation fails.
4. *"Does the human-baseline error sink the claim?"* — No; the number is wrong but the
   inequality (24% ≪ human) is robust to the correct figures. Partial ding, not a kill.

### Relevance to the bank (the SCAN-MCD/COGS structure-injection thesis)
This claim is a *supporting pillar* for the "no emergent/TTT escape yet" position, not a
counterexample to it. It CONFIRMS that the strongest 2025 TTT+synthetic-data system on the
hardest abstraction benchmark still lands at 24% — i.e., test-time training is **not** a
demonstrated emergent escape to systematic compositional/abstract reasoning. Nothing here
reopens the door for a brain-inspired bet; if anything it closes a candidate (TTT alone).

**One honest caveat for the bank's framing:** ARC-AGI-2 is fluid-abstraction, not the
SCAN/COGS/CFQ grammar-compositional splits the program actually targets. So this result is
*adjacent evidence* (TTT under-delivers on hard generalization broadly), not a direct test of
the SCAN-MCD claim. Don't over-cite it as if it spoke to COGS specifically.

**Sources:**
- Primary: ARC Prize 2025 Technical Report — https://arxiv.org/html/2601.10904v1
- ARC Prize 2025 Results & Analysis — https://arcprize.org/blog/arc-prize-2025-results-analysis
- NVIDIA dev blog (NVARC writeup) — https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/
- ARC-AGI-2 human-baseline details — https://arcprize.org/arc-agi/2 ; arxiv 2505.11831 (ARC-AGI-2 paper)

---

## Voter 3/3 (adversarial) — claim verification: TTT augmentation-dependence (Akyürek et al. 2024)

**Claim under review:** "The TTT gain is structurally dependent on hand-designed augmentations
and per-task data construction, not pure emergent in-weight adaptation: dropping the augmentation
transformations causes a 55% drop, indicating the lift is driven by injected geometric task structure."
**Source:** arXiv:2411.07279v1 ("The Surprising Effectiveness of Test-Time Training for Abstract Reasoning"), §3.3 / Fig 3.

**Verdict: REFUTED (as stated — interpretation overreaches; the number is real but the conclusion is contradicted by the same table).**

### What's verified
- The quote "Dropping transformations applied to augment data hurts by 16 tasks (55% decrease)" is VERBATIM accurate.
- The 55% is correct: full TTT = **29 tasks**, no-transformations = **13 tasks** on the 80-task dev set (16/29 = 55%).
- Augmentation IS the largest *single* component ablation in their table.

### Why the INTERPRETATION is refuted (two independent disconfirmers)
The full ablation hierarchy (Fig 3) refutes "not pure emergent in-weight adaptation / lift driven by injected geometric structure":

| Condition | Tasks |
|---|---|
| No TTT (base fine-tuned model) | **5** |
| TTT, per-task LoRA, **NO geometric augmentation** | **13** |
| Full TTT (+ augmentation) | **29** |

1. **In-weight adaptation alone is load-bearing and emergent.** With ZERO geometric augmentation, the per-task LoRA update still lifts 5 → 13 tasks (2.6×). That 8-task gain owes nothing to "injected geometric structure." So the claim's "not pure emergent in-weight adaptation" is false: the in-weight update is the *foundation* augmentation builds on, not an inert carrier.
2. **By the claim's own "biggest-drop = real driver" logic, TTT beats augmentation.** Removing TTT entirely (29 → 5, −24 tasks, ~83%) is a far bigger collapse than removing augmentation (29 → 13, −16, 55%). The claim cherry-picks the augmentation ablation and ignores the larger no-TTT ablation.
3. **"Injected geometric task structure" is the verifier's gloss, not the paper's.** The transformations are rotations/flips/color-perms applied to the *given few-shot examples* to give the LoRA more data to fit — a data-efficiency/coverage device for the in-weight update, NOT a symbolic grammar of the solution. The paper frames them as an empirical augmentation, not as injecting the task's structure. Conflating "data augmentation that helps a parameter update generalize" with "structure injection" is the overreach.

### Bottom line for the bank
For the program's "structure-injection-in-disguise" thesis, TTT is a WEAK pillar at best: its core mechanism is an emergent in-weight adaptation (5→13 with no augmentation), and augmentation is an additive data-efficiency booster, not a symbolic-structure injection that displaces the learning. Do NOT bank "TTT = structure injection." The honest read: TTT is a genuine (if expensive) emergent in-weight adaptation whose absolute ceiling on hard abstraction is still low — which is a *different* and more defensible point than "the gain is injected structure."

**Sources:**
- Primary: arXiv:2411.07279v1, §3.1 (D_TTT construction, Step 1 leave-one-out + Step 2 rule-based transformations), §3.3 / Fig 3 (ablation table). https://arxiv.org/html/2411.07279v1
- GitHub (implementation, confirms aug = invertible geometric/color transforms on examples): https://github.com/ekinakyurek/marc
