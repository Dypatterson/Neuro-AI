# Research note: Continual-learning benchmarks & evaluation principles the project hasn't engaged

Date: 2026-05-24
Brainstorm batch: `unconsidered-paths`
Research angle: **External validation for a memory-first substrate.** The project's eval has been intrinsic (cap-coverage, ΔE, Recall@K on WikiText, masked-token completion). It is at risk of Goodharting its own metrics. There is a mature literature of continual-learning, catastrophic-forgetting, compositional-generalization, and associative-recall benchmarks the substrate could be measured against — most of which can be run on a MacBook Pro, and several of which are *purely cue-driven* (no controller, no task ID, no RL policy → anti-homunculus clean).

---

## Key findings (with URLs)

### A. The "three scenarios" taxonomy — required reading
- **van de Ven, Tuytelaars & Tolias 2022** (Nature Mach. Intell.) — distinguishes task-incremental / domain-incremental / class-incremental learning. Empirical comparison of major CL methods on Split-MNIST and Split-CIFAR-100 across all three. <https://www.nature.com/articles/s42256-022-00568-3> · arXiv precursor <https://arxiv.org/abs/1904.07734> · code <https://github.com/GMvandeVen/continual-learning>
  - **Project mapping.** A memory-first substrate with no task labels is *natively a class-incremental or domain-incremental learner* — the hardest of the three. Task-incremental, where a task ID picks a head, would require a homunculus. Recording which scenario the project targets is itself a clarifying step the notes haven't taken.

### B. Frameworks the project should adopt rather than reinvent
- **Avalanche (ContinualAI)** — PyTorch-based, includes Split/Permuted/Rotated MNIST, Split-CIFAR-10/100/110, CUB200, Tiny-ImageNet, Omniglot, CORe50, OpenLORIS, Stream-51. Built-in metrics: accuracy, forgetting, BWT, FWT, time, memory. <https://avalanche.continualai.org/> · baselines repo <https://github.com/ContinualAI/continual-learning-baselines>
- **Mammoth (aimagelab)** — 70+ CL methods, 20+ datasets, modular, easy to debug. Includes CaSpeR (continual spectral regularizer) and the Dark Experience Replay family. <https://github.com/aimagelab/mammoth>
- **CLEAR** (Lin et al. NeurIPS 2021) — first CL benchmark with *natural* temporal distribution shift (YFCC100M, 2004-2014). Smooth concept drift instead of artificial splits. <https://clear-benchmark.github.io/> · arXiv <https://arxiv.org/abs/2201.06289>

### C. Embodied / robot-stream benchmarks
- **CORe50** (Lomonaco & Maltoni 2017) — 50 objects, 11 sessions, RGB-D, robot-like manipulation views. Class- and domain-incremental scenarios out of the box.
- **OpenLORIS-Object** — robotic-vision dataset with *quantified* environmental factors (illumination, occlusion, pixel size, clutter) → enables ablation on which factor breaks the substrate. <https://lifelong-robotic-vision.github.io/dataset/object.html>

### D. Compositional-generalization / role-binding (Phase 5's natural test bed)
- **bAbI** — 20 textual QA tasks; "solved" = >95% on each. Lightweight: bAbI-1k has only 1k examples per task. Tasks 1, 4, 5, 11, 12 test role-binding and entity tracking directly. <https://research.facebook.com/downloads/babi/>
- **CLUTRR** (Sinha et al. EMNLP 2019) — diagnostic for *systematic generalization* of kinship-relation inference over short stories. Evaluates on held-out *combinations* of logical rules. GNNs on symbolic input beat BERT/MAC. <https://arxiv.org/abs/1908.06177> · <https://github.com/facebookresearch/clutrr>
- **SCAN / gSCAN / COGS** — compositional command-to-action mapping; gSCAN grounds it in a grid world and tests eight kinds of generalization (novel adjective-noun pairs, novel adverbs, longer sequences). <https://arxiv.org/abs/2003.05161> · <https://arxiv.org/abs/2109.12243>
- **RuleTaker / LogicNLI** — deductive reasoning over synthetic rules + facts in NL; ~707k instances; explicit multi-step proofs. <https://allenai.org/data/ruletaker>

### E. Memory-and-recall benchmarks
- **MQAR (Multi-Query Associative Recall)** — synthetic key→value recall benchmark from Stanford Hazy Research's *Zoology* / *Based* line of work. Literally the task an associative substrate should ace by construction. Mamba degrades as #KVs grows; attention is the SOTA reference. <https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis> · <https://hazyresearch.stanford.edu/blog/2024-03-03-based>
- **NarrativeQA** — 355 stories (avg 57K, max 404K tokens), 10.5k questions requiring narrative-level understanding. <https://aclanthology.org/Q18-1023>
- **LongMemEval / LoCoMo / MEQA** (2024-2025) — long-term-memory chat / multi-hop event QA. <https://arxiv.org/abs/2410.10813> · <https://aclanthology.org/2024.acl-long.747/>
- **Long Range Arena** — 1K-16K token sequences across text/image/maths. <https://arxiv.org/abs/2011.04006>

### F. Continual learning *with* Hopfield-style networks — directly load-bearing for this project
- **Sparse Quantized Hopfield Network (SQHN)** (Nicola Plebe & Bogacz, *Nature Communications* 2024). Online-continual, sparse one-hot hidden code, neuro-genesis. Beats SOTA on *associative memory under noise* and on episodic memory. <https://www.nature.com/articles/s41467-024-46976-4>
- **Sequential Learning in the Dense Associative Memory** (McAlister, Robins, Szymanski, *Neural Computation* 37(10), 2025) — first systematic sequential-learning benchmark of Modern Hopfield Networks; shows phase transitions in DAM behaviour. <https://arxiv.org/abs/2409.15729>
- **Dense Associative Memory with Epanechnikov Energy** (Hoover et al. 2025) — alternative energy that may change capacity/forgetting trade-off. <https://arxiv.org/abs/2506.10801>
- **VAE + Modern-Hopfield CLS model** (2024-2025) — 89.71% Split-MNIST class-incremental (upper baseline 95.55%). Exactly the architectural family of this project. <https://arxiv.org/html/2507.11393>
- **HiCL: Hippocampal-Inspired Continual Learning** (2025) <https://arxiv.org/abs/2508.16651>
- **Autonomous retrieval for continuous learning in associative memory networks** (2024) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12418250/>

### G. Metric formalisms the notes don't engage with
- **GEM metrics** (Lopez-Paz & Ranzato 2017) — Average Accuracy (ACC), Backward Transfer (BWT = mean over i<T of R_{T,i} − R_{i,i}), Forward Transfer (FWT = mean over i>1 of R_{i-1,i} − b_i where b_i is a random init baseline). <https://arxiv.org/abs/1706.08840>
- **Pareto Continual Learning** (2025) — stability/plasticity as multi-objective; report the *Pareto frontier* of (forgetting, accuracy) rather than a scalar. <https://arxiv.org/abs/2503.23390>
- **Representation forgetting via linear probing** (Davari et al. CVPR 2022) — train a fresh linear classifier on frozen features before vs after a new task; if representation is preserved, classifier accuracy is preserved. Disambiguates *representational drift* from *catastrophic forgetting in the head*. <https://openaccess.thecvf.com/content/CVPR2022/papers/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.pdf>
- **FOREVER / Ebbinghaus-style forgetting curve** (2026) — report forgetting as a curve indexed by parameter-update magnitude, not by step count. <https://arxiv.org/html/2601.03938v1>

---

## Concrete ideas the project could adopt now

### Idea 1 — "Phase 5 sanity check": bAbI tasks 1, 4, 5, 11, 12 + CLUTRR-k=2,3
Why this is a sanity check, not a graduation experiment: Phase 5's headline is intrinsic (ΔE between role-prior and content-prior). If role-binding actually works, **the substrate should be able to answer simple role-filler questions** with cue-driven retrieval alone. Concretely:

- bAbI task 1 ("single supporting fact"): given "Mary went to the kitchen. John went to the bedroom. Where is Mary?", retrieve the binding `agent⊛Mary ⊛ location⊛kitchen`, unbind by `agent⊛Mary ⊛ location^{-1}` (FHRR conjugate) → expected fill `kitchen`. No controller needed; the substrate's *retrieval dynamic* is the entire mechanism.
- bAbI task 4 ("two-argument relations"): cleanest binding-arity test.
- bAbI task 11 ("basic coreference"): role-binding under indirection.
- CLUTRR length-2 and length-3: tests systematic combination of relations (mother-of-father-of-X), exactly what FHRR composition `A⊛B` is supposed to support.

**Expected baseline numbers.** bAbI joint-training SOTA is >99% (Relation Net family, ~2017-2018). Reasonable subhuman targets for a *non-trained* substrate doing pure retrieval: 70-80% on tasks 1 and 4, 50-60% on 11. CLUTRR: random ≈ 1/N relations; symbolic GNN reaches >90% in-distribution; *out-of-distribution by path length* is the diagnostic — if substrate maintains accuracy as path length grows from 2→3→4, that is the strongest possible evidence that binding is compositional rather than associative-by-content.

**Anti-homunculus check.** bAbI/CLUTRR can be cast as: (i) read sentences as bindings into the substrate, (ii) form the query as a (partial) binding, (iii) retrieve, (iv) unbind. No supervisor, no task-ID, no policy. The "answer" is whichever filler vector the substrate settles to. This is the same dynamics as masked-token completion, just on a corpus that *requires* binding rather than rewarding content-matching.

### Idea 2 — Adopt Avalanche as the secondary harness
Specifically run Split-MNIST and Split-CIFAR-10 in **class-incremental** scenario and report (ACC, BWT, FWT) alongside intrinsic Phase metrics. Don't replace headline. The VAE+MHN CLS paper above gives a direct baseline (89.7% Split-MNIST CIL). If the substrate scores <70%, that's a clear sign the consolidation pipeline isn't doing what we think.

**Anti-homunculus check.** Class-incremental on MNIST is cue-driven: the cue is the image, the substrate retrieves a class label. No task ID. Compatible.

### Idea 3 — MQAR as the *unit test* for the associative memory
MQAR is essentially the synthetic benchmark Modern-Hopfield-style retrieval was designed for. Run it at varying K (#stored key-value pairs) and report *recall vs K*. Should monotonically degrade and the degradation curve is the substrate's effective capacity. Compare to: (a) a random-codebook control (Phase 4 protocol), (b) attention (perfect recall), (c) Mamba-2 (degrades at moderate K). This *separates the codebook-prior from the retrieval mechanism* — both are part of Phase 5.

**Why this is high value.** MQAR is small (synthetic; runs in seconds on MPS), has known theoretical scaling, and gives a *capacity curve* — a much richer signal than a single recall@K number. The current substrate has no published capacity curve.

### Idea 4 — Linear-probe sanity check on the codebook
Right now the project measures cap-coverage (geometric) and ΔE (energy). It does NOT measure: **does the codebook produce features a linear classifier can use?** Adopt Davari et al.'s linear-probing protocol:
1. Freeze substrate at end of each Phase 5 training run.
2. Train a single linear layer from {token → codebook code → embedding} on a held-out classification task (e.g., POS tagging or coarse semantic class on WikiText).
3. Report linear-probe accuracy. If cap-coverage rises but linear-probe accuracy doesn't, the geometric metric is decoupled from any downstream utility — a Goodhart signal.

### Idea 5 — Report a Pareto frontier instead of a scalar
The current Phase-5 headline is ΔE (single scalar). Per ParetoCL (2025), the *right shape* of a stability-plasticity finding is a Pareto front. For this project:
- x-axis: catastrophic-forgetting metric (e.g., recall@K on the *first* batch of stored tokens after T more batches are consolidated).
- y-axis: plasticity metric (recall@K on the *most recent* batch).
- Generate the curve by sweeping a consolidation hyperparameter (replay rate, learning-rate-on-consolidation, codebook freeze probability).

A point that is Pareto-dominated by a random-codebook baseline is *not* a graduation event. A point that pushes the frontier is.

---

## Anti-homunculus screen (which benchmarks are clean?)

| Benchmark | Cue-driven? | Needs task-ID / policy? | Verdict |
| --- | --- | --- | --- |
| bAbI 1/4/5/11/12 | yes (text → retrieval) | no | **CLEAN** |
| CLUTRR | yes (story → query) | no | **CLEAN** |
| MQAR | yes (key → value) | no | **CLEAN — strongly recommended** |
| Split-MNIST/CIFAR class-incremental | yes (image → label) | no (CIL hides task ID) | **CLEAN** |
| Split-MNIST task-incremental | yes | **needs task ID** | **dirty — skip** |
| CORe50 / OpenLORIS | yes (image stream) | depends on scenario; new-classes is clean | mostly clean |
| CLEAR | yes | no (natural temporal stream) | **CLEAN** |
| Habitat / MiniGrid memory | yes but requires policy | **needs RL policy** | dirty — skip until phase 7+ |
| SCAN / gSCAN | yes | no (mapping is the task) | clean, but requires *generating* an action sequence; needs an output decoder — borderline |
| Long Range Arena | yes | no | **CLEAN** but probably outside MBP budget |
| NarrativeQA / LongMemEval | yes | no | **CLEAN** but heavy |

---

## Surprises

1. **The closest published work is sitting on our shelf already.** SQHN (Nature Comms 2024) and the VAE+MHN CLS paper (2024-25) are *directly* in this project's architectural family and both report on Split-MNIST CIL. The project hasn't compared against either. That's a one-week comparison study sitting unrun.
2. **bAbI is more relevant than WikiText for Phase 5.** WikiText rewards content-matching; bAbI tasks 1/4/11 *require* role-binding to solve. The current eval setup is biased toward measuring the thing the substrate is *already good at*, not the thing Phase 5 is supposed to add.
3. **MQAR is the unit test we never wrote.** Stanford has been using it for two years to discriminate "real" associative recall from "approximate-via-content" in subquadratic LMs. The Phase 4 random-codebook control answers a *different* question than MQAR's capacity curve.
4. **The "headline = scalar" frame is itself out of date.** ParetoCL (2025) argues that any single-scalar headline in CL is a Goodhart trap. Reporting a frontier (stability axis × plasticity axis) is now the methodological norm for new CL papers in 2025.
5. **Linear probing has been the canonical sanity check since 2022 (Davari et al.).** The project's notes have a long discussion of cap-coverage as a geometric metric; there is no parallel discussion of representational utility. This is a meaningful gap.
6. **"Sequential Learning in the Dense Associative Memory"** (McAlister 2025) found *phase transitions* in DAM behaviour during sequential learning. If those transitions appear in this project's substrate too, they would be a much more interesting headline than ΔE.

---

## Risks & failure modes (which benchmarks would expose the substrate as content-matching?)

- **MQAR at high K**: if the substrate's recall curve looks like Mamba's (sharp degradation) rather than attention's (flat), the codebook is acting like an approximate content store, not a true associative memory. This is the **most direct test for the failure mode the project is worried about**.
- **CLUTRR out-of-distribution by path length**: a content-matching system will hit a wall when going from train-length 2 to test-length 4. A genuine binding system shouldn't.
- **bAbI task 11 (coreference)** — requires indirect role binding (`pronoun ↔ entity` link). Content-matching fails.
- **Split-CIFAR-100 CIL with 10×10 splits and no replay**: catastrophic forgetting is *brutal* without rehearsal; would expose whether the codebook genuinely consolidates or just lossy-compresses.
- **CLEAR's natural-shift setting**: if the substrate's intrinsic metrics rise on the early time-window and fall on the late one, that's a clean demonstration of distribution-shift overfitting.

A self-protective failure mode to watch: the project could "pass" any of these by **finetuning to them** rather than treating them as held-out external probes. Pre-commit to evaluation protocols in advance, the way GEM specified ACC/BWT/FWT in 2017.

---

## Most promising leads (in priority order)

1. **MQAR as a recurring unit test.** Cheap, decisive, theoretically grounded. Should be Phase-5-week-1 work.
2. **bAbI tasks 1/4/11 as the Phase 5 sanity check.** If role-binding works, these should be solvable cue-only. If not, the design needs to change.
3. **Replicate SQHN's online-continual MNIST setup on the project's substrate.** This is the closest published baseline; the comparison is the highest-information benchmark we can run.
4. **Adopt GEM's (ACC, BWT, FWT) triad as a secondary metric block.** Even if not the headline, it's the lingua franca of CL since 2017.
5. **Linear-probe the codebook.** One-day experiment. Either confirms cap-coverage matters downstream, or surfaces a Goodhart problem.
6. **Read McAlister et al. 2025 ("Sequential Learning in the Dense Associative Memory") before designing the next Phase 5 experiment.** It is the literal prior art for "do MHNs catastrophically forget when trained sequentially."
7. **Plan a CLEAR-style natural-shift test for after Phase 5 graduates.** This is the right "is the substrate real?" external check for Phase 6/7.

---

## What to put in STATUS.md as a follow-up

A new "External-validation deferred" section noting that the substrate has *not* been measured on any of: MQAR, bAbI, CLUTRR, Split-MNIST CIL, CORe50, CLEAR. This is not a Phase 5 blocker, but its absence should be acknowledged so that "Phase 5 graduated" doesn't quietly come to mean "Phase 5 graduated on its own intrinsic metrics, untested externally."
