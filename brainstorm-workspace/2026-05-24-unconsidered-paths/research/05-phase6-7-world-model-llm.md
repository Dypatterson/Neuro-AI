# Phase 6 + Phase 7 — Memory-First, Anti-Homunculus World Model and LLM Paths

**Research date:** 2026-05-24
**Angle:** Phase 6 ("predictive world model and latent rollouts") and Phase 7 ("LLM interface") paths that keep the world model as an *energy term* and the LLM as a *voice* — never as a controller or seat of identity. The literature is dominated by the opposite shape (Dreamer/MuZero = controller, MemGPT = LLM-as-self-editor). The interesting work is the smaller body of energy-based / settling / variational-inference work that fits anti-homunculus discipline natively.

---

## TL;DR for the project

1. The Phase 6 architecture that fits this substrate is **not** Dreamer-style. It is closer to **"Planning as Descent" (Dec 2025, arXiv:2512.17846)** + **EBWM (NeurIPS 2024, arXiv:2406.08862)** + **Modern Hopfield as the attractor backbone (arXiv:2506.11043, 2502.10122)**. Planning is *gradient descent on a goal-conditioned energy landscape over latent trajectories*. There is no policy network. The "rollout" is a settling process. This is bit-for-bit the anti-homunculus stance.
2. **Diffusion models are formally identical to modern Hopfield networks** when trained on discrete patterns (Ambrogioni 2023; Hoover/Krotov 2025, arXiv:2505.21777). This is huge for the project: a Phase 6 latent transition module trained as a diffusion model *is already an extension of the Phase 4/5 Hopfield substrate*. There is no "two competing memory systems" problem — they are the same object viewed from two angles (encoding ↔ retrieval).
3. **Active inference + energy-based memory is the gap.** EFE-as-variational-inference (arXiv:2504.14898, Apr 2025) is exactly the right shape, but the published work uses small generative models, not Hopfield/FHRR substrates. **This is a genuinely under-explored convergence the project can claim.**
4. Phase 7: the literature is sharply bifurcated. **MemGPT/Letta makes the LLM the memory manager** — this is the anti-pattern. The right shape is closer to **sparse memory finetuning (arXiv:2510.15103)** + **retrieval-augmented in-context cueing** where the LLM is a frozen decoder driven by settled substrate state, with replay as the only continual-learning channel.

---

## 1. Key findings by technique

### 1.1 JEPA / V-JEPA 2 — *anti-homunculus compatible, but reframe needed*

- V-JEPA 2 (Meta, June 2025, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)) is action-free and predicts in latent space using an energy-based loss. V-JEPA 2-AC plans by "finding a sequence of actions that minimizes the L1 distance between the imagined future state and the target state in the learned representation space, with this L1 distance called the goal-conditioned energy function, optimized using cross-entropy method."
- **For this project:** the JEPA encoder + predictor is exactly the shape of a latent transition module. The energy function is the right abstraction. The only homunculus risk is the cross-entropy *optimizer* over action sequences — replace with gradient descent on energy (see Planning as Descent below) and it is clean.
- LeCun's broader EBM program ([deepsense.ai overview](https://deepsense.ai/resource/world-models-explained-jepa-energy-based-learning-and-the-limits-of-llms/), [innobu](https://www.innobu.com/en/articles/jepa-world-models-energy-based-models-ai-architecture.html)) is the cleanest published statement of "world model = energy landscape, not policy" — the project should claim this lineage explicitly.

### 1.2 DreamerV3 / IRIS / TWM — *fundamentally a controller*

- DreamerV3 trains an actor-critic on imagined trajectories sampled from the world model ([emergentmind topic](https://www.emergentmind.com/topics/dreamerv3), [GitHub](https://github.com/danijar/dreamerv3)). The actor *is* a policy network. This is the homunculus shape — a learned function that, given a latent state, outputs "the action to take." There is no version of this that survives the anti-homunculus filter.
- 2025 work like DreamerV3-XP ([arXiv:2510.21418](https://arxiv.org/html/2510.21418v1)) and DreamerNav ([PMC12510832](https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832/)) keep the actor-critic structure intact and bolt on exploration uncertainty.
- **For this project:** Dreamer is a useful *baseline* (compute-matched, controller-style world model) but **not** an architectural template. The Phase 6 design should be able to explain why a Dreamer agent fails on a contextual-completion task that the energy-settling world model solves.

### 1.3 Active Inference / Expected Free Energy — *anti-homunculus by construction*

- **[arXiv:2504.14898](https://arxiv.org/abs/2504.14898) — "Expected Free Energy-based Planning as Variational Inference"** (Apr 2025). Quote: "EFE-based planning arises naturally from minimizing a variational free energy functional on a generative model augmented with preference and epistemic priors, casting planning under uncertainty itself as a form of variational inference." This is the cleanest anti-homunculus statement in the recent literature — planning is *not* a separate module, it *is* the inference dynamics.
- **EFE-GLean** ([MDPI Entropy 27/8/846](https://www.mdpi.com/1099-4300/27/8/846), Aug 2025) — continuous-state active inference with goal-conditioned latent rollouts. Uses dynamic planning + information gain; T-maze and continuous-domain experiments. The architecture infers low-dimensional latent posterior trajectories rather than picking actions from a policy.
- pymdp + deep active inference lineage ([arXiv:2201.03904](https://arxiv.org/pdf/2201.03904), [arXiv:1907.03876](https://arxiv.org/pdf/1907.03876)) — "Deep Active Inference as Variational Policy Gradients" already showed that policy selection can be cast as inference (gradient descent on free energy).
- **For this project:** this is the *theoretical frame* Phase 6 should claim. Energy = free energy. Settling = variational inference. Rollouts = sampling from the posterior over latent trajectories. **The project should write the Phase 6 design doc with explicit citations to 2504.14898 and the MDPI EFE-GLean paper** so the anti-homunculus stance is grounded in 2025 literature, not just LeCun's program.

### 1.4 Successor Representation — *promising but mostly orthogonal*

- 2025 hippocampus SR work: [Zhou/Sibille/Dragoi PubMed 40866358](https://pubmed.ncbi.nlm.nih.gov/40866358/) — "Generative emergence of non-local representations in the hippocampus" (Aug 2025) shows rapid (~1-2 lap) emergence of temporally-compressed hippocampal theta sequences via re-purposing pre-existing sleep motifs. This is *exactly* the replay-as-prior story the project already endorses.
- [bioRxiv 2025.06.11.658893](https://www.biorxiv.org/content/biorxiv/early/2025/06/15/2025.06.11.658893.source.xml) — after learning, SR aligns with high-level abstract properties, not low-level features. The SR becomes a *conceptual* predictive map.
- **For this project:** SR-over-atoms is *not* a separate Phase 6 module — it is what the Phase 4 codebook already does, plus a temporal-discount edge structure. The right framing: the codebook is the SR; replay + the Hopfield landscape provide the prediction. **Don't add a separate SR module.**

### 1.5 Latent Diffusion as a world model — *anti-homunculus compatible*

- **Diffusion ≡ Modern Hopfield** when patterns are discrete (Ambrogioni 2023, [arXiv:2309.17290](https://arxiv.org/abs/2309.17290); Hoover, Krotov et al., May 2025, [arXiv:2505.21777](https://arxiv.org/abs/2505.21777) — "Memorization to Generalization: Emergence of Diffusion Models from Associative Memory"). This is the most important single finding for the project.
- "Efficient Planning with Latent Diffusion" ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b2ac1112e14fac8d07275a7f482e0c11-Paper-Conference.pdf)) — diffusion in a learned latent space, planning via guided sampling.
- "What Makes a Good Diffusion Planner" ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/ab6022d3d669b5baafa24c91d7c407a6-Paper-Conference.pdf)) — diffusion planners as the SOTA "generate-and-rank" approach.
- **For this project:** a small latent-space diffusion model trained on replayed atom trajectories *is* the Phase 6 transition model, and *it is the same object as the Phase 5 Hopfield landscape* viewed from a different angle. This makes the Phase 5 → Phase 6 transition mathematically clean rather than architecturally additive.

### 1.6 MuZero / EfficientZero / Stochastic MuZero — *structurally a controller*

- [arXiv:2411.04580 — "Demystifying MuZero Planning"](https://arxiv.org/abs/2411.04580) (2025) — MuZero's MCTS over learned dynamics works even when the dynamics drift, because the search corrects for errors. But MCTS is structurally a search procedure executed *by* something — there is no way to dissolve it into local dynamics without losing the algorithm.
- [UniZero arXiv:2406.10667](https://arxiv.org/pdf/2406.10667) — generalizes MuZero with transformer-based latent dynamics; still MCTS-driven.
- **For this project:** MuZero is the wrong architectural template. The "tree" is a homunculus that decides which branches to expand. Settling-as-rollout is the correct replacement. No version of soft-MCTS rescues this cleanly.

### 1.7 Energy-Based World Models (EBWM) — *anti-homunculus compatible, directly usable*

- **[arXiv:2406.08862 — "Cognitively Inspired Energy-Based World Models"](https://arxiv.org/abs/2406.08862)** (NeurIPS 2024). Trains an EBM to predict compatibility of (context, predicted future state). Introduces the **Energy-Based Transformer (EBT)** — autoregressive transformer variant tailored for EBMs. Three System-2 capabilities claimed: predictions actively influence internal processing, plausibility is evaluated, computation is dynamically allocated (settling time).
- EBWM scales better with data and GPU-hours than autoregressive transformers in CV; promising early NLP scaling.
- **For this project:** this is the most direct precedent for the Phase 6 design. The EBT architecture (autoregressive-style but with energy-based gradient descent at each step) maps cleanly onto FHRR-bound atom sequences. The "dynamic compute allocation" is the same idea as the project's settling-step budget.

### 1.8 Memory-augmented LLMs (Phase 7) — *MemGPT is the anti-pattern; sparse memory finetuning is the right shape*

- **MemGPT / Letta** ([Letta docs](https://docs.letta.com/concepts/memgpt/), [research.memgpt.ai](https://research.memgpt.ai/), [Letta v1 blog](https://www.letta.com/blog/letta-v1-agent)): the LLM is *the memory manager*. It edits its own persona blocks. **This is the homunculus shape made explicit** — the LLM is "the seat of self" and decides what to remember about itself. Directly violates Phase 7's "swapping LLM doesn't erase identity" goal.
- **Recurrent Memory Transformer / ARMT** ([arXiv:2207.06881](https://arxiv.org/abs/2207.06881), [LM-RMT GitHub](https://github.com/booydar/LM-RMT)): memory tokens are stored in Hopfield-style energy basins; ARMT extends this across all layers with horizontal propagation. The memory is *substrate-resident*, not LLM-resident.
- **Sparse Memory Finetuning** ([arXiv:2510.15103](https://arxiv.org/pdf/2510.15103)): updates a small set of parameters out of a large memory pool per forward pass. The LLM body stays fixed; the "memory layer" is what learns.
- **FOREVER** ([arXiv:2601.03938](https://arxiv.org/html/2601.03938v1)): replay-based continual learning aligned with human-inspired schedules, grounded in parameter-update dynamics. This is exactly the channel Phase 7 needs.
- **MemoryBench** ([arXiv:2510.17281](https://arxiv.org/html/2510.17281)): a 2025 benchmark for memory and continual learning in LLM systems. Worth using for Phase 7 evaluation.
- **For this project:** the Phase 7 path is *not* "LLM with memory tools." It is "LLM as frozen decoder driven by substrate state; replay updates a small memory layer, never the LLM body." The identity-survives-swap test is then literal: swap GPT-N for GPT-(N+1), nothing in the substrate changes.

### 1.9 Tool use as latent commitment — *anti-homunculus compatible if reframed*

- 2025 tool-use literature ([MeCo arXiv:2502.12961](https://arxiv.org/pdf/2502.12961), [adaptive tool use ACL 2025](https://aclanthology.org/2025.acl-long.655.pdf)) frames tool calling as "the LLM decides when it needs help" — homunculus.
- Reasoning Agentic RAG survey ([arXiv:2506.10408](https://arxiv.org/html/2506.10408v1)) — same shape, decision-making embedded in retrieval.
- **For this project:** the right reframe is that a tool call is *what happens when the substrate settles into an attractor that has a tool-binding role*. Tool calls are emergent from settling, not the output of a "should I use a tool" classifier. The work to do is to *demonstrate* this — there is no 2025 paper that articulates it.

### 1.10 World model + replay convergence — *mostly extant in Dreamer lineage*

- WMAR (World Models with Augmented Replay, [arXiv:2401.16650](https://arxiv.org/pdf/2401.16650)) — Dreamer + augmented replay buffer for continual RL.
- Nature Comm 2025 ([s41467-025-65181-5](https://www.nature.com/articles/s41467-025-65181-5)): replay can occur *without* sharp-wave ripples in spatial memory. Ripples and replay are dissociable. Useful for the project: the replay channel doesn't have to coincide with a discrete consolidation event.
- A Unified Dynamic Model for Learning, Replay, and SWRs (J. Neurosci, [jneurosci 35/49/16236](https://www.jneurosci.org/content/35/49/16236)) — single dynamic system gives rise to all three. Anti-homunculus by construction.
- **For this project:** the existing replay/consolidation design is already aligned with these results. The Phase 6 transition model should be trained *on* the replay buffer — same buffer, no new mechanism.

### 1.11 Modern Hopfield ↔ Attention ↔ EBM unification — *the unifying mathematical thread*

- **[arXiv:2506.11043 — Non-Linear Attention via Modern Hopfield Networks](https://arxiv.org/abs/2506.11043)** (June 2025): "an energy landscape is defined whose gradient corresponds to the attention computation." Introduces "context wells" — stable token configurations. Standard transformer attention emerges from optimizing an MHN energy function.
- [arXiv:2603.06875 — Stochastic Attention via Langevin Dynamics](https://arxiv.org/abs/2603.06875): attention as gradient descent on classical energy; Langevin sampling gives stochastic attention. **Temperature controls retrieval vs generation** — cold = exact retrieval, hot = open-ended generation. This is the dial the project has been looking for.
- [arXiv:2502.10122 — Modern Hopfield Networks with Continuous-Time Memories](https://arxiv.org/abs/2502.10122) (Feb 2025): continuous-time memory + probability density energy function. Worth reading carefully for FHRR substrate.
- [arXiv:2411.08590 — Hopfield-Fenchel-Young Networks](https://arxiv.org/abs/2411.08590): unified family of energy functions covering classical, modern, sparse-Hopfield, and many transformer attention variants.
- Hoover et al. ICLR 2025 workshop ([openreview OBQwZaO4pt](https://openreview.net/pdf?id=OBQwZaO4pt)): "New Frontiers in Associative Memory" — the central community convergence.
- **For this project:** **diffusion = modern Hopfield = transformer attention = energy descent**. The Phase 6 transition model, the Phase 5 retrieval, and any Phase 7 attention-style mechanism over substrate state are *all the same operation at different temperatures*. This is the unifying claim the project should make architecturally.

---

## 2. Concrete Phase 6 Architecture Proposal

**Goal:** small JEPA/EBWM-style latent transition + coherence energy term + energy coupling to existing landscape. **No actor, no critic, no MCTS.**

### Components

1. **Latent encoder φ.** Input: a window of FHRR-bound atoms (the existing substrate state). Output: a low-dimensional latent z ∈ ℝ^d (d ≈ 64-128). Trained with JEPA-style masked prediction loss (V-JEPA 2 lineage). Implementation: small transformer or MLP-mixer over FHRR atoms.
2. **Latent transition predictor T.** Input: (z_t, optional cue from substrate). Output: predicted ẑ_{t+1}. Trained on the replay buffer with target φ(state_{t+1}). EBWM-style — produces a *distribution* over futures via energy, not a point estimate.
3. **Coherence energy E_coh.** A scalar function E_coh(z_t, z_{t+1}) = ||z_{t+1} - T(z_t)||² (or a learned compatibility energy à la EBWM). Low when the latent rollout is consistent with the predictor; high when not.
4. **Memory energy E_mem.** This is the *existing* Hopfield + codebook + replay-prior energy on the FHRR substrate. Already implemented.
5. **Coupling.** The total energy at rollout time is `E_total(state, z_rollout) = E_mem(state) + λ_coh · E_coh(z_rollout) + λ_dec · ||φ(state) - z_rollout[0]||²` — the last term anchors the rollout to the current substrate state.
6. **Rollout = settling.** A "rollout" is gradient descent on E_total jointly over (substrate state, latent trajectory z_1, ..., z_K). When it settles, the converged substrate state is the "completed context." No tree, no actor, no value head.

### Training

- Train φ and T together on the replay buffer using JEPA-style masked-region prediction. **No reward signal anywhere.**
- The coherence weight λ_coh is a *hyperparameter*, not a learned controller. (If it must be adaptive, make it a function of local energy gradient — i.e., a temperature, not a decision.)

### Citations grounding each piece

| Component | Citation |
|---|---|
| Encoder φ + JEPA-style training | V-JEPA 2 ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985)) |
| Latent transition T as EBM | EBWM / EBT ([arXiv:2406.08862](https://arxiv.org/abs/2406.08862)) |
| Coherence energy term | Planning as Descent ([arXiv:2512.17846](https://arxiv.org/html/2512.17846)) |
| Energy coupling to memory | Modern Hopfield ↔ Attention ([arXiv:2506.11043](https://arxiv.org/abs/2506.11043)); Hopfield-Fenchel-Young ([arXiv:2411.08590](https://arxiv.org/abs/2411.08590)) |
| Rollout = settling, no controller | EFE-as-VI ([arXiv:2504.14898](https://arxiv.org/abs/2504.14898)) |
| Diffusion ≡ Hopfield equivalence | Hoover/Krotov 2025 ([arXiv:2505.21777](https://arxiv.org/abs/2505.21777)) |
| Replay buffer as training corpus | Unified dynamic model J. Neurosci ([35/49/16236](https://www.jneurosci.org/content/35/49/16236)) |

### Anti-homunculus check (Phase 6)

- **Who decides which rollout to run?** No one. The system settles. λ_coh is a fixed temperature.
- **Who decides when the rollout has converged?** A local convergence criterion (Δstate < ε for N steps) — a measurement, not an arbiter.
- **Who decides which completion to commit to?** The settled state *is* the commitment. There is no selection step.
- **Where does "value" live?** It does not exist as a separate term. If preference shows up, it shows up as a *prior* in the coherence energy (e.g., goal-conditioned: bias toward latents near a target embedding) — exactly the EFE construction in [arXiv:2504.14898](https://arxiv.org/abs/2504.14898). Value is a bias on the landscape, not an actor.
- **Does the world model decide?** No. The world model *is* the landscape. It is shape, not agent.

This check passes cleanly.

---

## 3. Concrete Phase 7 Architecture Proposal

**Goal:** LLM as encoder (text → workspace cue) + decoder (settled state → tokens). Replay is the only continual-learning channel. **Identity survives LLM swap.**

### Components

1. **LLM encoder (frozen).** Input: user text. Output: a contextualized embedding sequence. Mapped via a small learned projection P_enc into a set of FHRR atoms or a cue vector. The LLM is *not* fine-tuned. The projection P_enc is.
2. **Substrate.** Existing FHRR + Modern Hopfield + codebook + replay + Phase 6 latent landscape. The cue from P_enc enters as a bias on the energy landscape (a "context well" in the [arXiv:2506.11043](https://arxiv.org/abs/2506.11043) sense). The system settles.
3. **LLM decoder (frozen).** Input: settled substrate state, mapped via a small learned projection P_dec into a prompt prefix / soft prompt / embedding sequence for the LLM. The LLM autoregressively generates tokens conditioned on this state.
4. **Replay channel for P_enc, P_dec, and substrate.** Successful interactions are written to the replay buffer (substrate state + LLM output text). Sleep/consolidation re-trains P_enc and P_dec (and updates the codebook + Hopfield substrate per existing Phase 4-5 mechanisms). The LLM body is never updated.
5. **Identity = the substrate, not the LLM.** Persona / preferences / accumulated experience live in the codebook + replay distribution + Hopfield landscape. Swapping the LLM swaps P_enc and P_dec (or retrains them from a frozen embedding alignment task on a small held-out corpus) but does not touch the substrate.

### Why this fits the rules

- LLM = voice. P_enc / P_dec are the vocal cords. Substrate is the self.
- Persistence is in the substrate. The LLM cannot store anything about the user across sessions; the substrate must.
- Tool calls (if any) are *emergent* from settling: a settled state whose role is bound to a tool slot triggers a tool call. The LLM does not "decide" to call a tool; the substrate's settled commitment expresses itself as a tool call when decoded.

### Citations grounding each piece

| Component | Citation |
|---|---|
| LLM frozen + small projection | Sparse Memory Finetuning ([arXiv:2510.15103](https://arxiv.org/pdf/2510.15103)) |
| Memory substrate-resident, not LLM-resident | RMT / ARMT ([arXiv:2207.06881](https://arxiv.org/abs/2207.06881)) |
| Replay channel for continual learning | FOREVER ([arXiv:2601.03938](https://arxiv.org/html/2601.03938v1)); WMAR ([arXiv:2401.16650](https://arxiv.org/pdf/2401.16650)) |
| Tool call as commitment, not decision | Reframe of [adaptive tool use ACL 2025](https://aclanthology.org/2025.acl-long.655.pdf) (no 2025 paper says this directly — claimable contribution) |
| Cue = context well | Non-linear attention via MHN ([arXiv:2506.11043](https://arxiv.org/abs/2506.11043)) |

### Anti-homunculus check (Phase 7)

- **Who decides what to say?** The substrate settles; the LLM decodes the settled state. No "selection" of output beyond LLM sampling temperature — which is a setting, not a controller.
- **Who decides what to remember?** No one. Replay writes everything; consolidation prunes by energy/coverage (existing Phase 4-5 mechanism), which is a local dynamic, not an arbiter.
- **Who decides when to call a tool?** Tool calls fall out of settling — a role binding in the settled state decodes to a tool call. No "tool need classifier."
- **What happens if you swap the LLM?** P_enc and P_dec need to be re-projected (a small alignment task on a held-out corpus). The substrate is untouched. Identity persists.

This check passes cleanly. The MemGPT design *fails* this check on every line.

---

## 4. Benchmarks that would prove "Phase 6 latent rollouts beat one-step retrieval"

The project needs benchmarks where the headline metric is sensitive to *multi-step latent inference*, not single-step retrieval. Candidates:

1. **Contextual completion under occlusion** — present a window with a multi-token gap; the gap requires inferring an intermediate atom that is not in the immediate context (multi-hop). Existing Phase 4-5 substrate should fail; Phase 6 rollouts should succeed.
2. **OGBench** ([Park et al. 2024](https://arxiv.org/abs/2410.20092)) goal-conditioned cube-manipulation tasks — used by Planning as Descent ([arXiv:2512.17846](https://arxiv.org/html/2512.17846)) to show 95% vs 68% over prior. A robotics task is off-shape for the project's substrate but the *form* of the benchmark (goal-conditioned, no reward) is the right shape.
3. **LongBench** ([arXiv:2308.14508](https://arxiv.org/pdf/2308.14508)) / **MemoryBench** ([arXiv:2510.17281](https://arxiv.org/html/2510.17281)) — multi-document QA where the substrate must integrate facts across a long span. The Phase 7 LLM-as-decoder design is testable here: substrate + frozen LLM decoder vs LLM-only.
4. **WikiText-103 multi-step continuation** — sample a 64-token prefix, mask a 16-token middle, ask the system to complete the middle conditioned on the suffix. Compare: (a) one-step Hopfield retrieval, (b) Phase 6 latent rollout, (c) LLM-only, (d) substrate + LLM (Phase 7).
5. **AgentLongBench** ([arXiv:2601.20730](https://arxiv.org/pdf/2601.20730)) — long-context agent tasks via environment rollouts. Good for Phase 7 identity-persistence tests (swap the LLM mid-task; does substrate state survive?).
6. **Tolman-Eichenbaum maze / T-maze** — used in EFE-GLean and active inference work. Small enough to be a Phase 6 unit test for "settling-as-rollout produces hidden-state-disambiguating behavior."

**Specifically as a Phase 6 graduation experiment:** WikiText-103 multi-step continuation, *with the Phase 5 codebook-prior controls already in place*. Headline metric: Δ exact-match recall on the masked span (Phase 6 vs Phase 5 substrate only). Drill-downs: rollout depth where the benefit appears, energy convergence statistics, cap-coverage of the inferred span.

---

## 5. Anti-homunculus screen — full table

| Technique | Out-of-box | Reframable as energy | Fundamentally controller |
|---|---|---|---|
| V-JEPA / V-JEPA 2 | ✓ (energy loss) | — | — |
| EBWM / EBT | ✓ | — | — |
| Modern Hopfield + non-linear attention | ✓ | — | — |
| Stochastic attention via Langevin | ✓ | — | — |
| Diffusion world models | ✓ (score = energy gradient) | — | — |
| Planning as Descent | ✓ | — | — |
| EFE-as-Variational-Inference | ✓ | — | — |
| EFE-GLean / continuous active inference | ✓ | — | — |
| Sparse memory finetuning | ✓ (substrate-resident) | — | — |
| Successor Representation | ✓ (already a measurement) | — | — |
| DreamerV3 | — | reframe: drop actor, use energy → loses Dreamer's identity | (as-is) |
| MuZero / EfficientZero | — | reframe: MCTS → settling → loses MuZero | (as-is) |
| MemGPT / Letta | — | reframe: LLM no longer manages memory → loses MemGPT | (as-is) |
| Tool-use ReAct lineage | — | reframe: tool call = settled commitment → unclaimed gap | (as written) |
| Recurrent Memory Transformer | ✓ if memory tokens treated as energy basins (consistent w/ ARMT) | — | — |

---

## 6. Surprises

1. **Diffusion ≡ Modern Hopfield is now mainstream**, not a curiosity. The Hoover/Krotov May 2025 paper ([arXiv:2505.21777](https://arxiv.org/abs/2505.21777)) is in the "associative memory frontier" workshop track at ICLR 2025. The project has a *very* clean architectural story to tell because of this equivalence.
2. **2025 active inference work has fully embraced "planning = inference, not optimization"** ([arXiv:2504.14898](https://arxiv.org/abs/2504.14898), [MDPI EFE-GLean](https://www.mdpi.com/1099-4300/27/8/846)). The "policies emerge from variational inference" stance is no longer fringe. The project should cite these explicitly to ground Phase 6's anti-homunculus posture.
3. **The Letta team is doubling down on LLM-as-self-editor** ([Letta v1 blog](https://www.letta.com/blog/letta-v1-agent), 2025). This is the dominant commercial direction and **the cleanest possible foil for Phase 7**. The project should position Phase 7 explicitly against MemGPT — same task domain, opposite identity locus.
4. **Replay-without-ripples is real** ([Nature Comm 2025 65181-5](https://www.nature.com/articles/s41467-025-65181-5)). Replay and ripples are dissociable. Implication for the project: don't tie the consolidation event to a discrete trigger; let it be a continuous low-level process (which aligns with the project's existing design).
5. **There is no 2025 paper articulating "tool call as latent commitment from settling."** All the 2025 tool-use work treats the LLM as the decider. **This is a claimable contribution.**
6. **There is no 2025 paper combining active inference with Hopfield/FHRR-style energy memory.** The EFE-VI papers use small neural generative models, not associative-memory substrates. **This convergence is a claimable contribution.**

---

## 7. Promising leads — papers to read in full

Ranked by load-bearing-ness for Phase 6/7 design:

1. **[arXiv:2512.17846 — Planning as Descent](https://arxiv.org/html/2512.17846)** (Dec 2025) — *the* paper to read for the Phase 6 design. Read the architecture and training-loss sections carefully. The "identical computation during training and inference" stance is exactly what the project wants.
2. **[arXiv:2504.14898 — EFE as Variational Inference](https://arxiv.org/abs/2504.14898)** (Apr 2025) — the theoretical frame.
3. **[arXiv:2406.08862 — Cognitively Inspired EBWM / EBT](https://arxiv.org/abs/2406.08862)** (NeurIPS 2024) — the closest published architectural template.
4. **[arXiv:2505.21777 — Diffusion ≡ Associative Memory](https://arxiv.org/abs/2505.21777)** (May 2025) — the unifying mathematical bridge.
5. **[arXiv:2506.11043 — Non-Linear Attention via MHN](https://arxiv.org/abs/2506.11043)** (June 2025) — context-well framing for Phase 7 LLM cue.
6. **[arXiv:2603.06875 — Stochastic Attention via Langevin](https://arxiv.org/abs/2603.06875)** — temperature dial between retrieval and generation; useful for Phase 6 rollout sampling.
7. **[arXiv:2502.10122 — Continuous-Time Modern Hopfield](https://arxiv.org/abs/2502.10122)** (Feb 2025) — relevant if the project moves toward continuous-time substrate dynamics.
8. **[arXiv:2411.08590 — Hopfield-Fenchel-Young Networks](https://arxiv.org/abs/2411.08590)** — unified energy-function family; useful taxonomy.
9. **[arXiv:2510.15103 — Sparse Memory Finetuning](https://arxiv.org/pdf/2510.15103)** — the right shape for Phase 7 substrate-LLM coupling.
10. **[MDPI Entropy 27/8/846 — EFE-GLean continuous active inference](https://www.mdpi.com/1099-4300/27/8/846)** (Aug 2025) — concrete continuous-state implementation.
11. **[arXiv:2506.09985 — V-JEPA 2](https://arxiv.org/abs/2506.09985)** (June 2025) — the JEPA encoder/predictor scaling story.
12. **[ICLR 2025 NFAM workshop proceedings](https://openreview.net/pdf?id=OBQwZaO4pt)** — the community convergence point. Worth skimming the table of contents.
13. **Hippocampus 2025 — [PubMed 40866358](https://pubmed.ncbi.nlm.nih.gov/40866358/)** — "generative emergence of non-local representations" — the replay-as-prior biological story for Phase 6.

---

## 8. Gap flags (claimable contributions)

1. **Active inference + Hopfield/FHRR energy memory** — published EFE-VI work uses small NN generative models. No one has done it on an associative-memory substrate. The project's Phase 6 design proposed above *is* this convergence. Write the Phase 6 design doc with explicit citations to [arXiv:2504.14898](https://arxiv.org/abs/2504.14898) so this lineage is visible.
2. **Tool call as settling commitment** — no 2025 paper says this. The Phase 7 design proposed above is the first articulation.
3. **LLM identity persistence under swap via substrate-resident memory** — Letta deliberately makes the LLM the locus of identity. The project's Phase 7 explicitly inverts this. Position the Phase 7 design as a direct contrast.
4. **Phase 6 architecture where the world model IS the energy landscape, not a separate predictor + actor** — the "Planning as Descent" paper is the closest prior art (Dec 2025); it does this for robot manipulation, not for episodic-memory-style contextual completion. The project's Phase 6 design extends this to a memory-first substrate.

---

## 9. Strongest single recommendation

**The Phase 6 design document should be written this week, citing primarily:**
- arXiv:2512.17846 (Planning as Descent) for the planning-as-settling stance,
- arXiv:2406.08862 (EBWM / EBT) for the energy-based transition predictor,
- arXiv:2505.21777 (Diffusion ≡ Hopfield) for the mathematical bridge to the existing Phase 5 substrate,
- arXiv:2504.14898 (EFE-VI) for the variational-inference theoretical frame,
- arXiv:2506.09985 (V-JEPA 2) for the encoder/predictor training shape.

These five citations together make the anti-homunculus stance defensible in the published literature, give a clear architectural template, and establish that the project's Phase 6 design is a *combination* of recent (2024-2025) ideas rather than a from-scratch invention. The combination itself is the contribution.
