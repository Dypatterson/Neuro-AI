---
date: 2026-04-09
project: personal-ai
tags:
  - overview
  - subject/cognitive-architecture
  - subject/alife
  - project/personal-ai
---

# Personal AI — Project Briefing

A memory-first, workspace-architecture AI companion. The goal is not to build a smarter general-purpose model; it is to build a small, local system that grows alongside one specific person over time, remembers what matters, and can explore ideas in ways that feel more like thinking than like autocomplete. This project emerged from a several-hour architectural exploration on 2026-04-09 that started with "can I build an AI agent with SNNs?" and refined over ~20 turns into a coherent design Dylan converged on through his own reasoning.

> **Read this first.** Everything below is a snapshot of where the thinking landed on 2026-04-09, not a build specification. See the **Posture** section immediately below before treating any architectural choice in this document as committed.

---

## Posture (added 2026-04-09, session 2)

This briefing is scaffolding, not a spec. Dylan has explicitly stated that the current phase of this project is *learning and working through ideas*, not building to the outline. The architecture described in this document — memory substrate, Global Workspace, dual consolidation signal, LM-as-voice, all of it — is where the reasoning landed on the day the briefing was written. It is not a commitment. If future exploration reveals that the workspace is the wrong frame, or the LM doesn't belong in the stack at all, or the whole thing should be reorganized around something not yet thought of, the briefing should be rewritten without hesitation and without treating the current version as lost ground.

**Implications for how to use this file:**
- The Architecture table, the module list, the Open Questions — all of these are *conversation starting points*, not things that constrain what the next session is allowed to explore.
- Building is not the current goal. Understanding is. The build will happen, but only when Dylan says the ideas are ready, and whatever gets built then may look nothing like what is described here.
- Claude should not steer conversations toward "which direction gets us closer to code." It should steer toward "which direction reveals the most interesting thing we don't yet understand."
- No framing of the current architecture as load-bearing. If a new idea contradicts it, the new idea wins by default and the old architecture gets reexamined, not defended.

---

## The Core Dissatisfaction

Dylan's critique of mainstream LLM development, in his own words:

- "Advanced token predicting machine" is an oversimplification. *And* "some form of understanding and reasoning going on" is also a stretch. Both extremes overclaim.
- True thinking requires the ability to remember and reflect on past instances. Current LLMs don't have persistent continuity across interactions; whatever "understanding" they have is locked in frozen weights.
- LLMs cannot be genuinely creative in the sense that matters — the sense where a new question arises from lived experience, past memories resurface and become relevant, and a path opens up that wasn't there before. They can produce combinatorially novel text but not *generative curiosity rooted in continuity*.
- The industry's bet is "build a god that knows everything." Dylan's bet is "build a companion that knows *me*." These are not the same project.

The system should not have to start smart. It should have the capacity to *become* knowledgeable alongside its user, through real shared experience.

---

## The Intellectual Arc (how we got here)

This is logged because the *sequence* of the reasoning is valuable — Dylan refined a vague intuition into a precise architecture across the conversation, and the refinement path matters for resuming the work.

1. **SNNs as agent substrate?** Dylan asked about spiking neural networks for an AI agent. Conclusion: viable for RL-style agents on his hardware (snnTorch + Gymnasium + CartPole) but not for language modeling; SNNs and next-token prediction are architecturally mismatched.
2. **Consciousness and self-learning?** Separated the two: consciousness is philosophically intractable, self-learning is a well-defined engineering target (curiosity-driven exploration, world models, episodic memory, meta-learning). Mentioned Global Workspace Theory as the engineering-friendly consciousness-inspired framing.
3. **Local LM with NCAs?** Clarified which NCA — Dylan meant Neural Cellular Automata but was also curious about Neural Circuit Policies / Liquid networks. Liquid foundation models from Liquid AI noted as one of the more interesting biologically-inspired LM directions currently shipping.
4. **NCAs integrated deep within a language model as regeneration/visualization?** Yes, theoretically possible. Three candidate integration slots identified:
   - **Perception NCA** (input side — replace the vision encoder with an NCA that iteratively refines 2D input into token-like vectors, with natural locality and damage-repair properties)
   - **Memory sidecar** (middle — cross-attention bridge to a persistent NCA grid that serves as regenerative mental scratchpad between the LM's layers)
   - **Imagination canvas** (output side — LM emits commands to an NCA canvas that grows visual content; LM can read back from the canvas to reason about what it drew)
   Of the three, perception is most tractable, canvas is most visually striking and novel, memory sidecar is the deepest research question and worth leaving for later.
5. **Transformer refresher.** Tokens → attention (query/key/value) → layer stack → next token prediction. Stateless between forward passes. Everything the model "knows" is in frozen weights or the current context window.
6. **Dylan's actual critique sharpened.** The missing thing is not NCAs specifically — it is a *substrate where the system can lay out possibilities and move around in them*, where retrieval from memory is itself a creative act, and where persistence allows the system to grow through experience. NCAs were a gesture toward that substrate, not the precise answer.
7. **Latent space as the framing.** LLMs already have rich latent spaces (word2vec-style geometric concept relationships). The problem is that the only operation performed on those spaces is "consult them briefly on the way to producing the next token." All that geometric structure collapses back into a probability distribution every step. What Dylan wants is a module *allowed to wander* in latent space, not constrained by next-token pressure.
8. **Hopfield connection.** Hopfield networks store patterns as attractors in an energy landscape; retrieval is the act of settling into a configuration, and the settled state can be a *blend* of multiple stored patterns. Ramsauer et al. (2020), "Hopfield Networks is All You Need," proved that modern continuous Hopfield networks have exponential storage capacity *and* that their update rule is mathematically identical to transformer attention. Transformer attention is already Hopfield retrieval, viewed through a different lens — but it runs on an ephemeral landscape built fresh from the current context every forward pass, then thrown away.
9. **Memory and exploration might be the same operation.** Dylan's insight, arrived at in conversation: when you have a creative thought, a path opens up *because* a forgotten memory became relevant. The opening of the path and the surfacing of the memory are not two events; they are the same event. This collapses two design requirements (a memory store + an exploration mechanism) into one requirement (*a memory whose act of retrieval is itself capable of producing new combinations that weren't stored*). That is exactly what a modern Hopfield network does.
10. **Query/key/value ↔ cue/landscape/content.** Mapped transformer attention to protein folding: the query is the cue (the protein entering the landscape), the keys are the landscape (the terrain that determines where queries can settle), the values are the content retrieved. Dylan worked this out himself after some guided questioning.
11. **Query should belong to the idea, not the token.** Dylan's reframing. In the standard transformer, queries are computed from token embeddings — the system can only go looking for things based on the surface form of the current word. Dylan wants queries computed from whatever the system is currently *holding in mind*, which may not correspond to any specific word. This decouples retrieval from the linguistic surface.
12. **Global Workspace as the translation layer.** Dylan brought it back at the right moment. The workspace is where "an idea" exists in a form separable from any particular module. Queries into the long-term memory come from the workspace, not from the LM directly. Results return to the workspace where the LM and other modules can read them. The LM talks to memory *through* the workspace.
13. **Substrate properties.** Walked through what operations the substrate needs to support: comparable, combinable, deformable, decomposable, translatable, persistent. Dylan's insight: deformable follows from comparable + combinable (a small weighted blend is a deformation), so the three are not independent. The essential properties are *comparable, combinable, translatable, and persistent*. This matches vector-space substrates and (especially) modern Hopfield substrates cleanly.
14. **Selective consolidation.** Dylan's instinct: persistence is essential but only for things that are "possibly relevant," otherwise everything would be impossible. This is the *stability-plasticity dilemma*, and his proposed solution is the same principle the hippocampus uses — fast plastic store for recent experience, slow stable store for things that earn their way in, selective writing based on what matters. He rederived the architecture of biological memory from first principles.
15. **Dual consolidation signal: surprise + user reinforcement.** Dylan proposed combining prediction-error-based surprise (unsupervised, captures novelty the user didn't flag) with explicit user reinforcement (supervised, captures importance the system didn't predict). The two signals fail in opposite directions and cover each other. *Disagreement between the two signals becomes a meta-learning signal* — the system learns what to learn by watching where its own sense of importance diverges from the user's.

---

## The Architecture (as it stands)

> Reminder: this is a snapshot, not a spec. See Posture section above.

Every component below corresponds to something that exists in the research literature in some form. The novelty is the *combination* and the *framing* — building this as a personal companion that grows with one person over time, not as a benchmark-beating general-purpose model.

| Module | Role | Candidate implementations |
|---|---|---|
| **Language model** | The "voice." Handles external communication, linguistic reasoning, output generation. Small, local, possibly frozen. Does not need to be smart on day one. | Llama 3.2 1B–3B, Qwen 2.5 3B, Liquid Foundation Model (1.3B or 3B). Run via Ollama or HuggingFace transformers with MPS. |
| **Persistent memory substrate** | The "landscape." Long-term, stable, selective. Keys accumulated from past experience shape the terrain; values hold the content. Retrieval is generative — blends stored patterns when cues land between them. | Modern continuous Hopfield network (Ramsauer et al. 2020). Alternates to investigate: sparse distributed memory (Kanerva), vector symbolic architectures, predictive-coding-style energy models. |
| **Fast ephemeral store** | Working memory. High-bandwidth, overwritten constantly. Holds whatever is currently happening. Roughly what attention already provides inside a transformer. | Could be the LM's own context window + a lightweight scratch buffer. |
| **Global workspace** | The stage / translation layer. Where "an idea" lives in a form separable from any single module. Queries into long-term memory originate here. Results return here. The LM reads from and writes to the workspace, as does the memory. | Shared embedding space + competitive selection mechanism. Details deliberately unresolved — this is one of the open design questions. |
| **Consolidation rule** | Decides what moves from the ephemeral store into the persistent memory. Two signals combined: prediction-error-based surprise (unsupervised) + explicit user reinforcement (supervised). | Surprisal = negative log probability of observed content under the LM's predictions. User reinforcement = explicit flags in the interface, or implicit signals like dwell time, repeated queries, etc. |
| **Meta-learning loop** | Emerges for free from disagreement between the two consolidation signals. When the user flags something the system didn't find surprising → pay more attention to this *type* of thing. When the system flags something the user doesn't care about → adjust what counts as signal. | No implementation yet. Empirical question once the base system runs. |

**The important property.** Personality in this architecture does not live in the language model. It lives in the *memory* — in what the system remembers, what it has consolidated, what its landscape has been shaped by. The LM is the voice; the memory is the self. This means two instances of the same base LM, living alongside different users for six months, should diverge into genuinely different companions without any fine-tuning. The learning happens through consolidation, not through gradient updates to the LM weights.

---

## Key Concepts to Carry Forward

- **Gap between prediction and reality = surprise = learning signal.** One quantity (prediction error / surprisal) drives both exploration (go toward things you don't understand) and consolidation (remember the things that surprised you). Same mechanism, two angles. Also the leading theory of what dopamine is doing in biological brains (reward prediction error hypothesis).
- **Stability-plasticity dilemma.** Pure plasticity → catastrophic forgetting. Pure stability → no learning. Real minds solve this via dual stores + selective consolidation. Any learning system has to solve it one way or another.
- **Modern Hopfield networks as the "secret" inside attention.** Every transformer is running a Hopfield retrieval on an ephemeral landscape every forward pass. *Liberating* that operation — giving it a persistent landscape, longer settling time, more flexibility — is one of the most principled paths toward the architecture Dylan wants.
- **Query should come from the idea, not the token.** This single reframing separates "LLM that can only think about the word in front of it" from "system that can think about anything currently in the workspace." The Global Workspace is the mechanism that makes this possible.
- **Retrieval as creativity.** In a continuous energy landscape, retrieving from memory can produce blends that were never explicitly stored. Two ideas combining is not an extra operation bolted onto memory; it is what memory *does* in this kind of system. This is the operation Dylan has been reaching for the whole conversation.
- **Personality in the memory, not the model.** The frozen LM is shared infrastructure. The consolidated memory is what makes the system *this* companion, belonging to *this* person.

---

## References to Read (in rough priority order)

| Resource | Type | Why |
|---|---|---|
| Ramsauer et al. (2020) — "Hopfield Networks is All You Need" | Paper | The mathematical bridge between attention and Hopfield retrieval. Already partially read; Dylan was working through the confusion about the "new update rule" vs. the classical one. Sections to re-read carefully: Introduction, "A New Hopfield Network," experiments. |
| Baars, *In the Theater of Consciousness* (1997) or Dehaene, *Consciousness and the Brain* (2014) | Book | The clearest prose statements of Global Workspace Theory. One is enough. |
| Pathak et al. (2017) — "Curiosity-driven Exploration by Self-supervised Prediction" | Paper | Foundational paper for prediction-error-as-intrinsic-reward. Short, clear, directly relevant to the consolidation rule. |
| Franklin et al. — LIDA architecture papers | Paper set | Canonical software implementation of Global Workspace Theory. Useful as a reference point for what a full workspace architecture looks like when actually built. |
| Kanerva (1988) — *Sparse Distributed Memory* | Book / papers | Alternative memory substrate with different properties than Hopfield networks. Worth knowing about before committing to a substrate. |
| Liquid AI — LFM model cards / technical reports | Docs | If Dylan wants to use a non-transformer base model, this is the easiest off-the-shelf option for a biologically-inspired LM. |
| Predictive coding / free energy principle literature (Friston et al.) | Papers | Deeper theoretical frame for "prediction error as the central currency of cognition." Optional but illuminating. |
| LeCun (2022) — "A Path Towards Autonomous Machine Intelligence" | Position paper | The most-cited mainstream critique of current LLMs. Overlaps substantially with Dylan's own critique. Worth reading to see the vocabulary and where he agrees/disagrees. |

---

## Open Questions Left on the Table

These are the questions we ended the session on — the places to pick up when the next chat starts.

1. **What does *writing* into the persistent landscape look like?** We worked out the reading operation in detail (query from workspace → settle into keys → return blended value to workspace). We never worked out the complement: when a new experience happens and it earns its way into long-term memory via the consolidation signal, what does the *write* operation actually do to the landscape? Does it carve a new valley? Deform an existing one? Merge with a nearby pattern? There is a hunch that reading and writing may be more entangled than they look, the same way memory and exploration turned out to be — but this wasn't worked through.
2. **What generates the query when the query belongs to "the idea"?** Dylan recognized that the workspace solves the translation/connection problem, but it's not yet clear what concrete mechanism lives inside the workspace to extract an idea from current activity and turn it into a cue that can be matched against the memory's keys. "Extract and embed the idea itself" is the phrase, but the operation is unspecified.
3. **What concretely wins the competition for the workspace?** In Global Workspace Theory there's a bidding process — modules compete for the stage. The criterion for winning is doing a lot of work in the theory. Relevance, surprise, reward, emotional charge, sheer activation? The choice of criterion is "the most personality-shaping decision you'd make in the whole architecture." Dylan hasn't named his criterion yet.
4. **How exotic does the substrate need to be, really?** Vector embeddings give comparable/combinable/persistent cleanly. Hopfield nets give all of those *plus* the blended-retrieval-as-creativity property. But vector embeddings are dramatically easier to build with. Does the first working version need Hopfield, or does it start with a vector store and grow into Hopfield only when Dylan can *feel* what the vector store is missing?

---

## Four Candidate Directions for the Next Session

Claude offered these at the close of the chat. Dylan has not chosen one yet. Listed in the order they were offered. **These are starting points for exploration, not a menu Dylan has to pick from — a better question is allowed to replace any of them.**

1. **Workspace mechanics.** Dig into how the workspace actually translates between the LM's token space and the memory's concept space. Closest to engineering; would sharpen what code eventually gets written.
2. **Failure modes.** What could go wrong in this architecture, how it would break, what to watch for, how to know if it's working. Often where the most important learning happens — forces you to think about the system as a real thing instead of an idea.
3. **Simplest possible prototype on the M5 Pro.** The smallest thing that captures the essence of this architecture even if it's missing most of the sophistication. Something Dylan could touch and live with and learn from in the next week or two.
4. **Experience of using it.** What would the system feel like from the user's side? What kinds of conversations would it enable? How would it fit into Dylan's daily life? Sometimes the more important conversation, because the use case shapes the architecture more than the architecture shapes the use case.

Recommendation from Claude, not Dylan's decision yet: build the boring version first (small local LM + Obsidian vault as memory substrate + reflection loop with surprise-based consolidation), live with it for a few months, notice what it's missing, and then reach for the exotic substrates (Hopfield, sparse distributed memory) to fill the specific gaps that experience reveals. The ambitious version is real and worth pursuing eventually; the path to it goes through the simpler version first. *This recommendation is held loosely — the Posture section above takes precedence.*

---

## Relationship to Other Projects

- **Second Brain (Obsidian vault):** The existing vault is already, structurally, the beginnings of a memory substrate for this project. Notes tagged by subject, cross-links, session digests, a consolidation-adjacent system (decisions, connections, reflection entries). The personal-ai project may end up *using* the vault as its initial long-term store, with the LM reading from and writing to it through a tool interface. That is probably the concrete v1.
- **Cellular Automata project:** Shares philosophical DNA (gradualism, curiosity-driven build, skepticism of mainstream "make it bigger" approaches, interest in emergence from simple substrates). Differs in goal — the CA project is a biosphere / world-building project where intelligence is *emergent* from a substrate; personal-ai is a *companion* project where the substrate is deliberately engineered. They are adjacent, not overlapping. Some insights from one will cross to the other. Dylan should not feel pressure to unify them.
- **Code Garden:** No direct overlap, but the "autonomous generative process running in the background" pattern is similar. The code garden is a good reference for what "system that does its own thing between user interactions" looks like at small scale.

---

## Hardware & Practical Context

- **Current hardware:** M5 MacBook Pro (incoming / recently arrived). 24 GB unified memory. MPS backend available for PyTorch. Sufficient for everything in the v1 plan — small local LMs (1B–3B parameter range), vector-store memory, and modest Hopfield experiments all fit comfortably.
- **No GPU cluster required.** Every technical direction in this briefing is accessible on a single laptop for the exploratory phase. Scaling concerns are not relevant to v1.
- **Not bottlenecked by compute.** The bottleneck is architectural clarity and willingness to sit with a confusing question. Both of which Dylan has.

---

## Framing Commitments

- **Exploration before build.** The current phase is learning and working through ideas. Building is a future phase, not the current one. Any session that tries to steer toward "let's lock this down and start coding" is steering wrong unless Dylan explicitly says he's ready. See Posture section.
- **The outline is not load-bearing.** Nothing in this briefing constrains what the next conversation is allowed to explore. If a new idea contradicts the architecture, the new idea wins by default.
- **Personal, not universal.** This is a companion for Dylan. Benchmarks and generality are not the point. If it helps one person think better, it has succeeded.
- **Start unsmart.** The system does not need to know things on day one. It should acquire knowledge through shared experience over time.
- **Memory first.** Architectural decisions should be made starting from the memory layer, not the LM. The LM is a replaceable component; the memory is the self. *(Provisional — subject to being overturned by the Posture commitment.)*
- **Curiosity-driven.** Same commitment as the CA project. The right question right now is "what would be interesting to see," not "what would constitute a contribution."
- **Grow the architecture through living with it.** The exotic pieces (Hopfield substrate, full workspace, meta-learning loop) earn their place by solving gaps that experience with the simpler version reveals. They don't go in up-front on speculation.

---

## Status

*Updated 2026-05-09.*

**This briefing is preserved as the 2026-04-09 early-thinking snapshot.** The architecture has substantially refined since it was written — most notably the 2026-05-04 commitment to emergent codebooks with FHRR substrate at D=4096 (validated empirically in Phase 1) and the 2026-05-07 finding that settled-state modulus is not a reliable ambiguity signal in the current implementation (mixed states unstable to float32 numerical asymmetry). Live state for design and implementation lives in `brain/session-context.md` and `journal/2026-05/`. The Posture section above still applies to architectural choices that haven't been built yet.

**Implementation status (2026-05-07):** Phase 1 complete. Phase 2 Sessions 1 & 2 complete. Repo at `github.com/Dypatterson/personal-ai` (private). Phase 1 substrate (FHRR via torchhd, D=4096), encoding module, validation report. Phase 2 Session 1 data pipeline (WikiText-2 vocabulary, windows, codebook). Phase 2 Session 2 Hopfield retrieval module + `RetrievalResult` dataclass + 24 passing tests + three diagnostic probes (`02a` v1 single-β, `02a` v2 β-sweep, `02b` trajectory dynamics on K-ambiguous cues). All Phase 2 substrate findings empirically validated and incorporated into `PHASE_2_SPEC.md` (Session 2 probe findings section, per-retrieval metrics, β values locked at {0.01, 0.1, 1.0}).

**Empirical findings carried forward from Phase 2 substrate characterization:**

- **Posture 4 (no renormalization in loop)** chosen as default for Hopfield retrieval; renorm-in-loop available as diagnostic flag only.
- **β regime structure at D=4096:** soft-blend (β ≤ 0.001) gives 0% retrieval with K-graded modulus collapse; sharp-retrieval (β ≥ 0.01) gives 100% retrieval with immediate commitment. Regimes are nearly disjoint on memorization condition.
- **Settled-state modulus is NOT a reliable ambiguity signal in this implementation.** The K=2 symmetric mixed state is mathematically a fixed point but unstable to float32 numerical asymmetry — only 1 of 3 tested cue groups maintained it across 15 iterations. K=5 and K=20 always commit to a single constituent.
- **Ambiguity is carried by softmax entropy and top-2 margin in the retrieval distribution** rather than by per-dimension modulus of the settled state. Phase 2 per-retrieval metric set records both endpoint and per-iteration trajectory data.
- The "trajectory carries the reasoning" framing is empirically grounded as of Session 2, not just an architectural intuition. Implication for Phase 5+: the soft-blend regime that would let geometry-as-temperature naturally express ambiguity coincides with retrieval failure at this D and landscape size.

**Post-Session-2 architectural review (2026-05-09):** Cross-paper synthesis run against thirteen research papers (Geometry of Consolidation, HEN, SQHN, Krotov-Hopfield 2016, Krotov hierarchical AM, Papyan neural collapse, MESH, Benna-Fusi, OCL-MIR, others). Three outputs landed in the docs. (1) Phase 2/3 diagnostic stack expanded — cap-coverage error and meta-stable-state rate added to the per-retrieval metric set, NC1 with unsupervised-case reformulation added as a Phase 3 collapse diagnostic, softmax entropy reframed as a feature/prototype mode classifier per Krotov-Hopfield 2016, empirical θ′(β) calibration recommended as a pre-Phase-3 spike (~2 days). (2) Phase 3 headline metric specified explicitly: Recall@K on masked-token contextual completion, regime-stratified, vs. shuffled-token control. (3) Two project-level commitments locked in — anti-homunculus filter as a hard rule that every diagnostic, metric, or actuator must measure or be expressible as a local geometric dynamic, and headline-vs-drill-down structure for every phase's evaluation. The diagnostics-vs-actuators distinction was identified as the next conceptual threshold; it must be worked out before Phase 3 starts, not before Phase 2 Session 3. Full synthesis: `notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md`. Companion file updates: `PHASE_2_SPEC.md` (settled-state diagnostics, post-hoc analyses, validation report deliverables); `emergent-codebook/experimental-progression.md`; `emergent-codebook/phase-3-deep-dive.md`; `emergent-codebook/consolidation-geometry-diagnostic.md`.

**Next phase:** Phase 2 Session 3+ — multi-session experimental driver implementation. First post-Session-2 session: persistence wrapper + minimal landscape population code + small validation run with new metrics. Subsequent sessions: full 540-condition sweep across (window size × mask count × mask position × landscape size × β × retrieval condition × objective), aggregation with Wilson confidence intervals + frequency stratification, plot generation, and report.
