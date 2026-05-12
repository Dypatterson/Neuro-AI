---
date: 2026-04-30
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# CTM/TSSP Literature Review and Architectural Convergence

## Entry Point

Eighth working session. Dylan surfaced four external sources found between sessions — two papers and their associated GitHub repositories. The session focused on understanding the Continuous Thought Machine architecture, the TSSP auxiliary loss, and their relationship to the personal-ai project. No open questions were advanced; the session was pure literature review and architectural mapping.

## Literature Review

### Darlow et al. — "Continuous Thought Machines" (Sakana AI, arXiv 2505.05522v4, May 2025)

Source: https://arxiv.org/html/2505.05522v4
Repo: https://github.com/giansha/continuous-thought-machines-classification

Core architecture: a neural network that introduces an internal time dimension ("ticks") decoupled from data dimensions. Two innovations: (1) neuron-level models (NLMs) — each neuron has private learned parameters and processes its own history of recent pre-activations through a small MLP, producing heterogeneous temporal dynamics across the population; (2) neural synchronization as the latent representation — the pairwise temporal correlations between neuron activity histories (computed as inner products of post-activation histories) are used directly as the representation for attention queries and output predictions.

Key results:
- 2D maze navigation without positional encodings: CTM builds internal spatial representation through iterative attention dynamics — functionally an internal world model constructed through "episodic future thinking."
- ImageNet classification: emergent "looking around" behavior where attention shifts across image regions over ticks, resembling human saccadic eye movements. No training signal for this behavior.
- Native adaptive computation: easy inputs resolve in fewer ticks, hard inputs use more. No halting mechanism — falls out of the architecture via a loss function that optimizes at the tick of minimum loss and the tick of maximum certainty.
- Parity task: learns interpretable sequential algorithms with attention heads scanning input systematically.
- All behaviors emerge from the same core architecture applied to different tasks.

Significant limitations identified:
- Fixed tick count T as hyperparameter — temporal dimension is imposed, not emergent.
- No language modeling results — all experiments are vision, algorithmic, or simple RL.
- Synchronization matrix S = Z·Z^T is symmetric — cannot natively represent directional temporal relationships (A leads B vs. B leads A).
- NLMs create per-neuron parameter costs and only process local history — no access to population-level temporal patterns.
- No memory across inputs — dynamics reset completely for each new input. No consolidation, persistence, or growth from experience.
- Scaling story unclear — ImageNet result (72.5% top-1) not competitive with modern architectures; authors acknowledge this is outside scope.

### Caldwell & Archon — "Thought-Space Self-Prediction (TSSP)" (DuoNeural AI Lab, April 2026)

Source: uploaded PDF (tssp_ctm_paper.pdf)

Core idea: CTM intermediate states (ticks 1 through N-1) receive no direct training signal — only the final output connects to supervised loss. TSSP adds an auxiliary loss: a lightweight prediction head f_θ forecasts the next hidden state h_{t+1} from the current state h_t, with stop-gradient on the target. Enforces temporal self-consistency in thought space.

Key results:
- 32M scale: +23% perplexity improvement over CTM baseline with full annealing schedule (warmup → hold → cosine decay).
- Scale-dependent inversion: at 300M parameters, constant-λ TSSP causes late-stage divergence — TSSP gradient overwhelms the near-converged CE gradient. Analogous to posterior collapse in VAEs.
- Annealed fix: warmup+hold+cosine-decay λ schedule prevents divergence and yields +31% improvement over standard transformer baseline at 300M.
- Recurrence-specific: identical TSSP applied between consecutive layers of a plain transformer *degrades* performance. Same-weight iterative execution is a structural prerequisite. This rules out TSSP as a general regularizer.
- Compress-expand topology emerges: hidden state norm contracts at first recurrence step (gathering/integrating context) then re-expands (projecting toward output). Discovered by the model, not engineered.
- World model connection: TSSP prediction head is formally equivalent to a latent transition function T: S → S — the model learns to predict its own thought dynamics, which is self-modeling.

### Shang — "Theater of Mind for LLMs: GWT-Based Cognitive Architecture" (arXiv 2604.08206, April 2026)

Source: https://arxiv.org/html/2604.08206v1
Repo: https://github.com/giansha/Global-Workspace-Agents

Implementation of Global Workspace Theory for multi-agent LLM coordination. Maps GWT literally: STM = stage, attention node = spotlight, heterogeneous agent swarm (generator, critic, meta-arbitrator, response) = audience. Four-phase cognitive tick: perceive → think → arbitrate → update.

Most interesting mechanism: entropy-based intrinsic drive — Shannon entropy over semantic clusters of recent winning thoughts, with automatic temperature regulation when entropy drops (stagnation). Formula: T_gen = T_base + α · exp(−β · H(W)).

Assessed as low relevance to personal-ai architecture. This is software engineering orchestration of LLM API calls, not substrate-level dynamics. The "workspace" is a prompt context, the "memory" is a RAG vector database. No settling, no landscape, no emergent dynamics. Useful primarily as a negative example — clarifies what a literal GWT implementation looks like and what our project is NOT doing. The entropy anti-stagnation mechanism is conceptually related to density-dependent gating but operates at a completely different level (token-level LLM outputs vs. landscape geometry).

## Architectural Convergence Analysis

### What the CTM validates about our architecture

The central convergence: dynamics-as-representation. The CTM team arrived at this through empirical engineering (tried snapshot representations, found them too constraining, switched to synchronization). We arrived at the same principle through phenomenological reasoning (the meta-loop is the settling dynamics, not a module watching the dynamics). Two independent paths, same structural conclusion.

Specific validations:

1. **Adaptive computation without a controller** — CTM's adaptive tick usage via loss function is a concrete implementation of the anti-homunculus principle. Easy inputs settle fast, hard inputs settle slow. No halting module decides. Validates engagement-dependent convergence rate.

2. **Sustained superposition is computationally productive** — CTM's unresolved synchronization states produce rich representations. Combined with Coconut's BFS emergence and our engagement-dependent convergence rate, this is now three independent systems demonstrating the same principle: premature commitment loses information; sustained competition enables deliberation.

3. **Emergent compress-expand topology** — TSSP's discovery that settling trajectories naturally develop gather-then-project patterns maps onto our three-layer architecture: fast perceptual stage (input) → consolidated landscape (compression/integration) → output (re-expansion). The pattern emerges from recurrence dynamics without being architecturally imposed.

4. **Temperature as a property of the dynamics, not a control parameter** — CTM's synchronization-based queries naturally vary in character based on the state of the dynamics. Parallel to our resolution that temperature is a geometric property of the query itself.

### The relationship: CTM as proof-of-concept for our settling layer

Key framing from this session: the CTM isolated the thinking mechanism and showed it works, but treated it as the whole system. Our architecture embeds the same mechanism as one layer — the fast perceptual/settling layer — inside a larger loop that includes consolidation, persistence, and growth.

Every major gap in the CTM maps to a resolved component in our architecture:

| CTM Gap | Our Resolution |
|---------|---------------|
| No consolidation — every input starts fresh | Consolidation writes SONAR embeddings into Hopfield landscape, gated by dual signal |
| No persistent landscape — structure from gradient descent only | Landscape accumulates experience over lifetime through selective consolidation |
| No dual signal — nothing decides what was worth remembering | Prediction-error surprise + user reinforcement, with meta-learning from disagreement |
| No temperature modulation — synchronization character fixed by learned weights | Temperature is geometric property of query; retrieval character modulated by landscape topology |
| No personality development — two identical CTMs produce identical dynamics | Personality lives in memory; two instances diverge through differential consolidation |

The CTM demonstrates that dynamics-based settling produces adaptive computation, emergent world models, and rich internal representations. Our architecture predicts all of this and adds the persistence layer that makes the dynamics personal.

### Pieces examined for potential borrowing

Three mechanisms were evaluated:

1. **Self-prediction along settling trajectory (from TSSP)** — Could settling coherence (predictability of the trajectory) serve as a quality signal for consolidation? Distinct from our existing prediction-error signal (which measures surprise about external input). A coherent settling trajectory might indicate clean retrieval from a well-formed attractor; an erratic trajectory might indicate poorly organized landscape regions. Assessed as potentially interesting but unclear whether it's redundant with the resolution property. Not committed.

2. **Temporal history buffer influencing settling dynamics** — NLMs process a rolling window of recent pre-activations. Raises the question: should Hopfield settling be Markovian (current state only) or carry a short temporal buffer? A buffer could produce richer dynamics than standard Hopfield convergence. Whether this needs to be engineered or emerges naturally at the right scale is unclear. Not committed.

3. **Synchronization readout mechanism** — CTM provides one concrete answer for how to read the dynamics: compute pairwise temporal correlations and project them through learned weights. Our equivalent (reading resolution and engagement from Hopfield settling) is conceptually clear but mechanistically less concrete. The CTM's readout might not transfer directly to a Hopfield substrate, but it demonstrates that the readout problem is solvable. Flagged as a gap worth sharpening.

### Three-way convergence: Coconut + CTM + our architecture

All three systems demonstrate the same principle from different substrates:

- **Coconut**: reasoning in continuous latent space (bypassing tokens) produces BFS-like parallel exploration. Sustained superposition across multiple reasoning paths.
- **CTM**: synchronization-based representation preserves full settling trajectory. Unresolved dynamics are computationally productive representations, not transitional noise.
- **Our architecture**: engagement-dependent convergence rate sustains superposition across competing attractors. High engagement + low resolution = slow settling = deliberation.

The shared principle: uncommitted dynamics are computationally productive, not a phase to be rushed through. No controller decides when to commit — geometry determines settling time.

The difference, again, is persistence. Coconut resets per sequence. CTM resets per input. Our landscape accumulates. When sustained superposition produces a novel blend, the dual consolidation signal can capture it and write it into the landscape. The dwelling produces something, and the something persists.

## What This Session Resolves

- No open questions resolved — session was literature review and architectural validation.
- Significant external validation: three independent systems (Coconut, CTM, our architecture) converge on dynamics-as-representation and sustained-superposition-as-computation.
- Relationship between CTM and our project clarified: CTM is a proof-of-concept for the settling dynamics layer; our project embeds that mechanism within a persistence/consolidation loop.

## What Remains Open

Same four open questions as session 7:
1. Per-source priors formation — how does source identity modulate engagement/resolution during retrieval?
2. Where do raw snags come from?
3. Whether engagement-dependent convergence rate needs explicit engineering or emerges from Hopfield dynamics at the right scale.
4. Edge cases where SONAR encodes different question types with identical geometry.

Plus one sharpened gap from this session:
5. Concrete readout mechanism for resolution and engagement from Hopfield settling trajectory — the CTM demonstrates one approach (synchronization matrix → projection), but whether this transfers to Hopfield substrate is unclear.
