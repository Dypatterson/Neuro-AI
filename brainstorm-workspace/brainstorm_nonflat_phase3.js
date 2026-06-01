export const meta = {
  name: 'brainstorm-nonflat-phase3-substrates',
  description: 'Divergent bio-analogous brainstorm: entirely different Phase-3 substrates beyond the flat codebook',
  phases: [{ title: 'Diverge' }, { title: 'Curate' }],
}

const SHARED = `
PROJECT: Neuro-AI — a neuroscience-inspired cognitive substrate. The BET (CONTEXT.md): the data-hungry / backprop / autoregressive path is the wrong bet; biology/neuroscience gives the answer. Every mechanism is chosen because it is BRAIN-ANALOGOUS. The system is a contextual-COMPLETION machine (settle into an attractor — "what fills this gap / what does this remind me of") NOT a next-token predictor. Working dir: /Users/dypatterson/Desktop/Neuro-AI-main.

CURRENT ARCHITECTURE (what we want ALTERNATIVES to): Phase 3 is the FOUNDATION layer — emergent structure grown from experience. It is currently a FLAT CODEBOOK: one FHRR hypervector per token, grown by co-occurrence statistics (Hebbian centroid / SPPMI), retrieved by a Modern Hopfield Network, bound/unbound by FHRR (Fourier Holographic Reduced Representations) circular convolution.

WHY WE NEED SOMETHING DIFFERENT: the ENTIRE flat-codebook growth family is now empirically EXHAUSTED (all NULL):
- first-order Hebbian co-occurrence centroid (Report 121) → paradigmatic NULL,
- second-order SPPMI + anti-collapse (Report 123) → NULL,
- directional/successor operator (Report 124, today) → NULL.
DECISIVE BOUND (Report 123 §4): the paradigmatic ("substitutability") structure — similar NON-co-occurring words like king/queen clustering, the structure the project's higher-level goals need — lives in the SUBDOMINANT modes of the co-occurrence operator. A GLOBAL SVD reaches it (+0.109) but LOCAL iterative growth runs to the DOMINANT collocational modes and CANNOT reach it (+0.021). This local-vs-global bound holds for the flat code REGARDLESS of the growth rule. The flat codebook is the wrong SUBSTRATE for paradigmatic/relational structure, not just the wrong growth rule.

THE BRAINSTORM ASK: generate ENTIRELY DIFFERENT, biologically-analogous Phase-3 SUBSTRATES — genuinely different representational architectures + growth dynamics that could grow paradigmatic / relational / compositional structure from experience, WITHOUT being a flat codebook of co-occurrence-grown vectors. (NOT in scope: a latent layer bolted ON TOP of the existing flat code — that is a separate effort already underway. We want the substrate itself reimagined.) GO WIDE. Prize genuine architectural novelty and biological grounding over safe incrementalism. This is DIVERGENT ideation — generate, do not prematurely converge.

SOFT CONSTRAINTS (for actionability, not to over-filter — flag tension, do not self-censor): bio/neuroscience-analogous; ANTI-HOMUNCULUS (every mechanism is a LOCAL geometric/energy/dynamical process or a measurement of one — never a supervisor/arbiter/if-metric-then rule); batch-offline-capable (a sleep/wake consolidation split; online error-driven writes are banned); contextual-completion (attractor settling) not token-prediction; ideally expressible over (or a principled reason to extend/replace) the FHRR+MHN substrate. The memorization FLOOR (role-selective recall) already works and is not in question — this is about the STRUCTURE layer.

DO NOT REINVENT (already killed/exhausted — cite if a new idea collapses to one): flat-codebook co-occurrence growth (121/123/124); error-driven & reconstruction codebook growth (Reports 017/018, below random); shaping the flat value codebook with a predictive/contrastive objective (Report 119 WS-InfoNCE → d_eff collapse, memory-not-learner); the Dorrell rectangular-support generalization route (does not port to FHRR).

For EACH idea you generate, give: (1) the BIO ANALOGY + the primary literature (name papers/researchers; web-research to ground it); (2) the MECHANISM (how structure forms from experience); (3) WHY IT IS ARCHITECTURALLY DIFFERENT from a flat codebook; (4) the BOUND-ESCAPE HYPOTHESIS — why this substrate could carry paradigmatic/relational structure that a flat co-occurrence code provably cannot; (5) PROJECT-FIT (anti-homunculus? batch-offline? FHRR-native or substrate-change?); (6) a CHEAP-TEST sketch (the cheapest substrate-free probe that would kill-or-greenlight it, in the spirit of the project feasibility oracles); (7) a REINVENTS-KILLED? flag. Web-research aggressively to ground the bio analogy and the mechanism — cite real work.
`

phase('Diverge')

const FAMILIES = [
  { key: 'hippocampal-entorhinal', brief: `Hippocampal-entorhinal STRUCTURED representational spaces: the Tolman-Eichenbaum Machine (Whittington et al. — factorized structure×content, grid×place conjunction), grid/place cell codes as a learned metric/topological space, the successor representation as a MAP (not a codebook), relational memory / schema / cognitive-map theory (O'Keefe, Behrens, Eichenbaum), replay-driven schema consolidation. How could a MAP-like / factorized-structure substrate grow paradigmatic structure a flat code cannot?` },
  { key: 'cortical-sparse-geometry', brief: `Cortical sparse-distributed codes & POPULATION GEOMETRY: HTM / cortical columns / sparse distributed representations (Numenta/Hawkins), sparse coding & overcomplete dictionaries (Olshausen-Field) but as a STRUCTURED manifold not a flat lookup, mixed selectivity & high-dimensional geometry (Fusi, Rigotti), neural manifolds / population-geometry / the "geometry IS the computation" view (Chung, Sompolinsky). Could a structured population geometry place substitutable items on a shared low-D manifold its DOMINANT axes capture?` },
  { key: 'assembly-attractor', brief: `ASSEMBLY & ATTRACTOR dynamics: the Assembly Calculus (Papadimitriou-Vempala-Maass — projection/association/merge of neural assemblies as a computational primitive), continuous-attractor networks (ring/plane attractors), structured-connectivity Hopfield (beyond isolated patterns — patterns with structure BETWEEN them), dynamical-systems / latent-dynamics memory. Could assembly overlap / attractor geometry encode substitutability relationally?` },
  { key: 'temporal-predictive-oscillatory', brief: `TEMPORAL / PREDICTIVE / OSCILLATORY coding: theta-gamma phase coding & phase precession, spike-timing (STDP) sequence learning, temporal-context / temporal-binding models, communication-through-coherence, predictive-coding as a DYNAMICAL substrate (not a bolt-on layer). How could TIMING/PHASE as a coding dimension carry relational/paradigmatic structure a rate/co-occurrence code misses?` },
  { key: 'developmental-selforganizing', brief: `DEVELOPMENTAL / SELF-ORGANIZING / PLASTICITY: self-organizing maps & topographic organization (Kohonen, von der Malsburg), metaplasticity & neuromodulation-gated plasticity (Benna-Fusi cascades, dopaminergic gating), STRUCTURAL plasticity / synaptogenesis / pruning (the wiring itself grows), self-organized criticality, homeostatic + Hebbian developmental dynamics. Could the substrate GROW ITS OWN TOPOLOGY so paradigmatic neighbors become wired-adjacent?` },
  { key: 'generative-freeenergy', brief: `GENERATIVE / FREE-ENERGY / WORLD-MODEL SUBSTRATES: active inference & free-energy-minimizing generative models (Friston) as the substrate itself, energy-based structured generative models, disentangled / slow-feature latents as the PRIMARY representation, the brain as a generative world-model. Could a generative/disentangling substrate factor experience into substitutable latent factors directly (king and queen sharing a royalty factor)?` },
  { key: 'wildcard-orthogonal', brief: `WILDCARD / genuinely ORTHOGONAL bio bets: dendritic computation & nonlinear dendritic subunits (Poirazi, London-Häusser), neuro-symbolic / emergent-syntax substrates, graph / relational neural substrates & relational reasoning, topological / sheaf / simplicial representations of concept spaces, glial/neuromodulatory FIELD computation, hyperbolic / non-Euclidean embedding geometry for hierarchy, anything genuinely orthogonal to all of the above. Surprise us — but ground it.` },
]

const ideas = await parallel(FAMILIES.map(f => () => agent(SHARED + `
YOUR FAMILY = ${f.key}.
${f.brief}
Generate 2-4 CONCRETE Phase-3 substrate candidates within this family (and adjacent to it). Web-research each to ground the bio analogy and mechanism in real literature (cite papers/researchers). For each candidate give all 7 fields from the SHARED spec. Be expansive and specific — concrete mechanisms, not vibes. End with a one-line "best bet in this family" pick.`,
  { label: `diverge:${f.key}`, phase: 'Diverge' })))

phase('Curate')

const ideasText = FAMILIES.map((f, i) => `\n===== FAMILY: ${f.key} =====\n${ideas[i] || '(no output)'}`).join('\n')

const [synthesis, critic] = await parallel([
  () => agent(SHARED + `
You are the CURATOR. Below are divergent idea memos from 7 bio-mechanism families. Pool, DEDUPLICATE, and curate them into a single ranked MENU of the most promising ENTIRELY-DIFFERENT Phase-3 substrates (beyond the flat codebook).
${ideasText}

Produce a ranked menu (aim for 8-14 distinct candidates, the strongest across families). For each: a crisp name, the bio analogy + key primary, the mechanism in 1-2 sentences, why-different-from-flat-codebook, the BOUND-ESCAPE hypothesis (why it could carry paradigmatic structure the flat code cannot), project-fit (anti-homunculus / batch-offline / FHRR-native-or-substrate-change), the cheap-test sketch, and a REINVENTS-KILLED flag (GENUINELY-NEW / SHARES-A-FAILURE-MODE / REINVENTS-KILLED). Then name the TOP 3-4 highest-upside genuinely-different bets and say WHY each is the strongest, and recommend the single cheapest divergent probe to run first. Be honest where a whole family collapses back toward co-occurrence statistics (the thing the bound is about).`,
  { label: 'curate:menu', phase: 'Curate' }),

  () => agent(SHARED + `
You are the COMPLETENESS CRITIC. Below are divergent idea memos from 7 bio-mechanism families.
${ideasText}
Your job is to find what is MISSING — not to rank what is present. Answer: (1) what whole FAMILY of biologically-analogous mechanisms for growing structure-from-experience did the 7 generators NOT cover (name it, cite a primary, sketch a candidate)? (2) which generated ideas are secretly the SAME idea or secretly collapse back to co-occurrence statistics (the bound's trap)? (3) what is the single most UNDER-EXPLORED, highest-upside bio bet for escaping the subdominant-modes bound, and why is it under-explored? (4) is there a framing in which the project whole "grow paradigmatic structure" goal is mis-posed, and a different bio-analogous capability target would dissolve it? Be the agent that asks what everyone else missed.`,
  { label: 'curate:completeness', phase: 'Curate' }),
])

return {
  menu: synthesis,
  whats_missing: critic,
  note: 'Divergent brainstorm of non-flat-codebook bio-analogous Phase-3 substrates. Pair with the Option-B latent-layer grounding (separate run). Both feed a user decision on the next architectural direction after the flat-code growth family was exhausted (Reports 121/123/124).',
}
