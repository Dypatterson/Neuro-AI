# Findings — mechanisms for GROWN hierarchical abstraction (2026-06-22)

*Deep-research session, 5 parallel angles + adversarial 3-vote-style per-claim verification.
Scope locked with the user: buildable shortlist; both fields, ML-leaning; injected-hierarchy
methods included as the ceiling. Plan: [research-plan.md](research-plan.md). Builds on the
06-21 frontier recon ([../2026-06-21-frontier-recon/findings.md](../2026-06-21-frontier-recon/findings.md))
and the §4½ reframe ([../../notes/RETROSPECTIVE-program-close-2026-06-21.md](../../notes/RETROSPECTIVE-program-close-2026-06-21.md)).*

> **Verified + extended (2026-06-22, 4-agent workflow):** all 9 load-bearing claims below were
> adversarially re-verified — none refuted, headline survives ~90% (3 precision fixes applied inline).
> New candidates (DreamCoder, grokking-as-gate-signal, energy-composition, SAE-null) and the
> pre-training-to-initialize design live in the companion:
> [verification-and-extensions.md](verification-and-extensions.md).
>
> **Round-2 re-verification (2026-06-22):** a deeper 5-agent + repo audit corrected several items here
> — CSL ~89.9 (not 90–91); neuro "stabilizer-not-manufacturer" SOFTENED; the grokking gate is undefinable
> on MCD; the splitting→hierarchy leap is unsupported; and our own bar (COGS-structural / MCD) is partly
> confounded (ReCOGS / covariate-shift). Full ledger + revised next steps:
> [verification-round-2-and-seams.md](verification-round-2-and-seams.md).

## Bottom line

The grown-vs-injected arbiter cuts the literature cleanly, and the cut is sobering: **every
mechanism that actually clears the hard compositional splits (COGS structural, CFQ-MCD) injects a
tree / grammar / algebra prior; every *purely emergent* latent-hierarchy mechanism either fails the
hard splits or has never been tested on them.** The §4½ unified bet — *grow a compositional
hierarchy level-by-level, with consolidation-as-stabilization gating each level* — is **almost
entirely untested in its faithful computational form**, and there is exactly one buildable,
anti-homunculus-clean mechanism that instantiates the *measured* stabilize→grow gate: **neuron
splitting at stationarity (Splitting Steepest Descent / Firefly)**, never run on compositional
generalization. That single untested cell is the headline recommendation. The neuroscience supports
the *stabilizer-not-manufacturer* half of the premise solidly, but the *stabilize-N-gates-N+1* half
is a biology-**motivated** engineering bet, not an established fact.

## 1. Verdict table

Tags: **grown** = hierarchy/structure is induced/emergent from the learning signal; **injected** =
tree/grammar/recursion supplied by architecture or init. **Buildable** = implementable on the small
SCAN-MCD/COGS seq2seq harness (`experiments/87/94/99`). "Hard-split evidence" = COGS *structural* /
CFQ-MCD specifically (the bar; vanilla seq2seq ≈0–18%).

| Candidate | grown / injected | buildable | Hard-split evidence | Verdict | Strongest disconfirmer |
|---|---|---|---|---|---|
| **Splitting Steepest Descent (Liu 2019)** | **grown (measured gate)** | **yes** | **untested** | **★ V-as-gate / U-on-comp-gen** | only ever grew *width for accuracy/compression*; never built a compositional level | 
| **Firefly arch. descent (Wu 2020)** | **grown (measured gate)** | **yes — grows width AND depth** | untested | ★ V-as-gate / U | its own finding: splitting-at-stationarity *alone* can't escape local minima (needs fresh neurons) |
| Neural Data Router (Csordás 2021) | grown (emergent routing) | yes | COGS 81%, SCAN 100%, CFQ **output-length** 81%; **no CFQ-MCD** | V-on-easy / U-on-hard | reports the easy CFQ output-length split, skips MCD — metric-shopped |
| CSL — induced QCFG + aug (Qiu 2022) | grown (grammar induced from data) | yes (as data-augmenter) | **CFQ-MCD ≈89.9%** (R2-corrected from "90–91"; COGS ↑) | V (with caveat) | grown *offline as a generator* feeding pretrained T5 — not grown inside the learner; T5+CSL-augmentation > T5–CSL ensemble |
| Universal Transformer / ACT | injected (weight-tied block) | yes | **CFQ-MCD 18.9% ≈ vanilla 17.9%** | R as hierarchy-grower | tied-block recurrence is statistically indistinguishable from vanilla on MCD |
| PonderNet / looped transformers | injected (tied block + halt) | yes | untested on hard splits (parity/addition only) | U | "more steps" = more iterations of the *same* function; no MCD number |
| ON-LSTM / PRPN / StructFormer | grown | partly (LM-shaped) | untested | U | induced trees are restart-inconsistent & fragile (Williams 2018; Htut 2018) |
| Gumbel-Tree / RL-SPINN latent trees | grown | no (classifier) | untested | R | TACL 2018: trees restart-inconsistent, "resemble no formalism," underperform a plain LSTM; RL collapses to >99% left-branching |
| DIORA / URNNG / compound-PCFG | grown | heavy (chart/generative) | untested as comp-gen solver | U | evaluated only as unsupervised parsers (PTB F1 / perplexity); no hard-split transfer |
| Hierarchical VAE / Ladder / slot hierarchies | grown | no (vision/generative) | **untested (clean null)** | U | never bridged to symbolic seq2seq; evaluated on image synthesis/gSCAN-visual |
| Net2Net / Progressive Nets / GradMax / DEN | grows *width/modules* | yes | untested | R as hierarchy-grower | added depth is identity-init or task-curriculum-gated — capacity, not compositional structure |
| LeAR (latent tree + algebra) | **injected** (hard-wired syntax↔semantics homomorphism) | moderate | **COGS 97.7%, CFQ 90.9%** | V-as-ceiling | the near-perfect number rides the injected algebra prior |
| Differentiable Tree Machine (sDTM) | **injected** (precomputed car/cdr/cons TPR) | moderate | 100% *synthetic* tree2tree; sDTM→COGS-ish | V-as-ceiling | flagship 100% is a self-generated tree task, not CFQ-MCD |
| NeSS / NSR (symbolic stack/recursion) | **injected** (symbolic engine + equivariance) | harder | SCAN 100%, PCFG 100%; **no MCD** | V-as-ceiling | recursion supplied by the machine; SCAN/PCFG easier than MCD |

## 2. Ranked buildable shortlist (grown candidates worth an experiment)

**The ceiling to approach** (so each candidate's gap is concrete): on **CFQ-MCD**, injected methods
reach ~90% (LeAR 90.9% MCD-mean [table-confirmed]; CSL ~89.9% [R2-corrected]) while flat/emergent sits at ~18% (UT 18.9% ≈ vanilla 17.9% [table-confirmed]); on **COGS
structural**, LeAR reaches 97.7% while plain seq2seq is 0–12%. **That ~18%→~90% / ~0%→~97% gap is what
a grown mechanism has to close to matter.**

1. **★ Stabilization-gated growth via neuron-splitting-at-stationarity (Splitting Steepest Descent /
   Firefly), run on SCAN-MCD/COGS.** This is the single faithful, buildable, anti-homunculus-clean
   instantiation of the §4½ unified frame: growth of new capacity fires *because* the current level's
   parameter descent has plateaued (a local functional steepest-descent move, not a supervisor's
   `if-then`), which is exactly "stabilize level N → grow level N+1." It is the one cell in the
   grown × tested matrix nobody has filled.
   - **Decisive controls:** (a) vs **scheduled growth** (same growth operator on a fixed iteration
     clock — isolates whether the *measured stability gate* is load-bearing vs growth-per-se); (b) vs
     **injected-hierarchy ceiling** (Tree-LSTM / LeAR — the gap to close); (c) vs **flat-baseline**
     (no growth). Win = gated-growth beats *both* scheduled-growth and flat-baseline on COGS
     structural / CFQ-MCD, CI-disjoint, ≥8 seeds.
   - **Honest caveat / pre-registered null:** Firefly's own result is that splitting-at-stationarity
     *alone* gets stuck — so a pure stabilize-then-grow gate may stall, and if only the *injected*
     hierarchy clears the bar, the grown door closes and the bank hardens. The experiment is decisive
     either way, which is the point.
   - Sources: Liu 2019 (arXiv 1910.02366), Wu 2020 (arXiv 2102.08574).

2. **Neural Data Router as the *abstractor* half, paired with (1).** NDR's emergent data-dependent
   routing (COGS 81%, SCAN 100%) is the closest thing to a *grown compositional process* in the ML
   sweep, but it reports only the easy CFQ **output-length** split (81%) and skips CFQ-MCD. Worth
   running NDR against CFQ-MCD directly (cheap: it's a modified UT on the existing harness) and as the
   composition operator inside the gated-growth loop.
   - **Decisive control:** NDR on CFQ-MCD vs the UT 18.9% bar — does emergent routing survive compound
     divergence, or is COGS-81% / output-length-81% another metric-shopped easy split? Source: Csordás
     2021 (arXiv 2110.07732).

3. **CSL-style induced grammar as a *grown* augmenter (lower priority, near-ceiling already).** The one
   genuinely grown mechanism that already touches the hard splits (CFQ-MCD ~90%), but it grows the
   grammar *offline as a data generator*, not level-by-level inside the learner. Use it as the **grown
   ceiling reference** rather than a novel bet; the open question it leaves is precisely whether the
   in-learner grown hierarchy of (1) can match what the offline-induced grammar achieves.
   Source: Qiu 2022 (NAACL 2022.naacl-main.323).

**Import as a coupling trick, not a candidate:** ProGAN / MSG **fade-in** (α: 0→1 blends a freshly
grown level into the stabilized one) is the clean way to add level N+1 without destabilizing level N —
worth importing into (1), but note it is *scheduled*, not a measured gate, so it supplies the coupling,
not the gate.

**Demoted (collapse into injected or untestable):** all pure latent-tree induction (ON-LSTM, PRPN,
Gumbel-Tree, DIORA, URNNG, compound-PCFG, StructFormer) — restart-inconsistent trees, untested on hard
splits; hierarchical VAE / slot hierarchies — no bridge to symbolic seq2seq; capacity-growers (Net2Net,
Progressive Nets, GradMax, DEN) — width not hierarchy; UT/PonderNet/looped — tied structure, MCD ≈ vanilla.

## 3. Neuroscience premise status (§4½, two halves)

> **R2 correction (2026-06-22):** Half (1) is **SOFTENED, not cleanly shored up** — an adversarial neuro
> re-check found genuine *manufacture* evidence (Wagner 2004 causal "restructuring"; Lewis & Durrant iOtA
> "builds schemata"; Wittkuhn 2025 replay *forms* successor representations). Honest reframing: consolidation
> does **both** — stabilizes existing traces AND **manufactures new abstraction by overlap-driven
> recombination**. See [verification-round-2-and-seams.md §7](verification-round-2-and-seams.md). The text
> below overstates the "does not manufacture" half.

**Half (1): "consolidation STABILIZES, it does not MANUFACTURE abstraction" — SHORED UP.** Synaptic
homeostasis (Tononi–Cirelli SHY) is a subtractive renormalizer; CLS-replay interleaves to *enable* a
slow neocortical learner; the actual abstractors are *separate* (the slow cortex; the entorhinal
structural basis in TEM, which factorizes structure while hippocampus binds specifics). The one source
that reads as "manufacture" — Lewis & Durrant's iOtA — is a theoretical model whose "strengthen-the-
overlap" mechanism is (*our reading*) mathematically indistinguishable from Hebbian stabilization over
interleaved replay, and a 2026 well-powered null-replication undercuts active sleep-driven abstraction. **Your own
record agrees** (138 replay carries compounding; 136 protection; 139 soft-anchor — all stabilizers; 147/150
manufacture-asks hurt). *Verdict: well supported.*

**Half (2): "the abstractor is a GROWN hierarchy with per-level stabilization GATING" — SPLIT.**
- *Grown, sequentially-maturing hierarchy (weak form):* **well supported.** The sensorimotor→
  association→PFC (S-A) axis matures in order, experience-dependently (large-N, reproducible);
  temporal-receptive-window hierarchy is immature in childhood; thalamocortical gradients track the
  same axis. A purely fixed/innate hierarchy is refuted.
- *Stabilize-N causally GATES forming-N+1 (strong, load-bearing form):* **UNESTABLISHED.** The
  "cascade of critical periods" is offered explicitly as a *framework/hypothesis* (Larsen/Sydnor 2023);
  the documented mechanisms are *local* (per-area PV/E:I maturation) and could be staggered by
  independent regional or thalamic clocks rather than bottom-up gated. The sharpest disconfirmer: a
  2025 cataract-reversal study where higher-level categorical coding *survives a permanently degraded
  lower level* — higher representation formed without a stabilized level below. The best gating
  evidence, **Tse 2007**, shows a pre-existing stable schema *accelerates integration* of new items —
  schema-gated *assimilation*, **not** consolidation literally building a new tier.

*Net for the build:* lean on "grown, sequentially-maturing, stabilizer-enabled hierarchy" as
well-motivated; treat "each level's stabilization gates the next" as a **biology-motivated engineering
bet**, presented as such — not as established neuroscience. This is exactly the honesty the §4½ reframe
already adopted.

## 4. Headline recommendation

**Build the one untested cell: stabilization-gated growth driven by neuron-splitting-at-stationarity, on
the SCAN-MCD / COGS-structural harness, against an injected-hierarchy ceiling and a scheduled-growth
control.** It is the only mechanism in the entire sweep that (a) instantiates the §4½ "stabilize level N →
grow level N+1" gate as a *measured*, anti-homunculus-clean local dynamic (growth fires because descent
plateaued — not a supervisor), (b) is buildable on the existing harness, and (c) has *never been run on
compositional generalization*. The program has validated the stabilizer (136/138/139) and a flat
abstractor (133); this is the first faithful test of the grown hierarchy itself.

The result is decisive in both directions, matching the project's discipline:
- **If gated-growth beats scheduled-growth *and* the flat baseline on CFQ-MCD / COGS-structural** →
  the first evidence that a hierarchy can be *grown* (not injected), and the structure-injection bind
  reframes to "injected **or** stabilization-grown."
- **If only the injected ceiling (Tree-LSTM/LeAR/DTM) clears the bar, or the gate stalls** (the Firefly
  warning) → a clean earned negative that *closes the grown door* and hardens the bank — and it lands on
  the discriminating MCD arena, not a toy.

Either outcome is a real, citable result. The convergent disconfirmers (emergent trees are
restart-inconsistent and fail the hard splits; capacity-growth ≠ structure-growth; the neuro gating
premise is unestablished; splitting-alone stalls) make this a **high-risk, high-information** bet — which
is precisely the kind of decisive experiment the §4½ ★ reopen-door was reserved for.

---

### Provenance / caveats
- 5 angles, ~46 verified claims, adversarial claim→disconfirmer→verdict per claim. Hard numbers
  (UT 18.9% CFQ-MCD [table-confirmed R2]; LeAR 97.7%/90.9%; CSL ~89.9% [R2-corrected]; COGS-structural seq2seq 0–12%) are primary-sourced.
- Several neuroscience publisher PDFs returned HTTP 403 to direct fetch; the S-A axis, cascade framing,
  PV/E:I mechanism, SHY, CLS, Tse 2007, and the cataract dissociation were each corroborated across
  multiple independent results, but a few mechanistic quotes are excerpt-level (flagged U where so).
- Metric-shopping is the main distortion in the ML literature: papers report SCAN-length / COGS and
  quietly omit CFQ-MCD-mean, the one split where everything still fails. Treat any "solves
  compositionality" claim that omits an MCD number as untested-on-the-hard-split.
