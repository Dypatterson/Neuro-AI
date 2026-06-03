# Seed design — the growing relational-graph energy substrate

**Branch:** `design/growing-graph-substrate`. **Status:** design exploration, nothing built or run.
**Origin:** the WF4 completeness critic's single un-invented direction + the 2026-06-02 novelty audit.
**Read first:** [`README.md`](README.md) (the full session arc), [`invented-substrate-slate.md`](invented-substrate-slate.md).

This is the seed, not a frozen precommit. Its job is to make the design *specific enough to attack* — to
turn "growing graph substrate" into a concrete object with a named admissibility crux and a first cheap
kill-test. Freeze a real precommit (CLAUDE.md preamble + headline metric + controls) before any numbered
run.

---

## 1. The design in one paragraph

A memory atom is **a node's position in a graph the experience stream grows.** Per-occurrence experience
instantiates nodes and adds/strengthens **edges by a local co-activation rule** (structural plasticity =
the substrate's *topology* changes; the plastic object is the graph, not weights on a fixed lattice). The
energy is a **graph-diffusion / Laplacian-smoothness functional**; recall is **local diffusion settling**
from a cue (Andersen–Chung–Lang-style push, runtime ∝ community size, *not* a global eigendecomposition).
**Abstraction = graph communities** that emerge as the graph densifies — found locally, never as a global
eigenvector. **Paradigmatic class** (king/queen) = co-membership of a community even with no direct edge.
**bind vs bundle** = an edge that *persists across contexts* (a structural tie) vs one that *decays*
(transient co-fire). Generation = **annealed sampling** of the diffusion (not descent to its argmin).

## 2. Why this is the one shape that dodges the central wall

The 121–132 bound is about a **fixed linear operator over a flat type-averaged code**: a local projection
is one step of power iteration → the dominant collocational mode; the paradigmatic signal lives in
subdominant modes that local single-projection can't reach. **Communities of a grown topology are not
eigenvectors of a fixed operator.** Local diffusion (ACL) provably recovers a community in time ∝ its
size, with **no global eigendecomposition** — so "the partition global k-means/SVD reaches" has, here, a
*provably-local* analogue. And the code is per-occurrence-by-construction (a node's graph position), so
the **type-collapse low-pass filter never runs** (no `C += Pᵀ@P` averaging step). Both load-bearing walls
are sidestepped *by the substrate's form*, not by a cleverer reader.

## 3. The honest prior: the closest published cousin already works — via the banned means

**CSCG (Clone-Structured Cognitive Graph, George et al., Nat. Commun. 2021; "Space is a latent sequence,"
Sci. Adv. 2024)** already realizes the *outcome*: graph learned from sequences; **clones = context-specific
copies = the per-occurrence/un-collapse move**; community detection on its transition matrix recovers
modular abstraction; generative; transitive inference + schema transfer; hippocampal. It is the
**non-backprop cousin of TEM** (which this project already tested in 124/125/129).

**So the deliverable is NOT a new capability — it is the admissible realization of a validated one.**
CSCG learns by **EM (global forward–backward over sequences)**. That is the crux:

> **The genuinely-unbuilt thing = the LOCAL / ENERGY / GROWN realization of the CSCG-class outcome — no
> EM, no global eigendecomposition, no backprop.** Everything novel and everything hard lives here.

## 4. The admissibility crux (where this dies if we're not careful)

Three places the design can silently re-import a banned global computation. Each must have a named local
substitute *before* a build is licensed:

| Banned global step | The CSCG/standard version | Required local substitute |
|---|---|---|
| Structure learning | EM / Baum–Welch forward–backward | a **local co-activation edge rule** (GNG/competitive-Hebbian: connect co-active nodes; age/decay unused edges; insert nodes at local error) |
| The partition | global spectral clustering / Lloyd k-means | **local diffusion push** (ACL personalized-PageRank / heat-kernel), seeded from a cue, runtime ∝ community size |
| Generation | sampling a globally-normalized model | **annealed local sampling** of the diffusion (Langevin/Gibbs on the graph energy; temperature = a broadcast neuromodulatory scalar, *not* a per-node router) |

**Anti-homunculus checks (must pass cleanly):**
- *Who decides an edge exists?* Local co-activation + a fixed decay/age schedule — not a controller.
- *Who decides a node splits/is added?* A fixed local error/bimodality threshold (generic scaffold,
  precommitted, never tuned to make king/queen split) — the C.2.2 splitting-tension lineage applies.
- *Who decides community membership?* Nobody reads it and acts; it is the settled basin of local
  diffusion — a *measurement* of a local dynamic, not an arbitration.
- *Is there a global error / downward reconstruction weight?* Must be **no** (the LPR/Millidge trap that
  killed the one WF4 reject). Diffusion settling is descent on a graph-Laplacian energy with no learned
  transpose.

**The single sharpest risk:** if recovering the communities that carry paradigmatic structure *requires*
a global pass (full PageRank to convergence, full eigendecomposition, EM to a global optimum), the local
version re-collapses to the SR null. The whole bet is that **strongly-local diffusion is enough.**

## 5. The decisive first oracle (cheap, substrate-free, before any build)

**Question:** does grown-per-occurrence-**local-diffusion** community structure separate a planted
paradigmatic class where the **SR null** (fixed / type-collapsed / global-eigenvector) does not?

**Design (planted-first, then real):**
1. **Planted corpus** with a known paradigmatic class {A,B} that **never co-occur** but always share
   second-order neighbors (same role-slots), plus planted collocational distractors (the subdominant-mode
   wall, planted-recoverable).
2. **Build two objects from the same stream:** (a) the **SR / type-transition operator** read by its top
   eigenvectors (the 124/125 null, the negative control); (b) a **grown per-occurrence graph** (local
   co-activation edges, node-per-token-instance or clone-per-context) read by **local diffusion** (ACL
   push from each cue, no global eigendecomposition).
3. **Headline = CLASS-level differential** (NOT pair-specific B-KILL — verified hostile): does A's diffusion
   community overlap B's (and the rest of the planted class) **above** a frequency/degree-matched
   non-paradigmatic class and above a within-class label-shuffle? Multi-seed.
4. **Pre-committed KILL gates:** (a) **re-collapse** — if the local-diffusion read sits at the SR-null
   floor, OBJECT-limited is falsified for this substrate and it inherits the bound; (b) **global-leak** —
   if a *global* PageRank/eigendecomposition is needed to get the signal (local push at realistic
   truncation fails), the locality claim is dead; (c) **anchor** — a global community-detection flashlight
   must recover the planted class (else the planted signal is broken); (d) **collocational-not-paradigmatic**
   — corr(co-occurrence, community-lift) must stay low (the Report-123 gate), else it's collocation again.
5. **Honest ceiling:** a PASS = "a *local* route to the community structure a *global* method also finds"
   — a real first escape of the single-projection bound, **not** "beats the global method." Exceeding the
   global community-detection ceiling = inflation artifact.

LLN pre-test (minutes, run first): does raw local-diffusion overlap of A-cloud vs B-cloud exceed the
SR-eigenvector cosine? If not, don't build the full cell.

## 6. Open design questions for the branch

1. **Node granularity:** pure per-occurrence (one node per token instance — huge, sparse) vs **clones**
   (bounded context-specific copies, CSCG-style — the natural bound on growth). Clones are probably the
   admissible-and-tractable answer; they are also the explicit fixed-capacity/consolidation lever.
2. **Edge rule:** competitive-Hebbian (connect two nearest/most-co-active) vs co-activation-count with
   decay. Must be local + bounded + decay-pruned (stability–plasticity).
3. **Diffusion read:** personalized-PageRank push vs heat-kernel vs label-propagation — which is the most
   biologically-plausible *and* strongly-local?
4. **Energy form:** is the graph-Laplacian smoothness functional the right Lyapunov, and does its settling
   = the diffusion read (so "settling IS the computation")? Make the energy explicit.
5. **Growth bound / consolidation:** how does replay/consolidation prune and merge clones to keep capacity
   bounded (the fixed-capacity tension)? This is where the project's existing replay machinery plugs in.
6. **Relationship to the FHRR substrate:** is the graph a *replacement* for FHRR+Hopfield, or does FHRR
   bind node-identity/edge-labels (a hybrid)? The user authorized a literal new substrate; a hybrid may be
   the cheapest path to reuse.
7. **Continual/multimodal:** cross-modal edges = same local rule (a fresh discriminative channel for the
   wall) — defer until the monomodal text oracle is settled, but design the node/edge schema to admit it.

## 7. What would make this graduate vs. be a null

- **Graduate:** the local-diffusion community read separates the planted (then real) paradigmatic class
  *class-level*, beats the SR null CI-disjoint, needs no global pass, stays collocational-free, multi-seed
  — i.e. **grown-local-diffusion escapes the SR null.** Honest ceiling: a local route to the global
  community code, the first genuine escape of the single-LAYER/single-projection bound.
- **Null (most likely per the skeptics):** local diffusion on the grown graph re-collapses to the SR null
  (the SPPMI/transition normalization is corpus-wide; a per-occurrence row is a context-masked copy of the
  type row; local truncation loses the subdominant signal) → the bound is the *reader's* even here, and
  the object-limited hypothesis is finally falsified. That is itself a clean, valuable result (it closes
  the last un-probed degree of freedom).

## 8. First moves on the branch

1. Freeze a real precommit (headline = class-level differential; controls = SR-null negative control,
   global-community anchor, freq/degree-matched class, within-class shuffle, collocational-gate; the 4
   kill-gates above).
2. Build the substrate-free oracle (§5) — reuse the exp61/63/65/68 corpus + pair list + planted-corpus
   harness; add a local-diffusion (ACL push) reader and a grown-graph builder. Planted-first.
3. Run the LLN pre-test; then the planted cell; then (only if it survives) the real-WikiText cell.
4. Adversarial 3-lens verification before banking ANY positive (the 129 discipline). Audit for the three
   global-leak traps (§4) explicitly.
