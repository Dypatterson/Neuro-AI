# Phase 5 Unified Briefing — for research subagents

## What the project is

A neuroscience-inspired cognitive substrate built around three pieces:
**FHRR** (Fourier Holographic Reduced Representations — a vector symbolic
architecture using complex unit-magnitude vectors with element-wise
multiplication as binding and circular convolution / addition as
bundling), **Modern Hopfield Networks** (Ramsauer 2020; energy-based
exponentially-stable associative memory with softmax retrieval), and an
**emergent codebook** (atoms that grow via Hebbian reinforcement during
retrieval, then are pruned / consolidated). The architectural target is
*contextual completion*, not next-token prediction. The system is
explicitly **not** a transformer or vector DB — it must produce apparent
"decisions" as local geometric dynamics, never as a supervisor module
that arbitrates between subsystems (the project's binding
*anti-homunculus filter*).

## Where things stand

- **Phases 1–4 graduated.** Phase 4 (post-death substrate, frequency-
  weighted consolidation) graduated at n=10 multi-seed with CI on
  W=3 meta-stability.
- **Phase 5 is the active phase and has not graduated.** Phase 5 is
  about **K-branch energy-guided structural retrieval**: for a cue,
  seed K parallel HAM-settling branches; the goal is that branches
  seeded by *role-binding* similarity (Condition B, "role prior") land
  at lower final-state energy than branches seeded by *content* similarity
  (Condition A, "content prior"). The headline metric is
  `ΔE = E_content − E_role`, paired per cue, with CI strictly > 0 AND
  magnitude ≥ 5.5e-3 (a pre-committed substrate-noise floor at D=4096,
  β=10, N≈1064), n_seeds ≥ 10.

## What has failed and why

Across reports 041–061 (Apr–May 2026), the project has tried multiple
paths to make role-prior branches actually find lower-energy states than
content-prior branches:

1. **Naive K-branch retrieval (default design).** Branches collapse —
   per-branch state diversity on the post-death substrate is
   essentially zero (6.7e-6 to 2.7e-3 across seeds × K=4). All branches
   converge to the *same* basin regardless of prior, so ΔE ≈ 0.

2. **Pre-death substrate scale-up (Path A).** N pushed from 12 to 1064
   atoms. SNR diagnostic shows signal scales 50× but noise scales 95× —
   SNR is approximately invariant in N. Scaling doesn't rescue.

3. **Varner-style log-prior softmax-bias spike (Path C, latest).** An
   additive per-pattern log-multiplicity boost. At gain=1 on the
   slow-store substrate: ΔE = +0.0089, CI [+0.0065, +0.0116], 10/10
   seeds positive — *crosses* the magnitude floor (1.615× over). BUT:
   - `random_lowest` = 0.37 — random priors land at the lowest-energy
     branch 37% of the time (chance is 33% for 3 conditions). Role
     barely distinguishes from random.
   - `hit_role` = 0.003, `rank_role` ≈ 443 / 489 atoms. The role-target
     basin retrieval is essentially absent.
   - 2×2 ablation: the log-prior channel is the dominant lever
     (γ-only sub-floor at 0.44×; gain-only above floor at 1.33×).
   - No-schema-store control amplifies ΔE to +0.0915 (16.6× floor) —
     classified as an **arbitration-shape positive control** by the
     anti-homunculus filter, not portable evidence. It just shows the
     log-prior spike *can* manufacture energy margins when paired with
     `argmax(content_sim) → boost-that-atom`.

**Core diagnosis the project has converged on.** The substrate does not
appear to have role-target attractors. The log-prior bias produces
energy margins by reshaping the retrieval logits, not by causing the
flow to settle into a different basin. The role-target basin retrieval
metrics (`hit_role` ≈ 0, `rank_role` near random) are the smoking gun
that the apparent ΔE win is not structural retrieval.

## Path decisions currently open

- **Path B' (Phase 5 closure + pivot)**: close Path C, document Phase 5
  as not-graduated, pivot to surprise / prediction-error-driven replay.
- **Path A (closure-paper evidence)**: capacity-proportional scale-down
  probe as mechanistic explanation of the energy-margin artifact.
- The user has rejected closure framing — they believe better mechanisms
  exist. **The research brief is to find them.**

## Hard architectural constraints (do not violate)

- **Anti-homunculus filter.** Every mechanism must be a local geometric
  dynamic or a measurement of one — never an arbitration over them.
  No `if X then do Y` rule that reads a metric and triggers a response.
  No supervisor module that picks which subsystem wins. "Branch
  selection" must remain energy-only.
- **Substrate is FHRR + Modern Hopfield.** Don't propose replacing it
  with transformers, vector DBs, or LLM-as-memory.
- **Pure-Python reference backend must survive** (i.e. avoid CUDA-only
  proposals; PyTorch with MPS/CUDA is the production path).
- **Hebbian for runtime, error-driven only in batch offline passes** —
  online error-driven codebook updates are banned by an existing rule.
- **Substrate-pure metrics gate, readouts are drill-downs.** Don't
  propose evaluating on R@K or top1; the headline must be a
  substrate-intrinsic quantity (energy, distance, dispersion).

## Parked / high-leverage ideas the project has already named

- **Frequency-weighted Benna-Fusi α** — never built; named as the key
  experiment for the architecture's "compression → abstraction" claim
  (multi-timescale α tied to atom frequency).
- **Consolidation-geometry regime classifier** (d̄, d_eff per atom) —
  designed but not built. Would classify atoms as feature/prototype
  regime based on cap geometry.
- **Empirical θ′(β) calibration spike** — would replace 1/β
  approximation with calibrated mapping per Vangara & Gopinath E1
  protocol.

## Papers the project is already drawing on

Vangara & Gopinath *Geometry of Consolidation* (2026), Kashyap et al.
*HEN* (2024), Alonso & Krichmar *SQHN* (Nature Comm 2024), Krotov &
Hopfield *Dense Associative Memory* (2016), Papyan et al. *Neural
Collapse* (2020), Krotov *Hierarchical Associative Memory* (2021),
Sharma/Chandra/Fiete *MESH* (2022), Benna & Fusi (2016), Aljundi et al.
*OCL-MIR* (2019), Dawid & LeCun *Latent Variable EBMs* (2023), Saighi
HRR/replay synthesis (2024), Plate's original HRR work, Kanerva's HDC
foundation.

## What "better than where we are" probably looks like

A mechanism that puts **actual role-target basins into the substrate**,
so that role-seeded settling flows there *naturally*, not because a
prior bias reshaped the logits. The current design assumes role
attractors emerge from FHRR binding + Hebbian consolidation + MHN
retrieval. They evidently don't on this substrate at this scale. The
mechanism gap is somewhere in:

- How roles get *stored* (binding may not produce distinguishable
  basins at this density / β / D).
- How retrieval *explores* (MHN softmax flow may erase the role/content
  distinction during settling).
- How consolidation *selects* what becomes a schema (post-death
  survivors reflect content-retrieval reinforcement breadth, not role
  structure).

The brainstorm should generate ideas about each of these mechanism
gaps, grounded in current literature and in the project's existing
architectural primitives (FHRR, MHN, emergent codebook, replay).
