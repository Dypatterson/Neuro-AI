# Implementation Context — Phase 5 Brainstorm

Scope: what's already in the toolbox in `src/energy_memory/`, with an eye to
what moves are cheap (reuse existing primitives), what's expensive (new
infrastructure), and where new measurements or dynamics can be slotted in
without violating the anti-homunculus rule.

All paths absolute under `/Users/dypatterson/Desktop/Neuro-AI/`.

---

## 1. Substrate-level numerical primitives (`src/energy_memory/substrate/torch_fhrr.py`)

`TorchFHRR` is the FHRR substrate. Vectors are unit-magnitude complex
tensors of shape `[D]` (typical D = 4096); batches are `[N, D]`. Lives on
MPS/CUDA/CPU. Cheap operations available everywhere:

- `bind(l, r)` = elementwise complex multiply. O(D), no sync.
- `unbind(b, role)` = `b * role.conj()`. Exact algebraic inverse.
- `normalize(v)` = elementwise unit-magnitude. O(D).
- `permute(v, shift)` = cyclic roll along last dim. Composes additively
  (`permute(permute(v, a), b) == permute(v, a+b)`). Exactly invertible
  via `permute(v, -shift)`. This is the VSA permutation operator (Plate
  / "Attention as Binding" arXiv:2512.14709). **Useful primitive** —
  it's an angle-preserving group action distinct from bind.
- `bundle(vectors)` and `weighted_bundle(vectors, weights)`: superposition
  + normalize.
- `similarity(l, r)`: scalar cosine. **Forces a CPU sync.**
- `similarity_matrix(query, patterns)`: `[N]` real tensor of cosines via
  `(patterns.conj() * query).real.mean(-1)`. No sync.

### Already-exploited algebraic identities

- **`d_eff` via `(tr G)² / tr(G²)`** instead of eigvalsh. Same value as
  participation ratio of centered Gram, but: (a) O(N²·D) not O(N³), (b)
  **differentiable through autograd**. This is the key load-bearing
  identity — it's why Candidate B's repulsion force can be cheap.
- **`substrate_energy_anti` and `repulsion_force`** are already wired:
  `H_anti = -α·log(d_eff)`, with `repulsion_force` returning the per-atom
  descent direction via a single `torch.autograd.grad` call. PyTorch's
  Wirtinger convention gives `-grad` as descent direction. This means
  **autograd flows through pattern positions** in this codebase — not
  a common move, and a strong adjacency to differentiable-VSA / iEnergy
  techniques.
- `alpha_anti` is fixed at construction; per the design note's
  precondition it is **not** adapted from observed d_eff. Same for the
  `repulsion_step_size` in `ReplayConfig`. This pattern (fix
  meta-parameters at construction, let the substrate do the work) is
  the project's anti-homunculus discipline.

### Hot/cold for substrate ops

- Cheap, GPU-async: `bind`, `unbind`, `normalize`, `permute`,
  `similarity_matrix`, `d_eff`.
- Sync-forcing: `similarity()` (returns float), `top_k()`, anything
  printing a tensor.
- New per-atom geometric properties that are cheap to compute from
  patterns + Gram: any reduction over rows of `patterns @ patterns.conj().T / D`
  (mean, max, top-k, kth-quantile, entropy of normalized row). The
  `_coverage_redundancy_instantaneous` function already does row-max;
  the same Gram is sitting there for free.

---

## 2. Modern Hopfield retrieval (`src/energy_memory/memory/torch_hopfield.py`)

`TorchHopfieldMemory` stores `[D]` patterns and retrieves via iterative
softmax settling. Key facts:

- `retrieve(query, beta, max_iter, tol, kernel)` runs **all** `max_iter`
  iterations on-device then replays convergence detection with a single
  batched CPU sync. Converged state captured on-device via
  `torch.where(frozen_mask, ...)`. This is the GPU-pipeline discipline
  from CLAUDE.md.
- Kernels: `"softmax"` (default) and `"lsr"` (Lagrangian conjugate of
  truncated-quadratic). Both have the same per-iter shape.
- `score_bias` (used via the `TracedHopfieldMemory` subclass): a
  `[n_patterns]` tensor subtracted from `β·scores` before softmax. **This
  is the universal injection point** for per-atom modulation of
  retrieval. The pipeline already routes two independent biases through
  it (Saighi A_k and step-3 E_i-weight). Adding a third per-atom signal
  is a one-line change at the call site.
- `_pattern_matrix()` returns the cached `[N, D]` stack; invalidated on
  store/remove. Mutations to patterns require `invalidate_cache()`.

### `TracedHopfieldMemory` (`phase4/trajectory.py`)

Subclass that returns a `TrajectoryTrace` alongside the retrieval result.
Trace captures, per settling step: top-k indices, top-k weights,
softmax entropy, energy. Derived signals already wired:

- `engagement()` = mean entropy across snapshots.
- `resolution()` = final top score (max cosine to any stored pattern).
- `gate_signal()` = `engagement * (1 − resolution)` — the unresolved-
  but-forceful gate.

**Adjacent toolbox**: the trace also carries `final_state` (the basin
the settling landed in), `query`, and `converged`. Anything that wants
trajectory-derived per-cue features (curvature of energy descent, basin
switching across iterations, top-k churn) has the data it needs without
new instrumentation.

---

## 3. Consolidation state (`src/energy_memory/phase4/consolidation.py`)

`ConsolidationState` holds one row per stored pattern with these per-atom
fields, all `[N]` tensors on device:

| field | dtype | semantics |
|---|---|---|
| `u` | `[N, m] float32` | Benna-Fusi u-chain (Eq. 10/11). u_1 is fast, u_m is slow. m=6 default. |
| `below_threshold_steps` | int32 | Counter for legacy death-window mechanism. |
| `A` | float32 | Saighi & Rozenberg per-atom self-inhibition. Grows on retrieve, subtracted from β·score. |
| `retrieval_count` | int32 | Per-atom retrieval counter. Powers `alpha_freq_lambda` (freq-weighted α). |
| `r_ema` | float32 | EMA of `r_inst` (coverage redundancy proxy). Powers Candidate A's reinforcement modulation. |

**This is the load-bearing per-atom state surface.** Adding a new
per-atom signal (provenance tag, binding precision, age-of-first-
retrieval, …) is a copy-paste of `retrieval_count`'s scaffolding:
init zeros, grow on `add_pattern`, update somewhere, expose via a
method that returns a `[N]` tensor.

### Anti-homunculus-compatible per-atom additions

The existing state shapes give a template: state-evolves-by-local-rule
(u-chain dynamics, A_k accumulator), or state-is-measurement-of-geometry
(`r_ema` is just an EMA of the Gram row-max). New per-atom signals that
would pass the audit:

- **Binding precision**: `1 − mean_pairwise_distance` of an atom's W
  unbound fillers. Already implemented in
  `phase5/role_fidelity.compute_role_fidelity` for schemas — exact same
  computation works per atom if the atoms are role-bound.
- **Per-atom phase coherence**: `|mean(p_i)|` or some other within-vector
  statistic of the FHRR phases.
- **Provenance**: distance to the source-query that birthed the atom. The
  `Layer2Attractor.source_query` field already holds this for layer 2;
  consolidation atoms don't carry it yet but the slot is obvious.
- **Per-atom autograd-driven flow**: anything expressible as
  `∂(scalar_energy)/∂p_i` is one `torch.autograd.grad` call away
  (existing pattern from `repulsion_force`).

### What `step_dynamics` does per replay tick

1. Bidirectional u-chain Laplacian: `Δu = α·(-2u + u_left + u_right)`.
2. Optional per-row α scaling via `alpha_freq_lambda · retrieval_count`,
   clamped at CFL bound (`α_eff ≤ 0.5`).
3. Optional inhibition decay `A *= (1 − decay)`.
4. Optional `r_ema` update from current `pattern_matrix` (Gram row-max
   on `[N, D] @ [D, N]` → row-max-over-others, EMA-blended at
   `coverage_ema_rate=0.01`, halflife ~100 steps).
5. Death-counter increment (legacy; suppressed when Candidate A is on).

### `effective_strength()` and the death-by-asymptotic-decay design

`effective_strength_i = Σ_k w_k · u_{i,k}` with default weights `2^(1-k)`.
The `retrieval_weight_bias()` method then maps this to a per-atom
softplus bias `softplus((ε − |E_i|) / τ)` with fixed-at-construction
ε=0.05, τ=0.02 — atoms with effective_strength near zero get
exponentially suppressed in retrieval softmax. This **replaces** the
binary `dead_indices()` + `garbage_collect()` controller; when
`coverage_lambda > 0`, `garbage_collect()` is a hard-coded no-op.

---

## 4. Replay loop pipeline (`src/energy_memory/phase4/replay_loop.py`)

`UnifiedReplayMemory` glues substrate + memory + consolidation. The per-cue
hot path is `retrieve_and_observe(query, beta, max_iter, tol)`:

1. Build per-atom `score_bias` by summing Saighi A_k + step-3 E_i-weight.
   **Either or both can be zero.** Returns None when nothing's active.
2. `memory.retrieve_with_trace(query, bias=...)` — traced retrieval (see §2).
3. Compute `gate = trace.gate_signal()`; if above `store_threshold`,
   add trace to the bounded `ReplayStore` (with optional tag-overlap
   collapse and inhibition-of-return suppression bookkeeping).
4. Reinforce `consolidation.u[top_idx, 0]` by `retrieval_gain` (Candidate
   A multiplies by `1 − coverage_lambda · r_ema[idx]` here).
5. `accumulate_inhibition(top_idx)` increments A_k for the winner.

Per replay cycle (every `replay_every` retrievals):

1. Sample `replay_batch_size` traces from `ReplayStore` ∝ `gate × tag_count × suppression`.
2. Re-settle each via traced retrieval with current bias stack.
3. If new resolution > `resolve_threshold`, hand off to `candidate_handler`
   which actually stores the pattern and gets back an index. Each new
   atom enters consolidation at `u_1 = novelty_strength`, with `r_ema`
   initialized to its **geometric equilibrium** (`r_inst` against the
   augmented pattern matrix) per A1 of the discovery-channel design.
4. `_step_substrate_dynamics()`: one call to `consolidation.step_dynamics`
   (with `pattern_matrix` snapshot for r_ema update) and, if B is on,
   one step of `patterns += step · repulsion_force(patterns)` followed
   by `normalize` + cache invalidate.

### Where new dynamics slot in cleanly

- **New per-atom score_bias term**: add a method on
  `ConsolidationState` returning `[N]`, add it to the components list
  in `_score_bias()`. Single touch point, doesn't change any other
  call site.
- **New per-cycle substrate update**: append to `_step_substrate_dynamics`.
  The pattern is: compute force from a substrate-energy gradient,
  step + normalize + invalidate.
- **New per-atom state**: extend `ConsolidationState.__init__` / `add_pattern` / `remove_pattern` to carry the new `[N]` field. Update it in `step_dynamics` from `pattern_matrix` (geometry-derived) or `reinforce()` (event-driven).
- **New replay sampling weight**: the `ReplayStore._priorities()` method is the single point that determines what gets replayed. Already weights by `gate × tag_count × suppression`; another multiplicative or log-additive factor is trivial.

---

## 5. Phase 5 surface (`src/energy_memory/phase5/`)

### `ham_aggregator.HAMAggregator`

Coupled multi-scale settling. Per iteration: each scale does one Hopfield
step → decodes its masked position → contributes a distribution over
candidate tokens; a consensus (geometric mean of log-distributions, or
arithmetic) is computed; each scale's state is biased toward
`bind(masked_position, expected_codebook_vec_under_consensus)` with
strength `α=0.3`. Convergence on consensus delta < tol. Same on-device
pattern as `retrieve()`: all iters run, sync once at end.

**Key data flow**: the consensus distribution over the **decode token
vocabulary** is the natural carrier for cross-scale "agreement." This
is where layer-2 attractors hook in.

### `ham_with_layer2.HAMWithLayer2`

Extends the aggregator: each layer-2 attractor is a profile
(distribution over decode vocabulary) plus a scalar strength. Per
iteration:

1. Compute raw consensus as base HAM.
2. `_apply_layer2`: cosine-sim between raw_consensus and each layer-2
   profile → softmax with `β_l2 = 10` and log-strength bias →
   activations `[L]`. Build `layer2_signal = Σ a_l · profile_l`.
3. `modified_consensus = (1 − λ_l2) · raw + λ_l2 · layer2_signal`.
4. Feed modified consensus back as the top-down bias for the next scale
   iteration.

Layer 2 has its own decay/reinforcement loop (`Layer2State`):
`strength_decay=0.995/step`, `reinforcement_gain=0.05 · activation_weight`
on the top activated attractor when its weight exceeds uniform. Prune
when `strength < min_strength=0.05`.

**`Layer2Attractor.source_query`** is currently an untyped tuple slot;
nothing in the loop uses it for dynamics but it's stored.

### `role_fidelity` (β prior)

Pure functions, no state:

- `compute_role_fidelity(schema_bindings: [N, W, D])`: per-schema mean
  pairwise FHRR distance among W unbound fillers. Schemas with cleanly
  separated role decompositions → `f ≈ 1`; collapsed → `f ≈ 0`. Uses
  the same per-row Gram pattern as `_coverage_redundancy_instantaneous`.
- `fidelity_weighted_prior(cue, schemas, fidelities, p=1, q=1)`:
  `normalize(Σ_i (cue·s_i)_+^p · f_i^q · s_i)`. Continuous weighted
  superposition over the **full** schema store — no top-k cutoff, no
  categorical branch. Replaces the failed categorical
  `top_k_by_effective_strength` selector from report 049.

**Important**: this is a pure measurement + superposition. It's
state-free. Adding new per-schema scalar weights (or per-position
weights) into the same superposition is trivial — the function signature
already takes a `[N]` fidelity tensor, and adding another `[N]` weight
just becomes another multiplicative term.

---

## 6. What's expensive vs cheap

### Cheap (existing primitives, ≤ O(N·D) or O(N²) per call)

- Per-atom Gram and any row-reduction (max, mean, top-k, entropy of row).
  This is what `r_inst` and the schema role-fidelity already do.
- `d_eff` via the algebraic identity, including its autograd.
- Adding a per-atom score_bias component (one tensor add).
- Adding a per-cycle substrate force (one autograd.grad + normalize + cache invalidate).
- Per-atom state with EMA dynamics (one elementwise blend per cycle).

### Moderate (O(N² · D) per call but already paid)

- Computing the centered Gram for `d_eff` (`patterns @ patterns.conj().T`).
- Computing `r_inst` for all atoms (same Gram, different reduction).
- Settling iterations of Hopfield with N patterns: O(max_iter · N · D)
  per cue.

### Expensive (would be new infrastructure)

- Per-cue eigendecomposition of the substrate — not needed, the d_eff
  identity sidesteps it.
- Anything requiring full pairwise pattern-of-pattern computation
  (O(N³)) — none currently in hot paths; avoid.
- Per-atom autograd flows where the energy depends on *all* patterns —
  already paid for B's repulsion; adding a second autograd-driven force
  doubles that cost. Practical, but not free.
- Per-cue computation against the *full* schema store (β prior already
  does this; it's O(N · D), fine for current N ~ thousands).

---

## 7. Surfaces with strong adjacency to other techniques

These are the unusual or load-bearing pieces — places where the
implementation naturally talks to a broader literature.

- **Autograd through pattern positions**: `repulsion_force` already
  flows gradients through complex unit-magnitude patterns. This is
  adjacent to differentiable VSA, iEnergy / Equilibrium Propagation,
  any "patterns are parameters" formulation. The infrastructure is
  in place to add additional energy terms and let `autograd.grad`
  give the descent.
- **Per-atom EMA of a Gram statistic** (`r_ema`): this is the shape
  of any "slow-timescale per-element geometric property." Mean-field
  RG, slow-fast decoupling, persistent-homology-of-the-fly
  measurements all fit this template.
- **Coupled multi-scale settling with consensus** (HAM): direct Krotov
  2021 territory. Bidirectional message passing through a shared
  vocabulary distribution. Adjacent to belief-propagation / VFE /
  predictive coding hierarchies.
- **FHRR + Hopfield + role-binding**: the substrate provides
  algebraically-clean role/filler decomposition (bind/unbind) with an
  exact inverse. The `compute_role_fidelity` measurement quantifies
  how well a stored pattern preserves that decomposition. Adjacent to
  tensor-product representations, neural-symbolic systems, and the
  "Attention as Binding" thread.
- **Permutation as a separate group action**: `permute()` exists but is
  not currently wired into any phase-5 dynamics. It's a tool sitting
  in the box for directed/asymmetric encodings.
- **Continuous-death-via-retrieval-bias** (step 3 E_i-weighted): the
  full replacement of binary deletion by a softplus on effective
  strength. Adjacent to mortality-without-controller designs; the
  garbage-collect call is already no-op'd when this is on.

---

## 8. Pipeline shape for inserting new measurements/dynamics

The codebase has two natural insertion points per cycle and one per atom-add:

```
add_pattern():
  - u_1 ← novelty_strength
  - r_ema ← r_inst against current substrate (A1)
  - retrieval_count, A ← 0
  [INSERT: any new per-atom state, init from geometry or zero]

retrieve_and_observe(cue):
  - bias = sum(consolidation.inhibition_bias(),
               consolidation.retrieval_weight_bias(),
               [INSERT: new per-atom score_bias term])
  - trace = memory.retrieve_with_trace(cue, score_bias=bias)
  - consolidation.reinforce(top_idx, magnitude·(1 − λ·r_ema[top_idx]))
  - consolidation.accumulate_inhibition(top_idx)
  [INSERT: new event-driven per-atom updates from trace]
  - if cycle: run_replay_cycle()

run_replay_cycle():
  ...sample / re-settle / candidate_handler...
  _step_substrate_dynamics():
    - consolidation.step_dynamics(pattern_matrix)
      [INSERT: new geometry-derived per-atom updates inside step_dynamics]
    - if B: patterns += step · repulsion_force(patterns); normalize; invalidate
      [INSERT: new substrate-energy-gradient force, summed with repulsion]
```

Phase 5 retrieval (`HAMWithLayer2.retrieve`) is structurally separate
from the Phase 4 replay loop — it consumes the consolidated substrate
but doesn't drive consolidation dynamics. New per-cue HAM measurements
fit naturally between the raw consensus and the layer-2 modification
step; new layer-2 dynamics fit in `Layer2State`.

---

## 9. Files load-bearing for the brainstorm

- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/substrate/torch_fhrr.py` — primitives, d_eff, repulsion
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/memory/torch_hopfield.py` — softmax-settling, `score_bias` injection
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase4/trajectory.py` — trace + engagement/resolution/gate
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase4/consolidation.py` — per-atom state, u-chain, A, r_ema, retrieval_count
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase4/replay_loop.py` — bias stack, replay sampling, candidate handoff, substrate dynamics step
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase5/ham_aggregator.py` — coupled multi-scale settling
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase5/ham_with_layer2.py` — layer-2 attractor pathway
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase5/role_fidelity.py` — pure β prior + fidelity measurement
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase2/encoding.py` — `build_position_vectors`, `encode_window`, `decode_position`
- `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase2/metrics.py` — `cap_coverage`, `meta_stable_rate`, `wilson_interval`
