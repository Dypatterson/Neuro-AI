---
date: 2026-05-20
project: personal-ai
tags:
  - brainstorm
  - subject/cognitive-architecture
  - phase-5
status: research-brief
---

# D — Unexplored Diagnostic-Actuator Pairs (Phase 5 Graduation Alternatives)

## Frame

The 2026-05-09 synthesis named five diagnostic-actuator pairs as the
architectural threshold-crossing — each pair is supposed to convert a
diagnostic METRIC into a slow-timescale local ACTUATOR (continuous
geometric dynamic), not into a rule that reads the metric and triggers
a response.

Status snapshot:

| # | Pair | Status |
|---|---|---|
| 1 | high spread ~ reduced consolidation | ✅ CLOSED (A+B continuous coverage-weighted reinforcement + −α·log d_eff repulsion) |
| 2 | high drift ~ replay-pressure | ❌ no design note |
| 3 | bimodality ~ splitting-pressure | ❌ K-branch state_divergence chase, β implementation, empirically falsified (uniform f_i at D=4096) |
| 4 | metastability ~ replay-prioritization | ❌ no design note |
| 5 | low cap-coverage ~ restructuring-pressure | ❌ no design note |

Phase 5 has been chasing pair #3 via K-branch state_divergence and just
hit a substrate-encoding-level wall (report 050). This brief asks: what
would close #2, #4, or #5 instead, and which is the cleanest pivot
given that A+B+A1+A1′ already produced a substrate primed for these
dynamics?

The methodology below is the 2026-05-09 grammar:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of. Diagnostic and actuator are the
> same physical process viewed at different temporal resolutions.

For each pair I name (a) the diagnostic the substrate already
measures, (b) the local dynamic that the diagnostic is a snapshot of,
(c) the anti-homunculus shape check, (d) the literature ground, and
(e) the implementation distance from what the substrate has *today*.

---

## Pair #2 — drift / replay-pressure

### Diagnostic side (what the substrate already measures)

- `codebook_drift()` in
  [src/energy_memory/phase34/reencoding.py:107](../../../src/energy_memory/phase34/reencoding.py)
  — mean cosine distance between two codebook snapshots. Already
  computed during Hebbian-online training. *Global* metric.
- Per-atom drift is a trivial reduction: store the last-replay codebook
  snapshot per atom (or per stored pattern's codeword indices) and
  measure `1 − cos(p_i_now, p_i_at_last_replay)`. The substrate has
  every ingredient; the per-atom view isn't surfaced anywhere yet.
- The HAMAggregator (`src/energy_memory/phase5/ham_aggregator.py`)
  already exposes an `energy()` API; the *change* in an atom's local
  energy between replay events is a second drift proxy ("how much has
  this atom's basin shifted since I last visited it?").

### Actuator side (the local dynamic)

**Per-atom drift IS replay-pressure** — they are the same quantity at
different timescales. Replay is biased toward atoms whose local
geometry has changed most since they last contributed to retrieval.

**Formulation.** Let `δ_i(t)` be atom i's per-atom drift state:

> `dδ_i/dt = η · ||p_i(t) − p_i(t − Δt_i)||²   −   λ · δ_i · 𝟙{atom_i replayed}`

where `Δt_i` is "time since atom i was last replayed" and the second
term resets δ_i when the atom *is* replayed. The replay-buffer's
sampling weights *are* the δ_i values (continuous, not thresholded):

> `P(replay_i) ∝ δ_i(t) · w_i`

where `w_i` is the existing effective-strength weight (A1′ output).
There is no "high drift → trigger replay" rule. δ_i is a per-atom
energy that *is* the replay-pressure. When δ_i is integrated into the
softmax over replay candidates, drift modulates selection by a
continuous gradient.

**Connection to MIR (Aljundi 2019).** MIR samples replay items by the
loss-rise predicted under the foreseen update. The project's δ_i is
the *measured* counterpart of MIR's *predicted* interference: atoms
whose patterns have moved (drift) are exactly the atoms whose loss
would have risen under the consolidation steps that moved them. The
project's substrate has every ingredient — Hebbian updates are local,
patterns are stored, and the geometry the loss is computed against is
the substrate itself. Concretely: δ_i can be expressed as
`Δ(self-retrieval-energy)` over the replay interval, which is a
single energy-difference evaluation per atom per cycle.

**Connection to InfoRS (Sun 2022).** Surprise + learnability is the
two-knob form. The project's δ_i covers the *learnability* knob
(things that moved are things the consolidation update can still
move). The *surprise* knob — `−log P(p_i | substrate \ p_i)` — is
already cheap on the substrate: it's the inverse of the
self-retrieval-cap-coverage we just discussed in pair #5. So a
two-knob `δ_i · surprise_i` weight is GPU-cheap and anti-homunculus
clean.

### Anti-homunculus check

- δ_i is a continuous running estimate per atom (not a global
  schedule). PASS.
- Replay weights are a softmax over δ_i · w_i (no threshold, no
  membership flag). PASS.
- No "controller reads drift, decides to replay" — replay sampling
  *is* a function of δ_i; the dynamic *is* the selection. PASS.
- The "decay on replay" term is the dual of A's r_ema rate: δ_i pays
  down when atom i is visited, the way r_ema accumulates when atom i
  is reinforced. Same shape as A+B. PASS.

### Substrate readiness

- ✅ Codebook drift measurement exists (global).
- ✅ Replay store with weighted sampling exists
  ([phase4/replay_loop.py:191](../../../src/energy_memory/phase4/replay_loop.py)).
  Current weights are `gate × tag_count × suppression`. Adding δ_i
  factor is a one-line multiplication.
- ✅ Self-retrieval energy is a single substrate call.
- ⏳ Per-atom drift state (δ_i array) needs to be added to
  `ConsolidationState`. ~30 lines.
- ⏳ A snapshot of each atom's "last-replay" pattern needs to be kept.
  Either a parallel tensor (~D × N floats) or computed implicitly via
  the energy-difference formulation (no snapshot needed).

**Distance from substrate-already-built: ~1 day of implementation,
~1 design note, the same n=5 falsification discipline as A+B.**

---

## Pair #4 — metastability / replay-prioritization

### Diagnostic side (what the substrate already measures)

- **Per-retrieval metastable flag** —
  `branch.meta_stable = top_score < 0.95` already computed in
  [experiments/40_phase5_branching.py:779](../../../experiments/40_phase5_branching.py)
  and `meta_stable_rate` in
  [phase2/metrics.py:69](../../../src/energy_memory/phase2/metrics.py).
- **Per-atom metastability proxy** can be derived from
  `_coverage_redundancy_instantaneous` (max-over-others, A1′) +
  effective-strength: an atom whose `max_j |G_ij|` is high AND whose
  effective_strength is near the noise floor is metastable —
  competing for activation but not winning. The substrate already
  computes this every replay cycle.
- **HEN's metastable-state diagnostic** (Kashyap 2024): cue with a
  binding, settle, count fraction that don't converge to any stored
  pattern. The HAMAggregator and Hopfield retrieve both expose
  per-iteration state already.

### Actuator side (the local dynamic)

**Metastability IS replay-prioritization** — the per-atom adaptation
variable `a_i` of Candidate C (Saighi-style self-inhibition, redundancy-
coupled decay; see
[notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md](../../../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md))
is the natural local dynamic. But re-framed for *replay*, not retrieval:

> Replay frequency for atom i ∝ `m_i(t)` where m_i evolves as:
>
> `dm_i/dt = ζ · 𝟙{atom_i is in metastable settling} − μ · m_i · 𝟙{atom_i replayed}`

m_i accumulates when atom i is involved in metastable settlings (its
contribution to a softmax that fails to peak); m_i pays down when atom
i is replayed. There is no "metastable rate > X → schedule replay"
rule. The replay store's priority function (currently
`gate × tag × suppression`) gains an `m_i` factor.

**The substrate already counts per-atom "softmax-contribution-but-not-
winner" naturally** in any retrieval that uses softmax weights. The
per-atom weight `w_i = softmax_i(scores)` is bounded ≤ 1; per-atom
metastable contribution is `w_i · (1 − max_j w_j)`. An atom with
high w_i in a low-max retrieval is "metastable" in the HEN sense
without any threshold being read.

**Connection to MIR via metastability.** MIR predicts which items
would be maximally interfered by the next update. The project's
metastability flag is the *empirical* signature of an item currently
being interfered with — its score is high but not winning, which means
*some other atom is being reinforced at its expense*. So
metastability-driven replay is MIR with `O(0)` extra cost: the
substrate already runs the retrievals that surface metastable atoms.

**Connection to Benna-Fusi (2016).** The "bidirectional fast↔slow"
recipe: items that lose fast-variable signal but haven't yet flowed
into slow variable are exactly the metastable items. m_i is the
fast-variable accumulator; replay flushes it into slow.

### Anti-homunculus check

- m_i is per-atom, continuous, with no threshold. PASS.
- Replay priority is a multiplicative factor in the existing softmax
  over candidates — no rule "if metastable then prioritize." PASS.
- The diagnostic (meta_stable_rate from HEN) is the *aggregate* of
  m_i across the population; it never causes any action. PASS.
- No "metastable detector" module. m_i evolves automatically as
  retrievals happen. PASS.

### Substrate readiness

- ✅ meta_stable flag exists per retrieval.
- ✅ Per-atom softmax weights exist inside every retrieve() call —
  they are currently computed and *discarded* after returning the
  top-1.
- ✅ Replay-store priority function is already a multiplicative
  composition of factors.
- ⏳ Surface the per-retrieval softmax weights from
  `TorchRetrievalResult` (currently only top score is returned). ~10
  lines.
- ⏳ Add `m_i` array to ConsolidationState with EMA update. ~30 lines.
- ⏳ Multiply into replay-store priority. ~3 lines.

**Distance from substrate-already-built: ~half a day of plumbing,
~1 design note. Lowest implementation effort of the three.**

The diagnostic side is *already running every retrieval cycle* — the
information is currently being thrown away after each call. This pair
is the most "the substrate already does this; we just haven't read
the value."

---

## Pair #5 — cap-coverage / restructuring-pressure

### Diagnostic side (what the substrate already measures)

- `cap_coverage()` in
  [phase2/metrics.py:63](../../../src/energy_memory/phase2/metrics.py)
  — fraction of retrievals whose top score exceeds a cap threshold.
  Existing per-eval-call metric.
- **Per-atom cap-coverage** is computable cheaply: for each atom i,
  measure the self-retrieval cap = `cos(retrieve(cue=p_i), p_i)`. An
  atom whose own pattern doesn't get cap-covered by retrieving its own
  cue is the local geometric signature of "this atom needs
  restructuring." The substrate has every primitive needed.
- Vangara-Gopinath bound: `cap_coverage ≈ (θ′/d̄)^(d_eff/2)`. With
  d_eff and d̄ both computable from the consolidated substrate, the
  *expected* cap-coverage per atom is a structural quantity; deviation
  from it is the local restructuring signal.

### Actuator side (the local dynamic)

**Restructuring is neuro-genesis governed by local cap-coverage
failure** — SQHN's (Alonso & Krichmar 2024) neuro-genesis is the
literature precedent. The substrate already does discovery-channel
addition of new atoms via convergent retrievals; this is a *constrained*
form of neuro-genesis. The cap-coverage actuator says: the discovery
channel's gating signal *is* the cap-coverage gradient.

**Formulation.** Let `c_i = cos(retrieve(cue_i), p_i)` be atom i's
self-retrieval cap-coverage at the current substrate. Define the
restructuring-pressure field on the substrate as:

> `H_restruct = β · Σ_i (1 − c_i)² · 𝟙{c_i < cap_expected_i}`

where `cap_expected_i` is the Vangara-Gopinath baseline for atom i's
local d_eff. Then add the gradient as a substrate-energy term, parallel
to B's −α log(d_eff):

> `dp_i/dt += −∂H_restruct/∂p_i`

This is a continuous force on each atom toward its own cap. If c_i
keeps dropping, the gradient grows; eventually the atom can no longer
restore its cap unilaterally, and the discovery channel's
`add_pattern()` event fires when an atom's *retrieval residual*
(retrieval target minus current state) has accumulated a continuous
norm above the substrate's noise floor — *not* a threshold being read,
but the natural limit of "this atom is being pulled toward two
different basins and can't satisfy both."

**Alternative formulation (closer to PAM/Dury 2602).** Restructuring
fires when an atom's *associative neighbors* — the patterns it
predictably co-retrieves — don't fit its current basin. The local
geometric quantity is the predictability residual of the atom's
co-retrieval graph. The substrate has co-retrieval data (every replay
cycle produces co-occurrence pairs); aggregating it per-atom is cheap.

**Connection to capacity-bottleneck consolidation (Dury 2603).** The
"forced abstraction under capacity ceiling" recipe: when an atom
can't fit its associations into one basin, splitting (= restructuring)
is the only resolution. This is also the literature's grammar for the
*bimodality / splitting* pair (#3) the project has been chasing —
which suggests pair #5 and pair #3 are the same dynamic at different
scales. (Splitting is restructuring-via-creation; cap-coverage failure
is the local geometric signal for both.)

### Anti-homunculus check

- c_i is local per atom. PASS.
- H_restruct is a substrate-energy term, evaluated everywhere
  substrate energy is read (parallel to B). PASS.
- Atom addition (discovery-channel `add_pattern`) is a limit of the
  dynamic, not a triggered action — but ONLY if the discovery-channel
  gating is itself local-geometric, not a wall-clock schedule. The
  current discovery-channel gate signal in
  [phase4/replay_loop.py](../../../src/energy_memory/phase4/replay_loop.py)
  uses `gate_signal` which is the trajectory's resolution metric — a
  local geometric quantity per trace. PASS, assuming the gate is
  routed through cap-coverage residual.
- cap_expected_i must be set once at substrate construction or per-
  retrain from theoretical considerations, not adapted from observed
  c_i. PASS-if-disciplined (mirrors A+B's α/λ commitment).

### Substrate readiness

- ✅ cap_coverage measurement exists (aggregate).
- ✅ Self-retrieval is a single substrate call per atom.
- ✅ d_eff / d̄ are computed by `consolidation_geometry_diagnostic.py`.
- ✅ Discovery channel exists — adds new atoms when convergent
  retrievals indicate.
- ⏳ Per-atom c_i array as continuous running estimate. ~30 lines.
- ⏳ H_restruct gradient as substrate-energy contribution (parallel
  to B). ~50 lines + autograd.
- ⏳ Discovery-channel gate composition with cap residual signal. ~20
  lines.
- ⏳ Vangara-Gopinath cap_expected_i: needs θ′(β) calibration spike
  (pre-Phase-3 open item). Without it, use a single global expected
  cap; coarser but functional.

**Distance from substrate-already-built: ~2 days of implementation +
~1 design note + θ′(β) calibration as a side-quest (or skipped with a
single global expected-cap constant).**

This pair has the longest distance, but it's also the *deepest*
architectural payoff — closing it would give the project the
restructuring dynamic that's been called out as the
"compression-to-abstraction" mechanism (open pre-Phase-3 commitment,
high-leverage brainstorm idea 5). It would also resolve pair #3
(bimodality → splitting is restructuring-by-creation) without re-
opening the role-fidelity-at-D=4096 wall that just falsified β.

---

## Ranked recommendation

### #1 (best Phase 5 pivot) — **Pair #4: metastability / replay-prioritization**

**Why.**

1. **The substrate already does it** — every retrieve() call computes
   per-atom softmax weights and a meta_stable flag. The "implementation"
   is mostly *surfacing* values that are already being computed and
   discarded. This is the strongest "the architecture has already
   built it; we just haven't tested it" signal of the three.
2. **Anti-homunculus shape is the cleanest of the three.** No new
   energy term, no new gradient — just multiply an existing replay
   priority by a per-atom accumulator that has the same shape as the
   suppression variable already in `ReplayStore`. The 2026-05-09 note
   literally gives this pair's non-controller reading as "replay
   buffer is energy-ranked; metastable trajectories carry higher
   energy by construction" — that's nearly verbatim what `m_i` is.
3. **Headline metric exists and is independent of the K-branch
   chase.** `Δ meta_stable_rate at W=3` is already the current Phase 4
   headline (D1). Re-purposing it for Phase 5 — measuring the impact
   of m_i-weighted replay on meta_stable_rate — keeps a known-clean
   metric. The discipline note
   ([2026-05-16-substrate-vs-readout-metric-discipline.md](../../../notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md))
   already certified D1 as substrate-pure.
4. **Falsification is symmetric to A+B+A1′.** Pre-commit: m_i-weighted
   replay reduces meta_stable_rate at W=3 by Δ ≤ −0.1 at n=10,
   CI-disjoint from zero, without regressing D1. Same n=5-then-n=10
   discipline.
5. **Literature ground is multi-source.** MIR provides the
   theoretical priority signal; HEN provides the diagnostic; Benna-
   Fusi provides the fast↔slow accumulator dynamic; Saighi provides
   the per-atom adaptation shape; all four converge on the same `m_i`.
6. **No new structural-divergence chase.** The K-branch state_divergence
   has reached substrate-encoding-level structure (uniform f_i at
   D=4096). Pair #4 pivots away from that wall while still closing a
   2026-05-09 pair.

**Headline:** Δ meta_stable_rate at W=3 under m_i-weighted replay, n=10,
CI-disjoint from zero. (Substrate-pure; same discipline as Phase 4 D1.)

### #2 — **Pair #2: drift / replay-pressure**

Second-cleanest. Drift measurement exists globally; per-atom δ_i is a
modest extension. The dynamic form is symmetric to A+B (running
estimate per atom; multiplied into existing replay weights). The
MIR/InfoRS literature ground is the strongest of the three for
"selection by interference, not by frequency" — directly addresses the
project's open question about consolidation selectors beyond strength-
by-frequency. ~1 day to implement.

Slightly worse than #4 because (a) the drift snapshot per atom needs
new state, vs. #4 which just surfaces existing softmax weights, and
(b) the headline metric is less obviously substrate-pure (ΔR@K under
m_i replay is a readout; whether *replay-pressure-weighted*
consolidation improves it needs a separate stratification).

### #3 (deepest payoff, longest distance) — **Pair #5: cap-coverage / restructuring**

This is the highest-leverage architectural close — it would give the
project its restructuring dynamic and arguably subsume pair #3
(splitting = restructuring-by-creation) without re-opening the role-
fidelity wall. But it requires:

- A new substrate-energy term with autograd (parallel to B).
- θ′(β) calibration spike (still open pre-Phase-3 commitment).
- Discovery-channel gating composition.

Not the right *Phase 5* pivot — too much scope. **The right move on
this one is to spec it as the Phase 6 architectural target** and
treat the θ′(β) spike as its prerequisite.

---

## Summary table

| Pair | Diagnostic ready? | Actuator shape | Anti-homunculus | Effort | Lit ground |
|---|---|---|---|---|---|
| #4 metastability ~ replay-priority | ✅ already computed every retrieval | per-atom m_i EMA, multiplied into existing replay softmax | cleanest of three | ~½ day plumbing | MIR, HEN, Benna-Fusi, Saighi |
| #2 drift ~ replay-pressure | ✅ codebook drift (global); per-atom is trivial reduction | per-atom δ_i EMA, multiplied into replay softmax (parallel to #4) | clean — same shape as A+B | ~1 day | MIR, InfoRS |
| #5 cap-coverage ~ restructuring | ✅ cap_coverage exists; per-atom c_i needs adding | new substrate-energy term H_restruct + discovery-channel gating | clean if c_i is local + cap_expected is fixed | ~2 days + θ′(β) calibration | SQHN, Vangara-Gopinath, Dury 2603 |

---

## Recommendation

**The best alternative Phase 5 headline is pair #4 (metastability ~
replay-prioritization) because the substrate already computes its
diagnostic side every retrieval call (per-atom softmax weights and
meta_stable flag), the actuator side is a single per-atom EMA
multiplied into the replay-store priority that already exists, the
anti-homunculus shape is the cleanest of the three open pairs, and the
headline metric (Δ meta_stable_rate at W=3) is the same substrate-pure
quantity that just graduated Phase 4 on D1 — letting Phase 5 graduate
on a re-application of the project's most disciplined metric to a
mechanism the architecture has already 90% built.**

The sequencing then becomes:

1. **Phase 5 (re-scope):** close pair #4 with `m_i`-weighted replay.
   Same n=5-then-n=10 discipline as A+B. Headline:
   Δ meta_stable_rate at W=3, CI-disjoint, plus D1 non-regression.
2. **Phase 5.5 (if time):** close pair #2 by adding `δ_i` as a second
   replay factor; test multiplicatively with m_i. Both are MIR-shape;
   they should reinforce.
3. **Phase 6 (architectural):** close pair #5 with the cap-coverage /
   restructuring dynamic, requires θ′(β) calibration as prerequisite.

Pair #3 (bimodality / splitting) does NOT get re-opened in isolation
— its closure is folded into pair #5 (restructuring-by-creation) as
the right scale. The K-branch state_divergence chase ends as a
falsified-by-substrate-encoding result, with the architecture-level
diagnostic insight (uniform f_i at D=4096) preserved as evidence that
the role-fidelity-via-pairwise-distance route was the wrong
operationalization, not that the bimodality/splitting pair itself is
wrong.
