# 2026-05-13 — FEP audit checklist

Companion to the anti-homunculus filter from [2026-05-09-papers-diagnostics-and-actuator-dynamics.md](2026-05-09-papers-diagnostics-and-actuator-dynamics.md).

The anti-homunculus filter is a necessary condition but is judgement-heavy:
"who decides X" can be argued either way for borderline mechanisms. The Free
Energy Principle (FEP) supplies a stricter, more mechanical check that the
brainstorm of 2026-05-13 (Idea 1, citing Spisak & Friston, May 2025,
arXiv:2505.22749) recommends adopting as a habitual practice.

## The rule

Every proposed addition to the architecture must be expressible as one
of:

1. **A free-energy quantity F** that the substrate already minimizes via
   its existing dynamics; or
2. **A measurement of (1)** — a diagnostic whose value reads, but does
   not steer, the gradient flow.

A mechanism fails the FEP audit if it requires *inspecting* a quantity
and *triggering* a different update rule based on its value. The
inspection/trigger split is the signature of a hidden controller.

## The checklist (write this out before coding any new mechanism)

For each new diagnostic or actuator, the design note must answer:

- [ ] **What free-energy quantity does this contribute to?**
      Name F explicitly. If you can't, the mechanism is not anti-homunculus.
- [ ] **What is the gradient response?**
      ∂F/∂x where x is the state being updated. The actuator IS this
      gradient — not a separate "decision module" that reads F and picks
      an action.
- [ ] **Does any branch read a metric and pick between dynamics?**
      If yes, redesign. Replace the branch with a continuous quantity
      that contributes to F. (Example: "if entropy > τ then blend else
      retrieve" → make entropy contribute to F such that blend vs.
      retrieve is a single energy minimum on a varying landscape.)
- [ ] **Is the trigger a constant or a learned function of state?**
      Constants (thresholds set once at construction) are fine — they
      are part of the architectural prior, not a controller. Learned
      branching functions are controllers.
- [ ] **Anti-homunculus restatement.**
      "Who decides X, where does the decision live in the dynamics?"
      The answer should be a single geometric quantity, not a routing
      table.

## Translation table from 2026-05-13 brainstorm

This is the working dictionary the project uses when re-auditing existing
diagnostics. Add to it as new mechanisms come online.

| Diagnostic                  | FEP quantity                                 | Gradient response                                  |
| --------------------------- | -------------------------------------------- | -------------------------------------------------- |
| High drift                  | KL divergence from prior generative model    | Replay pressure drives KL toward 0                 |
| High mean dispersion d̄     | Precision of consolidation likelihood        | Low precision → smaller update weight on u_1       |
| Bimodality in codebook      | Complexity cost of redundant attractors      | Hebbian/replay gradient penalizes redundancy       |
| Metastability of settling   | Prediction error on settling trajectory      | Higher error ↑ replay weight (Idea 4b graded u_1)  |
| Low cap-coverage            | Surprise at reconstruction                   | Surprise gradient drives codebook restructuring    |
| Replay tag_count            | Posterior probability of trace relevance     | Sampling weight ∝ tag_count (Idea 4a)              |
| Suppression (IoR)           | Local prior penalizing recently-sampled item | Sampling distribution renormalized (Idea 4c)       |

## When this checklist is required

- New design notes for any phase mechanism: required before implementation.
- New experiment proposals: required for any mechanism added to the
  experimental setup, not just the substrate.
- Refactors: not required, but if a refactor changes a control-flow
  branch into a continuous quantity (or vice versa), call it out.
- One-shot diagnostics that don't affect dynamics: not required (they
  are pure measurements of F by definition).

## Worked example: the existing replay gate

Existing mechanism: `gate = engagement × (1 - resolution)`, with a store
threshold and a sampling distribution.

- F contribution: gate is a measurement of the local free-energy
  *change rate* — high engagement with low resolution means the system
  is still climbing the gradient, so storing it for replay continues
  the descent.
- Gradient response: traces with high gate enter the store; sampling
  weight ∝ gate × tag_count × suppression; settling dynamics descend
  the same energy landscape that produced the gate value. No
  inspect-and-trigger branch.
- Anti-homunculus restatement: "who decides which trace gets replayed?"
  → the priority weight, computed from local per-trace state, sampled
  by multinomial. No supervisor.

## Worked example: the tag_count upgrade just landed (Idea 4a)

- F contribution: each repeated observation of an overlapping query is
  evidence about the posterior probability that the trace's resolution
  matters. `tag_count` is the running count of that evidence.
- Gradient response: priority = gate × tag_count × suppression, used as
  the sampling distribution.
- Inspect-and-trigger branch? **There is one in the overlap-collapse
  add() code**: if cosine > threshold, increment instead of append.
  This is a borderline call. The threshold is constant (architectural
  prior, not learned). Decision: passes, because the branch is on a
  constant threshold and produces the same kind of object (a tagged
  store entry) on both sides. If we later want to make this fully
  continuous, the formulation would be a soft kernel weight on the new
  trace distributed over existing entries — that is a future refinement.

## Failure modes this checklist catches

1. **Threshold-based mode switching** ("if metric > τ then dynamics A
   else dynamics B"). The checklist forces a redesign into a continuous
   F that picks the energy minimum.
2. **Supervisor modules that route between subsystems** (the explicit
   anti-homunculus violation; the FEP check makes it formal).
3. **Learned gating functions** that read state and emit a routing
   decision. These fail the "constant threshold" sub-check.
4. **Diagnostics that secretly steer**: a metric whose value is logged
   for monitoring but is also read by a different module to adjust its
   update. The audit forces the question of which side of the
   diagnostic/actuator split each consumer is on.

## When the checklist is *not* sufficient

The FEP gradient criterion does not itself prove a mechanism is correct
or that the chosen F is the right one. It only verifies that no hidden
controller has been imported. A passing audit plus a failed experiment
means the F is wrong; a passing audit plus a successful experiment
means the F is plausible and the dynamics are clean. Both are needed.
