---
date: 2026-05-16
project: neuro-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
  - methodology
  - metric-discipline
---

# Substrate vs. readout: top1 is a drill-down, not a graduation criterion

## Entry point

Mid-session re-framing prompted by a question from Dylan during the A_k decay sweep on Colab. We had been treating Δtop1 as if it were a load-bearing Phase 4 verdict (reports 029-035), but Δtop1 is not in the design spec's headline list. This note resolves that drift and establishes a standing discipline going forward.

The trigger: report 035's n=10 falsification of report 034 was framed around Δtop1 = −0.051 with CI strictly negative — making A_k look like a failure. Under the design-spec headlines (ΔR@K + Δcap-coverage), the n=10 result is actually *neutral* (ΔR@10 indistinguishable from baseline; Δcap-coverage CI crosses zero). The "dramatic failure" framing was a methodology error.

## The architecture separates substrate from readout

[PROJECT_PLAN.md Phase 2-7](../../docs/PROJECT_PLAN.md) makes the architectural decomposition explicit:

| Phase | Responsibility | Produces |
|---|---|---|
| 2-4  | Substrate: FHRR + codebook + replay/consolidation | A *settled attractor* given a context |
| 5    | Structure/abstraction: roles, binding, hierarchy | A *richer attractor* with relational structure |
| 6    | Predictive world model: JEPA-style rollouts | A *selected continuation* from K candidates |
| 7    | LLM interface | A *literal word* decoded from the chosen latent state |

The literal token only exists in Phase 7. Substrate's job is to settle to a *coherent neighborhood*, not to pick the word.

## What the metrics actually measure

For the FHRR substrate at Phase 4:

| metric | what it measures | role |
|---|---|---|
| **R@K** | "is the correct completion in the top-K plausible neighbors of the settled attractor?" | substrate quality — basin correctness |
| **cap-coverage** | "across the test set, does the substrate resolve to many distinct confident attractors?" | substrate quality — expressive resolution |
| **top1** | "did the substrate's argmax readout happen to be the literal wikitext token?" | substrate + naive readout (where naive readout = `argmax(score)`, a placeholder for Phase 6/7) |

The first two are pure substrate properties. The third **conflates** substrate behavior with a placeholder readout that the architecture explicitly plans to replace.

## What this means for Phase 4

[phase-4-unified-design.md:276-282](../emergent-codebook/phase-4-unified-design.md) is unambiguous: the headlines are **R@K AND cap-coverage**. top1 appears in that document only as a drill-down.

Re-reading the report trail through this lens:

| Report | Stated finding | Re-framed under design spec |
|---|---|---|
| 026 | ΔR@10 +0.010 verified; cap-coverage CI crosses 0 | Same — but headline #1 verified, headline #2 still open |
| 029 | ΔR@10 +0.0145; "new blocker": Δtop1 −0.026 | Headline #1 trending positive; "drill-down anomaly" not "new blocker" |
| 030 | Δtop1 regression "is Phase 3, not Phase 4" | Diagnostic finding about a drill-down, not a Phase 4 verdict |
| 032 | n=10: ΔR@10 +0.009 (CI crosses 0); Δtop1 −0.018 | Headline #1 sober but on the right side; drill-down behavior consistent |
| 034 | Seed 1: top1 +0.009, R@10 +0.022, cap_t05 +0.022 | All three improved on one seed; need n=10 verification |
| 035 | n=10: Δtop1 −0.051 (CI strictly negative); ΔR@10 +0.009; Δcap_t05 −0.026 (CI crosses 0) | **Neutral on both design headlines**; drill-down shows substrate distortion direction |

The framing shift dissolves "report 034 was sample-lucky on top1 → falsified at n=10" into the cleaner "report 034 was sample-lucky on cap_t05 → n=10 shows no cap-coverage improvement". Same conclusion, but for the right metric.

## Why top1 crept in as if it were a headline

Two reasons:

1. **It's intuitive**: "did the system get the right word?" is easy to explain. cap-coverage requires a paragraph to define.
2. **It was unexpectedly negative**: surprise begets attention, attention begets metric inflation. The Δtop1 regression in report 029 was surprising enough that it spawned blocker #6'. Blocker status created a false sense that it was load-bearing.

Neither reason is a sound architectural justification.

## Is top1 useless then?

No — it's a *drill-down*, exactly as the design spec calls it. It tells us:

- **Direction of substrate distortion**: A negative Δtop1 under online Hebbian means drift is moving the substrate in a way that disrupts naive argmax. That IS informative about *what* the mechanism is doing geometrically, even if it isn't a verdict on whether the mechanism is *working* for the design headlines.
- **Interaction effects worth investigating**: If a new mechanism (e.g. A_k) makes Δtop1 worse than baseline, that tells us the mechanism is shaping the substrate in a particular way (mass redistribution off correct rank-1 basins). The system-level consequences of that depend on what Phase 6/7 do with the redistributed mass — but the *substrate-shaping* signal is real.
- **Pre-Phase-6 sanity check**: We don't have Phase 6/7 yet, so naive argmax is our only end-to-end readout. A monotonically collapsing top1 *might* indicate the substrate is becoming useless to *any* readout, not just argmax. The threshold for "concerning" is sustained collapse, not a one-report regression.

What it does NOT tell us: whether Phase 4 has graduated. That requires R@K + cap-coverage CI-disjoint-from-zero per the design spec.

## Standing discipline going forward

1. **Phase 4 graduation requires R@K AND cap-coverage CI-disjoint-from-zero.** This is the design spec's bar. top1 movement neither blocks nor enables graduation.
2. **Reports lead with the design-spec headlines.** When a result is reported, the first table must show R@K and cap-coverage. top1 may appear lower as a drill-down with explicit framing.
3. **"Blocker #6'" is reshaped**: not "top1 regression must be fixed", but "top1 regression is a substrate-shaping diagnostic to watch — Phase 4 is responsible for substrate quality (R@K + cap-coverage), and what happens to literal-token readout is downstream". Track it but don't gate on it.
4. **Surprising drill-downs warrant investigation, not promotion.** When Δtop1 surprises us, we investigate the mechanism. We do NOT add it to the headline set.
5. **Phase 6/7 deferred metrics are not Phase 4's responsibility.** When the system has rollout-based or LLM-based readout, the metric definition of "correctness" changes. Phase 4 is being measured on what it produces *for those phases to operate on*, not on what argmax outputs today.

## What this changes for the current sweep

The decay sweep currently running is asking the wrong question if framed as "does decay flip Δtop1 back positive?" The right question is:

> Does any decay value move **Δcap_t05** CI-disjoint-from-zero (positive)?

cap-coverage is the design-spec headline #2 that's been chronically un-verified since report 026. If A_k's theory of operation (push mass off dominant attractors) actually works at the substrate level, it should help cap-coverage. If decay doesn't move cap-coverage at any value, A_k is genuinely not earning its place at the substrate level.

When the sweep returns, the analysis table should be:

```
decay | Δcap_t05 (headline) | ΔR@10 (headline) | Δtop1 (drill-down)
```

And the verdict criterion is the first two columns, not the third.

## Status implications

- This note is methodology, not a result. It does NOT change any report or close any blocker.
- Report 035 should have a small addendum reframing its conclusions under design-spec headlines (will add when decay results are in, so the reframe lands once not twice).
- Blocker #6' in STATUS.md will be edited to reflect "drill-down to watch" rather than "load-bearing failure to fix" — once we have the decay data and write report 036 (or whatever consolidates the next step).

## Anti-homunculus / first-principles check

This re-framing strengthens the anti-homunculus discipline rather than weakening it. The previous framing implicitly assumed Phase 4 had a job (produce the right top1) that the architecture does not assign it. Re-framing returns top1 to the layer that's architecturally responsible for it (Phase 6/7) and lets Phase 4 be evaluated against what it is architecturally responsible for (substrate quality).

In the same way that the anti-homunculus filter says "don't add a supervisor that picks which subsystem wins," this discipline says "don't measure a subsystem against an output it isn't responsible for producing." Both are about respecting the architectural separation of responsibilities.
