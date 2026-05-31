# Phase 3 — Consolidation-Write Sub-Program (design spec)

> Status: **ACTIVE** (opened 2026-05-30, user-approved). Supersedes the Frame-B
> level-DiD program (CLOSED — not economically rescuable, Report 117; three rescue
> levers closed, Reports 116/117/118). Grounded in the research-grounded forward
> plan: [brainstorm-workspace/2026-05-30-research-grounded-plan/](../../brainstorm-workspace/2026-05-30-research-grounded-plan/)
> (`00-diagnosis-synthesis.md`, `02-forward-plan-FINAL.md`, `03-grill-and-audits.md`).
> Confidence in the root cause: **medium** (reached by elimination, not yet a positive
> write-then-read). This spec is the citation target for the experiment preamble.

## The question

Does a **consolidation (learning) objective** write role-discriminative role→target
structure into the substrate that a **native** readout recovers above chance —
without a homunculus? Two independent project lines converge on the consolidation
**write** as the locus: Phase-5 role-binding (`hit_role=0.000` exactly across
D1/D3/E1/M1, sharpening monotonically worse = the anti-overlap signature of an
*absent* write; Reports 062–064) and Phase-3 Frame B (consolidation *injects*
variance σ_A=0.157 vs frozen σ_C=0.069; Reports 116–118).

## Locus (grill-verified at primary — do not mis-cite)

The per-event consolidation **write** is the pull/push contrastive blend at
`src/energy_memory/phase2/error_driven_learner.py:147-162` (mirror
`reconstruction_learner.py:147-162`): pull the correct atom toward
`normalize(Σ slot_query)`, push the self-mined `predicted_id` atom away.
It is **NOT** `phase4/consolidation.py:549/636` — that `members.mean` is the
tr(Σ)/anti-collapse **diagnostic** (docstring :538-540). Both the WF-1 diagnosis and
its synthesis mis-located the write; the surgical edit lands at the learner lines.

**"Zero-margin write" is a misnomer.** A margin/push term already exists and already
**failed** (phase3b, Fail Rate 1.0, seed 17) for three confounds the re-run must fix:
(i) the negative was **self-mined** via `sims.argmax()` at `error_driven_learner.py:106`
— a runtime metric-reader = **anti-homunculus thermostat**; (ii) the readout was
**top-1 cosine**, not basin-membership; (iii) **single seed**. EBM-2006's flat-surface
collapse is therefore a **loose analogy**, not a proof, for this margin-bearing blend;
the surviving "never wrote a recoverable basin" verdict rests on **elimination**
(Reports 062–064), not EBM.

## §Headline metric

**One headline, three named reads** (never one banner straddling reads of opposite
empirical status):

- **R1 — value-codebook `top_index_hits` Selectivity-Δ** *(the graduation headline)*:
  `Δ = top_index_hits(true-role cue) − top_index_hits(role-shuffled cue)` over the
  value/atom codebook, on a **held-out** split of the 4-role toy.
  `recoverability ≡ top_index_hits` (the Hopfield pre-decode argmax basin equals the
  cued target index; Report 066:19) — **NOT** top-1 cosine.
  **Two-floor PASS rule:** the absolute true-role arm Wilson-CI **lower bound > chance
  (1/N = 0.25 on the 4-role toy)** **AND** the Δ Wilson-CI **> 0**. A Δ>0 produced by a
  **sub-chance** shuffled arm is the anti-overlap signature and is **REJECTED**, not a
  pass. Ship `entropy` + `margin` drill-downs so `tix=0` is disambiguated from a
  sharp-wrong-basin (Report 066:36).
- **R2 — key-only-over-bound-pairs recoverability**: the read Reports 065/066 KILLED
  (`tix=0/3072` FHRR + GHRR-native); re-probed by **G-0/BTSP** with a margin write.
- **R3 — Phase-5 role-binding read**: re-opened only after a recoverable role basin is
  proven (R1). Distinct from the paused Phase-5′ ΔE headline.

This is a **drill-down**, not graduation, at any scale until the headline re-point is
recorded in STATUS (done 2026-05-30) and the report cites this spec.

## §Required controls (all on the same held-out test set)

1. **random-codebook ablation** — must collapse Δ→0 (and give a near-zero G-A
   refit-vs-production gap; a leaky readout that recovers under random codebook aborts).
2. **no-negatives / no-swap ablation** — must collapse Δ→0 for any contrastive/swap write.
3. **role-shuffle = read-time role-filler PAIRING shuffle** over fixed **distinct**
   position-roles (the `fixedpoint_free_shuffled_role` / `deranged_role` form in
   `phase5/natural_source_protocol.py`), **NOT** a codebook-relabel (which is
   C.3-gauge-vacuous, the trap that retired the old control, STATUS). Plus a
   precommitted C.3-style E-arm (identity-shuffle → byte-identical; random-shuffle → Δ
   scatters around 0 from **above** chance).
4. **content-matched non-positional shuffle** — because role≡position at
   `encoding.py:38`, random-codebook + no-negatives exclude neither position-legibility
   nor content-overlap; this is the control that makes Δ>0 a **role** result.
5. **perfect-cue / clean-codebook UPPER-BOUND arm** — bounds the ceiling so a null is
   attributable to "objective didn't write" vs "toy has no recoverable structure even in
   principle" (Report 066:74-75 perfect-cue=100%).
6. **lower-D / N-primary arm (G-C)** — N↓ predicted to raise recoverability (the real
   capacity lever), D↑ flat-to-better, **D↓ a predicted-null control** (lowering D at
   fixed N makes 1/√D crosstalk *worse* — "lower-D rescues" is mechanistically backwards).

## The immediate gate (five run-first adjudicators)

| Gate | What it tests | Routing |
|---|---|---|
| **G-A** frozen-substrate refit-readout | readout vs objective | refit recovers → **readout** defect (un-park GSBC, write-side moot); refit fails → structure never written |
| **G-0** BTSP key-only probe (cards first) | the binding-algebra wall (065/066, NOT excluded) | key-only basin forms → algebra rescuable; still null → algebra wall → bundle-first/substrate |
| **G-B** swap-negative phase3b A/B (behind a bundle-first margin-existence gate) | was phase3b's null the bad negative? | swap-neg Δ>0 → surgical; null while bundle-first holds → contrastive can't carve the landscape |
| **G-C** N-primary capacity surface | capacity vs objective | N↓/D↑ rescue → capacity (GSBC un-parks, mild-rebuild) |
| **G-D** R1 Selectivity-Δ | did a candidate write deposit a recoverable basin? | Δ>0 with controls collapsing + upper-bound non-trivial → **surgical confirmed** |

**Output = an evidence-backed four-way fork:** readout-fix / surgical-in-place /
mild-rebuild (capacity, swap substrate only) / rebuild-from-Phase-1. The rebuild
trigger fires **only** if G-A fails ∧ R1 null across contrastive **and** predictive
families ∧ G-C shows no N↓/D↑ rescue ∧ G-0 key-only fails.

## Candidate structure-writing objectives (batch-offline only; runtime error-driven BANNED)

Ranked, each cites ≥1 source_id + a principle check; link_only/absent carded from
primary before load-bearing (carding is licensed to return "does-not-transfer"):
**swap-reconstruction** (`arxiv:2412.19847` ArSyD; precommitted role-swap negative — the
FHRR-native form of the rank-1 negative that avoids the phase3b self-mined trap),
**predictive/JEPA** (`pdf:pam-2026`, `arxiv:2502.05164`, `arxiv:2501.14174`; cue-derived
self-target, EMA/stop-grad), **FEP single-phase self-orthogonalizing** (`arxiv:2505.22749`;
anti-Hebbian = the missing margin), **mixture-prior EM** (`arxiv:2406.07141`; negatives-free),
**Dorrell rectangular-support** (`arxiv:2410.06232`; Report 068 deferred the training step —
OPEN, fence-clear under Selectivity-Δ). **BTSP** (bioRxiv 2025.05.15.654220) for G-0.

## Anti-homunculus check (binding)

PASS by construction: every gate is an **offline/batch statistic** or a measurement of a
local dynamic; the new write is the **offline ReconstructionLearner-style** pass (runtime
error-driven BANNED, STATUS Live policies); the swap-negative is drawn by **fixed
role-equivalence-class membership + seed-fixed permutation, NEVER `sims.argmax`** (the
existing `:106` argmax-mined negative is the thermostat being **removed**); the replay
scheduler stays tension-driven; the Krotov-Latham dense-AM is admissible **only** as a
control architecture (its non-local rule BANNED as a write). **BANNED:** recall-gated
plasticity whose gate reads a metric and branches. **Phase-5′ fence stays down** — a
compliant read path terminates in a `top_index_hits` equality count and MUST NOT feed
scores into `mhn_energy`/`min`-over-branches → ΔE.
