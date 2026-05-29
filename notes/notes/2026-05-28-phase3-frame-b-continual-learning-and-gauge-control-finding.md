---
name: phase3-frame-b-continual-learning-and-gauge-control-finding
date: 2026-05-28
project: personal-ai
phase: Phase 3 — Growing Codebook (reframe)
status: design-note / north-star (Frame B headline PROPOSED, pending user sign-off on exact criterion)
supersedes: notes/emergent-codebook/path-gamma-mechanism-family-survey.md (Γ2-vs-Γ3 framing)
tags:
  - notes
  - subject/cognitive-architecture
  - subject/phase-3
  - subject/continual-learning
---

# Phase 3 reframe — continual learning (Frame B) is the north star; the C.3 shuffled-token control is gauge-vacuous

**This note is the Phase 3 anchor.** It (1) records a structural finding that
the current C.3 graduation control cannot test what it claims, (2) reframes
Phase 3 around its actual title — *Growing Codebook / continual learning* —
as the pinned headline, and (3) demotes the previous mechanism-redesign
program (Path γ / Γ1 / Γ2 / Γ3) to a *gated sub-step* whose outcomes both
route back to this frame.

## TL;DR

1. **The C.3 "shuffled-token control" is a gauge transformation on i.i.d.
   random atoms, so `E[Δ] = 0` by construction** — in every regime stratum,
   regardless of corpus or mechanism. The control permutes *which random atom
   each token-id wears*; the atoms are i.i.d. ([torch_fhrr.py:65](../../src/energy_memory/substrate/torch_fhrr.py))
   and the permutation is independent of them, so standard and control are
   identically distributed. The chain of Path C / Path α / Γ1 nulls measured
   against this control is **uninterpretable as evidence about corpus-specific
   learning.**
2. **Phase 3's real claim is continual learning** ("Growing Codebook,"
   [PROJECT_PLAN:160](../../docs/PROJECT_PLAN.md)): atoms encode and refine
   from an experience stream over time. The static "consolidate-then-score a
   fixed random codebook" framing of C.3 is *why* the corpus-specificity
   control collapsed into a gauge symmetry.
3. **New headline (Frame B, proposed):** completion quality improves *with
   exposure* in a way that depends on real corpus co-occurrence — tested
   against a **corpus-structure-destroying** control (stream shuffle), not a
   codebook relabel.
4. **Gate 0 (Frame A):** one cheap run of *plain pull/push* against a valid
   control, with both outcomes pre-committed to route into Frame B. This
   confirms the gauge finding and decides which Frame-B path to build, without
   becoming its own program.

## Part 1 — The gauge-vacuous control finding

### Claim

The C.3 graduation control (`control_mode="shuffled-token"`,
[c3_phase3_exit_criterion.py:641-646](../../experiments/c3_phase3_exit_criterion.py))
builds `codebook_ctrl[i] = A_{π(i)}` — a row-permutation of the standard
codebook — then runs the identical pipeline (landscape, consolidation,
per-condition regime, eval). Standard vs. control therefore has expected
Δ = 0 in every stratum, for any corpus and any consolidation mechanism.

### Proof sketch (exchangeability)

Let `X = (A_1, …, A_V)` be the codebook atoms and `W` the (un-permuted)
token-id corpus. `E[Δ_stratum] = 0` holds under **two** conditions, both of
which the code satisfies (verified line-by-line 2026-05-28; see
§Confidence):

- **(C1) Equivariance.** `F` reads atoms *only* through codebook rows — no
  fixed token-id-indexed external reference. Equivalently, jointly
  relabeling token-ids by `π` and permuting codebook rows by `π` leaves `F`
  invariant: **`F(P_π X, W) = F(X, π(W))`** per realization. (Verified:
  consolidation, regime stratification, and eval all index *into* the
  codebook; eval scores against the control's *own* codebook; positions and
  the mask vector are slot-indexed, not token-id-indexed.)
- **(C2) Exchangeability.** The `A_i` are **i.i.d.** random-phase FHRR
  vectors ([torch_fhrr.py:65](../../src/energy_memory/substrate/torch_fhrr.py))
  and `π` is seeded independently of `X`, so `P_ρ X =d X` for every fixed
  bijection `ρ`, giving `E_X[F(X, ρ(W))] = E_X[F(X, W)]`.

Combining C1 + C2 over the random `π`:
`E[F(P_π X, W)] = E_π E_X[F(X, π(W))] = E_X[F(X, W)]`, i.e.
`E[Δ_stratum] = 0` **exactly**, per stratum. (The note's earlier one-liner
"`P_π X =d X` hence `F(P_π X) =d F(X)`" was too quick — it glossed that the
permuted codebook is fed *un-permuted* token-ids, coupling the relabel to
the corpus; C1 is the lemma that closes that gap. The single falsifier to
check on any future code change: *is there any object read by the pipeline
that is indexed by raw, un-permuted token-id and NOT itself permuted by `π`?*
If yes anywhere, C1 fails and `E[Δ]` need not be 0.)

The *realized* Δ is nonzero only because, for a **fixed** draw `X`, the
permutation is not a symmetry of that realization — it is a symmetry of the
distribution. That realized difference is per-seed noise; it averages to zero.

### Retrodiction (corroboration, *consistent-with* — not the load-bearing argument)

| Prediction from `E[Δ]=0` | Observed in committed reports |
|---|---|
| Δ centers near 0, large per-seed spread | typical Δ≈+0.02–0.03, **σ≈0.13**, range −0.19..+0.465 ([Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md)) |
| sign **reverses** across seed sets | v5 D=4096 seeds 30–39 = **−0.017**; D=2048 = **−0.011**; D=16384 reversed |
| ≈50% of seeds positive (a fair coin) | Γ1 per-seed paired robustness = **5/10 = 50.0%** ([Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md)) |
| "+0.055 disjoint" is a tail draw | the walk-back's *own* conclusion |
| mechanism **strength** is irrelevant | "stronger pull doesn't help; weaker doesn't help; signal floor is intrinsic" ([112:187](../../reports/112_phase3_c3_wikitext_graduation_walkback.md)) |
| pooled n=30 "margin 0.003" is a pooling fallacy | true SEM ≈ 0.13/√30 ≈ 0.024 → +0.020 is **<1 SEM from 0** |

Every anomaly the v3→v6 chain chased is *consistent with* differencing two
identically-distributed conditions.

> **Diagnosticity caveat (added 2026-05-28 after adversarial verification).**
> Retrodiction is weaker than prediction: a *small real signal*
> (+0.02–0.04 with per-seed σ≈0.13) reproduces **every** row of this table
> equally well, so the table does **not** distinguish `E[Δ]=0` from a weak
> real effect. The headline is carried by the **exchangeability proof**
> (C1+C2 above) and confirmed empirically by **Gate 0**, *not* by this
> table. Two row-level caveats: (a) the "≈50% positive" row cites Γ1's
> 5/10 (a *mechanism*, [113:81](../../reports/113_path_gamma_gamma1_headline_gate.md)),
> whereas the gauge control's own rate is PathC 6/10 ([113:82](../../reports/113_path_gamma_gamma1_headline_gate.md)) —
> both non-diagnostic; (b) the "+0.020 / margin 0.003" row conflates two
> D=4096 cells (default/spread vs calibrated/tight). The *sharper* knife
> than the "pooling-fallacy / SEM" framing is **pseudo-replication**: the
> committed disjoint-CI gate ran per-*trial* Wilson intervals on counts
> pooled across seeds, but all trials in a seed share one codebook, so it
> under-states variance and would manufacture "disjoint" cells even under
> `E[Δ]=0`. That harness bug is now **fixed** (per-seed inference; the atom
> seed is the unit) at [c3_phase3_exit_criterion.py](../../experiments/c3_phase3_exit_criterion.py)
> `_delta_ci_stats` / `graduation_per_seed`, with regression tests.

### What it does NOT mean

Consolidation is **not** failing. Absolute recall is ~0.20–0.26 at
R@5/vocab=1000 — ~50× chance. Both arms show strong contextual completion;
the corpus signal is real and present. The control is simply blind to it
because it is symmetric across the two arms.

### What it invalidates

- Path C / Path α / Γ1 nulls are uninterpretable as evidence about
  corpus-specific learning.
- The Γ2 motivation ("two per-atom mechanisms failed → the unit is
  exhausted") is **not established**.
- The [Path γ mechanism-family survey](../emergent-codebook/path-gamma-mechanism-family-survey.md)
  Γ2-vs-Γ3 framing is superseded (it was downstream of the broken control).

What it does **not** invalidate (scope, sharpened 2026-05-28): the
**graduation *decisions*** of Reports 112/113/114 still stand — "does not
graduate" holds under *either* `E[Δ]=0` or a small-real-signal hypothesis,
because σ/μ ≈ 3.3 fails the per-seed robustness clause regardless — as do
the **absolute-recall** and basin-tightening measurements. What is
overturned is narrower: the *corpus-specificity interpretation* of those
nulls, and the Γ2 mandate built on it. (One concrete over-interpretation
the gauge lens correctly flags: Report 114's lr_cr 0.20/0.50 "atom-vs-atom
repulsion actively degrades the codebook" was an n=3 noise artifact — it
collapses to t≈1.2 at n=60.)

### Confidence + confirmation

High — carried by the **exchangeability proof** (C1+C2), not the
retrodiction. The proof was independently verified 2026-05-28 by adversarial
review (4 analytical lenses + 1 empirical probe + synthesis): atoms are
i.i.d. ([torch_fhrr.py:65-67](../../src/energy_memory/substrate/torch_fhrr.py)),
`π` uses a disjoint RNG (`random.Random(seed+70000)`) that never advances
the substrate generator, and the permuted codebook is threaded consistently
through encode → landscape → consolidate → stratify → eval with **no**
token-id-indexed object escaping the relabel (C1 holds). The empirical probe
reproduced both Gate 0 predictions at smoke scale: identity-permutation →
**byte-identical** per-stratum outcomes (4a), and across seeds Δ scatters
around 0 with sign flips and no consistent offset (4b). Verdict:
**sound-with-caveats** (the caveats are the editorial ones above — the
retrodiction is corroborative, not diagnostic). Because the proof contradicts
the committed-report *interpretation*, **Gate 0 confirms it empirically as a
side-effect** by running the old gauge control as one of its conditions
(predicts Δ≈0, with a pre-committed STOP-and-re-derive branch if not)
alongside the valid control.

## Part 2 — Reframe: Phase 3 is continual learning

Phase 3 is titled **"Growing Codebook"**; its purpose is *"atoms that drift,
stabilize, split, decay, and **consolidate from experience**"*
([PROJECT_PLAN:160-163](../../docs/PROJECT_PLAN.md)). Two senses are tangled:

- **Sense A — atom refinement (fixed vocab).** Atoms inside a fixed set of
  token-slots refine from experience. *This is what C.3 tests, and what Path
  C / Γ1 / Γ2 / Γ3 are about.*
- **Sense B — novelty allocation (vocab grows).** New words/concepts get new
  atoms over time; the codebook grows in size. *This is the project's vision
  ("novelty encoded over time"); it is largely unphased today — closest to
  Γ5 (Hyperseed); atom-splitting is the Phase-3-tracked / Phase-5-executed
  fragment.*

**Frame B (continual learning) is the pinned north star.** Sense A is a valid
*slice* of it (the asymptotic snapshot), tested first as Gate 0.

### Bag-vs-stream clarification (so Frame B does not contradict the replay design)

The *input corpus* is an ordered stream (experiences arrive in time). *Replay*
remains a prioritized bag (`ReplayStore.sample()` = priority-weighted
multinomial, [replay_loop.py:330](../../src/energy_memory/phase4/replay_loop.py)).
Frame B exercises the *input* axis only; the no-ordered-replay commitment is
untouched. (This is also why SFA / Γ3 was set aside: it needed ordered
*replay*, which the architecture deliberately lacks.)

## Part 3 — Frame B headline metric + graduation criterion (PROPOSED — needs sign-off)

> **REFINED 2026-05-28 → see [2026-05-28-frame-b-exposure-slope-headline-design.md](2026-05-28-frame-b-exposure-slope-headline-design.md).**
> The "slope (and asymptote)" sketch below is superseded by the precise
> **within-seed slope-DiD** design (the *asymptote* = the endpoint, which is
> exactly the high-variance Gate 0 DiD, so it is demoted to a drill-down; the
> *slope* differences out the per-seed codebook intercept that made the
> endpoint underpowered). The sketch is retained below for history.

> **PROPOSED headline:** *Masked-token contextual-completion quality improves
> with cumulative exposure to the corpus stream, and the improvement depends
> on real corpus co-occurrence.*
>
> Operationalized as the **exposure–recall slope**: Recall@K measured at
> checkpoints along the consolidation stream; the headline is the slope (and
> asymptote) of that curve, compared against the corpus-scramble control.

> **PROPOSED graduation criterion (both clauses, mirroring the revised C.3
> structure):**
> 1. The real-corpus exposure–recall curve is CI-disjoint above the
>    corpus-scramble control's curve at the asymptote, n ≥ 10 seeds.
> 2. Per-seed paired robustness ≥ 70% (≥ 7/10 seeds with real > scramble at
>    asymptote).

These exact numbers are **proposed, not yet binding** — they are the main
thing requiring user sign-off before they enter the design spec.

## Part 4 — Required controls

- **Retired:** the gauge-vacuous "shuffled-token = codebook row-permutation"
  control. Reason: Part 1. (Likely a semantic drift from the PROJECT_PLAN's
  "shuffled-token control" — which most naturally meant *shuffle the token
  stream* — into "permute the token→vector map" during the 2026-05-26 Path α
  reframe.)
- **New primary control — corpus-stream shuffle** (decided 2026-05-28:
  *global* token-stream shuffle). Permute the flat token sequence before
  windowing: preserves unigram marginals, destroys co-occurrence. Run the
  identical pipeline. Real-beats-scramble is then a genuine test of
  corpus-specific learning. (Not perfectly atom-matched — that matched-ness
  was the gauge control's fatal flaw.)
  - **Landscape confound (resolved via DiD, not held-fixed-landscape).** A
    positive whole-pipeline Δ could be the *landscape* being corpus-sensitive
    (Phase 2), not *consolidation* (Phase 3). A first draft held the landscape
    fixed on real windows and shuffled only the consolidation corpus —
    **rejected on review**: consolidation generates its signal *by retrieving
    through the landscape* ([_consolidate_codebook step 3](../../experiments/c3_phase3_exit_criterion.py)),
    so "real landscape + shuffled cons corpus" is a degenerate noise-injection
    regime that inflates Δ → false pass. **Fix: run real and shuffled worlds
    self-consistently and difference out the landscape with no-consolidation
    baselines (difference-in-differences).** See Gate 0 precommit:
    [2026-05-28-gate0-frame-a-valid-control-precommit.md](2026-05-28-gate0-frame-a-valid-control-precommit.md).
- **Secondary / drill-down — novelty-introduction.** Introduce held-out
  tokens partway through the stream; measure exposures-until-retrievable in
  real vs. scrambled context. (Directly probes Sense B.)

## Part 5 — Gate 0 (Frame A): one run, pre-committed branches

**Mechanism:** the *existing Path C consolidation stack* — pull/push
(`use_pull_push=True`, `use_context_residual=False`) **+ C.2.1–C.2.5 at Path C
values**, so condition A reproduces prior work exactly. No redesign.

**Conditions (matched-world difference-in-differences, n ≥ 10 atom seeds; full
spec in the [Gate 0 precommit](2026-05-28-gate0-frame-a-valid-control-precommit.md)):**
- **A** real world (landscape+cons+test all real), Path C stack;
- **B** shuffled world (landscape+cons+test all from one global stream
  shuffle), Path C stack;
- **C** real world, **frozen codebook** (Phase 2 baseline);
- **D** shuffled world, **frozen codebook** (Phase 2 baseline);
- **E** gauge confirmation (fixed atom seed × perm seeds + identity unit test).

**Headline read (DiD):** per seed, `[(A)−(C)] − [(B)−(D)]` on stratum-pooled
Recall@K — consolidation's benefit on real *minus* on shuffled. Pass = CI > 0
**and** ≥70% per-seed. This differences out the landscape (Phase 2) effect,
isolating consolidation's corpus-specificity, with each world self-consistent
(no mismatched regime). Secondary: (A)−(B) whole-pipeline; (A)−(C)
consolidation-on-real. E confirms the gauge finding.

**Pre-committed branches (all route to Frame B — Gate 0 cannot terminate the
program; full table in the [Gate 0 precommit](2026-05-28-gate0-frame-a-valid-control-precommit.md)):**
- **G0→pass** (DiD CI > 0, ≥70% per-seed): consolidation is corpus-specific →
  build **Frame B on the existing Path C stack**; Γ1/Γ2/Γ3 stay closed.
- **G0→weak** (DiD mean > 0 but CI overlaps / <70%): underpowered, **not** a
  redesign trigger → escalate n before deciding.
- **G0→null-cons** (DiD ≈ 0 but (A)−(B) > 0): landscape carries structure,
  consolidation does not → redesign **targeting consolidation**, in the Frame
  B frame.
- **G0→dead** ((A) ≈ (B)): pipeline captures no structure → deeper redesign.
- **G0→confound** (gauge arm E fails 4a/4b): Part 1 is wrong → stop and
  re-derive.

## Part 6 — Anti-homunculus pre-check (for Frame B's eventual mechanism)

Gate 0 introduces no new mechanism (a control is a data manipulation), so no
reviewer pass is needed for it. **Frame B's continual mechanism will need a
pass** when specified, with two known risks:
- **The clock.** Online/continual processing tempts a scheduler ("now
  consolidate / now allocate"). Consolidation must stay a local geometric
  event (buffer-fill), allocation (if Sense B) a local residual-energy event —
  never a controller reading a clock.
- **Allocation threshold (Sense B).** "Allocate a new atom when residual >
  θ" must be substrate-derived (1/β-equivalent), not externally set.

## Part 7 — Relationship to prior work

- **Phase 5′ remains paused.** No ΔE / bridge / M2 / matrix / headline /
  graduation work.
- **Survey candidates repositioned:** Γ1 closed (atom-vs-atom repulsion;
  finding stands independent of the control bug). Γ2 / Γ3 / Γ4 / Γ5 are *not*
  the active deliverable; they re-enter only via G0→null, in the Frame B frame.
- **The gauge finding is independent of all of the above** — it is a property
  of the test harness, not any mechanism.

## Part 8 — Binding edits + open questions

**Edits that pin this (made / to make):**
- `notes/emergent-codebook/phase-3-deep-dive.md` §Headline + §Graduation:
  retire the gauge control, point to this note as the superseding frame.
- `STATUS.md`: walk-back banner + recent-updates entry.

**Open questions for sign-off (before the criterion becomes binding):**
1. Exact Frame B headline — exposure–recall *slope/asymptote* vs.
   time-to-encode-novelty as the primary number? (Part 3 proposes slope.)
2. Frame A first (Gate 0), or skip to building Frame B directly? (User chose
   Gate 0-embedded-in-B on 2026-05-28.)
3. ~~Global stream shuffle vs. within-window shuffle?~~ **RESOLVED 2026-05-28:
   global token-stream shuffle, with the landscape held fixed on real windows
   (see Part 4 landscape-confound note). Within-window shuffle is held as a
   stricter drill-down if Gate 0 passes.** Precommit:
   [2026-05-28-gate0-frame-a-valid-control-precommit.md](2026-05-28-gate0-frame-a-valid-control-precommit.md).
