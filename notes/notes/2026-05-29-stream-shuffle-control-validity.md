---
name: stream-shuffle-control-validity
date: 2026-05-29
project: personal-ai
phase: Phase 3 — Growing Codebook (Frame B)
status: validity argument (analytical) — branch-independent prereq, no run required
parent: 2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md
tags:
  - notes
  - subject/cognitive-architecture
  - subject/phase-3
  - subject/continual-learning
---

# Stream-shuffle control validity — it does NOT difference out the co-occurrence signal

**Purpose.** The retired C.3 gauge control had `E[Δ] = 0` *by construction* — it
was a gauge symmetry, so it was blind to the very thing it claimed to test
(corpus-specific learning). The new primary control for Frame B is the **global
token-stream shuffle**. This note argues, analytically, that the stream-shuffle
is **categorically different**: it is **not** a gauge symmetry, its `E[Δ]` need
not be 0, and a positive real−shuffle Δ is genuine evidence of corpus-specific
learning. This is a **validity argument** (analytical), a **branch-independent
prereq** — it holds regardless of the window-vs-slope branch and requires no run.

## Part 1 — Why the retired gauge control had `E[Δ] = 0` (and why)

The retired control (`control_mode="shuffled-token"`,
[c3_phase3_exit_criterion.py:641-646](../../experiments/c3_phase3_exit_criterion.py))
built `codebook_ctrl[i] = A_{π(i)}` — a **row-permutation of the codebook**. It
permuted *which i.i.d. atom each token-id wears*, leaving the corpus token-id
sequence `W` untouched. Per the anchor note's Part 1 exchangeability proof
([2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md:59-90](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md)),
`E[Δ_stratum] = 0` holds under two conditions, both satisfied by the code:

- **(C1) Equivariance** (anchor :66-72): `F` reads atoms *only* through codebook
  rows — no fixed token-id-indexed external reference — so jointly relabeling
  token-ids by `π` and permuting codebook rows by `π` leaves `F` invariant:
  `F(P_π X, W) = F(X, π(W))` per realization.
- **(C2) Exchangeability** (anchor :73-76): the atoms `A_i` are **i.i.d.**
  random-phase FHRR vectors
  ([torch_fhrr.py:65](../../src/energy_memory/substrate/torch_fhrr.py)) and `π`
  is seeded independently of `X`, so `P_ρ X =d X` for every fixed bijection `ρ`.

Combining (anchor :78-86):
`E[F(P_π X, W)] = E_π E_X[F(X, π(W))] = E_X[F(X, W)]`, i.e. `E[Δ_stratum] = 0`
**exactly**, per stratum. The realized Δ is nonzero only as per-seed noise (the
permutation is a symmetry of the *distribution*, not of a fixed realization), and
it averages to zero. **The control was symmetric across the two arms — blind to
the corpus signal by construction.**

## Part 2 — The stream-shuffle is categorically different (not a gauge symmetry)

The stream-shuffle control permutes the **corpus token ORDER**, not the
token→atom map. Concretely (per the anchor note Part 4, :233-238): permute the
flat token sequence before windowing, then **re-window**. This:

- **Preserves unigram marginals** — each token-id appears the same number of
  times, so the per-token frequency distribution is identical across worlds.
- **DESTROYS co-occurrence** — re-windowing a reordered stream breaks the
  within-window token pairings that carry the corpus's second-order structure.

Crucially, the two worlds **share the SAME codebook / atoms** (the real and
shuffle arms are **paired on the atom seed `s`** — only the corpus order changes;
see the headline-design spec, where `world="shuffled"` is seeded `s+80000` with
the *same atom set*,
[2026-05-28-frame-b-exposure-slope-headline-design.md:66-71](2026-05-28-frame-b-exposure-slope-headline-design.md)).

Therefore the real−shuffle difference is **not a relabeling of exchangeable
atoms** (the gauge move). The atoms are held identical; what changes is the
**co-occurrence structure of the data**. Consolidation generates its signal **by
retrieving through the landscape** (anchor :243-244,
[_consolidate_codebook step 3](../../experiments/c3_phase3_exit_criterion.py)) —
it *reads through* the corpus's co-occurrence geometry. Destroying that geometry
while holding the atoms fixed is a substantive change to the input the dynamics
read, not a symmetry of their distribution.

**Why the two C-conditions of Part 1 do NOT apply here:**

- **C2 (exchangeability) is irrelevant.** C2 was about the atoms being i.i.d. so
  that *relabeling them* leaves the distribution unchanged. The stream-shuffle
  does not relabel atoms — it leaves them identical. There is no atom permutation
  to be invariant under.
- **C1 (equivariance) does not bridge the two worlds.** C1 said permuting atoms
  and token-ids *together* is a symmetry of `F`. The stream-shuffle has no
  matching atom permutation to pair with the order change; it changes only `W`'s
  internal ordering. There is no `π` such that `F(X, W_real) = F(X, W_shuffle)`
  by symmetry, because `W_shuffle` is not a relabeling of `W_real` — it is a
  genuinely different (co-occurrence-destroyed) sequence over the same alphabet.

Hence **`E[Δ]` need not be 0**, and a positive real−shuffle Δ is **genuine
evidence of corpus-specific learning**: it can only arise if the dynamics
extracted something from the real co-occurrence structure that is absent once
that structure is scrambled. (This is exactly why the anchor note flags the
stream-shuffle as "not perfectly atom-matched" — and notes that *matched-ness was
the gauge control's fatal flaw*, anchor :237-238.)

## Part 3 — The single falsifier to check on any future code change

The gauge control's fatal flaw was an object indexed by raw, un-permuted
token-id that escaped the relabel (anchor's C1 falsifier, :83-86). The
stream-shuffle has a **dual falsifier**:

> **Falsifier.** Is there any object the pipeline reads that **carries
> co-occurrence structure** yet is **IDENTICAL across the real and shuffled
> worlds**?

If **yes** anywhere, the control **leaks**: that object lets the real arm and the
shuffle arm share co-occurrence information, so destroying the corpus order no
longer fully removes the signal, and the real−shuffle Δ understates (or, if the
leak is in the baseline, miscredits) the true corpus-specific effect. Concrete
things to audit on any future change:

- The **landscape / Hopfield memory**: it must be built from the *same world's*
  windows it is later used to score (the headline-design spec builds and freezes
  per-world; the rejected "real landscape + shuffled cons corpus" draft was
  exactly a co-occurrence leak — anchor :239-249). Each world must be
  self-consistent.
- Any **cached window set, co-occurrence matrix, or precomputed statistic**
  shared across the two arms.
- The **test set**: it is held fixed across worlds *by design* (it is the same
  held-out probe), which is fine — it carries co-occurrence but it is the
  *measurement instrument*, not part of the consolidation input being
  differenced. The differencing is over the *consolidation/landscape* corpus,
  not the probe. (The DiD's frozen-codebook arms C/D control for any test-set
  co-occurrence common to both worlds.)

The clean state is: the **only** co-occurrence-bearing object that differs
between worlds is the consolidation+landscape corpus order, and the **only**
co-occurrence-bearing object identical across worlds is the measurement
instrument (the fixed test set), whose contribution is differenced out by the
DiD.

## Part 4 — Optional cheap empirical probe (proposal, no run required)

To confirm the argument empirically at smoke scale (analogous to the gauge
control's 4a/4b probes), the cheapest decisive test:

- **Planted-co-occurrence synthetic corpus.** Build a tiny synthetic corpus with
  a deliberately planted co-occurrence pair (token `a` always near token `b`)
  on top of otherwise-uniform unigram draws, so that **unigram-only statistics
  match** between the real and stream-shuffled versions by construction.
- **Predicted reads if the control is valid:**
  - `β_real > β_shuffle` (or, at the endpoint, recall_real > recall_shuffle on
    the planted pair) — the dynamics learn the planted co-occurrence in the real
    world but not the shuffled one.
  - Unigram marginals **identical** across the two worlds (sanity: the shuffle
    really did preserve first-order and destroy only second-order structure).
- **Predicted reads if the control LEAKS:** `β_real ≈ β_shuffle` despite the
  planted pair — pointing straight back to the Part 3 falsifier (find the
  co-occurrence-bearing object shared across worlds).

This probe is optional; the validity argument stands on Parts 1–3 analytically.
The probe would *confirm* (not establish) it, and would be the right first check
if any future refactor touches the world-construction or landscape-build path.

## Status

**Validity argument (analytical).** Branch-independent prereq — holds regardless
of the window-vs-slope decision and requires no run. It does **not** sign off any
threshold and does **not** itself constitute a graduation result; it establishes
that the stream-shuffle is a *sound control* (unlike the retired gauge control),
which is a precondition for any Frame B / Gate 0 claim that leans on it.
