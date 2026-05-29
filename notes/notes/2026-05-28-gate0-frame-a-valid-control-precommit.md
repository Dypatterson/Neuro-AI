---
name: gate0-frame-a-valid-control-precommit
date: 2026-05-28
project: personal-ai
mechanism: Gate 0 — existing Path C consolidation stack vs. a VALID (matched-world DiD) control
phase: Phase 3 — Growing Codebook (Frame B reframe)
status: precommit (GATE / diagnostic — does NOT graduate Phase 3 under any outcome)
parent: 2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md
tags:
  - notes
  - subject/cognitive-architecture
  - subject/phase-3
  - subject/continual-learning
---

# Gate 0 precommit — does the existing consolidation stack learn corpus structure against a VALID control?

## Status

**Gate / diagnostic precommit.** Gate 0 does **NOT** graduate Phase 3 under
any outcome — graduation requires the full Frame B continual-learning
criterion ([anchor note Part 3](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md)).
Gate 0's only job is to (a) **confirm the gauge finding empirically** and
(b) **select which Frame B path to build** via pre-committed branches. No new
mechanism is introduced (the control is a corpus-data manipulation; pull/push
+ C.2.x are existing and reviewer-PASSed), so **no new anti-homunculus
reviewer pass is required.**

## Design history (why DiD, not a single-arm control)

A first draft held the Hopfield landscape fixed on real windows and only
shuffled the consolidation corpus. **Rejected on review:** the consolidation
loop generates its training signal *by retrieving through the landscape*
([_consolidate_codebook step 3](../../experiments/c3_phase3_exit_criterion.py)),
so "real landscape + shuffled consolidation corpus" feeds shuffled cues
through real memory — a degenerate noise-injection regime that can push the
codebook *below* no-consolidation and **inflate Δ → false G0→pass.** The fix
is the standard real-vs-shuffled-corpus control run in **matched, self-
consistent worlds**, with no-consolidation baselines differencing out the
landscape effect (difference-in-differences).

## Experiment preamble (per CLAUDE.md)

> **Active phase:** Phase 3 — Growing Codebook (Frame B reframe,
> [anchor note](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md)).
> **Headline metric per [anchor note Part 5](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md):**
> the **consolidation corpus-specificity DiD** `[(A)−(C)] − [(B)−(D)]`
> (stratum-pooled Recall@K, per atom seed). **This is explicitly a GATE/
> diagnostic, not the Frame B graduation headline** — it measures whether the
> existing mechanism carries corpus signal against a *valid* control, which
> selects the Frame B build path.
> **Required controls per same spec:** matched-world global token-stream
> shuffle (B, D); no-consolidation Phase 2 baselines (C, D); gauge-control
> confirmation (E).
> **Last verified result:** [Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md)
> (Γ1 gate FAIL) and [Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md)
> (Path C walk-back) — both against the now-retired **gauge-vacuous** control,
> hence uninterpretable for corpus-specificity.
> **Why this experiment now:** the [anchor note](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md)
> shows the C.3 shuffled-token control is a gauge transform (`E[Δ]=0`). Before
> any mechanism redesign (Γ2/Γ3) or Frame B build, re-test the simplest
> existing mechanism against a control that can actually detect corpus
> structure.

## What Gate 0 measures

The diagnostic question: **does the existing consolidation stack extract
corpus co-occurrence structure** — i.e., does it add *more* to completion in a
real (structured) corpus than in a frequency-matched, co-occurrence-destroyed
corpus?

## Conditions

One **global token-stream shuffle** defines the shuffled world: flatten the
wikitext token stream, permute the order (seeded `atom_seed + 80000`),
re-window at the same `window_size`. Preserves unigram frequencies; destroys
co-occurrence. Each world is run **self-consistently** — landscape,
consolidation corpus, and held-out test all drawn from the *same* world — so
there is no mismatched regime.

| | World | Landscape | Consolidation | Test set | Role |
|---|---|---|---|---|---|
| **A** | real | real | Path C stack | real held-out | real, consolidated |
| **B** | shuffled | shuffled | Path C stack | shuffled held-out | shuffled, consolidated |
| **C** | real | real | **frozen codebook** | real held-out | Phase 2 baseline (real) |
| **D** | shuffled | shuffled | **frozen codebook** | shuffled held-out | Phase 2 baseline (shuffled) |
| **E** | real | real | Path C stack, **codebook row-permuted** | real held-out | gauge confirmation |

A, B, C, D: atom seeds 0..9 (paired — at seed *s*, A/C share the real corpus
and atom set; B/D share `shuffle(s)` and the atom set). E: atom seed 0 ×
permutation seeds 0..9, plus an identity-permutation unit test.

**Mechanism (A, B, E):** the existing **Path C consolidation stack** —
pull/push (`lr_pull=0.1`, `lr_push=0.05`, `use_pull_push=True`,
`use_context_residual=False`) + C.2.1–C.2.5 at Path C values
(`α_anti=0.01`, `repulsion_step_size=0.05`). No redesign.

## Operating point

Match the Γ1 headline / Path C wikitext point exactly, so condition **A**
reproduces the prior standard condition and the *only* change vs. prior work
is the control:

| Knob | Value |
|---|---|
| corpus | wikitext-2-raw-v1, vocab_cap=1000, window=8 |
| D | 4096 |
| β | 10 |
| K (Recall@K) | 5 |
| landscape_size | 64 |
| n_consolidation_events | 1000 |
| theta_prime_mode | both (default + calibrated) |
| n_seeds | 10 (atom seeds 0..9); E = atom seed 0 × perm seeds 0..9 |

## Falsifiable criterion + pre-committed branches

**Primary metric — consolidation corpus-specificity (DiD).** Per atom seed
*s*, on **stratum-pooled** Recall@K (regime strata are degenerate for the
frozen-codebook arms C/D, so strata are drill-down only):

```
DiD_s = [Recall_A(s) − Recall_C(s)] − [Recall_B(s) − Recall_D(s)]
```

i.e. (consolidation benefit in the real world) − (consolidation benefit in
the shuffled world). Pass requires **both**:
1. CI on mean DiD over the 10 seeds (t- or bootstrap interval) **strictly
   above 0**; and
2. per-seed robustness ≥ 70% (≥ 7/10 seeds with `DiD_s > 0`).

**Secondary reads:** `(A)−(B)` = whole-pipeline corpus-sensitivity;
`(A)−(C)` = consolidation benefit on real (does consolidation help at all).

**Gauge-confirmation (E):**
- **4a (mandatory local unit test, not a Colab run):** gauge control with the
  *identity* permutation must be **byte-identical** to condition A — proves
  the control's only effect is the permutation.
- **4b:** across the 10 permutation seeds at fixed atom seed 0, mean Δ vs. A
  within ~1 SEM of 0 (predict ≈0; per-seed spread expected).

| Outcome | Signature | Pre-committed next step (all route to Frame B) |
|---|---|---|
| **G0→pass** | DiD CI > 0 **and** ≥70% per-seed | Consolidation **is** corpus-specific. → Build **Frame B on the existing Path C stack**; open question is continual novelty over time, not mechanism redesign. **Γ1/Γ2/Γ3 do NOT reopen.** |
| **G0→weak** | DiD mean > 0 but CI overlaps 0 **or** <70% per-seed | Real-but-underpowered/weak — **not** a redesign trigger. → Escalate to n=30 (or pooled) before any branch decision. Distinguishes "weak signal" from "no signal." |
| **G0→null-cons** | DiD ≈ 0 (mean ≈0) **but** (A)−(B) > 0 | Landscape carries corpus structure; **consolidation does not.** → Frame B mechanism redesign, **targeting consolidation specifically**, evaluated in the Frame B frame. |
| **G0→dead** | (A) ≈ (B) | The whole pipeline captures no corpus structure. → Deeper redesign (landscape/substrate), still in the Frame B frame. |
| **G0→confound** | 4a not byte-identical, OR 4b mean Δ not ≈0 | The gauge finding (anchor Part 1) is wrong. **STOP** and re-derive before any further step. |

## Anti-homunculus

No new mechanism. The control is a corpus-data manipulation; the consolidation
update is the existing pull/push + C.2.x (already reviewer-PASSed). The
"apparent decision" of which atoms move lives in the existing buffer-fill +
local-geometry dynamics, unchanged. **No new reviewer pass required.**

## What this precommit does NOT permit

- **No graduation claim under any outcome.** Gate 0 is a gate; graduation
  needs the full Frame B continual criterion.
- **No mechanism redesign under G0→pass.** A pass *closes* the redesign track.
- **No redesign under G0→weak without n-escalation first.**
- **No Frame B build under G0→null-cons / G0→dead without re-entering
  candidate selection (in the Frame B frame).**
- **No operating-point sweep.** Single op point (the Γ1 headline point) so A
  reproduces prior work. Op-point robustness is a later Frame B question.
- **No within-window shuffle** in this run (held as a stricter drill-down if
  Gate 0 passes — anchor Part 8 #3).
- **No Phase 5 work of any kind.**

## Implementation surface

- [experiments/c3_phase3_exit_criterion.py](../../experiments/c3_phase3_exit_criterion.py):
  - Add a **global corpus-stream-shuffle** mode: shuffle the flat wikitext
    token stream (seeded `atom_seed+80000`), preserving unigram frequencies,
    *before* any windowing — so landscape, consolidation, and test for the
    shuffled world (B, D) are all drawn self-consistently from the shuffled
    stream. (Simpler than decoupling landscape from consolidation — no driver
    plumbing change to the landscape/consolidation split.)
  - C, D (no-consolidation): existing `run_consolidation=False` path with the
    appropriate world's landscape + test; frozen random codebook.
  - E (gauge-confirmation): existing `control_mode="shuffled-token"` + a
    permutation-seed parameter + an identity-permutation option for 4a.
- New unit test: identity-permutation gauge control byte-identical to A (4a).
- Companion Colab notebook (parallel procs), per the project Colab workflow.

## Cost

- A, B, C, D: 4 × 10 = 40 procs. E: 10 procs. **~50 procs**, comparable to the
  Γ1 headline gate (~1 hr on A100/L4). Plus the local 4a unit test (seconds).

---

*See also: [anchor note](2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md),
[phase-3-deep-dive.md](../emergent-codebook/phase-3-deep-dive.md) (§Headline
superseded banner), [Report 112](../../reports/112_phase3_c3_wikitext_graduation_walkback.md),
[Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md).*
