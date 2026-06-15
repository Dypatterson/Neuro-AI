# PRE-COMMIT — Bet B: the SCAN discriminating regime (retrospective §6, after toy-land closed)

*Status: DRAFT for user review (2026-06-15). Supersedes the modular-arithmetic
compositional regime ([betb-compositional-discriminating-regime-precommit.md](betb-compositional-discriminating-regime-precommit.md)),
which was empirically shown NULL — toy-land can't host the discriminating regime
(that doc §8.1). Extends [CONTEXT-B.md §8](../CONTEXT-B.md).*

---

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** Bet-B continual compounding-transfer ([CONTEXT-B.md §8](../CONTEXT-B.md))
> in the **discriminating / compositional regime** ([RETROSPECTIVE §6:156-167](RETROSPECTIVE-two-bets-2026-06-06.md))
> — now on a real benchmark where "simple methods fail to compose" is documented.
>
> **Headline metric (Stage 1 graduation) per [§3](#3-headline--controls):** **exact-match
> accuracy on the SCAN add-primitive-`jump` test split.** A restructuring consolidation
> (+ replay) must lift it CI-disjointly above the replay-only baseline — i.e. manufacture
> the compositional generalization protection/replay cannot.
>
> **Headline metric (Stage 0 = the gap-diagnostic, run FIRST, make-or-break):** reproduce
> the **documented failure** — our small seq2seq scores **≈100% on the random (simple)
> split but ≈0–5% on the add-`jump` split.** If it does NOT fail the jump split at our
> scale, the regime isn't discriminating here → fix scale/model before any mechanism.
>
> **Required controls per [§3](#3-headline--controls):** random/simple split (sanity,
> ~100%); plain-train baseline (the documented failure); replay-only; consolidation-only;
> replay+consolidation; **redundancy guards** (GECA-style data augmentation; a known
> equivariance/meta-seq2seq reference) — the "you're just a known SCAN fix" control.
>
> **Last verified result:** [Report 139](../reports/139_betb_replay_x_consolidation_bennafusi/report.md)
> (recombinant pass on the non-discriminating toy) + the 2026-06-15 modular-arithmetic
> NULL (toy substrate too brittle: bimodal compositions, multi-task grokking collapse).
>
> **Why this experiment now:** the retrospective §6 names a compositional benchmark as
> THE way to move the central thesis from UNCONFIRMED to tested, and 2026-06-15 closed the
> cheap toy path. SCAN is the canonical regime where the simple method demonstrably fails,
> at laptop scale.

**Stage 0 is a VALIDITY diagnostic, not a graduation** (preamble rule) — the SCAN analog
of Report 136's gap-diagnostic and the 2026-06-15 toy Step-1. Mechanism work (Stage 1)
is gated on Stage 0 reproducing the gap.

---

## 1. The benchmark + the gap (VERIFIED 2026-06-15, data in `data/scan/`)

SCAN (Lake & Baroni 2018): natural-language commands → action sequences. Input vocab (13):
`after and around jump left look opposite right run thrice turn twice walk`. Four primitive
verbs (jump/walk/run/look) combine with modifiers (twice/thrice/around/opposite/left/right)
and connectives (and/after) via 12 templates.

**The add-primitive-`jump` split (`data/scan/tasks_{train,test}_addprim_jump.txt`):**
- Train: 14670 lines. **`jump` appears ONLY as the isolated primitive** (`IN: jump OUT:
  I_JUMP`, 1467× for balance); **0 compositions of `jump`.** The other verbs DO appear
  composed → the model learns every template, just never with `jump`.
- Test: 7706 lines, **all compose `jump`** into the learned templates (`jump twice`,
  `jump around left`, …).
- **Documented result:** RNN seq2seq ≈ 0–1% exact-match here, ≈ 100% on the random split.
  *(Verify it reproduces at our scale — Stage 0.)*

This is the literal "T3 = compose a known primitive into known templates" we tried to build
in modular arithmetic — but here the gap is real, documented, and substrate-robust.

**Mapping to the project's mechanism vocabulary** (CONTEXT-B §8 Terminology):
- **Replay** = re-presentation/rehearsal of training pairs (incl. balanced primitive). The
  *plain* training already includes the isolated `jump`; replay ≠ a fix.
- **Consolidation** = an offline RESTRUCTURING pass with a fixed local non-reconstruction
  objective that **factorizes/aligns the verb-slot representation** so `jump`'s encoding
  lands in the same role-subspace as the verbs seen in templates → the decoder's learned
  template circuitry then applies to `jump`. The thesis: this manufactures the compositional
  generalization that replay alone does not.

---

## 2. Staged plan (gap-diagnostic-first — the discipline that paid off 2026-06-15)

- **STAGE 0 — build + reproduce the gap (make-or-break, ~1 build session).**
  - SCAN loader (tokenize, vocab, padded batches, exact-match-sequence eval).
  - A small seq2seq (1–2 layer GRU enc-dec + attention, teacher forcing) — laptop/MPS-sized.
  - Train on the add-`jump` train split; eval on BOTH random(simple) and add-`jump` test.
  - **GATE:** random-split ≈100% AND jump-split ≤ ~5% (multi-seed). Reproduces the documented
    failure ⇒ the gap is real at our scale ⇒ Stage 1 licensed. If jump-split is already
    high, our model/regime isn't discriminating → adjust (smaller model / no attention /
    the documented config) before any mechanism.
- **STAGE 1 — the mechanism (only if Stage 0 reproduces the gap).** The 2×2 (replay ×
  restructuring-consolidation) on the SCAN jump split; headline = jump-split exact-match.
  Consolidation recipe is SWAPPABLE (charter) and must pass anti-homunculus review before
  build. First candidate(s) deferred to Stage-1 design (e.g. a slow offline alignment of
  verb-slot encodings via a fixed local decorrelation/role-consistency loss).

---

## 3. Headline & controls (Stage 1)

- **Headline:** add-`jump` test exact-match accuracy. PASS = replay+consolidation lifts it
  CI-disjointly above replay-only (and above consolidation-only) — multi-seed (≥8),
  bootstrap CIs. The interaction framing of CONTEXT-B §8 applies (RC beats both parents).
- **Controls (mandatory):**
  1. random/simple split — sanity ceiling (~100%) for every arm.
  2. plain-train — the documented failure (the floor / the gap).
  3. replay-only, consolidation-only, replay+consolidation — the 2×2.
  4. **Redundancy guards (the §6/§ retrospective-§4 lesson):** a GECA-style data-augmentation
     arm and/or a known equivariance/meta-seq2seq reference. If our consolidation merely
     matches a known SCAN fix, **bank as REDUNDANT** (the 133/139 outcome) — novelty = a
     measured delta or a mechanism that is local/emergent where the known fix is engineered.
- **Anti-homunculus:** the consolidation is a fixed local loss over replayed encodings — no
  module that reads "is this jump?" and routes. Replay is content-blind (or surprise-weighted
  by a smooth local signal, Report 128). No `if verb-held-out then align`.

---

## 4. Build checklist (`experiments/87_betb_scan_gap.py` proposed for Stage 0)

- [x] SCAN data fetched + structure verified (`data/scan/`, §1).
- [ ] Loader: vocab build, `IN:/OUT:` parse, padded batching, exact-match-sequence eval.
- [ ] Small GRU seq2seq (enc-dec + attention; greedy decode for eval).
- [ ] Train on add-`jump` train; eval random + jump splits; multi-seed.
- [ ] **Stage-0 gate:** random ≈100% ∧ jump ≤~5% → reproduces the documented gap.
- [ ] Report under `reports/` (the project's done-gates), STATUS update.

---

## 5. Open questions before freeze

- **Model size / config to reliably reproduce the failure** at small scale (Lake & Baroni
  used a 1–2 layer LSTM, 200 units, dropout). Attention sometimes helps the model partially
  cheat the jump split — pick a config that *clearly* fails (the gap must be unambiguous).
- **Compute:** SCAN add-jump train ≈14.7k pairs; a small GRU is MPS-trainable in minutes/seeds.
  Stage 1 with multiple arms × ≥8 seeds may want the parallel-shard pattern (cf. exp85) or GPU.
- **Consolidation recipe (Stage 1):** the load-bearing design problem — a fixed local
  restructuring objective that aligns verb-slot encodings WITHOUT a supervisor and WITHOUT
  collapsing to data augmentation. Deferred to Stage-1 precommit; anti-homunculus review first.
- **COGS as the scale-up:** richer systematic-generalization splits; hold in reserve once
  SCAN gives a clean Stage-0 gap + a Stage-1 signal (or null).
