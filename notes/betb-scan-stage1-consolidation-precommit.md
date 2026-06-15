# PRE-COMMIT — Bet B / SCAN Stage 1: does a restructuring consolidation manufacture composition?

*Status: DRAFT for user review + anti-homunculus check (2026-06-15). Gated on
[Report 140](../reports/140_betb_scan_gap/report.md) (Stage 0 PASS — the gap is real).
Charter: [betb-scan-discriminating-regime-precommit.md](betb-scan-discriminating-regime-precommit.md)
+ [CONTEXT-B.md §8](../CONTEXT-B.md). This is the GRADUATION attempt for the Bet-B thesis.*

---

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** Bet-B continual compounding-transfer in the discriminating
> regime — the thesis test: *does a brain-shaped restructuring consolidation manufacture
> structure a simple method cannot?* ([RETROSPECTIVE §6](RETROSPECTIVE-two-bets-2026-06-06.md)).
> **Headline per [§3](#3-the-experiment--headline):** **add-`jump` test exact-match,
> lifted CI-disjointly above the no-consolidation baseline** (which Stage 0 fixed at
> ≈0.003). **Controls per [§4](#4-controls--the-redundancy-guard):** baseline (Stage-0
> floor); the consolidation arm; **redundancy guard = GECA-style augmentation**;
> ablations (random-align, align-wrong-tokens). **Last verified:** Report 140 (gap real,
> headroom ≈0.995). **Why now:** Stage 0 licensed it — the simple method demonstrably
> fails, so a positive here is NOT the §4 task-selection confound.

**Data-space recipe note (CONTEXT-B §8 Terminology):** SCAN add-`jump` is a single static
train set; "replay" (re-presentation) is already in plain training. So the 2×2 collapses to
**RC vs R = baseline vs baseline+consolidation** (Report 138's Δ_structure form) — the
load-bearing delta. The consolidation must be a genuine RESTRUCTURING op (not re-presentation).

---

## 1. The failure mechanism (diagnosed) → what consolidation must do

`jump` appears in training ONLY as the isolated primitive (`jump → I_JUMP`); `walk/run/look`
appear isolated AND in every template. So the net learns a **verb-slot pathway** (route a verb
through modifier machinery: twice/around/opposite/left/right) that is shaped by walk/run/look
but **never by `jump`** — `jump`'s encoder representation sits outside the verb-slot-filler
subspace. At test, `jump twice` → the modifier machinery mis-handles `jump`'s off-manifold
representation → ≈0% (Report 140).

**What a restructuring consolidation must manufacture:** pull `jump`'s representation INTO the
shared verb-slot-filler subspace — i.e. make `jump` **substitutable** for `walk/run/look`. The
only training signal that `jump` is a verb: it has **isomorphic standalone behavior** (a
one-token command producing one action), exactly like the other primitives. **This is the
Bet-A king~queen paradigmatic-substitutability problem** — here behaviorally testable, and
under Bet-B's lifted rules (backprop/global allowed; only one-shot closed-form SVD fenced).

---

## 2. Leading recipe — ROLE/FILLER alignment of distributional-peer primitives

An **offline consolidation pass** with a **fixed local objective** over replayed data:

1. **Identify the peer set STRUCTURALLY (not by supervision):** the tokens that appear as a
   *complete one-token command* in the replay buffer (`{jump, walk, run, look}` — `jump`
   included, because `jump → I_JUMP` IS in train). This is a structural read of the data, a
   fixed dynamic — NOT a metric-reading supervisor and NOT `if token==jump`.
2. **Restructure:** a learned linear split of each peer token's embedding into
   `role` ⊕ `filler`, trained offline by a fixed loss:
   - **filler keeps identity:** `filler(t)` must still decode the token's own standalone action
     (so `jump`≠`walk` is preserved — no collapse);
   - **role is shared/aligned:** minimize the variance of `role(t)` across the peer set (pull
     the verb-role together) — a decorrelation/alignment term, the local "keep-together".
   The encoder then reads the realigned embedding so `jump`'s role rides the verb-slot pathway.
3. **Anti-homunculus check (written out, per CLAUDE.md):**
   - *Who decides `jump` is a verb?* No one — the alignment loss is applied UNIFORMLY to the
     structurally-defined one-token-command set. `jump` is included by the same rule as `walk`.
   - *Where is the "decision"?* In the geometry — the alignment term pulls role-components
     together; `jump` follows because it is in the set. No `if/then`, no metric read, no
     per-token branch.
   - *Fence:* the alignment is an ITERATIVE local loss (gradient steps), NOT a one-shot
     closed-form SVD/eig of a materialized operator. Legal under Bet B.
   - **Open worry to surface (not hide):** the recipe is *motivated by* the known failure.
     Guard: the objective is GENERAL (align distributional peers, preserve identity), not
     jump-specific; it must also help a SECOND split (§4) or it is over-fit to this one.

*Swappable alternative (if recipe-1 nulls):* a slot/bottleneck autoencoder over encoder states
that factorizes role/filler unsupervised; same headline + same guards.

---

## 3. The experiment & headline

- **Arms:** `baseline` (Stage-0 config, no consolidation; floor ≈0.003) · `consolidation`
  (baseline + the §2 offline pass) · plus the §4 guards/ablations.
- **Headline:** add-`jump` test exact-match, `consolidation − baseline`, **≥8 seeds, bootstrap
  CI; PASS = CI-lo > 0** AND the absolute lift is non-trivial (e.g. ≥0.30, well above the
  0.003 floor). Drill-down: does it also hold the random-split ceiling (no regression)?
- **Disposition (decisive either way):**
  - CI-lo > 0 and ≫ GECA-floor-of-triviality → **the thesis gets its first real positive**: a
    restructuring consolidation manufactures compositional generalization the simple method
    can't, in a regime where the simple method provably fails. *Then* harden (2nd split, COGS).
  - Lift ≈ GECA / a known fix → **bank as REDUNDANT** (the 133/139 outcome) — real but not
    distinctive; novelty would need the local/emergent form to BEAT the engineered one.
  - Null → iterate the recipe (swappable, charter-binding); a clean null here is cheap.

---

## 4. Controls & the redundancy guard

1. **baseline** — Stage-0 floor (≈0.003), same seeds.
2. **GECA-style augmentation** (the redundancy guard) — the known SCAN fix (good-enough
   compositional augmentation: swap `jump` into template slots that other verbs occupy). If our
   consolidation merely matches this, it is REDUNDANT. Novelty = a measured delta OR a
   mechanism that is local/emergent where GECA is an engineered preprocessing step.
3. **random-align ablation** — align random tokens (not the peer set) → must NOT help (proves
   the peer-structure is load-bearing, not generic regularization).
4. **align-but-collapse ablation** — drop the filler-identity term → should HURT (verbs merge,
   `jump twice` might decode as `walk twice`); proves identity-preservation is necessary.
5. **second-split generalization** — re-run on a different held-out primitive / the
   `around-right` template split; the recipe must generalize, not be jump-tuned.

**Anti-homunculus:** the consolidation is a fixed local loss over a structurally-defined set;
replay is content-blind. No supervisor, no task-identity read. (Pending the
`anti-homunculus-reviewer` agent — run BEFORE build.)

---

## 5. Build plan (`experiments/88_betb_scan_consolidation.py`)

**Anti-homunculus review: PASS (2026-06-15, agent `a0c17e48621a908db`)** — cleared to build
as specified, conditional on TWO binding honesty guards (verdict FLIPS to FAIL if dropped):

> **BUILD CONDITION 1 (non-negotiable):** the peer set is **computed in code from the train
> buffer** via the one-token-command predicate and **asserted `== {jump,walk,run,look}`** as a
> test — NEVER written as a hardcoded literal the loss consumes. (Else the arbiter moves from
> runtime into the experimenter's setup = the subtler homunculus.)
> **BUILD CONDITION 2 (non-negotiable):** the §4.5 second-split control reuses the
> **byte-identical** predicate + loss, with **no jump-specific constant retuned**.

- [ ] Extend `exp87` Seq2Seq with a `role/filler` embedding split + the offline consolidation pass.
- [ ] **Peer set computed by the loader from train pairs (CONDITION 1); unit-test the predicate.**
- [ ] Arms: baseline / consolidation / GECA / random-align / collapse. ≥8 seeds, bootstrap CI.
- [ ] Headline = jump-split exact-match Δ; ceiling-no-regression drill-down; 2nd-split check (CONDITION 2).
- [ ] Report under `reports/`; STATUS update.

## 6. Open questions before freeze

- **Where to apply role/filler** — input embedding only, or also encoder output? Start at the
  embedding (cleanest, the diagnosed locus); escalate if null.
- **GECA implementation fidelity** — a faithful-enough GECA for the redundancy guard (swap
  fragments across matched environments); or cite its published ~80–95% on add-jump as the bar.
- **Compute** — 5 arms × ≥8 seeds × 30 epochs × ~4s/epoch MPS ≈ tens of minutes; shard if needed.
