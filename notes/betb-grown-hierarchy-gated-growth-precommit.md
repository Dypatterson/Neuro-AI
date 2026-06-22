# PRE-COMMIT — Bet B / §4½: can a compositional hierarchy be GROWN (stabilization-gated) rather than INJECTED?

> ## ⚠ R2 REVISIONS (2026-06-22, post-verification — supersede the body below)
> A 5-agent re-verification + repo audit ([verification-round-2-and-seams.md](../brainstorm-workspace/2026-06-22-grown-hierarchy-mechanisms/verification-round-2-and-seams.md))
> changed this precommit materially. **Do NOT build the body as written.** The binding revisions:
> 1. **DEMOTED.** Two cheaper, higher-value moves come first: (a) **re-ground the bar on ReCOGS/SLOG and
>    treat MCD as covariate-shift** — COGS-structural's 0–12% floor is partly a decoder length/format
>    *artifact* (ReCOGS, TACL 2023), so a splitting result on raw MCD/COGS would be partly uninterpretable;
>    (b) **test the entropy soft-prior (Wold 2025)** on the cleaned bar — the cheapest test of "grown beats
>    injected without a tree."
> 2. **The headline mechanism is on shaky ground.** Splitting/Firefly optimize *fit*, not systematicity;
>    no growing-net work has ever cracked a compositional split; the parsimonious read of any gain is
>    "growth aided optimization," not "a hierarchy was grown." → **Headline metric changes to the
>    systematic-minus-iid GAP**, gated on three discriminators: a **tree-projection-score jump locked to
>    the split event** (Murty et al. ICLR 2023), **grow-then-freeze + ablate-the-grown-level** (selective
>    systematic drop), and a **capacity-matched + split-timing control** (plateau vs random-step vs
>    born-wide). Without these, a positive is indistinguishable from capacity/optimization.
> 3. **Gate signal: DROP the grokking progress-measure** (needs a known circuit MCD lacks). Use
>    **descent-stationarity (primary) + Local Learning Coefficient (confirmatory)**, HTSR-α / weight
>    intrinsic-dimension as the label-free fallback.
> 4. **Pretrain-atoms arm is largely Report 149 `exp96` relabeled** (`--clause-aug` already gave
>    0.158→0.297). Reframe as: *does gated-growth on top of the 0.297 atomic floor beat 0.297?* (C-vs-B).
> 5. **Buildability:** the harness is a single-layer **GRU** enc/dec — neuron-splitting on a GRU is a
>    substantial build, not a drop-in. Use the **Firefly gradient-grown variant** or **depth-stacking**, or
>    a small Transformer backbone. Note Reports 122/123 (Bet-A growth NULL) and 131 (capacity NULL-MONOTONE).
> 6. **Stronger alternative to consider first:** the neuro re-check SOFTENED "stabilizer-not-manufacturer"
>    — the brain *does* manufacture-by-overlap-recombination → **multi-scale successor-representation
>    replay** is a better-grounded grown mechanism, buildable on the replay substrate (caveat: Bet-A
>    banked SR-nulls, Reports 124/125 — the multi-scale/offline form is the unexercised part).

*Status: DRAFT, SUPERSEDED BY THE R2 BLOCK ABOVE — for user review + anti-homunculus check (2026-06-22). The graduation attempt
for the **★ §4½ reopen-door** ([RETROSPECTIVE-program-close §5](RETROSPECTIVE-program-close-2026-06-21.md):174).
Design source: [findings.md §2/§4](../brainstorm-workspace/2026-06-22-grown-hierarchy-mechanisms/findings.md)
+ [verification-and-extensions.md §4](../brainstorm-workspace/2026-06-22-grown-hierarchy-mechanisms/verification-and-extensions.md)
(4-agent-verified, headline survived ~90%). Charter: [CONTEXT-B.md](../CONTEXT-B.md). Arena: the
GECA-resistant MCD ([Report 146](../reports/146_betb_scan_mcd_stage0/report.md)).*

---

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** the §4½ corrected-premise experiment — *can a compositional hierarchy be
> **GROWN** (emergent, stabilization-gated) rather than **INJECTED** (Tree-LSTM)?*
> ([retrospective §5:174](RETROSPECTIVE-program-close-2026-06-21.md)). The program validated the
> **stabilizer** (136/138/139) and a **flat abstractor** (133) but never the grown hierarchy.
> **Headline per [verification-and-extensions.md §4](../brainstorm-workspace/2026-06-22-grown-hierarchy-mechanisms/verification-and-extensions.md)
> + [§3 below](#3-the-experiment--headline):** **MCD test exact-match of the stabilization-GATED-growth
> arm, lifted CI-disjointly above BOTH the flat baseline AND the SCHEDULED-growth control** (growth
> helps AND the *measured stability gate* is what makes it help), on an arena where the known fix
> (GECA) provably fails (Report 146). Broader thesis target = continual compounding-transfer FTSR
> ([CONTEXT-B §8:275-282](../CONTEXT-B.md)); this experiment tests its missing structural foundation,
> not FTSR directly.
> **Controls per [§4](#4-controls--guards):** flat baseline; scheduled-growth (gate-redundancy guard);
> GECA (known-fix redundancy, already in harness as `vanilla_geca`); random-grow ablation; the
> pretrained-atom-seed arms; injected-hierarchy ceiling.
> **Last verified result:** [Report 150](../reports/150_betb_scan_mcd_factored_lever3/report.md)
> (factored substrate × clause-composition consolidation NULL on MCD — retires 142 as the lever);
> the grown-hierarchy door was left explicitly open.
> **Why now:** it is the one **empty cell** in the grown × tested-on-hard-splits matrix — every
> hard-split clearer injects structure; every pure-emergent hierarchy is untested or fails — and the
> only buildable, anti-homunculus-clean **measured** stabilize→grow gate (neuron-splitting at
> stationarity) has never been run on compositional generalization.

---

## 1. The failure mechanism (diagnosed) → what a grown hierarchy must do

Two banked facts frame the target. (a) On MCD the hardness is **clause composition**, not the
verb-filler axis (Report 147 data-verification); GECA reaches ≈0% of MCD test (Report 146) — so a win
here is **not** the §4 task-selection confound. (b) **Emergent/learned depth has so far failed MCD**:
a vanilla deep seq2seq nulls, and the Report 132 depth-probe found depth does not compose
(L=1 ≥ L=2 ≥ L=4). So the open question is sharp: *not* "does more depth help" (it doesn't), but
**"does depth GROWN level-by-level under a stabilization gate behave differently from depth trained
jointly/from-scratch?"** — the §4½ claim that *stabilizing level N enables an abstractor to build a
composing level N+1*, which a flat or jointly-trained net never gets to do.

**What the grown-hierarchy mechanism must manufacture:** a composing layer N+1 that forms **only after
level N has stabilized**, where "stabilized" is a *measured local quantity* (descent plateau / progress
measure), not a schedule and not a supervisor's accuracy read — and that this gated, level-by-level
construction lifts MCD exact-match where flat and scheduled-growth do not.

---

## 2. Leading recipe — stabilization-GATED neuron growth

An in-learner growth loop over the existing MCD seq2seq (exp94 backbone), four interlocking parts:

1. **The gate signal (when to grow) — MEASURED, uniform.** Grow capacity when the current network's
   optimization has *stabilized*: parameter-descent stationarity (the Splitting-Steepest-Descent
   trigger, Liu 2019) and/or an information-theoretic **progress measure** that provably rises before a
   generalizing circuit crystallizes (grokking, Nanda 2023 — promoted in verification-and-extensions §2).
   The gate is a single thresholded scalar read of a *uniform* quantity over the whole net — never a
   per-task or per-token branch.
2. **The growth operator (how to grow) — local geometry.** Neuron splitting / Firefly architecture
   descent (Wu 2020 — grows **width AND depth**): the split direction is the functional-steepest-descent
   eigenvector of the per-neuron splitting matrix (escape the saddle the plateau sits on). Firefly's own
   caveat is priced in: splitting-*alone* can stall, so the operator also admits **fresh neurons by
   gradient** (the fence-clean path, see anti-homunculus below).
3. **The compose operator (optional) — substrate-native.** When a new level is added, levels combine by
   **energy summation** (Du/Liu product-of-experts; a Modern Hopfield net *is* an energy model, so this
   is ~1 line on the FHRR/Hopfield substrate). Off by default; on as a drill-down arm.
4. **The level-0 seed (optional) — pretrained atoms.** Pre-train ONLY the primitive lexicon / atomic
   mappings (never held-out compounds), giving the growth loop a stabilized non-degenerate level-0 to
   split from (the predicted cure for the Firefly cold-start stall). This is the user's
   pretrain-to-initialize idea, scoped to inject **representations (a), not the composition rule (b)**.

**Anti-homunculus check (written out, per CLAUDE.md):**
- *Who decides "level N has stabilized → grow N+1"?* No one — the growth fires when the **measured gate
  scalar** crosses a fixed threshold, applied uniformly. It is NOT a supervisor reading val-accuracy and
  flipping a switch. **The pretraining→growth handoff uses the SAME gate** (level-0 simply reaches the
  plateau in a stabilized state); there is no "pretraining finished" flag — if the design needs one, it
  has smuggled in a homunculus and is the wrong shape.
- *Who decides WHERE/what to grow?* The functional-steepest-descent criterion (local second-order
  geometry) or the gradient of a fresh neuron — a local dynamic, no `if/then`, no metric arbitration.
- *Fence honesty (CONTEXT-B one-shot-SVD/eig flashlight):* the splitting criterion is a **small, local,
  per-neuron eig that fires REPEATEDLY as the gate triggers** — not a one-shot closed-form SVD/eig of a
  materialized global co-occurrence operator. Argued within bounds, but **flagged**: the fence-clean
  fallback (Firefly gradient-grown fresh neurons, no closed-form eig) must reproduce any positive, or
  the positive is charged to the fenced shortcut.
- *Open worry to surface (not hide):* the honest prior is LOW — every disconfirmer says pure-emergent
  grown hierarchy fails the hard splits. The experiment is built to make a clean null cheap and
  bank-hardening (§3 disposition), not to manufacture a positive.

---

## 3. The experiment & headline

- **Arms** (≥8 seeds, `mcd1` primary; data `data/scan/mcd_split/`):
  - **A. flat baseline** — exp94 vanilla, no growth (the MCD floor).
  - **D. gated-growth (from scratch)** — §2 parts 1–2; **the headline grown arm**.
  - **S. scheduled-growth** — identical growth operator on a FIXED clock (gate disabled) — the
    **gate-redundancy control**.
  - **B. pretrained-atoms → flat** — atom-floor alone.
  - **C. pretrained-atoms → gated-growth** — seed × growth.
  - **Ceiling: injected hierarchy** — Tree-LSTM / LeAR-style (gap marker, ~90% CFQ / ~97% COGS).
- **Headline:** MCD test exact-match, **`D − A` with `D` also CI-disjoint above `S`**, bootstrap CI,
  ≥8 seeds. **PASS = (D − A) CI-lo > 0 AND (D − S) CI-lo > 0** — growth helps AND the measured gate is
  load-bearing — with a non-trivial absolute lift well above the flat floor.
- **Drill-downs (explain the headline, not compete with it):** `C − D` (is the atom-seed load-bearing,
  or redundant with what growth discovers — the 149b relabeling risk); gap to the injected ceiling;
  gate-signal variant (stationarity vs grokking-progress-measure); energy-compose on/off; growth trace
  (did depth actually grow, and after level-0 stabilized?).
- **Disposition (decisive either way):**
  - **D > A and D > S** on GECA-resistant MCD → **first evidence a hierarchy can be GROWN not injected**,
    with the stabilization gate load-bearing. *Then* harden (COGS-structural 2nd arena, scale).
  - **D ≈ S** (growth helps, gate inert) → the *measured stabilization-gate* is not load-bearing;
    growth-per-se / capacity is the lever. Demote the gating mechanism; bank as a scoped partial.
  - **D ≈ A / all arms ≪ ceiling** → grown hierarchy stalls on the discriminating arena; only injected
    clears the bar → **clean earned negative that closes the grown door and HARDENS the bank** (the
    honest expected outcome; collapses §4½ back into the hierarchy-injection bind, as that note warned).
  - **C > D** → pretrain-atoms is a required level-0 foundation (vindicates init-as-level-0; Firefly
    cold-start is the mechanism). **C ≈ D** → seed redundant; fold in for efficiency, claim no reopen.

---

## 4. Controls & guards

1. **flat baseline (A)** — the MCD floor, same seeds.
2. **scheduled-growth (S)** — the **gate-redundancy guard**: same growth operator, fixed clock, gate
   off. If D ≈ S, the measured stability gate adds nothing (only growth-per-se does).
3. **GECA (`vanilla_geca`, already in harness)** — known-fix redundancy: on MCD this is ≈0% (Report
   146), so a grown win is automatically non-redundant-with-GECA; kept as the published bar.
4. **random-grow ablation** — fire growth at random steps/directions (not gate-triggered, not
   steepest-descent) → must NOT match D, proving the gate+criterion are load-bearing, not generic
   capacity/regularization.
5. **fence-clean reproduction** — re-run any D-positive with gradient-grown fresh neurons (no
   closed-form eig); the positive must survive or it is charged to the fenced shortcut.
6. **leakage guard (atom-seed arms B/C)** — audit **compound-disjointness mechanically**: MCD shares
   atoms by construction (fine), but assert **zero test *compound* (even as a subtree)** appears in the
   pretraining set; plus a **shuffled-atom control** that must NOT help (else the "help" is leakage/memorization).
7. **mcd2 caveat** — mcd2's substrate was unstable (2/8 seeds collapsed, Report 150); `mcd1` is primary,
   mcd2 reported only with the premise-gate + paired controls.

**Anti-homunculus:** the gate is a uniform measured scalar; growth is local geometry; replay/pretraining
is content-blind. No supervisor, no task-identity read. (Pending the `anti-homunculus-reviewer` agent —
run BEFORE build.)

---

## 5. Build plan (`experiments/100_betb_grown_hierarchy_gated_growth.py` → Report 151)

**Anti-homunculus review: PENDING** — do not build until PASS. Anticipated binding honesty guards
(verdict FLIPS to FAIL if dropped):

> **BUILD CONDITION 1:** the stabilize→grow gate is a **measured scalar crossing a fixed threshold**,
> computed in code from training dynamics — NEVER a hand-placed "now grow" step and NEVER a val-accuracy
> read. The pretraining→growth handoff uses the **byte-identical** gate.
> **BUILD CONDITION 2:** the scheduled-growth control (S) reuses the **identical growth operator** with
> only the gate replaced by a clock — no operator retuning — so D−S isolates the gate alone.
> **BUILD CONDITION 3:** the atom-seed pretraining set is **generated in code and mechanically asserted
> compound-disjoint** from the MCD test (CONDITION-1-style audit), with a shuffled-atom control.

- [ ] Extend the exp94 Seq2Seq backbone with a growth loop (split/Firefly + measured gate).
- [ ] Gate module: stationarity + grokking-progress-measure, unit-tested as uniform scalars.
- [ ] Arms A/D/S/B/C + random-grow + GECA; injected-hierarchy ceiling (Tree-LSTM or cite LeAR).
- [ ] ≥8 seeds, bootstrap CI; headline = MCD exact-match `D−A` gated on `D−S`; drill-downs per §3.
- [ ] Leakage audit + shuffled-atom control (CONDITION 3); fence-clean reproduction (§4.5).
- [ ] Report under `reports/151_...`; STATUS walk-back FIRST, then update.

## 6. Open questions before freeze

- **Gate threshold calibration** — stationarity tolerance / progress-measure crossing point: pre-register
  on a held-out *random* split so the threshold is not MCD-tuned (a tuning-on-test homunculus).
- **Which gate signal is primary** — descent-stationarity (simplest, Firefly-native) vs the grokking
  progress-measure (sharper but heavier). Start with stationarity; progress-measure as the drill-down.
- **Injected ceiling fidelity** — build a small Tree-LSTM on MCD, or cite LeAR's published 90.9% CFQ /
  97.7% COGS as the gap marker (cheaper; fine for a gap, not for a paired control).
- **Compute** — 5 grown arms × ≥8 seeds, growth is heavier than vanilla; estimate then shard per-seed
  on MPS as in exp99 (60-shard precedent). Energy-compose and mcd2 are optional second-pass.
- **COGS-structural as the 2nd arena** — only after an mcd1 positive; pre-register it as the hardening
  step, not a fishing expansion.
