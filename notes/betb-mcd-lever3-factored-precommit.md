# Precommit — Lever 3: a clause-grain FACTORED SUBSTRATE on the GECA-resistant MCD arena

*Bet B. Written 2026-06-21. The program decision is RE-OPENED (Report 149). The user chose
"Lever 3" from [RETROSPECTIVE-addendum-2026-06-17 §4 item 3](RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md):
carry Report 142's substrate-shape lever to MCD. This note is the precommit + anti-homunculus
charter; it is cited by the experiment file and the eventual report.*

> **Anti-homunculus review: PASS-WITH-FIXES (2026-06-21, `anti-homunculus-reviewer`).** The
> mechanism is the right shape (fixed problem-generic channel scaffold + discovered content
> partition + fixed local consolidation loss + 3 NULL controls); no metric-reading supervisor, no
> test/label read at train time. The reviewer independently re-derived the biconditional predicate
> → `{jump,walk,run,look,left,right}` on all 3 splits with a 0.54+ Jaccard margin over every
> structural token (robust, not knife-edge), and confirmed it does NOT leak the MCD answer (which
> lives on the clause-combination axis the predicate never touches). The 4 required fixes are folded
> in below: **(1)** the biconditional is computed + asserted IN-CODE (CONDITION 1), never hardcoded;
> **(2)** `random_partition` is a CO-PRIMARY arm at n=8 + CI (§2/§4); **(3)** `verb_ids`/`struct_ids`
> are derived from the discovered predicate, not exp89's 4-verb literal (which mis-routes left/right);
> **(4)** the premise-gate floor + the 2×2 interaction test are pinned as numbers (§5/§6).

## Experiment preamble (CLAUDE.md, mandatory)

- **Active capability:** Bet-B Stage-1 mechanism on the GECA-resistant MCD arena (146) — a
  COMBINATION of *Consolidation-write* × *substrate-shape*. Carries Report 142's factored
  substrate (the strongest remaining pro-thesis datum, never tested on MCD).
- **Headline metric per [notes/betb-geca-resistant-regime-precommit.md:91-94](betb-geca-resistant-regime-precommit.md)
  + [notes/betb-mcd-stage1-mechanism-precommit.md:16-19,:102](betb-mcd-stage1-mechanism-precommit.md):**
  held-out MCD exact-match (greedy), per split (mcd1 primary, mcd2/mcd3 for generality);
  **PASS = `consolidation − factored_baseline` CI-disjoint > 0 AND `consolidation ≥ faithful-GECA`,
  ≥8 seeds, bootstrap CI.** Plus the **2×2 interaction**: consolidation must be inert on the
  holistic substrate (re-confirms 147) AND load-bearing on the factored substrate.
- **Required controls per [betb-mcd-stage1-mechanism-precommit.md:103-111](betb-mcd-stage1-mechanism-precommit.md)
  + [betb-geca-resistant-regime-precommit.md:95-96](betb-geca-resistant-regime-precommit.md):**
  `vanilla_plain` (in-house floor ~0.17/0.14/0.02, [reports/146:39](../reports/146_betb_scan_mcd_stage0/report.md));
  `factored_baseline` (λ/β=0, byte-identical matched-capacity); `vanilla_geca` (the redundancy
  decider); `*_random` split (must NULL); `*_nosplit` (must NOT match the mechanism);
  **`random_partition`** (route a random size-matched token set through the fill head — must NULL,
  proves the factorization is structural not generic capacity). B-KILL discipline per
  [CONTEXT-B.md:60-65](../CONTEXT-B.md).
- **The bar ([reports/146:42-44](../reports/146_betb_scan_mcd_stage0/report.md)):** beat the
  **general-purpose neural floor** (vanilla/GECA/MAML/T5 all <50% mean MCD). The ~99–100%
  AuxSeq/LeAR ceiling is **structure-injecting — NOT the target**; suspect leakage if approached.
- **Last verified result:** [Report 149](../reports/149_betb_scan_mcd_lever1_verification/report.md)
  (composition-as-inference TIES holistic; progress-allocation HURTS; structure-injection wins).
  [Report 147](../reports/147_betb_scan_mcd_ccc/report.md) (CCC on the HOLISTIC substrate NULLS:
  ccc−baseline −0.029 CI-disjoint negative; ccc ≈ ccc_nosplit). [Report 142](../reports/142_betb_scan_factored/report.md)
  (the SAME-shape consolidation went inert→load-bearing +0.079 5/5 when the substrate was factored,
  on SCAN add-jump).
- **Why now:** the user-chosen lever 3, on a CORRECT foundation. The literal 142 port is dead
  (see §1); this is its faithful reframe. Most-likely a high-value interpretable null that retires
  142 as the missing lever; small chance it flips 147 = first thesis-alive signal on the
  discriminating arena.

> **STATUS.md banner-drift flag (surface, do not paper over):** STATUS.md's headline field still
> reads the older FTSR continual-transfer metric ([STATUS.md:17](../STATUS.md), [CONTEXT-B.md:275-283](../CONTEXT-B.md)).
> The SCAN-MCD arc is a concrete instantiation of that target, but its operative headline is the
> per-split MCD exact-match defined in the two precommit notes cited above. This note uses the
> MCD-specific spec, per the CLAUDE.md rule that the design spec (not the STATUS banner) is the
> load-bearing source of truth.

## 1. Why the LITERAL 142 port is dead (data, this session)

`experiments/analyze_mcd_divergence_axis.py` (run 2026-06-21) measured the MCD divergence axis:

| axis | mcd1 | mcd2 | mcd3 |
|---|---|---|---|
| whole-command template novel | 100% | 100% | 99.9% |
| test cmds with ≥1 **primitive-filler-novel** clause | 12.2% | **0.0%** | 11.6% |
| per-clause filler-novel | 6.2% | **0.0%** | 5.8% |
| test clause-TEMPLATES seen in train | 6.2% | 13.6% | 6.3% |

**MCD's primitive-filler axis is saturated** (mcd2 = literally zero filler-holes). 142's substrate
factors the *filler* axis and its consolidation aligns the 4 primitive verbs' role-halves — on MCD
that has almost nothing to grab; a literal port would null for a **target-mismatch** reason, not a
real test. Worse, 142's peer predicate (`len(cmd)==1` standalone primitives) yields the **empty
set** on mcd1/mcd3 (no standalone single-token commands) — the literal mechanism cannot even define
its peer set here. So the literal port is doubly dead.

## 2. The faithful reframe — factored SUBSTRATE × MCD-matched CONSOLIDATION (the 2×2)

Carry 142's **substrate shape** (structure ⊥ content, encoder on the structure channel only,
content read by a separate head) but match the **consolidation** to MCD's actual axis (clause
COMPOSITION, the CCC operation that nulled on the holistic substrate, 147). The decisive design is
the **2×2 the retrospective asks for**:

|                | − consolidation            | + CCC consolidation                  |
|----------------|----------------------------|--------------------------------------|
| **holistic**   | `vanilla_plain` (~0.17)    | `ccc` (147: NULLS, −0.029)           |
| **factored**   | `factored_baseline` (new)  | `factored_ccc` (new — THE arm)       |

- **Headline:** `factored_ccc − factored_baseline` CI-disjoint > 0 (the 142-shape lift) **AND** the
  interaction (consolidation load-bearing on factored, inert on holistic) **AND** `factored_ccc ≥
  vanilla_geca`.
- **Thesis-alive signature (pinned, FIX 4):** `factored_ccc − factored_baseline` CI95 **lower
  bound > 0** (factored consolidation lifts) **AND** the holistic `ccc − vanilla_plain` CI95
  **includes or sits below 0** (re-confirms 147's inert/hurt) **AND** `factored_ccc ≥ vanilla_geca`
  **AND** all three NULL controls clean (§4). Anything else → null/down-rank per the ladder (§6).
- **CO-PRIMARY ARMS (FIX 2):** `random_partition`, `factored_ccc_random`, `factored_ccc_nosplit`
  run at the **same n=8 + bootstrap-CI treatment** as `factored_ccc` in the primary table — NOT as
  positive-only follow-ups. Given the 126/127/148→149 false-positive history these deciders are
  load-bearing, not optional.

## 3. The mechanism (concrete)

**Substrate `FactoredSeq2Seq` (ported from [experiments/89](../experiments/89_betb_scan_factored.py)):**
- Input embedding `e_t = [role_t (R=48) ; fill_t (F=16)]` (total 64 = the holistic embed, for
  capacity-fairness). PAD/SOS/EOS shared.
- **Encoder GRU runs on the ROLE channel only** → `enc_out_role (B,S,H)`, `h (1,B,H)`. The FILL
  channel is carried per-position (`fill_mem (B,S,F)`), never seen by the encoder.
- Decoder: attention over `enc_out_role` → `ctx_role`; the SAME attention weights over `fill_mem` →
  `ctx_fill`. **Structure head** `base: Linear(H → n_struct+1)` from `combine(d, ctx_role)` emits
  the non-content tokens + ONE "emit-a-content-action-here" slot logit (role-driven, content-blind).
  **Content head** `fillsel: Linear(F → n_content)` from `ctx_fill` selects WHICH action. Final
  content-token logits = slot-logit + fillsel. So **role decides the action skeleton / when-to-emit /
  EOS; fill decides which action fills each slot** — structure ⊥ content, architecturally.
- **The content/filler partition is DISCOVERED problem-generically** (§4), not hardcoded.

**Consolidation `consolidate_ccc_factored` (ported from [experiments/94](../experiments/94_betb_scan_mcd_ccc.py)):**
offline pass, decoder-frozen, encoder(role-emb + GRU) + W_* unfrozen, minimizing a fixed local loss
over replayed train batches:
`L = CE_whole + β·‖h − e_comp‖² + γ·CE(decode_from(e_comp), gold) + δ·hinge(e_comp apart)`,
where `e_comp = W_conj([pool(enc_out_role[L]) ; pool(enc_out_role[R])])` composes the **role**
(structure) clause-pools at the single top-level conjunction (`and`/`after`; single-clause →
`W_unary·pool(whole)`). β=γ=δ=0 ⇒ byte-identical to `factored_baseline`. This restructures the
STRUCTURE channel to be a learned composition of clause structures (content preserved in the fill
channel) — the 142 logic at the clause grain, matched to MCD's clause-combination divergence.

## 4. Problem-generic legality (anti-homunculus / CONTEXT.md §3)

The binding boundary, [CONTEXT.md:81-89](../CONTEXT.md): a *fixed, problem-generic* scaffold is a
LEGAL evolution-style scaffold; a scaffold *hand-shaped to the target answer* is a **design-time
homunculus**. For Bet B, the no-backprop clause is lifted ([CONTEXT-B.md:58](../CONTEXT-B.md)); what
binds is **no metric-reading supervisor** + **the problem-generic boundary**.

- **The role/fill split is by DIMENSION, not by token** — every token is embedded into both
  channels; the encoder structurally sees only role. A fixed 2-channel scaffold, identical across
  problems. LEGAL.
- **The content/filler partition is DISCOVERED in-code by a biconditional input↔output
  co-occurrence predicate** (verified this session): a content token is an input token `t` with an
  output token `o` such that `{i : o∈out_i} == {i : t∈cmd_i}` (perfect biconditional). On MCD this
  yields `{jump↔I_JUMP, walk↔I_WALK, run↔I_RUN, look↔I_LOOK, left↔I_TURN_LEFT, right↔I_TURN_RIGHT}`
  identically on all 3 splits — the deterministic-identity tokens, **discovered, not hand-shaped**,
  and it would yield the analogous set on any SCAN-family corpus. Asserted in code (CONDITION 1).
- **The clause split is a fixed content-blind string function** (first top-level `and`/`after`), the
  textbook compositional prior ([betb-mcd-stage1-precommit.md:92](betb-mcd-stage1-mechanism-precommit.md)),
  identical for train and test, reading neither the held-out set nor the labels nor any metric.
- **The mandatory NULL controls** (the design-time-homunculus / capacity deciders):
  1. **`random_partition`** — route a random size-matched token set through the fill head. MUST NULL.
     (Proves the *discovered* partition, not generic extra capacity, is what matters.)
  2. **`factored_ccc_random`** — consolidation with a random interior clause split. MUST NULL.
  3. **`factored_ccc_nosplit`** — consolidation with `W_unary·pool(whole)`, no split. MUST NOT match
     `factored_ccc` (the 147 `ccc≈ccc_nosplit` separator that falsified CCC).
- **Cleanliness gate** ([betb-mcd-stage1-precommit.md:114](betb-mcd-stage1-mechanism-precommit.md)):
  the win must survive the **minimal conjunction-only parser**; if only a deeper grammar-aware parser
  lifts, disclose injected grammar as load-bearing (slides toward structure-injection).

**Anti-homunculus 4-question check:** (a) what moves locally = the role-channel encoder manifold
under a fixed offline composition loss; (b) the apparent "decision" lives in the tension of the
channel split + the alignment/composition loss settling to a fixed point, not a routed choice;
(c) the clause split + filler discovery are fixed content-blind functions, no runtime metric read,
no per-example arbitration; (d) the three NULL controls above falsify "hidden arbitration / generic
capacity." → to be confirmed by the `anti-homunculus-reviewer` agent before code lands.

## 5. The PREMISE GATE (cheap, FIRST — decides whether the full run is worth it)

Build the factored substrate and run a 1–2 seed smoke BEFORE the n=8 mechanism run. **Pinned
thresholds (FIX 4 — fixed in advance, may NOT be retroactively relaxed):**
1. **In-distribution competence:** factored-substrate **TRAIN exact-match ≥ 0.90** (a competent
   MCD learner; the holistic seq2seq overfits train, 142 reached 1.0). Below 0.90 ⇒ the
   factorization breaks learning ⇒ premise-gate FAIL.
2. **Not degenerate on test:** `factored_baseline` **test EM ≥ 0.10** (≈ the vanilla floor band
   0.17/0.14; the headline is vs the matched `factored_baseline`, so a somewhat-lower-but-nonzero
   baseline is still interpretable). **`< 0.10` ⇒ premise-gate FAIL** ⇒ bank the premise null
   cheaply, do not run the mechanism.
3. **Clause-grain ceiling probe (the 142 "premise 0.26" analog, reported not gated):** the
   decompose-decode-concat oracle on the factored substrate (does it beat the holistic GRU's 0.0,
   [exp96](../experiments/96_betb_scan_mcd_oracle_ceiling.py))? A leaky factorization caps the
   headline; report the cap.

**GATE:** proceed to the full 2×2 + controls (n=8, mcd1+mcd2) only if (1) TRAIN-EM ≥ 0.90 AND
(2) `factored_baseline` test-EM ≥ 0.10 (1–2 smoke seeds). Report the gate numbers back to the user
before the expensive n=8 run.

## 6. Pre-registered verdict ladder (honest prior: LOW — see §7)

1. **Null with a competent substrate** (most likely) → "mechanism-absence" earned at the clause
   grain; retires 142 as the missing lever; the 147 null is no longer the only datum. HIGH-VALUE.
2. **Positive but `factored_ccc_random` or `factored_ccc_nosplit` also lifts** → capacity/tightening,
   not structure (the 147 failure mode at a new grain). Down-rank to non-mechanism.
3. **Positive, NULL controls clean, but `random_partition` lifts** → generic extra capacity. Down-rank.
4. **Positive, all controls clean, minimal parser suffices, ≥ GECA and the floor** → the genuine
   thesis-alive outcome (low prior). Verify with fresh seeds before any claim (the 148→149 lesson).
5. **Premise-gate fail** (factored substrate crippled) → bank cheaply, report the premise null.

## 7. Honest prior & the structure-injection bind (do not over-claim)

The literature shows **no emergent/replay/consolidation mechanism on any MCD/CFQ/COGS leaderboard**;
every winner injects structure (LeAR Tree-LSTM 90.9, ablate-tree→30.4; AuxSeq aux-supervision; NeSS
neuro-symbolic). A clause-grain factored substrate **leans structure-injecting** — so even a clean
positive is a *partial* structure-injection result, not a pure emergent win, and the thin train↔test
clause-template overlap (~6%) caps reachability. The high-value, most-likely outcome is the
**interpretable null** that converts 142 from "untested on MCD" to "tested, didn't transfer," which
is exactly what "test the thesis where it matters before banking the wall" requires. A
controls-killed positive is the more likely "positive."
