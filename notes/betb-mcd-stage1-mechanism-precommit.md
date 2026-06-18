# PRE-COMMIT — Bet B / SCAN-MCD Stage-1 mechanism: Clause-Compositional Consolidation (CCC)

*Status: DRAFT for the mandatory anti-homunculus review (2026-06-17), then build. Gated on
[Report 146](../reports/146_betb_scan_mcd_stage0/report.md) (MCD Stage-0 PASSED → a GECA-resistant arena,
Stage-1 licensed). Mechanism selected by a 4-angle design panel (`wf_dff3f687`) + a decisive data
verification (below). Arena = SCAN-MCD (user choice, 2026-06-17); autonomy = full (design → anti-homunculus →
build → test).*

---

## 0. Preamble (CLAUDE.md — mandatory)

> **Active capability:** Bet-B continual compounding-transfer — the Stage-1 restructuring-consolidation
> mechanism on the GECA-resistant SCAN-MCD arena (the genuinely-novel "does consolidation manufacture
> compositional generalization a known method can't" test, now testable-not-redundant per Report 146).
> **Headline per [betb-geca-resistant-regime-precommit.md:90-94](betb-geca-resistant-regime-precommit.md):**
> held-out MCD exact-match, **consolidation − no-consolidation baseline CI-disjoint > 0 AND consolidation ≥
> the faithful-GECA arm**, ≥8 seeds, bootstrap CI; on ≥ mcd1.
> **Controls:** vanilla floor (in-house) · faithful GECA (the redundancy decider) · λ=0 architecture-only
> baseline (byte-identical) · **random-bracketing control (the anti-"just capacity" / anti-structure-injection
> decider) must NULL** · minimal-parser (conjunction-only) as the clean primary, deep-grammar parser flagged
> if load-bearing.
> **Last verified:** Report 146 (MCD Stage-0 PASS). **Why now:** STATUS — Stage-1 licensed; this is the crux.

---

## 1. Why THIS mechanism — the decisive data finding (re-selected the family)

The design panel's 4 angles **converged** on a *filler-invariance* family (make operators act the same
regardless of which verb fills them: SBOR, OSC). **A direct grep on `data/scan/mcd_split/tasks_*_mcd1.txt`
falsifies that family for MCD:**

- **Verb×modifier cells are SATURATED in train** — the only empty cells are `jump+around`, `jump+left`
  (add-jump-shaped, where Report 144 shows **GECA scores 0.984–0.9998**, not ~0%). The verb-filler axis is
  *not* MCD's hardness.
- **MCD's hardness is the CONFIGURATION/COMBINATION axis:** **100% of test examples have a novel
  V/D-abstracted template** (38/38 test templates unseen in train); **88% have a novel operator/conjunction
  skeleton**; both train and test are ~entirely **2-clause conjunctions** (`and`/`after`) — the divergence is
  the novel *combination* of per-clause operator patterns (e.g. `opposite thrice and opposite thrice`).

So filler-invariance targets the saturated axis → would null-or-be-redundant. **The right mechanism targets
clause/operator COMPOSITION** — compose in representation space so a novel *combination* of seen clauses
generalizes. (Reachability ceiling: ~70% of mcd1-test has every leaf verb-phrase train-seen — well above the
floor.) This is the SCSC angle, minimized for cleanliness.

## 2. The mechanism — Clause-Compositional Consolidation (CCC)

**Substrate:** the exp87 GRU encoder + Luong-attention decoder, unchanged for the forward/decode path.

**Fixed, content-blind parser P (the most GENERIC compositional prior):** split the command at its single
top-level conjunction token (`and` / `after`) into `(clause_L, conj, clause_R)`. Literal surface markers
only; identical for train and test; no output/metric/held-out reference. *(Deeper grammar — `twice/thrice`
unary, verb-phrase structure — is NOT injected in the primary arm; see §4 cleanliness.)* **Single-clause
fall-through (AH-mandated spec):** the 17/1045 mcd1-test (and matching train) commands with NO conjunction get
a fixed, content-blind, train/test-identical fall-through `e_comp ≡ pool(whole)` (no compose) — a
degenerate-but-defined case, NOT a tuned branch. (Verified: mcd1-test = 1028 single-conjunction / 17
single-clause / 0 multi-conjunction → the single-top-level split is exhaustively well-defined.)

**Learned compose operator (content-blind, problem-generic):** `e_comp = W_conj · [pool(enc_out[L]) ;
pool(enc_out[R])]`, with one small learned matrix per conjunction type `W_AND, W_AFTER` (shared across all
commands — the analog of exp90's role-class map, in representation space). `pool` = mean-pool of encoder
hiddens over the clause's token span (index slice, no learned pooler). The composition *function* is LEARNED
from train (not hardcoded to "and=concat"); only the *split points* are given.

**Consolidation (offline restructuring phase, exp90 pattern):** after standard task-CE training to plateau,
freeze the decoder; unfreeze encoder + input embedding + `{W_AND, W_AFTER}`; Adam ~800 steps over replayed
train commands. Fixed content-blind loss:

```
L = CE_whole(x)                                   # keep competence
  + β · ||  h(x)  −  e_comp(x)  ||²                # SELF-CONSISTENCY: holistic encoder state = composed-from-clauses
  + γ · CE( decode_from(e_comp(x)), gold(x) )      # DECODE-CONSISTENCY: the composed state decodes correctly (decoder frozen, teacher-forced)
  + δ · hinge_apart( clause encodings )            # ANTI-COLLAPSE: distinct clauses stay apart (no map-all-to-one)
```

`h(x)` = the encoder's holistic final state (the decoder's normal init). Forcing `h = compose(clause-pools)`
+ `decode(compose) = gold` makes the command's drive-state a *learned composition of its clause encodings*; at
test a novel `(clause_L', clause_R')` lands on the seen manifold if the clauses are individually seen — the
recombination GECA cannot reach (it needs surface-substitutable compounds; verified ~0% MCD reach). β, γ, δ
fixed scalars (swept once over {0,1,3,10}-ish, the chosen fixed triple reported — NOT per-split tuned).
**λ≡(β,γ,δ)=0 ⇒ byte-identical to the architecture-only baseline** (default-off discipline).

## 3. Anti-homunculus analysis (to be vetted before build)

- **Who decides what:** the split is a fixed generic string function (conjunction markers); the composition is
  a learned matrix; the consolidation is a single averaged scalar loss. **No supervisor, no metric-branch, no
  per-example arbitration, no read of test labels or the held-out set.**
- **Where the apparent "decision" lives:** in the *tension* of the self/decode-consistency loss — the encoder
  manifold settles so that holistic = composed; "compositionality" emerges as a fixed point, not a routed
  choice. The compose operator is a fixed-arity scaffold (legal structural prior, CONTEXT.md §3) whose
  *function* is learned.
- **Problem-generic boundary:** "whole = compose(parts) at conjunction boundaries" is the textbook
  compositional prior, NOT hand-shaped to the MCD answer. The **random-bracketing control** (split at a random
  position) is the forcing function: if it lifts equally, the win is capacity/regularization, not
  conjunction-composition → down-rank. The **minimal (conjunction-only) parser is the clean primary**; any
  reliance on deeper injected grammar is disclosed as structure-injection (off-target).
- **Fence:** iterative gradient only; no closed-form SVD. ✓

## 4. Headline, controls, cleanliness gates

- **Headline:** mcd exact-match, **CCC(consolidation) − baseline(λ=0) CI-disjoint > 0 AND CCC ≥ faithful GECA**,
  ≥8 seeds, bootstrap CI, on ≥ mcd1 (then mcd2/mcd3 for generality).
- **Mandatory arms:** (1) `vanilla_plain` in-house floor (0.17/0.14/0.02); (2) `ccc_baseline` λ=0 (CCC model,
  no consolidation — matched-capacity, byte-identical to standard train); (3) `ccc` consolidation (conj split)
  — THE MECHANISM; (4) `faithful GECA` on the same MCD split (token-GECA in-house lower bound + cite published
  51/30/12 strong fragment-GECA) — the **redundancy decider**; (5) **`ccc_random` control** — CCC machinery
  with the conjunction split replaced by a random interior split point (same W keyed by the real conjunction);
  **must NULL** (the anti-conjunction-alignment decider); (6) **`ccc_nosplit` control (AH-mandated)** — the
  consolidation with `e_comp = W_unary·pool(whole)`, NO clause split; **must NOT lift as much as `ccc`** (the
  "genuine *clause* composition vs generic representation-tightening" separator — the label-shuffle/B-KILL
  discipline that caught the two prior false positives).
- **Anti-homunculus review: PASS-WITH-FIXES** (2026-06-17, `anti-homunculus-reviewer`): mechanism is the right
  shape; the 3 fixes above (single-clause spec, minimal-parser headline, `ccc_nosplit` arm) are folded in.
- **Cleanliness gate (panel-mandated):** the win must survive the **minimal conjunction-only parser**; if only
  a deeper grammar-aware parser lifts, disclose injected grammar as load-bearing → verdict slides toward
  structure-injection.
- **Honest bar:** beat the general-purpose neural floor (vanilla/GECA/MAML/T5 all <50% mean MCD). The
  ~99–100% AuxSeq/LeAR ceiling is structure-injecting — NOT the target; do not chase it (suspect leakage if
  approached). **A NULL is acceptable, expected iterate-fuel** — it maps which axis of compound divergence
  representation-space composition can/cannot reach.

## 5. Build plan (staged — smoke before the full matrix)

- [ ] **Anti-homunculus review of THIS design** (mandatory, before any code) → `anti-homunculus-reviewer`.
- [ ] Build `experiments/94_betb_scan_mcd_ccc.py` (fork exp87/exp90 + exp93 MCD loaders). λ=0 byte-identical check.
- [ ] **Smoke on mcd1** (n=1, short) — is there ANY signal over baseline? If flat-null, bank the mapped wall cheaply.
- [ ] If signal → **full run**: 8 seeds × {mcd1[,mcd2,mcd3]} × arms {vanilla, baseline, CCC, GECA, random-bracket}.
- [ ] Report (all 5 done-gates) + STATUS + memory. Headline vs floor AND vs GECA; cleanliness gates reported.

## 6. Open risks (on the record before the build)

- **Decoding from a composed vector** may underperform the attention-over-all-positions path; if `decode_from`
  loses the per-position attention the composition can't regenerate long sequences — mitigated by keeping
  attention over the original `enc_out` and only replacing the decoder *init* with `e_comp`.
- **Most-likely outcome is a partial/null** that maps the composition axis (honest, per charter). The
  random-bracketing + GECA arms make a null *interpretable* (a mapped wall), not a wash.
