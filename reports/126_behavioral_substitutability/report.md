# Report 126 — Behavioral substitutability probe (Reframe-B) → NULL: the 121-125 bound is CAPABILITY-level, not a codebook-cosine metric artifact

**Status:** Phase-3. A substrate-free **DRILL-DOWN reframe diagnostic** (NOT graduation) on the
EXISTING graduated 055-058 heteroassociative memory (read-only). Frozen pre-commit:
[phase-3-behavioral-substitutability-probe-precommit.md](../../notes/emergent-codebook/phase-3-behavioral-substitutability-probe-precommit.md).
Harness: `experiments/66_behavioral_substitutability_probe.py`. Floor (055-058) untouched.
5 substrate seeds, bootstrap CIs over the n=40 SimLex≥5 pair set; the adversarial label-shuffle
+ multi-seed are load-bearing (see §4).

**Verdict: NULL for the strong "metric-artifact" hypothesis.** The 121-125 codebook-cosine bound
(paradigmatic structure unreachable by local growth; +0.109 global / ≤+0.021 local, route-invariant)
is **NOT a mere metric artifact.** An INDEPENDENT behavioral measurement — does the graduated memory
admit a paradigmatic partner (queen) as a completion of its mate's contexts (king-contexts)? — agrees
with the codebook-cosine read: **paradigmatic substitutability is not robustly present as a
pair-specific capability.** What para>rand admission exists is **dominated by para-set membership
(content-word/syntactic hubness), not the specific paradigmatic relationship** (the label-shuffle
kill-test). **PASS 0/5 seeds.** → The reframe does not rescue the program; the escape genuinely needs
new structure (the iterated-TEM / eligibility families). The reframe question is **closed cheaply and
decisively.**

---

## 1. Why this probe (the reframe) and why it is the right cheap first move

The entire 121→125 arc measures paradigmatic structure as **static codebook-vector cosine** (king/queen
lift). CONTEXT.md defines the project's capability as **contextual completion** ("what fills this gap").
The completeness-critic (`brainstorm-workspace/2026-06-01-nonflat-phase3/whats_missing.md` §4 Reframe-B)
+ the domain-expert adjudication flagged a possible category error: substitutability may exist
BEHAVIORALLY (king-contexts admit queen as a low-energy completion) even when king/queen codebook
vectors are not cosine-close — and that capability is **not subject to the subdominant-modes bound** (it
reads the heteroassociative memory's geometry, never a grown codebook centroid). This probe tests that
directly on the EXISTING graduated memory, reusing existing machinery, before any new substrate is built.

**Mechanism (the key identity):** the admission of partner b under a's contexts, read through the
graduated write+L2 memory H, reduces to **how similar a-contexts are to b's training contexts in
heteroassociative key space** — i.e. second-order context similarity read through the completion
dynamic. This is independent of whether the memory *generalizes* (it sidesteps the "real-text held-out
≈ chance" memorizer limit), and it never forms a codebook centroid.

## 2. Method (frozen pre-commit §2-§4)

WikiText-2, W=6, **max_vocab=2000 (V=2002 — the EXACT vocab of the 121-125 arc)**, D=2048. Graduated
memory = write+L2 (`heteroassociative_write` over a frozen buffer + `CueDecorrelator(l2)`), value
codebook = the 2000 decodable tokens (chance 1/2000). Pairs from `exp63.select_pairs`: **para** = SimLex≥5
cooc≤2 (n=40, non-co-occurring ⇒ a positive can't be co-occurrence); **rand** = matched low-cooc (n=40,
negative control); **collo** = SimLex≥5 cooc>2 (n=5, contrast); **king/queen** (named). For each pair (a,b),
cue with a's center-context windows (a masked); read the completion's admission of b two ways — **score**
(cos of the settled completion to v_b, minus a frequency-matched random token = the "low-energy
alternative" framing) and **rank** (percentile rank of b among all 2000 completions − 0.5 = the "measured
by recall" framing). Symmetrized a↔b. 5 seeds; bootstrap CIs (4000) over the 40 pairs.

**Pure diagnostic measurement** (anti-homunculus-exempt): nothing branches/gates/writes on the reads. The
rank/recall co-headline is reported precisely to stay on the right side of the G-D panel's "top_index not
energy" (Phase-5′) fence — the reads characterize the existing completion dynamic, they do not arbitrate.

## 3. Result (5-seed aggregate; deterministic; `_probe_seed{0..4}.json`)

Memory calibration **C0 valid 5/5** (write_l2 true recall **0.990** ≫ store-as-is 0.634 ≫ 5×chance) — a
clean graduated regime where write+L2 is the operative mechanism.

| arm | score Δ (mean) | rank Δ (mean) |
|---|---|---|
| **collo** (cooc>2, n=5) | +0.0239 | +0.117 |
| **para** (SimLex≥5, non-cooc, n=40) | **+0.0117** | **+0.076** |
| **para_shuffled** (same para tokens, broken pairing) | **+0.0041** | **+0.031** |
| **rand** (matched low-cooc, n=40) | +0.0022 | +0.021 |
| king/queen (named; queen sparse) | +0.0239 (sd 0.017) | +0.180 (sd 0.12) |

Gates across seeds: **C0 5/5 · B1 (para>0) 5/5 · B2 (para>rand) 3/5 · B3 (context-specific) 5/5 ·
B4 (pair-specific, the kill-test) 1/5 · PASS 0/5.**

- **para admission is real but small** (B1 5/5): para sits above rand. But the **para−rand differential is
  fragile** (B2 3/5 — once both read variants must clear CI-lo>0, two seeds fail).
- **The adversarial label-shuffle is decisive (B4 1/5).** para_shuffled (+0.0041) sits roughly *halfway*
  between rand (+0.0022) and para (+0.0117): **most of the para>rand admission is para-set MEMBERSHIP**
  (these are mid-frequency content nouns that generically admit one another in completion), **not the
  specific paradigmatic relationship.** The pair-specific residual (para−shuffled) is +0.0077 on the score
  read (CI-lo>0 in only 3/5 seeds) and essentially absent on the rank read (1/5). The pre-committed bar —
  **both** read variants pair-specific — is met in **1/5** seeds.
- **king/queen, the canonical pair, is not robust:** rank Δ ranges −0.024→+0.285 across seeds (sd 0.12)
  because queen has only 31 center-occurrences (capped at 20) — too sparse to anchor a per-pair estimate.

## 4. The adversarial controls were load-bearing (methodological headline)

In the single best seed, **C0 ∧ B1 ∧ B2 ∧ B3 all pass** — i.e. the probe *without* the label-shuffle and
*without* multi-seed would have read as a clean **PASS** and produced the large over-claim "*the graduated
memory behaviorally admits paradigmatic substitutes → the 121-125 bound is a metric artifact*." The
**within-para label-shuffle** (same tokens, broken pairing — a sharper control than rand-pairs, which
differ from para in token identity/frequency) showed the signal is **para-set hubness, not pair-specific**;
**5 seeds** showed B2 itself is only 3/5. This is the "do it right / adversarially verify / multi-seed is
the bar" discipline (CLAUDE.md) catching a false positive — exactly the *"confidence from a single seed"*
failure mode the working agreement names.

## 5. Drill-down: a capacity/regime artifact that would have faked a positive

The headline runs in the graduated regime (n_train≈5000 ≈ 2.4×D). The first attempt used a "realistic
full-corpus" memory (background=20000 ⇒ n_train≈29000 ≈ 14×D): there the heteroassociative memory
**saturates** — write_l2 true recall collapses to 0.32 (< store-as-is 0.67) ⇒ **C0 INVALID** (correctly
rejected by the frozen gate, not relaxed). Notably the saturated regime showed a *stronger* para>rand
(+0.017) AND a strong king/queen (+0.045) — i.e. **the "metric-artifact positive" appears precisely in the
over-capacity regime where the memory is broken**, and vanishes in the clean regime. That the apparent
signal is regime-dependent on memory saturation is further evidence it is not a real capability.

## 6. One precommit-specification finding (surfaced, not papered over)

The frozen C0 second clause ("write_l2 true-rate ≥ store-as-is true-rate") was justified in the precommit
as "the 055/056 write-marginal direction." That is a **mis-mapping**: the 055/056 write-marginal is on the
**role-Selectivity-Δ**, not on raw true-recall. The clause therefore tests memory-capacity (write_l2 not
saturated), not the graduation property. It did **not** affect the verdict — the headline regime passes C0
decisively (0.99 ≫ 0.63) and the conclusion is a clean NULL — but the precommit text should be read with
this correction. Logged per the "contradiction is a finding" rule.

## 7. Disposition

- **The reframe is closed (NULL for the strong hypothesis), decisively and cheaply.** The 121-125
  codebook-cosine bound is **capability-level, not a metric artifact**: an independent behavioral
  (completion-admission) measurement agrees paradigmatic substitutability is not robustly present. This
  *strengthens* the 121-125 result — two orthogonal metrics now concur.
- **Honest residual (not banked as a positive):** there is a *weak, sub-bar* pair-specific admission on the
  score read (+0.0077, 3/5 seeds) that the rank read and king/queen do not confirm; flagged as
  possibly-real-but-below-the-pre-registered-bar and dominated by para-set hubness — NOT a foothold.
- **Next:** revert to the structural escapes the bound demands. The adjudicated lead remains the
  **iterated-TEM local-reachability oracle** (fixed slots + bounded MHN-settling content→structure
  equilibration, backprop-free — the domain-expert's recommendation) and/or the **eligibility-gated /
  three-factor consolidation** family (`whats_missing.md` §1, the only proposal that attacks the failing
  `corr(cooc,drift)` quantity by construction). The iterated version crosses toward a stripped-TEM build =
  the Phase-5 fence (the user's to lift).

## 8. Artifacts
- `experiments/66_behavioral_substitutability_probe.py`; `reports/126_behavioral_substitutability/_probe_seed{0..4}.json`
  (+ `.stderr`), `_probe_D2048.json` (= seed 0, the canonical headline). Deterministic (the freq-match draw
  was de-`hash()`-ed for reproducibility after the first sweep; numbers shifted < 0.001, verdict unchanged).
- Frozen pre-commit (C0 + B1-B4 gates, the label-shuffle kill-test pre-registered).
- The C0 over-capacity INVALID regime (background=20000) documented in §5 as a saturation artifact.
