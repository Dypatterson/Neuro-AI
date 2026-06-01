# HANDOFF — 2026-06-01 (PM-7; end of day; fresh-session orientation)

**Branch:** `consolidation/role-structure` · **HEAD:** today's commits 061683e (Report 126) + b7396f0
(exp67 + Step-2 precommit) + the about-to-be-committed Report 127 work. Memory edits live in `~/.claude/`.
Oracle `_*.json` are gitignored (data); `.stderr`/report.md committed. Pre-existing untracked left alone (§5).

**One line:** A long disciplined string of nulls (121–126: the flat-code/linear-local family exhausted,
the bound CAPABILITY-level) gave way to the day's **one positive — then the adversarial verification
narrowed it**: assembly **k-WTA** on the SPPMI 2nd-order operator clears the gate, BUT a plain global
**k-means** reproduces it ⇒ the real finding is **a NONLINEAR-PARTITION escapes the LINEAR-local read**
(operative bound = linear-projection-vs-nonlinear-partition, **not** global-vs-local, **not**
competition-specific). **NEXT = an online-local k-WTA vs offline global k-means head-to-head, headlined
on the label-shuffle B-KILL** (the only surviving competition-specific signal; locality+online is the
genuine differentiator over Lloyd k-means). Clean stopping point; nothing mid-run.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — charter (bet, gates, invariants).
2. **[STATUS.md](STATUS.md)** — bookmark (Active deliverable = the full 121→127 chain).
3. This file.
4. **[Report 127 §0.5](reports/127_nonlinear_competition/report.md)** (the verification update — the
   k-means refutation + the surviving claim) and **[Report 126](reports/126_behavioral_substitutability/report.md)**
   (the behavioral NULL). The Step-2 precommit (grill-hardened, Q1-Q6):
   [phase-3-nonlinear-competition-kill-test-precommit.md](notes/emergent-codebook/phase-3-nonlinear-competition-kill-test-precommit.md).

## 1. Where we are RIGHT NOW (today, PM-7 — a long session)
- Verified Reports 124/125 reproduce bit-identically (re-ran). Built+ran the **Reframe-B behavioral probe**
  → NULL (Report 126): the bound is capability-level, not a codebook-cosine artifact (the within-para
  label-shuffle caught a single-seed false-PASS).
- **Step 1 (exp67):** the Idea-1 static inverse-recall = Report 126 as a static cosine → lands on the 126
  NULL. Idea-1/2/3/7 cluster CLOSED.
- **Grilled the Step-2 precommit** (/grill-with-docs, Q1-Q6). Q1 = the prior upgrade: nonneg
  similarity-matching escapes PCA → manifold-tiling part-based (Sengupta 2018), the class NMF/E used.
- **Built+ran the nonlinear-competition kill-test (Report 127, exp68):** k-WTA on `build_S` clears the
  gate (label-shuffle 4-5/5, beats `grow_G` +0.002). A **6-agent adversarial verification** then showed
  **global Lloyd k-means one-hot reproduces it** (+0.250 ≈ +0.248) → the gate headline is a NONLINEAR-
  PARTITION property, NOT competition. **Survives:** any nonlinear partition of `build_S` beats the
  linear-local read (build_S-specific; k-means-on-random-Gram = +0.010; cosine-independent co-membership
  +0.300, perm-p 0.0000, broad 30/40 pairs; grow_G +0.002 fails on the SAME operator). NSM (soft
  competition) genuinely null. M_trans control-contaminated (inconclusive). Magnitude low-dim-inflated.
- **The single best thing all session: the discipline kept paying off.** The label-shuffle caught 126's
  false-PASS; the k-means control caught 127's overclaim. Without either, we'd have banked a false escape.

## 2. The genuine next move (un-built)
The corrected finding hands a SHARP, cheap next experiment. The project's genuine differentiator over a
global Lloyd k-means is NOT "partition vs projection" (k-means already proves that) — it is **LOCALITY +
ONLINE STREAMING** (no global Gram pass, brain-analogous = the non-negotiable). So:
- **An ONLINE/STREAMING local k-WTA vs an OFFLINE global k-means head-to-head on `build_S`**, both read
  with the static para-vs-random + label-shuffle, **headlined on the B-KILL margin** (the only surviving
  competition-specific signal: k-WTA's pair-specificity was modestly cleaner than k-means'). If the
  online-local writer **preserves** the B-KILL pair-specificity margin that the offline global k-means
  lacks → THAT is the load-bearing, novel, anti-homunculus-clean result worth opening + CARDING the
  assembly / sparse-Hopfield primaries (then a competitive-writer grounding precommit; build behind the
  fence). If it does not → competition adds nothing beyond global clustering; the partition-vs-projection
  reframe of the bound is the bankable finding, and the queue reverts to the iterated-TEM oracle.
- Write a frozen precommit first. Reuse `experiments/68` + `exp65`. Wire the **k-means control** into the
  harness (the verification ran it in /tmp; it must be in-harness for the next run).
- **Queued (unchanged):** the iterated-TEM local-reachability oracle; the eligibility×surprise
  two-timescale family (`whats_missing.md` §1). NSM (soft competition) needs a stable de-collapse that
  does NOT use rank-1 removal (which manufactures a no-learning artifact) before it can be re-tested.

## 3. Invariants the user holds (do not violate)
- **LOCAL growth is the MECHANISM, NON-NEGOTIABLE.** Global SVD/NMF/k-means/word2vec = DIAGNOSTIC
  FLASHLIGHTS only (the k-means here is a CONTROL, not the mechanism). "Do it right — no shortcuts." [[do-it-right]]
- Anti-homunculus (the k-WTA cap is a fixed precommitted top-k = clean; the reads are pure measurement);
  batch-offline (online TD/error-driven banned; Hebbian/anti-Hebbian + a fixed cap are allowed);
  FHRR-native (the kill-test is substrate-free; the FHRR-port is the deferred Stage-1 = fence); the
  055-058 FLOOR untouched.
- **Phase fence:** a competitive/TEM/latent BUILD is Phase-5 — the user's to lift; substrate-free oracles
  + grounding in-scope.
- **Anti-rationalization (the load-bearing habit):** the within-set label-shuffle + the competent-control
  (k-means, not just frozen-random noise) + multi-seed caught TWO false positives today. The verdict-bearer
  is the adversarial control, never the headline arm.

## 4. Banked side-findings (don't relitigate)
- The operative bound is **LINEAR-PROJECTION-vs-NONLINEAR-PARTITION** (grow_G power-iteration → dominant
  collocational mode; ANY hard partition of `build_S` reaches the paradigmatic structure) — NOT
  global-vs-local, NOT competition-vs-not. Magnitude is low-dim partition-inflated (recoverE 1.67; k-means
  inflates identically) → do NOT claim "k-WTA/partition beats SVD."
- The frozen-random no-learning control is an INCOMPETENT clustering — it does not exclude a competent
  non-assembly clustering (k-means). Always include a competent control.
- NSM de-collapse via rank-1-common-mode removal is itself a no-learning artifact generator (the same
  M_trans-style contamination). M_trans rows are peaky (top-1 = 36% of row energy) → frozen-random cap
  inflates; build_S rows are dense (top-1 = 1.5%) → clean.
- Literature: Sengupta-2018 manifold-tiling is UNCARDED (not in manifest); sparse-Hopfield primaries are
  link_only → motivating-only (Hard Rule 3), NOT load-bearing for a build.

## 5. Artifacts (uncommitted — today, PM-7)
- Report **127** (`reports/127_nonlinear_competition/report.md` + `_kill_test_wikitext.json`).
- `experiments/68_nonlinear_competition_kill_test.py` (k-WTA validated; NSM collapses — flagged).
- The grill-hardened Step-2 precommit (edited since b7396f0: Q1-Q6 + the 3a correction).
- STATUS (Active-deliverable walk-back + PM-7; two oldest entries migrated to the archive), this HANDOFF,
  memory (`~/.claude/`).
- Pre-existing untracked (leave them): `brainstorm-workspace/2026-05-30-research-grounded-plan/_wf{1,2}_raw.json`, `reports/gate0_2026-05-28/`.
