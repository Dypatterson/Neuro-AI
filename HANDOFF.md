# HANDOFF — 2026-06-01 (end of day; fresh-session orientation)

**Branch:** `consolidation/role-structure` · **HEAD:** today's commit "Reports 124 + 125: flat-code
growth family exhausted; TEM factorization is the live route" (Reports 124/125, `experiments/63` &
`65`, 2 precommits, brainstorm synthesis, STATUS/archive/HANDOFF). **Committed, NOT pushed.**
Memory edits live in `~/.claude/...` (separate from the repo). Oracle `_*.json` are gitignored
(data); `.stderr` run logs are committed. Pre-existing untracked left alone (see §5).

**One line:** The **flat-code growth family is EXHAUSTED** — the local-vs-global bound is
**route-invariant for LOCAL single-projection dynamics** (now 7 operators across 121/123/124/125).
The **one live route is a LOCAL writer for the TEM structure×content factorization** (Report 125,
Oracle E: the factorization *target* is real, the local *writer* untested). **NEXT = a TEM
local-reachability oracle.** Clean stopping point; nothing mid-run.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — the charter (bet, gates, invariants).
2. **[STATUS.md](STATUS.md)** — the bookmark (Active deliverable = the full chain).
3. This file.
4. **[Report 125](reports/125_escape_route_triage/report.md)** (the 4-oracle triage + the E
   disposition + the two guardrails) and **[Report 124](reports/124_r3_directional_oracle/report.md)**
   (the directional NULL + the contraction-artifact lesson). Design space:
   **[brainstorm-workspace/2026-06-01-nonflat-phase3/SYNTHESIS.md](brainstorm-workspace/2026-06-01-nonflat-phase3/SYNTHESIS.md)**.
   Frozen precommit for the next oracle's siblings:
   [phase-3-escape-route-oracle-triage-precommit.md](notes/emergent-codebook/phase-3-escape-route-oracle-triage-precommit.md).

## 1. Where we are RIGHT NOW
Today: ran the directional R3 oracle → NULL (124); the user lifted the Phase-5 fence; 2 workflows
(Option-B latent grounding + non-flat brainstorm) + a completeness critic converged on a few
bound-escapes + the **locality trap** (power iteration reaches the dominant collocational mode, not
the subdominant paradigmatic one); built+ran the **4-oracle escape-route triage** (Report 125):
- **B (SFA/SR, the LEAD), C (eligibility), D (order) = clean LOCAL `grow_G` NULLs.** The lead
  nulling confirms the locality trap. The bound is route-invariant for local dynamics.
- **E (TEM nonnegative slot-factorization) = a real GLOBAL-flashlight POSITIVE** — recovers ~all of
  M_trans's full-rank paradigmatic specificity in 8–32 dims (label-shuffle + random-nonneg controls
  confirm real). **But it's a global flashlight (NMF), not a local dynamic** — same epistemic status
  the global SVD had for the flat code: *target real, local writer untested*.

## 2. The genuine next move — a TEM LOCAL-reachability oracle (un-built)
The make-or-break, adjudicated by the verification. **Substrate-free, ~exp63 scale, no FHRR port,
no Phase-5 commitment.** ONE question: *does a LOCAL online Hebbian path-integration rule — fixed
random structural slots; content vectors Hebbian-bound to the slots they co-occur in across windows;
accumulated ONE WINDOW AT A TIME, with NO global factorization — recover most of E's +0.19, or null
to ≈0 like B/C/D?* The load-bearing distinction from E: E **factorized** M_trans globally (NMF); the
oracle must **accumulate** slot-occupancy locally/online.
- **PASS** (recovers most of +0.19) → the slot architecture breaks the locality trap → license a TEM
  build (then: open Whittington 2020 + the Hebbian-not-backprop locality question; FHRR-port Stage-1;
  the Phase-5 fence — the user's to lift).
- **NULL** (nulls like B/C/D) → the global NMF optimization was the part that mattered → TEM inherits
  the trap → kill the TEM build for ~zero cost.
- **Gates: g1 (CI-lo>+0.04), g2 (beat flat-SPPMI −0.0003 by +0.02), g4 (no collapse), PLUS a
  frequency-matched / label-shuffle collocational control — NOT g3** (dead at n=40; see §4). Expand
  the non-co-occurring SimLex set or replace g3 with a powered discrimination control. **Write a
  frozen precommit first.** Reuse `experiments/65` + `exp63` machinery.

## 3. Invariants the user holds (do not violate)
- **LOCAL growth is the MECHANISM, NON-NEGOTIABLE.** Global computations (SVD/NMF/word2vec) are
  DIAGNOSTIC FLASHLIGHTS only — Oracle E's NMF is used that way; it does NOT license a build.
  "Do it right — no thesis-compromising shortcuts." [[do-it-right]]
- Anti-homunculus; batch-offline (sleep/wake; online TD/error-driven banned); FHRR-native; the
  055-058 FLOOR untouched. **Phase fence: a TEM/latent BUILD is Phase-5 architecture — the user's to
  lift; grounding + substrate-free oracles are in-scope.**
- **Anti-rationalization:** FOUR metric subtleties were caught across 124/125 before banking (the
  stream-shuffle gauge leak; the gauge-free double-subtraction; the SVD low-rank contraction artifact;
  g3 unachievable at n=40 for static reads). Stay this suspicious — the verdict-bearer is the faithful
  `grow_G` LOCAL read, never the SVD/NMF flashlight.

## 4. Banked side-findings (don't relitigate)
- The **locality trap** (completeness critic): power iteration + local whitening (H_anti) converges
  to the dominant collocational mode, not the subdominant paradigmatic one. SFA ≡ SR (one eigenproblem).
- **g3 (`corr<0.15`) is unachievable for STATIC global reads at n=40** — the accepted +0.109 SVD
  anchor also fails it (corr pt 0.077, CI-hi 0.44). g3 is VALID for the `grow_G` LOCAL drift (its
  designed read-class; it correctly killed 123/124) — do NOT generalize the correction.
- **Do NOT claim "E beats the SVD" (+0.19 > +0.109)** — apples-to-oranges (different operators; a fair
  SVD-300 on M_trans gives +0.183). E's absolute cosines are low-dim-INFLATED (rand 0.47–0.67).
- Predict-context objectives (Γ1/PAM/WS-InfoNCE/raw-SR-cosine) are SPPMI-factorizers in disguise →
  bound-extends-to-them (Option-B grounding); the eligibility cheap surrogate (Oracle C) nulled.

## 5. Artifacts (uncommitted)
- Reports **124** (`_oracle_wikitext_full.json` + localiter), **125** (`_triage_wikitext.json`).
- `experiments/63_directional_successor_oracle.py`, `experiments/65_escape_route_triage.py`.
- Precommits: `phase-3-r3-directional-oracle-precommit.md`,
  `phase-3-escape-route-oracle-triage-precommit.md`.
- `brainstorm-workspace/2026-06-01-nonflat-phase3/` (SYNTHESIS + option_b_brief + brainstorm_menu +
  whats_missing) and `brainstorm_nonflat_phase3.js`.
- Pre-existing untracked (leave them): `brainstorm-workspace/2026-05-30-research-grounded-plan/_wf{1,2}_raw.json`, `reports/gate0_2026-05-28/`.
