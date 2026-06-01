# HANDOFF — 2026-05-31 (fresh-session orientation)

**Branch:** `consolidation/role-structure` · **HEAD:** `6e55e81`
**One line:** A major re-grounding + the genuine next gate (Phase-3 structure-gate **3b**) **ran and resolved to a NULL on the meaningful axis → RE-SCOPE the codebook-growth mechanism.** Phase 5 remains un-founded. No active build; clean stopping point.

---

## 0. Read first (in this order)

1. **[CONTEXT.md](CONTEXT.md)** — NEW this session. The stable charter: the bet, the three senses of "memory", the original six-phase gates, the bio North Star, and **the current crux** (§5). It is now read #1 in the CLAUDE.md session-start protocol. Read it before STATUS so STATUS is interpreted *against* the gates.
2. **[STATUS.md](STATUS.md)** — current bookmark (Active phase / headline / deliverable / blockers + Recent updates).
3. **This file.**
4. The load-bearing report + spec for the next move: **[Report 121](reports/121_phase3_structure_gate_3b/report.md)** + **[3b spec](notes/emergent-codebook/phase-3-structure-gate-3b-design.md)**; and the deep snapshot **[RE-GROUNDING-MAP.md](notes/RE-GROUNDING-MAP.md)**.

---

## 1. Where we are RIGHT NOW (the single most important thing)

The project's **"more than memory" thesis** comes down to one question — *does the codebook develop **paradigmatic** structure (similar words cluster), grown from experience?* — and **3b just answered it: NO, not under the current growth dynamics.**

- The graduated consolidation write (Reports 055–058) is the **Phase-3 contextual-completion FLOOR** (role-selective recall from sparse cues). It is real, multi-seed, integrated bit-identically, scaling settled (dense H, Report 120). **It is the floor, not the project's ceiling.** ("Memorization is the target" was a drift, now corrected everywhere — see §2.)
- **3b (Report 121):** the Hebbian codebook develops corpus-specific **COLLOCATIONAL** structure (co-occurring words cluster — ~by construction for a co-occurrence learner) but **NOT PARADIGMATIC** structure (similar, *non-co-occurring* words — the king/queen test — do **not** cluster; WikiText paradigmatic subset −0.0084, CI [−0.013, −0.004]; corr(log-cooc, drift) = +0.41/+0.82). → **The first-order Hebbian co-occurrence-centroid growth dynamic is the wrong SHAPE.**
- **Phase 5 (bind-vs-bundle discovery, atom-splitting, analogical retrieval) is UN-FOUNDED** — its mechanisms need paradigmatic structure, which does not emerge. Building Phase 5 now = a phase-order violation.

---

## 2. This session's arc (newest first)

| Commit | What |
|---|---|
| `6e55e81` | **Report 121** — Phase-3 structure-gate **3b RAN → paradigmatic NULL → re-scope.** |
| `0ddc240` | **Re-grounding** — corrected the "memorization is the target" **drift** (it mislabeled the Phase-3 floor as the project ceiling, brushing "not a vector DB"); stood up **`CONTEXT.md`** charter (read #1 in session-start). |
| `bf745ac` | **Report 120** — MESH scaling RESOLVED → **defer MESH**; H is intrinsically low-rank (factored-H is the cost fallback). |
| `cd277c0` | **Report 119** — WS-InfoNCE Stage-1 built + run → **F-COMPOSE FAIL, memory-not-learner** (held-out margin ≤ 0; value-codebook-shaping is a dead end for generalization). |
| `2dd0783` | Generalization-axis grounding — Dorrell doesn't port; within-scene JEPA does (design-only). |

The throughline: a sequence of grounded-before-built investigations that, via **pre-registration + adversarial verification**, turned three tempting "wins" into honest negatives — and surfaced that the project drifted into calling its Phase-3 floor the whole building. The re-grounding fixed the framing; 3b then tested the real thesis question and returned a clean re-scope signal.

---

## 3. The genuine next move (the re-scope target)

**Redesign the Phase-3 codebook-growth mechanism to cluster by CONTEXT SIMILARITY (second-order), not co-occurrence (first-order).** I.e. tokens with *similar neighborhoods* should cluster (paradigmatic), not just tokens that are *neighbors* (syntagmatic). Candidate directions named in CONTEXT.md §5 / the 3b spec: a **context-vector / SQHN / predictive-coding-style** update, replacing the co-occurrence centroid in `phase2/codebook_learner.py`.

**How to approach it (honor the disciplines that worked this session):**
- This is a **Phase-3 mechanism redesign**, NOT a Phase-5 build and NOT a larger re-run of the 3b null. (The pre-registered prior was explicit: a null = re-scope, not retry.)
- **Reuse the 3b harness** (`experiments/60_phase3_structure_gate_3b.py`) to evaluate any new growth mechanism: it already has the gauge-safe corpus-stream-shuffle control, the paradigmatic/collocational co-occurrence split, the random-pair specificity arm, and the d_eff collapse guard. The headline stays the **PARADIGMATIC** real-vs-shuffle gate (CI > 0). Don't reinvent it.
- **Ground before building** (grep `reports/` + read the design docs): the SQHN / predictive-coding / second-order-distributional literature is partly carded under `docs/ground-truth/`; check what's already been tried (e.g. `error_driven_learner` / `reconstruction_learner` are second-order-ish — see Reports 017/018) before treating a mechanism as new.
- **Anti-homunculus + gauge-safe controls are non-negotiable.** Any structure read must be an offline batch statistic; any control must be a data manipulation (the gauge-vacuous atom-relabel control is RETIRED).

**Alternative framing worth considering first:** is paradigmatic structure achievable from *this* substrate at all, or is the honest conclusion that the codebook is a *collocational* memory and the architecture's "more than memory" должен come from a different layer (e.g. Phase-4 hierarchy or the energy-term coupling)? That's a strategic fork for the user, not a foregone build.

---

## 4. Invariants to hold (the disciplines that paid off)

- **Memorization = the Phase-3 FLOOR, not the project ceiling/identity.** Never let "it's a memory" become "the project is a memory" (that's the forbidden vector-DB, `PROJECT_PLAN.md:276`). Compositional/paradigmatic structure is the Phase-5 deliverable (a 3→5 gradient, `experimental-progression.md §"What to test against"`).
- **Phase order.** Don't build a later phase on an unverified earlier gate. Phase 5 is gated behind a paradigmatic-structure foundation that does not yet exist.
- **Pre-register the interpretation** (especially "what does a null mean?") *before* a run, and **adversarially verify a PASS** — both caught false positives this session.
- **Bio North Star:** the bet is that data-hungry/backprop/autoregressive AI is wrong and biology (continuous learning, no homunculus, replay-as-sleep, energy-frugal) is the answer. Every mechanism must be brain-analogous and anti-homunculus-clean.
- **Mechanics:** `STATUS.md` < 20480 B (pre-commit hook); cite `experimental-progression.md` by **section anchor**, not line number (they shift); heavy `*.pt`/results JSON are gitignored; commit/push only when asked; the 3 untracked artifacts (`brainstorm-workspace/.../_wf1,2_raw.json`, `reports/gate0_2026-05-28/`) are pre-existing — leave them.

---

## 5. Open decisions (for the user)

1. **The re-scope itself** (the main one): pursue a context-similarity / second-order growth mechanism (§3), OR accept the codebook as collocational and seek "more than memory" from a different layer. **User's strategic call.**
2. **Pre-existing Phase-5′ `min_branch` aggregator** still FAILS anti-homunculus — must be de-arbitrated *before any Phase-5 reopen* (independent of 3b).
3. Minor: the 3b spec's NC1/inter-basin-separability drill-down (control #4) was not implemented — moot for the null verdict, but worth adding if a re-scoped mechanism produces a non-null paradigmatic signal.

---

## 6. Key artifacts

- **Charter:** `CONTEXT.md` (+ `docs/agents/domain.md`, `CLAUDE.md` session-start updated to read it first).
- **Deep snapshot:** `notes/RE-GROUNDING-MAP.md`.
- **Next-move spec:** `notes/emergent-codebook/phase-3-structure-gate-3b-design.md` (RESOLVED).
- **3b harness (reuse it):** `experiments/60_phase3_structure_gate_3b.py`.
- **Reports this session:** 119 (WS-InfoNCE), 120 (MESH), 121 (3b).
- **Memory (cross-session):** `neuro_ai_regrounding_2026_05_31`, `neuro_ai_memory_not_learner` (re-scoped), `neuro_ai_mesh_scaling_decision_open` (resolved), `neuro_ai_generalization_track_grounding` (resolved).
