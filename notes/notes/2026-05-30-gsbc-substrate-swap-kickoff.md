---
name: gsbc-substrate-swap-kickoff
date: 2026-05-30
project: personal-ai
status: KICKOFF SEED — Stage-0 precommit to be FINALIZED by the GSBC session (with the user). Not yet binding.
branch: GSBC/substrate-swap (off codex/phase5-prime-bundle-first-scene-memory @ 86cc9e4)
parent: brainstorm-workspace/2026-05-24-unconsidered-paths/research/01-alternative-vsa-algebras.md (Idea A)
tags: [notes, subject/substrate, subject/vsa-algebra, gsbc, diagnostic]
---

# GSBC substrate-swap — kickoff seed

**Read first (session-start):** [STATUS.md](../../STATUS.md), then
[Report 116](../../reports/116_frameb_corr_ac_bd_slope_drop.md) →
[117](../../reports/117_frameb_leveldid_feasibility_consolidation_variance.md) →
[118](../../reports/118_frameb_landscape_sweep_variance_irreducible.md), then this note.
This is a **seed**, not a binding precommit — the GSBC session finalizes Stage 0 with the user.

## Why we're here

Phase 3 Frame B hit an impasse: **three levers closed in one session** (slope-DiD dropped;
level-DiD not economically rescuable; landscape size does not tame the variance). The
consolidation-injected variance is **structural and substrate-intrinsic** (σ_A stuck ~0.15
regardless of landscape; Report 118). This **corroborates the "substrate-shape, not
mechanism-tuning" thesis** (Theme C of the 2026-05-24 unconsidered-paths brainstorm) — which
was independently reached from the **Phase 5** side (four role-binding retrieval families all
null). Two phases now point at the **FHRR substrate algebra** as the bottleneck.

## The bet: GSBC (Generalized Sparse Block Codes)

Spec at [research/01-alternative-vsa-algebras.md](../../brainstorm-workspace/2026-05-24-unconsidered-paths/research/01-alternative-vsa-algebras.md)
Idea A. D=4096, **B=64 blocks of L=64** (binary SBC: one nonzero/block; or GSBC: unit-ℓ₁/block).
Binding = **block-wise circular convolution**. Similarity = **ℓ∞-based** `s∞(x,y)=1−ℓ∞(x−y)`
→ **structural zeros**: non-overlapping bindings have *exactly* 0 similarity, which FHRR's
cosine provably cannot. Directly targets the documented failure "all role–cue overlaps look
the same in cosine." Refs: Hersche et al. 2023 (arXiv:2303.13957); Frady-Sommer 2020
(arXiv:2009.06734); code github.com/IBM/in-memory-factorizer.

## THE critical risk — read before writing any code

The **GHRR walk-back** ([Report 066]; status-log 2026-05.md:1608–1621) is the cautionary
template: the first GHRR cell stored non-FHRR vectors in the **FHRR-similarity Hopfield**, so
it *looked* like 100% but was algebraic-decode confound, not basin retrieval — it only became
honest once rebuilt with **GHRR-native primitives**. GSBC has its **own metric (ℓ∞)**.
Therefore, binding from line 1:
- Build **GSBC-native primitives** (block-sparse atom generation, block-circular bind/unbind,
  ℓ∞ similarity, GSBC-native cleanup). **Never** store GSBC vectors in the existing
  `torch_hopfield.py` (it assumes the complex/cosine inner product).
- A **structural-zero unit test must exist from the start** (non-overlapping bindings → exact 0;
  FHRR provably fails it). That test is the "is the algebra what it claims" gate.

## Anti-homunculus

PASSES **by construction** — the ℓ∞ zeros are a property of the *metric*, not a supervisor
threshold — **provided roles stay an algebraic property and are NOT hand-typed** ("this vector
is a role"). Typing is a homunculus unless the type is itself a geometric attractor of the
codebook (research §"Anti-homunculus screen — meta-observation"). **Run the finalized Stage-0
precommit past `/anti-homunculus-reviewer` before any mechanism lands.**

## Staged plan (de-risk the substantial decision cheaply)

- **Stage 0 — finalize this precommit** (variant, dims, headline test, "GSBC wins" threshold,
  anti-homunculus check, the GHRR-lesson guards). → anti-homunculus-reviewer pass.
- **Stage 1 — GSBC primitive + unit tests**, incl. the **structural-zero test**. Keep the
  pure-Python reference backend (PROJECT_PLAN rule); mirror the `torch_fhrr.py` API shape.
- **Stage 2 — the decisive head-to-head** (THE go/no-go): a *minimal* role-filler retrieval toy
  where FHRR is documented to fail (`hit_role≈0`), measured **natively** (no MHN / consolidation
  / WikiText). Does GSBC retrieve the right filler-by-role with a margin FHRR can't? Cheap
  (~1–2 days, CPU). **Kills or justifies the whole swap before any expensive integration.**
- **Stage 3 — integrate** into the MHN/eval pipeline with GSBC-native cleanup, **only if Stage 2
  passes**, then run the real protocol.

## OPEN DECISION (user's call — settle in Stage 0; sets what Stage 2 measures)

- **(a) Phase-5 role-separation** *(recommended)* — GSBC's structural zeros attack the documented
  FHRR failure most directly; sharpest, cheapest, most decisive Stage-2 toy.
- **(b) Phase-3 consolidation-variance** — weaker fit (today's variance is in the consolidation
  *dynamics*, not the similarity metric); would need Stage-3 integration even to test. Treat as a
  downstream hope, not the Stage-2 question.

## Substrate facts (don't reinvent / don't trip)

- `src/energy_memory/substrate/` has **no interface/ABC** — only `fhrr.py` + `torch_fhrr.py`.
  GSBC is a **parallel substrate**, not a drop-in. Mirror the `torch_fhrr.py` API.
- Reuse where it generalizes: `phase2/metrics.py` (cap-coverage, meta-stable rate, entropy, Wilson CIs).
- Do **not** remove the pure-Python reference backend; do **not** make the LLM the source of identity
  (PROJECT_PLAN non-negotiables).
- Phase 5′ stays **paused**; this branch does not touch it.
