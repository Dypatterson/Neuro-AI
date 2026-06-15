# Report 141 — Bet B / SCAN Stage 1: representation-alignment consolidation NULLs (and why)

**Status:** Stage-1 graduation attempt → **NULL**, with a precise, multi-probe diagnosis
(2026-06-15). **Experiment:** `experiments/88_betb_scan_consolidation.py` + 3 diagnostic probes.
**Charter:** [notes/betb-scan-stage1-consolidation-precommit.md](../../notes/betb-scan-stage1-consolidation-precommit.md)
(anti-homunculus PASS). Builds on Stage 0 ([Report 140](../140_betb_scan_gap/report.md)).

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer, discriminating regime — the thesis
  test (does a restructuring consolidation manufacture composition a simple method can't?).
- **Headline per [precommit §3](../../notes/betb-scan-stage1-consolidation-precommit.md):**
  add-`jump` exact-match, consolidation − baseline. **Result: Δ ≈ 0 (NULL).**
- **Why now:** Stage 0 licensed it (the gap is real, headroom 0.995). A positive here would NOT
  be the §4 task-selection confound. It nulled — but the *reason* is the contribution.

## The mechanism tested

`consolidation = offline pass: L_replay (CE on replayed train batches → keeps each verb decoding
to its own action = FILLER/identity) + λ·L_align (variance of the PEER-SET input embeddings →
pulls verbs together = shared ROLE)`. Peer set computed in code from the buffer (= `{jump,walk,run,
look}`, asserted — anti-homunculus BUILD CONDITION 1). The role/filler split was meant to emerge
from the loss interaction.

## Result + the diagnostic chain (why it nulls)

| probe | finding |
|---|---|
| **exp88 consolidation** (embedding align + replay) | baseline 0.0014 → consolidation 0.0010. **NULL.** |
| **premise probe** (jump←walk embedding swap, gold I_JUMP→I_WALK) | **0.9935** — the template slot works perfectly; `jump`'s embedding IS the sole locus. |
| **interpolation** (jump → walk / centroid, real I_JUMP gold) | **0.000 at every α** — no linear move into the verb cluster preserves `I_JUMP`. |
| **role-replace** (centroid + identity-subspace projection of jump, 3 constructions) | **all 0.000–0.0014** — no principled role⊕identity decomposition of the embedding composes with correct identity. |
| **whole-model consolidation** (fine-tune ALL params + align, λ∈{1,10,50}) | **NULL (0.0005–0.0008); `L_align` frozen at 30.85 even at λ=50.** |

**The mechanism, in one number:** at λ=50 the model *refuses* to reduce `L_align`. Reducing it
(making the verb embeddings similar) would break `L_replay`'s identity constraint (verbs must stay
decodable to distinct actions). **Role-alignment and identity-preservation are in irreducible
conflict** when the model has no factored role⊕filler representation — and a model trained with
`jump` in a single context never developed one. Post-hoc consolidation, on the embedding OR the
whole model, cannot manufacture the factorization: aligning the role destroys the identity.

## Reading (honest scope)

- **This is NOT "the gap is uncloseable."** It is: *the representation-alignment consolidation
  family does not close it, for a precise reason* (role/identity entanglement absent a factored
  representation). GECA-class **generative augmentation** (synthesize `jump`-compositions) is known
  to close it — but that is data-space augmentation, the redundancy wall (§4 guard), not a
  restructuring of representation.
- **It re-derives the Bet-A bound under Bet-B's lifted rules.** Backprop and global/iterative
  mechanisms were *allowed* here, and a post-hoc alignment (the SCAN analog of the local writer)
  still cannot reach the compositional factorization — the structure lives in a representation the
  model must be *built/trained* to have, not one post-hoc alignment can install. Cf. the 121–132
  local-vs-global bound: subdominant structure unreachable by the available means.
- **Premise + null together are the sharp result:** `jump`'s embedding is provably the locus
  (0.9935 with a full swap), yet no embedding gives structure + identity — so the locus is real but
  the fix requires factorization the post-hoc mechanism can't induce.

## Disposition — the next iterate (the wall names the spec)

The factorization must exist **during** learning, not be installed after. Two live forks:
1. **Architectural role⊕filler factorization, trained jointly** — build the split into the model
   (e.g. a bottleneck that forces verb identity into a separable subspace), trained on the task so
   `walk/run/look` establish the shared role pathway; THEN test whether a consolidation can align
   `jump`'s role while the architectural filler preserves identity. The principled "does the
   restructuring thesis survive with the right inductive bias?" test.
2. **Generative/analogical pseudo-replay** (synthesize `jump`-compositions by analogy to
   `walk`-compositions, rehearse them) — likely works but lands on the GECA redundancy wall;
   novelty would require it be emergent/local where GECA is engineered preprocessing.

Per the charter the recipe is swappable and a clean null is iterate-fuel, not a dead end. The
discriminating regime (Stage 0) stands as the durable asset; this report sharpens what a
graduating mechanism must do.

## Anti-homunculus / discipline

Mechanism was anti-homunculus-PASS pre-build (agent `a0c17e48621a908db`); BUILD CONDITION 1 (peer
set computed-in-code, asserted) honored. n=1 for the probes (the signals are at-floor and
consistent across 4 independent probes; a multi-seed run was not spent on a mechanism shown null by
the cheaper diagnostic chain — the gap-diagnostic-first discipline). The frozen-`L_align` result is
the load-bearing evidence and is deterministic.
