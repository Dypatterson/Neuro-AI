# 2026-05-30 — Research-grounded forward plan

Produced by two workflows, grounded strictly in `docs/ground-truth/` + the 21
primary PDFs + 33 new web sources, refuter- and grill-verified. **No binding doc
was edited — every STATUS/spec change is surfaced for sign-off.**

## Read in this order
1. **`02-forward-plan-FINAL.md`** — the deliverable: immediate gate, every-phase map (0–7 + 5′), rebuild branch, open user decisions, done-gates, risks.
2. `00-diagnosis-synthesis.md` — the verified diagnosis (root cause, fork, candidate objectives, findings to surface).
3. `01-brainstorm.md` — 10 ideas with anti-homunculus shapes + cheapest tests.
4. `03-grill-and-audits.md` — the 12-question grill verdicts + anti-homunculus / done-gate / terminology audits.

## Detail
- `context/` — 6 domain-expert corpus syntheses + the re-opened empirical record.
- `research/` — 6 web-research briefs (the new-source sweep).
- `_wf1_raw.json`, `_wf2_raw.json` — full structured outputs.

## Reusable tooling (created this session)
- `.claude/agents/domain-expert.md` — standing grounded domain-expert subagent.
- `.claude/workflows/neuro-ai-ground-and-brainstorm.js` — grounding + brainstorm pipeline.
- `.claude/workflows/neuro-ai-plan-and-grill.js` — plan + grill + harden pipeline.

## One-line bottom line
The consolidation **write** (`error_driven_learner.py:147-162`, not the
`consolidation.py` diagnostic) never laid down a recoverable role→target basin
(verdict by elimination). Recommended: **surgical-in-place**, gated by a cheap
write-then-read toy whose output is an evidence-backed four-way fork
(readout-fix / surgical / mild-rebuild / rebuild-from-Phase-1). Don't rebuild
before the toy.
