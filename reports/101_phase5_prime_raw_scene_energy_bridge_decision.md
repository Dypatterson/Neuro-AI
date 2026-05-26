# Report 101: Phase 5' Raw Scene-Energy Bridge Decision

**Date:** 2026-05-26
**Scope:** decision/precommit for the bundle-first bridge energy landscape
**Status:** decision recorded; next step is controls preflight only

## Decision

The first bundle-first bridge baseline is:

```text
raw_scene_energy_v0
score_bias = None
energy readout = raw scene-MHN energy from the existing run_branched_retrieval path
headline form = Delta E = E_content-prior - E_role-prior
```

This preserves the existing Phase 5 headline form. It does not claim that the
original Step-3-weighted Phase 5 landscape is obsolete. It says only that the
first bundle-first scene bridge must use the only grounded scene-level energy
currently implemented.

## Rationale

Report 100 showed that fixed bundle-first scene states can enter the existing
content-prior-vs-role-prior branching path and produce finite paired probe
energies.

The unresolved question was whether to invent a scene-level Step-3
`score_bias`. Do not do that yet.

Reasons:

1. The current Step-3 bias is defined over the Phase 4 consolidation atom
   landscape. Bundle-first scene states are constructed scene bundles, not
   Phase 4 atoms with an audited per-scene consolidation strength.
2. A scene-level `score_bias` would need a fixed, non-adaptive mapping from
   scene bundles to a legitimate scene/consolidation statistic. That mapping
   does not exist yet.
3. Deriving a bias from probe outcomes, candidate success, scene margins, or
   other diagnostics would create a route-selection or metric-feedback risk.
4. `score_bias=None` is falsifiable: required controls can still test whether
   content/role priors, random priors, K, gamma, and schema-store source matter.

## Non-Decisions

This report does not decide:

- that raw scene energy is the final Phase 5 landscape;
- that a scene-level Step-3 bias is invalid in principle;
- beta, gamma, K, surprise-branch, or seed-scale settings for a headline run;
- any n>=3 or n>=10 experiment;
- any graduation criterion change.

If a future scene-level Step-3 mapping is proposed, it must be precommitted as a
fixed mechanism before any evidence run. It must also include controls showing
that the bias is not derived from target labels, probe performance, or
diagnostic-triggered route selection.

## Binding For Next Implementation

The next implementation step is a preflight-only controls planner for
`raw_scene_energy_v0`.

It may build static plans and validate interfaces for:

- content-prior vs role-prior paired `Delta E`;
- random-schema branches;
- K=1;
- no-prior (`gamma=0`);
- no-schema-store;
- fixed source/query manifests and SHA anchors;
- exact device-independent config serialization.

It must not run retrieval, top1, headline, n=3, n=10, full-matrix, M1
escalation, M2, or graduation evidence.

The preflight must explicitly record any config inherited from Report 100
(`beta=30`, `gamma=0.5`, `k_main=4`, `include_surprise_branch=false`) and flag
any mismatch with the legacy Phase 5 headline defaults before a retrieval run is
allowed.

## Required Stop Conditions

Stop before a retrieval run if any of these are true:

1. The planner cannot express all required controls against the same fixed query
   set.
2. The no-schema-store control requires changing the headline metric rather than
   only the schema/prior source.
3. The bridge needs a scene-level Step-3 bias after all.
4. The config would silently drift from the existing Phase 5 headline without a
   documented precommit.
5. Any result would be interpreted as Phase 5 graduation or as replacing the
   required `Delta E` headline.

## Boundary

This report authorizes only the controls preflight planner. It does not
authorize or claim:

- a candidate/control gate;
- a retrieval run;
- M1 escalation;
- M2 implementation or run;
- a full matrix;
- a Phase 5 `Delta E` headline run;
- Phase 5 graduation;
- a top1-based graduation metric.

## Anti-Homunculus Check

Pass. The decision selects the currently grounded raw scene-MHN energy
landscape and rejects an ungrounded adaptive or diagnostic-derived bias. The
next step remains static preflight planning only.

## Next Step

Implement a preflight-only raw-scene bridge controls planner. It should output a
JSON artifact and report that prove the required controls can be represented
without running them.
