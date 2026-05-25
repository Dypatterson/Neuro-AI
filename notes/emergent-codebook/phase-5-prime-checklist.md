# Phase 5′ Checklist

Companion to [phase-5-prime-bundle-first-design.md](phase-5-prime-bundle-first-design.md).
This is a precommit checklist for bundle-first structural memory plus
range-shaped replay integration. It is not a Phase 5 graduation checklist.

Verified status requires n_seeds >= 10 with confidence intervals and matched
controls. n=3 MQAR runs are diagnostic gates only.

## A. MQAR Reproduction

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| A1 | Reproduce Report 067 bundle-first multi-role MQAR for `K_roles in {2,4,8}` and `N in {16,32,64,128,256,512}` | partial | Report 067, n=3 diagnostic |
| A2 | Extend to `K_roles=16` | partial | Report 079 covers fixed observed-prefix `K=16` across `N={128,256,512}` and `noise={0.0,0.10,0.15}`; Reports 080-081 cover available-prefix and trace-backed query-side context at the hard K=16 cell; Report 082 covers the less-synthetic replay-observed hard cell; full matrix still open |
| A3 | Report `top1`, Wilson CI, `scene_tix`, `content_tix`, entropy, and margin for every cell | partial | Reports 079-081 raw JSON includes top1/CI/scene/content/entropy/margins for fixed observed-prefix, available-prefix, and trace-backed diagnostics; Report 082 aggregate captures top1/CI/scene/content, with raw per-seed entropy/margins still pending |

## B. Cue Noise

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| B1 | Sweep cue noise over at least `{0.0, 0.05, 0.10, 0.15}` | partial | Report 079 covers fixed observed-prefix matched controls at `{0.0,0.10,0.15}`; `0.05` and full matrix still open |
| B2 | Report whether failures are scene-ID failures or content-cleanup failures | partial | Reports 075-079 show scene/context completion is the bottleneck; Report 079 candidate scene/content rates track top1 |

## C. Scene Token / Identity Robustness

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| C1 | Run no-scene-token baseline | partial | Report 075 records no-anchor / zero-weight skewed baselines near `~0.23` |
| C2 | Run optional scene-token condition | partial | Report 075 records random-anchor sweep; Report 076 records full-scene `context_bundle`; Reports 077-087 record strict partial-context, context-size, fixed observed-prefix, available-prefix, trace-backed, replay-observed, matched v2, and native-provenance preflight diagnostics |
| C3 | Confirm scene tokens do not inflate top1 through identity leakage | partial | Reports 077-086 random-role and deranged-role controls stay near zero despite high scene/context rates; shuffled-role residual remains bounded/dirty |
| C4 | Replace generated-scene trace construction with a replay-derived, trajectory-derived, learned, or naturally observed passive context trace before any full matrix | blocked | Reports 082-083 run replay-observed passive hard cells and are positive vs controls but degraded; Report 084 localizes a role-universe/learned-geometry mismatch; Report 085 passes the matched 4-role v2 preflight; Report 086 partially recovers the passive source but leaves learned-token context weak; Report 087 passes the synthetic controlled native-provenance preflight. Next allowed work is the fixed native-provenance candidate/control gate, not a full matrix |

## D. Co-Occurrence Statistics

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| D1 | Uniform filler sampling baseline | partial | Report 067 diagnostic |
| D2 | Skewed/natural co-occurrence condition if cheap | partial | Report 075 records skewed hard-cell and candidate-grid diagnostics |
| D3 | Report whether skew changes scene-MHN margins or content cleanup | partial | Report 075 records scene/content split for skewed hard cells; natural co-occurrence open |

## E. Controls

| # | Control | Required behavior | Status |
| -- | --- | --- | :---: |
| E1 | Random-role control | removes or bounds structural recall | partial |
| E2 | Shuffled-role control | removes role-specific structure | partial |
| E3 | Perfect-cue control | verifies storage and cleanup ceiling | partial |
| E4 | Bundle positive control | verifies algebraic bundle/unbind capacity | partial |
| E5 | Content cleanup positive control | verifies content-MHN cleanup independent of scene ID | partial |
| E6 | Deranged-role control | removes role-specific structure with no fixed role mappings | partial |

Controls are matched to candidate settings and reported alongside them. They
are diagnostics, not a route selector.

## F. Range-Shaped Replay Integration

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| F1 | `RangeShapedReplaySampler` lives under `src/energy_memory/phase4/` | done | `range_shaped_replay.py` |
| F2 | `sample_pairs()` returns `(role, atom, trace_idx)` with `None` for unbacked pairs | done | unit tests |
| F3 | Missing `encoder_terms` are ignored safely | done | unit tests |
| F4 | Backing-trace choice is priority-weighted | done | unit tests |
| F5 | Optional atom-support smoothing expands reachable atom support | done | unit tests |
| F6 | Single-binding rebind returns `TrajectoryTrace.encoder_terms` | done | unit tests |
| F7 | Window-preserving rebind returns `TrajectoryTrace.encoder_terms` | done | unit tests |
| F8 | `UnifiedReplayMemory` uses static config: `standard` or `range_shaped` | done | unit tests |
| F9 | Run downstream Phase 4 consolidation comparison | open | no downstream ΔE evidence yet |

## G. Verification Standard

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| G1 | n_seeds >= 10 for any verification claim | partial | Reports 075-087 pasted, attached, committed, or transcribed diagnostics are n=10 where applicable but not verification claims |
| G2 | Confidence intervals reported on headline | partial | Reports 075-086 record Wilson CIs for diagnostic cells where candidate/control retrieval is run; Report 087 is preflight-only |
| G3 | Leave-one-seed-out sensitivity reported | partial | Reports 082-083 and 086 include candidate leave-one-seed-out sensitivity for replay-observed context-source gates |
| G4 | Controls E1-E6 run on the same test set | partial | Reports 075-086 controls are matched where available; Report 087 fixes source rows/query schedule before the next control run; full matrix still open |
| G5 | Report explicitly says no graduation claim unless all gates pass | done | Reports 069-087 explicitly preserve no-graduation boundary |

## H. Anti-Homunculus Discipline

| # | Prohibition | Status |
| -- | --- | :---: |
| H1 | No controller chooses between storage/retrieval routes | binding |
| H2 | No metric-triggered replay sampler switching | binding |
| H3 | No best-of-N condition selection as graduation evidence | binding |
| H4 | Diagnostics are passive logs, not execution gates | binding |
