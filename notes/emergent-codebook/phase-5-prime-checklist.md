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
| A2 | Extend to `K_roles=16` | partial | Report 075 hard-cell diagnostic covers `K=16,N=512`; full grid still open |
| A3 | Report `top1`, Wilson CI, `scene_tix`, `content_tix`, entropy, and margin for every cell | partial | Report 075 records pasted top1/CI/scene/content for hard cells; full matrix still open |

## B. Cue Noise

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| B1 | Sweep cue noise over at least `{0.0, 0.05, 0.10, 0.15}` | partial | Report 075 records pasted candidate-grid extremes; full matched-control sweep still open |
| B2 | Report whether failures are scene-ID failures or content-cleanup failures | partial | Report 075 hard-cell split: scene-MHN is the bottleneck; full grid still open |

## C. Scene Token / Identity Robustness

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| C1 | Run no-scene-token baseline | partial | Report 075 records no-anchor / zero-weight skewed baselines near `~0.23` |
| C2 | Run optional scene-token condition | partial | Report 075 records random-anchor sweep; Report 076 records full-scene `context_bundle`; Reports 077/078 record strict partial-context and context-size diagnostics |
| C3 | Confirm scene tokens do not inflate top1 through identity leakage | partial | Reports 077/078 random-role and deranged-role controls stay near zero despite high scene_tix; shuffled-role residual remains bounded/dirty |

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
| G1 | n_seeds >= 10 for any verification claim | partial | Reports 075-078 pasted or attached diagnostics are n=10 but not verification claims |
| G2 | Confidence intervals reported on headline | partial | Reports 075-078 record Wilson CIs for hard cells |
| G3 | Leave-one-seed-out sensitivity reported | open | not yet run |
| G4 | Controls E1-E6 run on the same test set | partial | Reports 075-078 hard-cell controls are matched where available; full matrix still open |
| G5 | Report explicitly says no graduation claim unless all gates pass | done | Reports 069-078 explicitly preserve no-graduation boundary |

## H. Anti-Homunculus Discipline

| # | Prohibition | Status |
| -- | --- | :---: |
| H1 | No controller chooses between storage/retrieval routes | binding |
| H2 | No metric-triggered replay sampler switching | binding |
| H3 | No best-of-N condition selection as graduation evidence | binding |
| H4 | Diagnostics are passive logs, not execution gates | binding |
