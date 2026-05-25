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
| A2 | Extend to `K_roles=16` | open | `experiments/44_phase5_prime_bundle_first.py` supports it |
| A3 | Report `top1`, Wilson CI, `scene_tix`, `content_tix`, entropy, and margin for every cell | open | harness support required |

## B. Cue Noise

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| B1 | Sweep cue noise over at least `{0.0, 0.05, 0.10, 0.15}` | open | not yet run |
| B2 | Report whether failures are scene-ID failures or content-cleanup failures | open | requires `scene_tix` / `content_tix` split |

## C. Scene Token / Identity Robustness

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| C1 | Run no-scene-token baseline | open | not yet run |
| C2 | Run optional scene-token condition | open | harness support required |
| C3 | Confirm scene tokens do not inflate top1 through identity leakage | open | compare with shuffled-role and random-role controls |

## D. Co-Occurrence Statistics

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| D1 | Uniform filler sampling baseline | partial | Report 067 diagnostic |
| D2 | Skewed/natural co-occurrence condition if cheap | open | harness support required |
| D3 | Report whether skew changes scene-MHN margins or content cleanup | open | not yet run |

## E. Controls

| # | Control | Required behavior | Status |
| -- | --- | --- | :---: |
| E1 | Random-role control | removes or bounds structural recall | open |
| E2 | Shuffled-role control | removes role-specific structure | open |
| E3 | Perfect-cue control | verifies storage and cleanup ceiling | open |
| E4 | Bundle positive control | verifies algebraic bundle/unbind capacity | open |
| E5 | Content cleanup positive control | verifies content-MHN cleanup independent of scene ID | open |

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
| G1 | n_seeds >= 10 for any verification claim | open | not yet run |
| G2 | Confidence intervals reported on headline | open | harness supports CI; full run not yet done |
| G3 | Leave-one-seed-out sensitivity reported | open | not yet run |
| G4 | Controls E1-E5 run on the same test set | open | not yet run |
| G5 | Report explicitly says no graduation claim unless all gates pass | open | required for report 069 and later |

## H. Anti-Homunculus Discipline

| # | Prohibition | Status |
| -- | --- | :---: |
| H1 | No controller chooses between storage/retrieval routes | binding |
| H2 | No metric-triggered replay sampler switching | binding |
| H3 | No best-of-N condition selection as graduation evidence | binding |
| H4 | Diagnostics are passive logs, not execution gates | binding |
