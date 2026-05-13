# Session summary — 2026-05-12 / 2026-05-13

Spans a single working session across midnight. Entry point was a
codebase + results review and "identify and plan next steps" — the
review surfaced the Phase 3+4 re-encode regression as the Tier 1
priority, and the session pursued it through diagnosis, fix design,
implementation, validation, and follow-up Phase 4 work.

## What we set out to do

Dylan's commit 3aa23fc flagged that the online codebook updater
collapses retrieval to chance by ~2000 cues from random init. The
session goal: localize the cause, design a fix, validate the fix, and
unblock the Phase 4 work that the regression had been gating.

## What's settled

### 1. The original online error-driven updater is structurally unsafe

[`OnlineCodebookUpdater`](src/energy_memory/phase34/online_codebook.py)
runs error-driven contrastive pull/push inside a streaming cue loop.
Tested across two qualitatively different starting points:

| Starting codebook | Final top-1 (3500 cues) | Behaviour |
|---|---:|---|
| Random init (exp22, exp23) | 0.015 (= chance) | Catastrophic collapse |
| Pretrained Phase 3c (exp24b) | 0.020 (from 0.107) | 80% relative degradation |

The instability isn't a "random init" artefact. Pull dominates push by
about 20× per event, and pull pulls every target's codebook entry
toward approximately the same position-induced direction —
`avg(slot_query | target=t) ≈ rotated conj(position_mask)` — so
discrimination erodes regardless of how much structure the codebook
starts with. Full diagnosis:
[reports/phase34_stable_v2/findings.md](reports/phase34_stable_v2/findings.md).

### 2. The architecturally-correct runtime pathway is Hebbian, not error-driven

[`phase-3-deep-dive.md:140-145`](notes/emergent-codebook/phase-3-deep-dive.md)
specifies the runtime-vs-training asymmetry: batch error-driven for
codebook acquisition, online Hebbian reinforcement for runtime. The
session implemented the missing Hebbian pathway as
[`HebbianOnlineCodebookUpdater`](src/energy_memory/phase34/hebbian_online.py)
and validated it.

Result on the pretrained Phase 3c codebook (exp24b):

| Condition | top-1 | top-K | cap_t05 |
|---|---:|---:|---:|
| baseline (no online updates) | 0.107 | 0.322 | 0.015 |
| Hebbian (lr=0.01, t=0.5) | **0.107** | **0.322** | 0.015 |
| Error-driven (lr=0.1) | 0.020 | 0.205 | 0.015 |

Hebbian preserves the codebook exactly. Error-driven destroys it.
Full report: [reports/phase34_hebbian/findings.md](reports/phase34_hebbian/findings.md).

### 3. Hebbian unblocks Phase 4 replay

[`experiments/19_phase34_integrated.py`](experiments/19_phase34_integrated.py)
was modified to support `--updater-kind hebbian` and re-run with the
pretrained codebook. The result that had been hidden:

| Phase 4 condition | Original (broken error-driven) | Fixed (Hebbian + pretrained) |
|---|---:|---:|
| top-1 at 2000 cues | 0.022 (collapsed) | 0.124 |
| Replay candidates discovered | 2 | **126** |

Phase 4's replay-and-discover channel was being starved by the
destabilized codebook. With a stable substrate, it produces 63× more
discovery activity. Report:
[reports/phase34_hebbian_integrated/findings.md](reports/phase34_hebbian_integrated/findings.md).

### 4. The β regime split

Two distinct β regimes are in active use in the project, undocumented
until now:

- **β=10** — Phase 4 replay's design regime. Hopfield retrieval has
  high entropy across settling, the engagement gate
  (`entropy × (1 − resolution)`) fires, traces enter the replay store.
- **β=30** — HAM aggregation's design regime. Sharp prototype-mode
  retrieval. Phase 4's engagement gate effectively doesn't fire here.

The exp18 Phase 4 drift run at β=30 produced 0 candidates; at β=10 it
produced 33. The Phase 4 design's prior `phase4_unified_b10` directory
name encoded this dependency but the body of the design doc did not.
Updated in [notes/emergent-codebook/phase-4-unified-design.md](notes/emergent-codebook/phase-4-unified-design.md).

## What's still open

### Phase 4 headline metric not yet positive

The headline test "before vs. after N cycles with active drift" with
the default Phase 4 parameters:

| Setting | Result |
|---|---|
| drift=0.15, β=30, rt=0.85 | 0 candidates — gate doesn't fire |
| drift=0.15, β=10, rt=0.7 | 33 candidates — discovery fires, headline metrics tied/slightly worse than baseline |

Phase 4 mechanism is mechanically alive but headline-inert at default
settings. The 33 discovered patterns are 0.8% of the 4129-pattern
landscape — too sparse to move aggregate metrics. Two unused knobs that
could change this:

- `death_window=10000` is longer than any experiment we ran. Stale
  patterns never get pruned. Shorter `death_window` would let replay's
  new patterns *replace* stale ones rather than augment.
- `reencode_every` is 0 in exp18. The re-encode mechanism in
  [`phase34/reencoding.py`](src/energy_memory/phase34/reencoding.py)
  exists but is not invoked. Re-encoding stored patterns through the
  current codebook is the most direct Phase 4 rescue path.

Report: [reports/phase4_drift_headline_b10/findings.md](reports/phase4_drift_headline_b10/findings.md).

### Hebbian is preservation-only at current settings

Threshold sweep (single-scale W=2 with pretrained codebook):

| success_threshold | Hebbian fire rate | Top-1 Δ | drift |
|---:|---:|---:|---:|
| 0.5 | 0.06% | 0 | 0.0000 |
| 0.3 | 1.44% | 0 | 0.0000 |

In the multi-scale W=8 exp19 setup, Hebbian at t=0.5 *did* produce a
modest positive signal (topk +0.013 in condition B, cap_t05 +0.009 in
condition C) — Hebbian becomes visible at higher scale dimensionality
where each fire updates more atoms. But it remains a small effect.

The Hebbian path as currently designed is "verified-safe preservation
with mild positive signal in multi-scale settings." It isn't yet the
load-bearing continuous-learning mechanism the architecture eventually
needs. Open question: what regime of (lr, threshold, scale) makes
Hebbian actually drive learning without re-introducing instability?

### Tier 1 items still pending

From the original review (still unblocked but not yet pursued this
session):

- **5-seed HAM arithmetic validation** — Phase 5 HAM report flagged
  this as required before HAM replaces summed scores as default.
- **Layer-2 admission tightening** — Phase 5 layer-2 validation flagged
  recurrence-across-related-traces as the next admission rule.
- **PAM/Dury temporal-co-occurrence architectural question** — open
  Phase 5/6 design question from the 2026-05-11 papers note.

## What's saved in the repo

### New code

- [src/energy_memory/phase34/hebbian_online.py](src/energy_memory/phase34/hebbian_online.py) — Hebbian online updater
- [src/energy_memory/phase34/stable_online_codebook.py](src/energy_memory/phase34/stable_online_codebook.py) — V1 (repulsion) and V2 (mean-subtract + repulsion + low-lr) updaters; kept for the diagnostic trail, not the recommended path
- [experiments/22_codebook_health_diagnostic.py](experiments/22_codebook_health_diagnostic.py) — codebook health diagnostic over 6 conditions
- [experiments/22b_summarize_diagnostic.py](experiments/22b_summarize_diagnostic.py) — summarizer for exp22 JSON
- [experiments/23_stable_updater_diagnostic.py](experiments/23_stable_updater_diagnostic.py) — V1/V2 head-to-head test
- [experiments/24_hebbian_diagnostic.py](experiments/24_hebbian_diagnostic.py) — Hebbian vs error-driven vs baseline on pretrained codebook

### Modified code

- [experiments/19_phase34_integrated.py](experiments/19_phase34_integrated.py) — added `--updater-kind` flag, Hebbian path is the default

### Notes updated

- [notes/emergent-codebook/phase-3-deep-dive.md](notes/emergent-codebook/phase-3-deep-dive.md) — added "online error-driven contrastive updates collapse retrieval" row to the failure-mode table
- [notes/emergent-codebook/phase-4-unified-design.md](notes/emergent-codebook/phase-4-unified-design.md) — clarified what "codebook drift" means (replay re-encode + batch retrains, not online error-driven)

### Reports

- [reports/phase34_diagnostic/](reports/phase34_diagnostic/) — exp22 raw results
- [reports/phase34_stable_v2/findings.md](reports/phase34_stable_v2/findings.md) — collapse mechanism diagnosis
- [reports/phase34_hebbian/findings.md](reports/phase34_hebbian/findings.md) — Hebbian validation, error-driven safety failure
- [reports/phase34_hebbian_integrated/findings.md](reports/phase34_hebbian_integrated/findings.md) — exp19 with Hebbian + pretrained codebook
- [reports/phase34_hebbian_t03/](reports/phase34_hebbian_t03/) — Hebbian threshold sweep at 0.3
- [reports/phase4_drift_headline/](reports/phase4_drift_headline/) — Phase 4 drift @ β=30 (gate doesn't fire)
- [reports/phase4_drift_headline_b10/findings.md](reports/phase4_drift_headline_b10/findings.md) — Phase 4 drift @ β=10 (gate fires, headline not rescued)

## Recommended starting points for next session

### Highest leverage

1. **Phase 4 tuning: `death_window=1000` + `reencode_every=100`.**
   Both knobs already exist; just not used by exp18. This is the
   shortest path to a positive Phase 4 headline metric. ~1 hour.

2. **5-seed HAM arithmetic validation.** Phase 5 said HAM doesn't
   replace summed scores until it survives a Phase-4-style 5-seed
   validation. Independent of everything else. ~2 hours.

### Important but slower

3. **Multi-scale Hebbian threshold sweep.** The single-scale sweep was
   inconclusive. Multi-scale (exp19 W=8) is where Hebbian becomes
   visible. Test t ∈ {0.3, 0.4} there to see if there's a regime
   where Hebbian provides continued learning rather than just
   preservation.

4. **Layer-2 admission tightening** per the Phase 5 layer-2 validation
   recommendation: require recurrence-across-related-traces, not
   single-replay resolution.

### Conceptual, can wait

5. **PAM/Dury temporal-co-occurrence architectural question**
   (Phase 5/6 design pass).

6. **Diagnostic-actuator dynamic-form session** (from
   [2026-05-09 note](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md)).
   Still pending. Most relevant now that several thresholds and
   gating constants in the codebase (`store_threshold`,
   `resolve_threshold`, `success_threshold`, `death_threshold`,
   `repulsion_threshold`) are doing supervisor-shaped work.

## One-line takeaway

The Phase 3+4 regression resolved cleanly — the online error-driven
updater is unsafe on any init, Hebbian is its architecturally-correct
replacement, and with the substrate stable, Phase 4 replay finally
fires (63× more discovery activity than before). Phase 4's headline
metric is now the open question, with `death_window` and `reencode`
as the prescribed-but-untested knobs.
