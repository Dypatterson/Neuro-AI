---
date: 2026-05-24
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: spike-result
session-closes: P1-trace-schema-check
---

# Spike S1 — Replay-Trace Schema Check

Companion to the
[P1 role-bank dynamic-form spec](2026-05-24-phase5-p1-role-bank-dynamic-form.md)
and the
[Phase 5 rescue brainstorm](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md).
Closes the **substrate-validity gate Risk B** flagged by the reviewer-
agent audit of the P1 spec (2026-05-24).

## Question (from the P1 spec)

> Confirm the existing replay trace stores per-window encoder term lists,
> not just bundled output vectors. If the trace already carries
> encoder-time (atom, role) provenance, S1 closes Risk B. If not, scope
> the schema extension (estimated +30 LOC) and re-audit before P1 lands.

## Finding: **the trace does NOT carry encoder-time provenance.**

### `TrajectoryTrace` fields (the canonical trace dataclass)

[`src/energy_memory/phase4/trajectory.py:55-70`](../../src/energy_memory/phase4/trajectory.py:55):

```python
@dataclass
class TrajectoryTrace:
    query: "torch.Tensor"               # bundled vector — opaque post-bundle
    snapshots: List[TrajectorySnapshot] = field(default_factory=list)
    final_state: Optional["torch.Tensor"] = None
    final_top_score: float = 0.0
    final_top_index: Optional[int] = None
    converged: bool = False
    age: int = 0
```

`TrajectorySnapshot` (per-step co-activation pattern) carries
`top_k_indices`, `top_k_weights`, `entropy`, `energy` — all
*retrieval-time* quantities, none of which expose what (atom, role)
pairs went into the bundled query.

### Trace construction site

[`src/energy_memory/phase4/trajectory.py:263-270`](../../src/energy_memory/phase4/trajectory.py:263):

```python
trace = TrajectoryTrace(
    query=query.detach().clone(),       # bundled vector only
    snapshots=snapshots,
    final_state=state.detach().clone(),
    final_top_score=top_score,
    final_top_index=top_index,
    converged=converged,
)
```

The `query` tensor passed in is the *bundled* output of `encode_window`,
which has irreversibly summed the (position, atom)-bound terms. The
upstream (token_ids, positions) tuples never enter the trace.

### Where the provenance exists upstream

`encode_window` itself ([`phase2/encoding.py:29-33`](../../src/energy_memory/phase2/encoding.py:29))
*does* take the term list as input:

```python
def encode_window(substrate, positions, codebook, token_ids):
    terms = [substrate.bind(positions[index], codebook[token_id])
             for index, token_id in enumerate(token_ids)]
    return substrate.bundle(terms)
```

So the (atom_id = token_id, role_id = position_index) tuples are known
at every call site, but discarded immediately after bundling. The
callers I located (read-only audit):

| Call site | Context |
|---|---|
| `phase2/error_driven_learner.py:87, 130` | Phase 2 training; (positions, token_ids) live in caller scope |
| `phase2/reconstruction_learner.py:83, 130` | Phase 2 reconstruction; same |
| `phase5/ham_with_layer2.py:203` | Phase 5 K-branch settling; same |
| `phase5/ham_aggregator.py:109` | Phase 5 aggregator; same |
| `phase34/reencoding.py:59` | Phase 3↔4 re-encoding; same |
| `tests/test_phase4_replay_loop.py:83, 123` | Test harness; same |

In every case the term list is upstream of the bundle and could be
recorded with the call.

## What this means for P1

**Risk B does NOT close cleanly on existing infrastructure.** P1's
binding-count accumulation `c_{i,r}` requires per-window (atom, role)
term lists in the replay trace. They are not there.

**Schema extension required.** Two surgical additions:

1. **Extend `TrajectoryTrace`** with one optional field
   (~3 LOC in `trajectory.py:55`):
   ```python
   encoder_terms: Optional[List[Tuple[int, int]]] = None
   # List of (role_index, atom_id) pairs that fed the bundled query
   ```

2. **Extend `encode_window`** to optionally return the term list alongside
   the bundle, *or* expose a thin wrapper
   `encode_window_with_provenance` that returns `(bundle, term_list)`
   (~10 LOC in `phase2/encoding.py`). Backward-compatible — existing
   callers stay on `encode_window`; new P1-aware code paths use the
   provenance variant.

3. **Update Phase 4 consolidation + Phase 5 K-branch call sites** to
   pass the term list into the constructed `TrajectoryTrace`. Each call
   site needs ~3 LOC to plumb `(positions, token_ids)` → `encoder_terms`.
   ~6 call sites × 3 LOC = ~18 LOC.

**Total revised estimate: ~30 LOC for the schema extension** (close to
the spec's "+30 LOC" guess). Backward-compatible because the new field
is `Optional[…] = None` — existing trace consumers are untouched.

## Anti-homunculus check on the extension

The schema extension is **provenance plumbing**: passing through a
piece of data that the encoder already has in hand. It does not
introduce a new decision-making module. The (role, atom) tuple is a
substrate-time fact (what the encoder bound), not a metric-derived
categorization. ✅ PASS.

The risk to watch is in how downstream code *uses* the new field. If
any consolidation step writes `if role == r AND atom_active(i) > τ:`,
that's a controller — but that risk lives in the P1 implementation, not
in this schema extension. The schema extension only makes the data
available.

## Verdict for P1 unblocking

| Original P1 spec gate | Status |
|---|---|
| Risk B (encoder-time provenance available) | ❌ FAIL on existing infra — extension required |
| Risk A (Lyapunov of `diag(w)·X` MHN) | ⏳ pending Spike S2 (numerical check) |

P1 is **not yet unblocked**. Before P1 lands:

1. Schema extension PR (~30 LOC + tests) lands first.
2. Spike S2 runs and passes.
3. Then P1 main implementation (~120 LOC per the spec).

Sequence cost: ~½ day schema extension + ½ day S2 + 1 week P1 main =
~2 working weeks once P1 is selected as the Tier-1 commit.

## Linked notes

- [P1 role-bank dynamic-form spec](2026-05-24-phase5-p1-role-bank-dynamic-form.md)
- [Phase 5 rescue brainstorm](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)
- Sibling spike: [D3 Lyapunov analytical pass](2026-05-24-spike-D3-lyapunov-analytical.md)
