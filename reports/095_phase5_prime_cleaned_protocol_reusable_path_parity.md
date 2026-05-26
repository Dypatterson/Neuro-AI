# Report 095: Phase 5' Cleaned Protocol Reusable-Path Parity Gate

**Date:** 2026-05-25
**Scope:** Report 093 parity gate from the reusable cleaned-protocol path
**Status:** byte-identical parity pass

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This parity run
does not measure that headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This parity run
does not execute that graduation-control matrix.

**Last verified result:** Report 094 integrated the Report 092 cleaned
source/control protocol into reusable Phase 5' code and verified exact selected
query-plan and numeric preflight parity.

**Why this experiment now:** `STATUS.md` allowed at most one Report 093
reusable-path parity gate before scoping out. This is a drill-down parity run,
not Phase 5 graduation evidence.

## Command

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase5_prime_natural_source_control_cleanup_gate.py \
  --device cpu \
  --out reports/phase5_prime_natural_source_control_cleanup_gate_reusable_path_parity.json
```

The generated parity JSON was then compared against the committed Report 093
artifact:

```bash
shasum -a 256 \
  reports/phase5_prime_natural_source_control_cleanup_gate_reusable_path_parity.json \
  reports/phase5_prime_natural_source_control_cleanup_gate.json

cmp -s \
  reports/phase5_prime_natural_source_control_cleanup_gate.json \
  reports/phase5_prime_natural_source_control_cleanup_gate_reusable_path_parity.json
```

## Result

Both files have the same SHA-256:

```text
54d95700df01a43bb3c5d28a5f72f3f80cf49f2dd9f848ba860da90bebea98b9
```

`cmp -s` exited cleanly. The reusable-path output is byte-identical to the
committed Report 093 JSON. The duplicate generated JSON was not committed.

Printed aggregate parity:

| condition | top1 | CI | LOO |
| --- | ---: | --- | --- |
| candidate | 0.9512 | [0.9449, 0.9567] | 0.9492-0.9555 |
| random_role | 0.0000 | [0.0000, 0.0007] | 0.0000-0.0000 |
| deranged_role | 0.0004 | [0.0001, 0.0014] | 0.0002-0.0004 |
| fixedpoint_free_shuffled_role | 0.0010 | [0.0004, 0.0023] | 0.0007-0.0011 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000-1.0000 |

## Interpretation

The cleaned natural-source protocol extraction is behavior-preserving at the
full Report 093 gate level. No further protocol-local drill-down is needed
before scoping the next broader work.

This result remains diagnostic/integration evidence only:

- no Phase 5 graduation claim;
- no Phase 5 `Delta E` headline run;
- no full all-controls matrix;
- no M2 commitment;
- no MQAR/bAbI headline pivot;
- no adaptive source selection or metric-triggered fallback.

## Next Step

Stop the cleaned-protocol drill-down chain. The next session should scope the
larger follow-up explicitly rather than adding another local parity diagnostic.
