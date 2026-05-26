# Report 094: Phase 5' Cleaned Natural-Source Protocol Integration

**Date:** 2026-05-25
**Scope:** reusable code integration and parity tests only
**Status:** passed focused and full-suite verification

## Summary

Report 093 allowed an integration precommit, not another gate. This slice moves
the cleaned natural-source query/control protocol from report-local helpers into
reusable Phase 5' code:

`src/energy_memory/phase5/natural_source_protocol.py`

The implementation preserves the Report 092 protocol surface:

- non-special target exclusion;
- same-row duplicate target exclusion;
- target-frequency caps;
- deterministic selected-query plans;
- random-role, deranged-role, and fixed-point-free shuffled-role exact
  opportunity accounting;
- cleanup preflight SHA checks;
- selected protocol payload and source-plan plumbing for the gate script.

## Code Surface

Added:

- `src/energy_memory/phase5/natural_source_protocol.py`
- `tests/test_phase5_natural_source_protocol.py`

Updated:

- `src/energy_memory/phase5/__init__.py`
- `scripts/phase5_prime_natural_source_control_cleanup_preflight.py`
- `scripts/phase5_prime_natural_source_control_cleanup_gate.py`
- `tests/test_phase5_prime_natural_source_control_cleanup_preflight.py`

The report scripts now import the reusable helpers instead of carrying their
own protocol implementations. File loading, vocab construction, and artifact
writing remain in scripts.

## Verification

Compile check:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  src/energy_memory/phase5/natural_source_protocol.py \
  scripts/phase5_prime_natural_source_control_cleanup_preflight.py \
  scripts/phase5_prime_natural_source_control_cleanup_gate.py \
  tests/test_phase5_natural_source_protocol.py \
  tests/test_phase5_prime_natural_source_control_cleanup_preflight.py \
  tests/test_phase5_prime_natural_source_control_cleanup_gate.py
```

Focused tests:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_natural_source_protocol \
  tests.test_phase5_prime_natural_source_control_cleanup_preflight \
  tests.test_phase5_prime_natural_source_control_cleanup_gate -v
```

Result:

```text
Ran 9 tests in 0.511s

OK
```

Full suite:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

Result:

```text
Ran 377 tests in 1.086s

OK
```

The new parity test rebuilds the Report 092 `non_special_unique_target_freq_le_32`
protocol from the committed source artifact and checks exact equality for:

- `pass_criteria`;
- aggregate numeric support statistics;
- `selected_query_plan_by_seed`.

Additional manual parity check:

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase5_prime_natural_source_control_cleanup_preflight.py \
  --out /private/tmp/phase5_cleanup_preflight_refactor_check.json
diff -u \
  <(jq 'walk(if type == "object" then del(.token) else . end)' \
    reports/phase5_prime_natural_source_control_cleanup_preflight.json) \
  <(jq 'walk(if type == "object" then del(.token) else . end)' \
    /private/tmp/phase5_cleanup_preflight_refactor_check.json)
```

The normalized diff exits cleanly. The raw JSON is not byte-identical only
because `top_target_atoms[].token` labels are presentation fields derived from
the live repo-sample vocabulary, and that vocabulary changes as repository docs
change. Atom IDs, counts, selected plans, and numeric support diagnostics match.

## Boundary

No candidate/control gate was run. This report does not claim Phase 5
graduation, does not run the Phase 5 headline metric, does not start M2, and
does not authorize a full matrix.

The next allowed experimental step is at most a Report 093 parity gate from the
reusable path, with the same source rows, selected query plan, seeds, and five
conditions.
