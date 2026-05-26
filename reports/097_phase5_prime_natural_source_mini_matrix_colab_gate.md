# Report 097: Phase 5' Natural-Source Mini-Matrix Colab Gate

**Date:** 2026-05-26
**Scope:** fixed Report 096 mini-matrix gate on Colab L4
**Status:** diagnostic gate completed

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This gate does
not measure that headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This gate does not
execute that graduation-control matrix.

**Last verified result:** Report 096 froze the broader cleaned natural-source
mini-matrix plan without running retrieval.

**Why this experiment now:** Report 096 explicitly allowed one fixed gate that
consumes its exact planned cells and reports top1, Wilson CI, `scene_tix`, and
`content_tix`. This is still a drill-down gate, not Phase 5 graduation
evidence.

## Command

Run on the pushed branch in Colab/Safari terminal:

```bash
rm -rf /content/Neuro-AI
git clone --branch codex/phase5-prime-broader-followup-scope --depth 1 \
  https://github.com/Dypatterson/Neuro-AI.git /content/Neuro-AI
cd /content/Neuro-AI
PYTHONPATH=src:. python \
  scripts/phase5_prime_natural_source_mini_matrix_gate.py \
  --device auto \
  --out /content/phase5_prime_natural_source_mini_matrix_gate_colab.json
```

Runtime:

```text
python 3.12.13
torch 2.10.0+cu128
cuda True
gpu NVIDIA L4
```

Output artifact:

```text
/content/phase5_prime_natural_source_mini_matrix_gate_colab.json
```

SHA-256:

```text
c6220f7ab8e4011190a7798fca3d4afa9e8d977abb83786372d23b812a2a3eb5
```

Shape:

```text
aggregates 32
raw 320
device cuda
```

## Result

Noise order is `0.00`, `0.05`, `0.10`, `0.15`.

| condition | top1 by noise | scene_tix by noise | content_tix by noise |
| --- | --- | --- | --- |
| candidate | `0.9518`, `0.9527`, `0.9529`, `0.9512` | `4869/5120`, `4874/5120`, `4875/5120`, `4867/5120` | `4873/5120`, `4878/5120`, `4879/5120`, `4870/5120` |
| random_role | `0.0000`, `0.0000`, `0.0000`, `0.0000` | `4869/5120`, `4874/5120`, `4875/5120`, `4867/5120` | `0/5120`, `0/5120`, `0/5120`, `0/5120` |
| deranged_role | `0.0002`, `0.0002`, `0.0004`, `0.0004` | `583/5120`, `576/5120`, `570/5120`, `556/5120` | `1/5120`, `1/5120`, `2/5120`, `2/5120` |
| fixedpoint_free_shuffled_role | `0.0010`, `0.0010`, `0.0010`, `0.0010` | `576/5120`, `571/5120`, `562/5120`, `543/5120` | `5/5120`, `5/5120`, `5/5120`, `5/5120` |
| content_cleanup_positive | `1.0000`, `1.0000`, `1.0000`, `1.0000` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` |
| bundle_positive | `1.0000`, `1.0000`, `1.0000`, `1.0000` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` |
| perfect_cue | `1.0000`, `1.0000`, `1.0000`, `1.0000` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` | `5120/5120`, `5120/5120`, `5120/5120`, `5120/5120` |
| no_scene_token_baseline | `0.5809`, `0.5820`, `0.5840`, `0.5834` | `2961/5120`, `2967/5120`, `2977/5120`, `2987/5120` | `2974/5120`, `2980/5120`, `2990/5120`, `2987/5120` |

The candidate remains high and stable across the cue-noise sweep. The
role-negative controls remain clean at the planned operating point, with
`random_role` at zero, `deranged_role` at or below `0.0004`, and
fixed-point-free shuffled role at `0.0010`.

The no-scene-token baseline remains substantially above role-negative controls
but far below the scene-token candidate. This confirms that source/content
structure alone carries signal, while the scene token supplies the large
additional scene-addressing gain.

## Interpretation

This completes the Report 096 fixed mini-matrix gate and preserves the core
project boundary:

- no Phase 5 `Delta E` headline run;
- no graduation claim;
- no M1 escalation;
- no M2 commitment;
- no full matrix;
- no adaptive source selection or metric-triggered fallback.

Mechanistically, the result supports the same fixed local story as Reports
093-096: a non-synthetic native provenance context source can address scene
bundles robustly enough for cleanup, while role-negative controls stay near
zero once the source/control protocol removes same-row duplicates, special
targets, and fixed-point shuffled-role opportunities.

## Verification

Local runner compile passed before the Colab run:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_natural_source_mini_matrix_gate.py
```

Local CPU smoke passed for one cell/seed:

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase5_prime_natural_source_mini_matrix_gate.py \
  --device cpu \
  --max-cells 1 \
  --seeds 17 \
  --out /private/tmp/phase5_prime_mini_matrix_gate_smoke.json
```

Printed smoke aggregate:

```text
candidate ... top1=0.9551 CI=[0.9335,0.9699] scene_tix=489/512 content_tix=489/512
```

The Colab terminal readback reported the artifact SHA and aggregate/raw counts
above. The raw Colab JSON was produced in `/content`; this report records the
hash and aggregate table, not a repo-committed copy of the raw Colab artifact.

## Next Step

Do not add another cleaned-protocol local drill-down. The next useful step is
to decide whether this fixed mini-matrix evidence is enough to plan an
integration precommit for the bundle-first structural-memory path, or whether
one more preflight-only artifact is needed to map how this gate would connect
to the Phase 5 `Delta E` headline without changing the current headline
criterion.
