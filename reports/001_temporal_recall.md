# MVP 0 Temporal Recall

This report was generated manually from:

```bash
PYTHONPATH=src python3 experiments/synthetic_temporal_recall.py
```

The first experiment checks whether the kernel can retrieve temporally adjacent
items from a synthetic experience stream, then compares against a temporal
shuffle control that preserves the same item vectors but breaks order.

## Initial Result

Configuration:

- `dim = 512`
- `window = 2`
- `beta = 8.0`
- `seed = 7`

Result:

| Condition | Mean temporal Recall@4 |
|---|---:|
| Ordered stream | 1.000 |
| Temporal shuffle | 0.388 |
| Delta | 0.613 |

Interpretation: the MVP 0 temporal association channel is recovering lived
neighbors from the ordered experience stream, and much of that signal disappears
when order is shuffled. This is the first pulse check for the architecture.

Example ordered retrievals:

| Query | Expected temporal neighbors | Retrieved top items |
|---|---|---|
| `stair` | `doctor`, `ink`, `slip`, `window` | `window`, `doctor`, `ink`, `slip` |
| `slip` | `bandage`, `doctor`, `stair`, `window` | `stair`, `window`, `doctor`, `bandage` |
| `rain` | `bandage`, `crutch`, `letter`, `windowpane` | `windowpane`, `crutch`, `bandage`, `letter` |
| `letter` | `lamp`, `rain`, `silence`, `windowpane` | `windowpane`, `silence`, `lamp`, `rain` |

## Caveat

This is intentionally small and synthetic. The next experiment should introduce
content-similarity distractors so the kernel has to distinguish temporal
association from ordinary nearest-neighbor retrieval.

## Follow-Up: Similarity Distractors

Command:

```bash
PYTHONPATH=src python3 experiments/content_vs_temporal_distractors.py
```

Configuration:

- `dim = 512`
- `window = 2`
- `beta = 100.0`
- `family_noise = 0.18`
- `seed = 11`

Setup: `stair`, `ladder`, `ramp`, `step`, and `escalator` are intentionally
nearby in FHRR space. Only `stair` occurs next to `slip`, `doctor`, `table`,
and `candle` in the experience stream.

Result:

| Method | Temporal-neighbor Recall@4 |
|---|---:|
| Content nearest-neighbor | 0.000 |
| Temporal association memory | 1.000 |
| Advantage | 1.000 |

Content nearest-neighbor retrieved only the intentionally similar mobility
family distractors: `ramp`, `escalator`, `ladder`, `step`.

Temporal association retrieved the lived neighbors: `doctor`, `table`, `slip`,
`candle`.

Interpretation: the temporal channel can separate lived association from
content similarity in this controlled setting. This directly tests the
architecture's claim that "what happened near this" is a different retrieval
axis from "what looks like this."
