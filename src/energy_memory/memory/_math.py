"""Shared pure-Python math utilities for memory modules."""

from __future__ import annotations

import math
from typing import List, Sequence


def softmax(values: Sequence[float]) -> List[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def logsumexp(values: Sequence[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def normalized_entropy(weights: Sequence[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    raw = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)
    return raw / math.log(len(weights))
