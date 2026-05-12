"""Fourier Holographic Reduced Representation substrate.

This is a dependency-free reference implementation. It favors clarity and
determinism over speed; the same interface can later be backed by Torch/MPS.
"""

from __future__ import annotations

import cmath
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

Vector = Tuple[complex, ...]
T = TypeVar("T")


class FHRR:
    """Unit-modulus complex vector algebra for binding and bundling."""

    def __init__(self, dim: int = 512, seed: Optional[int] = None):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.rng = random.Random(seed)

    def random_vector(self) -> Vector:
        """Create one random unit-modulus complex hypervector."""
        return tuple(cmath.exp(1j * self.rng.random() * 2.0 * math.pi) for _ in range(self.dim))

    def random_vectors(self, count: int) -> List[Vector]:
        return [self.random_vector() for _ in range(count)]

    def perturb(self, vector: Vector, noise: float = 0.15) -> Vector:
        """Create a nearby vector by adding small phase noise per component."""
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        self._check_dim(vector)
        return tuple(value * cmath.exp(1j * self.rng.gauss(0.0, noise)) for value in vector)

    def bind(self, left: Vector, right: Vector) -> Vector:
        self._check_same_dim(left, right)
        return tuple(a * b for a, b in zip(left, right))

    def unbind(self, bound: Vector, role: Vector) -> Vector:
        self._check_same_dim(bound, role)
        return tuple(a * b.conjugate() for a, b in zip(bound, role))

    def inverse(self, vector: Vector) -> Vector:
        self._check_dim(vector)
        return tuple(v.conjugate() for v in vector)

    def normalize(self, vector: Sequence[complex]) -> Vector:
        """Project each component back to the complex unit circle."""
        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(vector)}")
        out = []
        for value in vector:
            mag = abs(value)
            out.append(1.0 + 0.0j if mag == 0.0 else value / mag)
        return tuple(out)

    def bundle(self, vectors: Iterable[Vector]) -> Vector:
        vectors = list(vectors)
        if not vectors:
            raise ValueError("cannot bundle zero vectors")
        for vector in vectors:
            self._check_dim(vector)
        sums = [0.0 + 0.0j for _ in range(self.dim)]
        for vector in vectors:
            for i, value in enumerate(vector):
                sums[i] += value
        return self.normalize(sums)

    def weighted_bundle(self, vectors: Sequence[Vector], weights: Sequence[float]) -> Vector:
        if len(vectors) != len(weights):
            raise ValueError("vectors and weights must have the same length")
        if not vectors:
            raise ValueError("cannot bundle zero vectors")
        for vector in vectors:
            self._check_dim(vector)
        sums = [0.0 + 0.0j for _ in range(self.dim)]
        for vector, weight in zip(vectors, weights):
            for i, value in enumerate(vector):
                sums[i] += weight * value
        return self.normalize(sums)

    def similarity(self, left: Vector, right: Vector) -> float:
        """Cosine-like similarity for unit complex vectors in [-1, 1]."""
        self._check_same_dim(left, right)
        score = sum((a.conjugate() * b).real for a, b in zip(left, right)) / self.dim
        return max(-1.0, min(1.0, score))

    def top_k(self, query: Vector, codebook: Dict[T, Vector], k: int = 5) -> List[Tuple[T, float]]:
        if k <= 0:
            return []
        scored = [(key, self.similarity(query, vector)) for key, vector in codebook.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def cleanup(self, query: Vector, codebook: Dict[T, Vector]) -> Tuple[T, float]:
        if not codebook:
            raise ValueError("cannot cleanup against an empty codebook")
        return self.top_k(query, codebook, k=1)[0]

    def _check_dim(self, vector: Vector) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(vector)}")

    def _check_same_dim(self, left: Vector, right: Vector) -> None:
        self._check_dim(left)
        self._check_dim(right)
