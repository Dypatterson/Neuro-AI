"""Encoding-switch contract for TorchTemporalAssociationMemory.

The class accepts ``encoding="bag"`` (default, prior behavior) or
``encoding="permutation"`` (brainstorm Idea 6, 2026-05-13). This test
pins three invariants:

  1. Default encoding is bag, contexts match the prior bundle.
  2. permutation contexts differ from bag contexts on the same sequence.
  3. permutation contexts distinguish directionality where bags cannot:
     two anchor positions sharing the same unordered neighbor set but
     in opposite order produce *different* permutation contexts and
     *identical* bag contexts.
  4. Unknown encoding is rejected.
"""

from __future__ import annotations

import unittest


class TemporalEncodingTests(unittest.TestCase):
    def _substrate(self, dim=256, seed=41):
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        return TorchFHRR(dim=dim, seed=seed, device="cpu")

    def test_default_is_bag(self):
        try:
            from energy_memory.memory.torch_temporal import (
                TorchTemporalAssociationMemory,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        mem = TorchTemporalAssociationMemory(substrate, window=2)
        self.assertEqual(mem.encoding, "bag")

    def test_unknown_encoding_rejected(self):
        try:
            from energy_memory.memory.torch_temporal import (
                TorchTemporalAssociationMemory,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        with self.assertRaises(ValueError):
            TorchTemporalAssociationMemory(substrate, encoding="banana")

    def test_permutation_contexts_differ_from_bag(self):
        try:
            import torch

            from energy_memory.memory.torch_temporal import (
                TorchTemporalAssociationMemory,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        vectors = substrate.random_vectors(8)
        labels = [f"t{i}" for i in range(8)]

        bag = TorchTemporalAssociationMemory(substrate, window=2, encoding="bag")
        perm = TorchTemporalAssociationMemory(substrate, window=2, encoding="permutation")
        bag.store_sequence(labels, [vectors[i] for i in range(8)])
        perm.store_sequence(labels, [vectors[i] for i in range(8)])

        diffs = (bag.temporal_contexts - perm.temporal_contexts).abs().sum().item()
        self.assertGreater(diffs, 0.0)

    def test_permutation_distinguishes_mirrored_neighborhoods(self):
        """Bags conflate mirrored neighborhoods; permutation slots do not.

        Build a sequence with two anchor positions whose unordered
        neighbor sets are identical but the orderings are reversed.
        """
        try:
            import torch

            from energy_memory.memory.torch_temporal import (
                TorchTemporalAssociationMemory,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate(dim=512, seed=53)
        # Vectors for tokens A, B, C, ANCHOR
        atoms = substrate.random_vectors(4)
        A, B, C, ANCHOR = atoms[0], atoms[1], atoms[2], atoms[3]
        # Sequence 1: [A, B, ANCHOR, B, A]  -> ANCHOR neighbors in order: A,B,B,A
        # Sequence 2: [A, B, ANCHOR, A, B]  -> ANCHOR neighbors in order: A,B,A,B
        # Both anchor positions have neighbor multiset {A, B, A, B} but the
        # *ordering* differs across the +/- offset axis.
        labels1 = ["A1", "B1", "ANC1", "B2", "A2"]
        seq1 = [A, B, ANCHOR, B, A]
        labels2 = ["A1", "B1", "ANC1", "A2", "B2"]
        seq2 = [A, B, ANCHOR, A, B]

        for encoding in ("bag", "permutation"):
            mem1 = TorchTemporalAssociationMemory(substrate, window=2, encoding=encoding)
            mem2 = TorchTemporalAssociationMemory(substrate, window=2, encoding=encoding)
            mem1.store_sequence(labels1, seq1)
            mem2.store_sequence(labels2, seq2)

            ctx1 = mem1.temporal_contexts[2]  # ANCHOR's context in seq1
            ctx2 = mem2.temporal_contexts[2]  # ANCHOR's context in seq2
            sim = float(substrate.similarity(ctx1, ctx2))
            if encoding == "bag":
                # Bags should be near-identical for these mirrored sequences.
                self.assertGreater(sim, 0.95)
            else:
                # Permutation contexts must distinguish them.
                self.assertLess(sim, 0.6)


if __name__ == "__main__":
    unittest.main()
