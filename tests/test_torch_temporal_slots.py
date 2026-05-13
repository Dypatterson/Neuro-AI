"""Tests for permutation-indexed temporal slot memory (brainstorm Idea 6).

Headline contract: directional information is preserved through bundle.
A bag-based memory cannot tell ``A before B`` from ``B before A``;
the permutation-indexed memory must be able to.
"""

from __future__ import annotations

import unittest


class PermuteSubstrateTests(unittest.TestCase):
    def test_permute_is_inverse_consistent(self):
        try:
            import torch

            from energy_memory.substrate.torch_fhrr import TorchFHRR
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = TorchFHRR(dim=128, seed=3, device="cpu")
        v = substrate.random_vectors(1)[0]
        for shift in (-3, -1, 1, 2, 5):
            shifted = substrate.permute(v, shift)
            recovered = substrate.permute(shifted, -shift)
            self.assertTrue(torch.allclose(recovered, v))

    def test_permute_composes_additively(self):
        try:
            import torch

            from energy_memory.substrate.torch_fhrr import TorchFHRR
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = TorchFHRR(dim=64, seed=5, device="cpu")
        v = substrate.random_vectors(1)[0]
        a, b = 3, -1
        nested = substrate.permute(substrate.permute(v, a), b)
        single = substrate.permute(v, a + b)
        self.assertTrue(torch.allclose(nested, single))

    def test_permute_preserves_unit_magnitude(self):
        try:
            import torch

            from energy_memory.substrate.torch_fhrr import TorchFHRR
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = TorchFHRR(dim=64, seed=7, device="cpu")
        v = substrate.random_vectors(1)[0]
        shifted = substrate.permute(v, 4)
        mags = shifted.abs()
        self.assertTrue(torch.allclose(mags, torch.ones_like(mags), atol=1e-6))


class PermutationSlotTemporalMemoryTests(unittest.TestCase):
    def _build(self, dim=512, seed=11, window=2, length=8):
        from energy_memory.memory.torch_temporal_slots import (
            PermutationSlotTemporalMemory,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=dim, seed=seed, device="cpu")
        vectors = substrate.random_vectors(length)
        labels = [f"tok_{i}" for i in range(length)]
        mem = PermutationSlotTemporalMemory(substrate, window=window)
        mem.store_sequence(labels, [vectors[i] for i in range(length)])
        return substrate, mem, labels, vectors

    def test_recovers_neighbor_at_signed_offset(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate, mem, labels, vectors = self._build()
        # At index 3, neighbors are 1,2,3,4,5 (excluding 3) within window=2.
        ctx = mem.context_for(3)

        plus1 = mem.query_offset(ctx, offset=+1, top_k=3)
        minus1 = mem.query_offset(ctx, offset=-1, top_k=3)
        plus2 = mem.query_offset(ctx, offset=+2, top_k=3)
        minus2 = mem.query_offset(ctx, offset=-2, top_k=3)

        self.assertEqual(plus1.top_label, "tok_4")
        self.assertEqual(minus1.top_label, "tok_2")
        self.assertEqual(plus2.top_label, "tok_5")
        self.assertEqual(minus2.top_label, "tok_1")

    def test_direction_is_distinguishable(self):
        """Bag-based memory cannot distinguish (A before B) from (B before A).
        Permutation-indexed memory must."""
        from energy_memory.memory.torch_temporal_slots import (
            PermutationSlotTemporalMemory,
        )
        try:
            from energy_memory.substrate.torch_fhrr import TorchFHRR
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = TorchFHRR(dim=512, seed=29, device="cpu")
        atoms = substrate.random_vectors(2)
        # Two sequences with identical atoms but reversed order
        seq_ab = [atoms[0], atoms[1]]
        seq_ba = [atoms[1], atoms[0]]

        mem_ab = PermutationSlotTemporalMemory(substrate, window=1)
        mem_ab.store_sequence(["A", "B"], seq_ab)
        mem_ba = PermutationSlotTemporalMemory(substrate, window=1)
        mem_ba.store_sequence(["B", "A"], seq_ba)

        ctx_ab_at_A = mem_ab.context_for(0)  # A's context: B at offset +1
        ctx_ba_at_A = mem_ba.context_for(1)  # A's context in BA: B at offset -1

        # The two A-contexts must differ in similarity to permute(B, +1)
        # vs. permute(B, -1).
        B = atoms[1]
        target_plus = substrate.permute(B, +1)
        target_minus = substrate.permute(B, -1)
        sim_ab_to_plus = float(substrate.similarity(ctx_ab_at_A, target_plus))
        sim_ba_to_plus = float(substrate.similarity(ctx_ba_at_A, target_plus))
        sim_ab_to_minus = float(substrate.similarity(ctx_ab_at_A, target_minus))
        sim_ba_to_minus = float(substrate.similarity(ctx_ba_at_A, target_minus))

        # AB context (B at offset +1) should align with B@+1 more than B@-1.
        self.assertGreater(sim_ab_to_plus, sim_ab_to_minus)
        # BA context (B at offset -1) should align with B@-1 more than B@+1.
        self.assertGreater(sim_ba_to_minus, sim_ba_to_plus)


if __name__ == "__main__":
    unittest.main()
