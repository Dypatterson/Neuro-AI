"""Tests for the HAM-style multi-scale aggregator."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestHAMAggregator(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase5.ham_aggregator import (
            HAMAggregator, HAMConfig, HAMScaleInput,
        )

        self.substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        self.codebook = self.substrate.random_vectors(30)
        self.mask_id = 28
        self.decode_ids = list(range(28))

        self.scales = {}
        for w in [2, 3, 4]:
            positions = build_position_vectors(self.substrate, w)
            memory = TorchHopfieldMemory(self.substrate)
            patterns = [
                (1, 2, 3, 4)[:w], (5, 6, 7, 8)[:w], (9, 10, 11, 12)[:w],
                (1, 5, 9, 2)[:w], (2, 6, 10, 3)[:w],
            ]
            for p in patterns:
                memory.store(
                    encode_window(self.substrate, positions, self.codebook, p),
                    label=str(p),
                )
            self.scales[w] = {
                "memory": memory,
                "positions": positions,
                "patterns": patterns,
            }

        self.HAMAggregator = HAMAggregator
        self.HAMConfig = HAMConfig
        self.HAMScaleInput = HAMScaleInput

    def _make_scale_inputs(self, target_window, masked_local_pos):
        inputs = {}
        for w, scale_data in self.scales.items():
            sub = list(target_window[:w])
            inputs[w] = self.HAMScaleInput(
                memory=scale_data["memory"],
                positions=scale_data["positions"],
                sub_window=sub,
                local_masked_pos=min(masked_local_pos, w - 1),
            )
        return inputs

    def test_retrieve_returns_consensus(self):
        agg = self.HAMAggregator(
            self.substrate,
            config=self.HAMConfig(beta=10.0, max_iter=6, alpha=0.3),
        )
        scale_inputs = self._make_scale_inputs((1, 2, 3, 4), 1)
        result = agg.retrieve(
            scale_inputs=scale_inputs,
            codebook=self.codebook,
            mask_id=self.mask_id,
            decode_ids=self.decode_ids,
        )
        self.assertIsNotNone(result.consensus)
        self.assertEqual(result.consensus.shape[0], len(self.decode_ids))
        self.assertAlmostEqual(
            float(result.consensus.sum().item()), 1.0, places=4,
            msg="consensus should be a probability distribution",
        )

    def test_final_states_per_scale(self):
        agg = self.HAMAggregator(self.substrate)
        scale_inputs = self._make_scale_inputs((1, 2, 3, 4), 1)
        result = agg.retrieve(
            scale_inputs=scale_inputs,
            codebook=self.codebook,
            mask_id=self.mask_id,
            decode_ids=self.decode_ids,
        )
        for s in scale_inputs:
            self.assertIn(s, result.final_states)
            mags = result.final_states[s].abs()
            self.assertTrue(
                torch.allclose(mags, torch.ones_like(mags), atol=1e-3),
                f"final state at scale {s} should be unit modulus",
            )

    def test_converges_within_max_iter(self):
        agg = self.HAMAggregator(
            self.substrate,
            config=self.HAMConfig(beta=30.0, max_iter=20, alpha=0.2, convergence_tol=1e-4),
        )
        scale_inputs = self._make_scale_inputs((1, 2, 3, 4), 1)
        result = agg.retrieve(
            scale_inputs=scale_inputs,
            codebook=self.codebook,
            mask_id=self.mask_id,
            decode_ids=self.decode_ids,
        )
        self.assertLessEqual(result.iterations, 20)

    def test_geometric_vs_arithmetic_consensus(self):
        from energy_memory.phase5.ham_aggregator import HAMAggregator, HAMConfig

        scale_inputs = self._make_scale_inputs((1, 2, 3, 4), 1)

        agg_geom = HAMAggregator(
            self.substrate,
            config=HAMConfig(consensus_mode="geometric_mean", max_iter=5),
        )
        agg_arith = HAMAggregator(
            self.substrate,
            config=HAMConfig(consensus_mode="arithmetic_mean", max_iter=5),
        )
        r1 = agg_geom.retrieve(
            scale_inputs=scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        r2 = agg_arith.retrieve(
            scale_inputs=scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        # Both should be valid distributions
        self.assertAlmostEqual(float(r1.consensus.sum().item()), 1.0, places=4)
        self.assertAlmostEqual(float(r2.consensus.sum().item()), 1.0, places=4)

    def test_invalid_consensus_mode_raises(self):
        from energy_memory.phase5.ham_aggregator import HAMAggregator, HAMConfig

        agg = HAMAggregator(
            self.substrate,
            config=HAMConfig(consensus_mode="bogus", max_iter=3),
        )
        scale_inputs = self._make_scale_inputs((1, 2, 3, 4), 1)
        with self.assertRaises(ValueError):
            agg.retrieve(
                scale_inputs=scale_inputs, codebook=self.codebook,
                mask_id=self.mask_id, decode_ids=self.decode_ids,
            )

    def test_empty_scale_inputs_raises(self):
        agg = self.HAMAggregator(self.substrate)
        with self.assertRaises(ValueError):
            agg.retrieve(
                scale_inputs={}, codebook=self.codebook,
                mask_id=self.mask_id, decode_ids=self.decode_ids,
            )

    def test_predict_top_k(self):
        from energy_memory.phase5.ham_aggregator import predict_top_k

        consensus = torch.tensor([0.05, 0.45, 0.3, 0.1, 0.1])
        top3 = predict_top_k(consensus, decode_ids=[10, 20, 30, 40, 50], k=3)
        ids = [t[0] for t in top3]
        self.assertEqual(ids[0], 20)
        self.assertEqual(set(ids), {20, 30, 40})


if __name__ == "__main__":
    unittest.main()
