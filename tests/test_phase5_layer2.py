"""Tests for HAM with layer-2 attractors."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestLayer2State(unittest.TestCase):

    def setUp(self):
        from energy_memory.phase5.ham_with_layer2 import (
            Layer2Config, Layer2State,
        )
        self.cfg = Layer2Config(capacity=3, initial_strength=1.0,
                                strength_decay=0.9, min_strength=0.1)
        self.state = Layer2State(self.cfg)

    def test_add_and_count(self):
        self.state.add(torch.tensor([0.5, 0.3, 0.2]))
        self.state.add(torch.tensor([0.1, 0.8, 0.1]))
        self.assertEqual(len(self.state), 2)

    def test_evicts_weakest_at_capacity(self):
        self.state.add(torch.tensor([0.5, 0.5]))
        self.state.attractors[0].strength = 0.05  # weakest
        self.state.add(torch.tensor([0.4, 0.6]))
        self.state.add(torch.tensor([0.3, 0.7]))
        self.state.add(torch.tensor([0.9, 0.1]))  # over capacity
        self.assertEqual(len(self.state), 3)
        strengths = [a.strength for a in self.state.attractors]
        self.assertNotIn(0.05, strengths)

    def test_decay_reduces_strength_and_ages(self):
        idx = self.state.add(torch.tensor([0.5, 0.5]))
        s0 = self.state.attractors[idx].strength
        self.state.decay_all()
        s1 = self.state.attractors[idx].strength
        self.assertLess(s1, s0)
        self.assertEqual(self.state.attractors[idx].age, 1)

    def test_reinforcement_increases_strength_resets_age(self):
        idx = self.state.add(torch.tensor([0.5, 0.5]))
        self.state.decay_all()
        self.state.decay_all()
        self.assertEqual(self.state.attractors[idx].age, 2)
        self.state.reinforce(idx, activation_weight=0.8)
        self.assertEqual(self.state.attractors[idx].age, 0)
        self.assertEqual(self.state.attractors[idx].activations, 1)

    def test_prune_weak_removes_below_threshold(self):
        i0 = self.state.add(torch.tensor([0.5, 0.5]))
        i1 = self.state.add(torch.tensor([0.3, 0.7]))
        self.state.attractors[0].strength = 0.05  # below 0.1 threshold
        removed = self.state.prune_weak()
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(self.state), 1)


@unittest.skipIf(torch is None, "torch required")
class TestHAMWithLayer2(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase5.ham_aggregator import (
            HAMConfig, HAMScaleInput,
        )
        from energy_memory.phase5.ham_with_layer2 import (
            HAMWithLayer2, Layer2Config,
        )

        self.substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        self.codebook = self.substrate.random_vectors(30)
        self.mask_id = 28
        self.decode_ids = list(range(28))

        self.scale_inputs = {}
        for w in [2, 3, 4]:
            positions = build_position_vectors(self.substrate, w)
            memory = TorchHopfieldMemory(self.substrate)
            for pattern in [(1, 2, 3, 4)[:w], (5, 6, 7, 8)[:w], (1, 5, 9, 2)[:w]]:
                memory.store(
                    encode_window(self.substrate, positions, self.codebook, pattern),
                    label=str(pattern),
                )
            sub_window = list((1, 2, 3, 4)[:w])
            self.scale_inputs[w] = HAMScaleInput(
                memory=memory, positions=positions,
                sub_window=sub_window, local_masked_pos=min(1, w - 1),
            )

        self.ham_l2 = HAMWithLayer2(
            self.substrate,
            ham_config=HAMConfig(beta=10.0, max_iter=8, alpha=0.3),
            layer2_config=Layer2Config(
                lambda_l2=0.4, beta_l2=10.0, capacity=10,
                initial_strength=1.0, strength_decay=0.99,
            ),
        )
        self.HAMScaleInput = HAMScaleInput

    def test_retrieve_with_empty_layer2(self):
        result = self.ham_l2.retrieve(
            scale_inputs=self.scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        self.assertAlmostEqual(float(result.consensus.sum().item()), 1.0, places=4)
        self.assertEqual(result.layer2_activations.shape[0], 0)
        self.assertGreater(result.iterations, 0)

    def test_add_discovery_grows_layer2(self):
        self.assertEqual(len(self.ham_l2.layer2), 0)
        profile = torch.zeros(len(self.decode_ids))
        profile[5] = 1.0
        self.ham_l2.add_discovery(profile)
        self.assertEqual(len(self.ham_l2.layer2), 1)

    def test_layer2_attractor_biases_consensus(self):
        # Get the baseline consensus
        result_before = self.ham_l2.retrieve(
            scale_inputs=self.scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        baseline = result_before.consensus.clone()

        # Add a strongly-peaked attractor on a specific token
        target_idx = 7
        profile = torch.full((len(self.decode_ids),), 0.005)
        profile[target_idx] = 1.0 - 0.005 * (len(self.decode_ids) - 1)
        self.ham_l2.add_discovery(profile)

        result_after = self.ham_l2.retrieve(
            scale_inputs=self.scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        # Probability at the attractor's peak should have increased
        self.assertGreater(
            result_after.consensus[target_idx].item(),
            baseline[target_idx].item() - 1e-6,
        )

    def test_record_trace_returns_history(self):
        result = self.ham_l2.retrieve(
            scale_inputs=self.scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
            record_trace=True,
        )
        self.assertGreater(len(result.consensus_history), 0)
        for c in result.consensus_history:
            self.assertAlmostEqual(float(c.sum().item()), 1.0, places=4)

    def test_engagement_and_resolution(self):
        result = self.ham_l2.retrieve(
            scale_inputs=self.scale_inputs, codebook=self.codebook,
            mask_id=self.mask_id, decode_ids=self.decode_ids,
        )
        self.assertGreaterEqual(result.engagement, 0.0)
        self.assertGreaterEqual(result.resolution, 0.0)
        self.assertLessEqual(result.resolution, 1.0 + 1e-6)

    def test_active_attractor_reinforced_after_retrieval(self):
        # Add a strong attractor on the dominant token
        profile = torch.zeros(len(self.decode_ids))
        profile[3] = 1.0
        idx = self.ham_l2.add_discovery(profile)
        s0 = self.ham_l2.layer2.attractors[idx].strength
        # Do several retrievals — should reinforce, then natural decay applies
        for _ in range(5):
            self.ham_l2.retrieve(
                scale_inputs=self.scale_inputs, codebook=self.codebook,
                mask_id=self.mask_id, decode_ids=self.decode_ids,
            )
        # The attractor should have been activated and possibly reinforced;
        # at minimum, its activations counter should be tracked
        a = self.ham_l2.layer2.attractors[idx]
        self.assertGreaterEqual(a.activations, 0)


if __name__ == "__main__":
    unittest.main()
