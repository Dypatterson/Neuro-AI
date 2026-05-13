"""LSR (Epanechnikov / log-sum-ReLU) kernel checks for TorchHopfieldMemory.

The LSR kernel is added as a sidebar experiment to test whether the
intermediate-beta regime described in Dense Associative Memory with
Epanechnikov Energy (arXiv:2506.10801) shows up at our scale. These
tests pin the basic contract; the phase diagram itself is mapped by
``experiments/25_lsr_kernel_sweep.py``.
"""

import unittest


class TorchHopfieldLSRTests(unittest.TestCase):
    def _substrate(self, dim: int = 256, seed: int = 11):
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        return TorchFHRR(dim=dim, seed=seed, device="cpu")

    def test_unknown_kernel_rejected(self):
        try:
            from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        memory = TorchHopfieldMemory[str](substrate)
        memory.store(substrate.random_vectors(1)[0], label="x")
        with self.assertRaises(ValueError):
            memory.retrieve(substrate.random_vectors(1)[0], kernel="banana")

    def test_lsr_kernel_retrieves_nearest_at_high_beta(self):
        """At high beta the LSR kernel must commit to the nearest stored pattern."""
        try:
            from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        patterns = substrate.random_vectors(5)
        memory = TorchHopfieldMemory[str](substrate)
        for index, pattern in enumerate(patterns):
            memory.store(pattern, label=f"item_{index}")

        query = substrate.perturb(patterns[2], noise=0.02)
        result = memory.retrieve(query, beta=64.0, kernel="lsr")

        self.assertEqual(result.top_label, "item_2")
        self.assertGreater(result.top_score, 0.95)

    def test_lsr_weights_have_compact_support(self):
        """Patterns with negative similarity to the cue receive zero weight under LSR."""
        try:
            import torch

            from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        patterns = substrate.random_vectors(8)
        memory = TorchHopfieldMemory[str](substrate)
        for index, pattern in enumerate(patterns):
            memory.store(pattern, label=f"item_{index}")

        # An anti-aligned cue: similarity to patterns[0] is strongly positive,
        # negated copies are strongly negative.
        cue = patterns[0]
        result = memory.retrieve(cue, beta=4.0, kernel="lsr", max_iter=1)
        weights = torch.tensor(result.weights)
        scores = torch.tensor(result.scores)
        # Every weight that is non-zero must correspond to a non-negative score.
        zero_mask = weights <= 1e-12
        negative_score_mask = scores < 0.0
        # Patterns with negative score should be zeroed (compact support).
        self.assertTrue(bool((zero_mask | ~negative_score_mask).all()))


if __name__ == "__main__":
    unittest.main()
