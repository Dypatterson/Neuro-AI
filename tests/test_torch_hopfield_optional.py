import unittest


class TorchHopfieldOptionalTests(unittest.TestCase):
    def test_torch_hopfield_retrieves_nearest_pattern_if_available(self):
        try:
            from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
            from energy_memory.substrate.torch_fhrr import TorchFHRR

            substrate = TorchFHRR(dim=256, seed=7, device="cpu")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        patterns = substrate.random_vectors(5)
        memory = TorchHopfieldMemory[str](substrate)
        for index, pattern in enumerate(patterns):
            memory.store(pattern, label=f"item_{index}")

        query = substrate.perturb(patterns[2], noise=0.02)
        result = memory.retrieve(query, beta=12.0)

        self.assertEqual(result.top_label, "item_2")
        self.assertGreater(result.top_score, 0.95)
        self.assertLess(result.entropy, 0.8)


if __name__ == "__main__":
    unittest.main()
