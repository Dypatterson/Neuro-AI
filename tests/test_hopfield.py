import unittest

from energy_memory.memory import HopfieldMemory
from energy_memory.substrate import FHRR


class HopfieldMemoryTests(unittest.TestCase):
    def test_retrieves_nearest_stored_pattern(self):
        substrate = FHRR(dim=512, seed=3)
        patterns = substrate.random_vectors(6)
        memory = HopfieldMemory[str](substrate)
        for i, pattern in enumerate(patterns):
            memory.store(pattern, label=f"item_{i}")

        result = memory.retrieve(patterns[2], beta=12.0)

        self.assertEqual(result.top_label, "item_2")
        self.assertGreater(result.top_score, 0.95)
        self.assertLess(result.entropy, 0.8)


if __name__ == "__main__":
    unittest.main()

