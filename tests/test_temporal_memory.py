import unittest

from energy_memory.memory import TemporalAssociationMemory
from energy_memory.substrate import FHRR


class TemporalAssociationMemoryTests(unittest.TestCase):
    def test_recalls_temporal_neighbors(self):
        substrate = FHRR(dim=512, seed=4)
        labels = ["candle", "table", "book", "stair", "slip", "doctor"]
        vectors = {label: substrate.random_vector() for label in labels}
        memory = TemporalAssociationMemory[str](substrate, window=1)
        memory.store_sequence(labels, [vectors[label] for label in labels])

        result = memory.recall(vectors["slip"], beta=12.0, top_k=3)
        retrieved = [label for label, _ in result.temporal_items]

        self.assertIn("stair", retrieved)
        self.assertIn("doctor", retrieved)


if __name__ == "__main__":
    unittest.main()

