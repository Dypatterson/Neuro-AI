import unittest

from energy_memory.memory import TemporalAssociationMemory
from energy_memory.substrate import FHRR


class SimilarityDistractorTests(unittest.TestCase):
    def test_temporal_neighbors_beat_content_neighbors(self):
        substrate = FHRR(dim=512, seed=11)
        labels = [
            "stair",
            "slip",
            "doctor",
            "ladder",
            "paint",
            "ramp",
            "cart",
            "step",
            "dance",
        ]
        family_base = substrate.random_vector()
        vectors = {label: substrate.random_vector() for label in labels}
        for label in ["stair", "ladder", "ramp", "step"]:
            vectors[label] = substrate.perturb(family_base, noise=0.18)

        memory = TemporalAssociationMemory[str](substrate, window=1)
        memory.store_sequence(labels, [vectors[label] for label in labels])

        content = substrate.top_k(
            vectors["stair"],
            {label: vector for label, vector in vectors.items() if label != "stair"},
            k=3,
        )
        temporal = memory.recall(vectors["stair"], beta=100.0, top_k=3).temporal_items

        content_labels = [label for label, _ in content]
        temporal_labels = [label for label, _ in temporal]

        self.assertIn("ladder", content_labels)
        self.assertIn("slip", temporal_labels)


if __name__ == "__main__":
    unittest.main()
