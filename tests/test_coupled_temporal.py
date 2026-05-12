import unittest

from energy_memory.experiments.synthetic_worlds import DISTRACTOR_STREAM, build_memory, distractor_vectors
from energy_memory.substrate import FHRR


class CoupledTemporalRecallTests(unittest.TestCase):
    def test_coupled_recall_records_stable_trajectory(self):
        substrate = FHRR(dim=512, seed=23)
        vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=0.10)
        memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=2)
        temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])

        result = memory.coupled_recall(
            vectors["stair"],
            temporal_query,
            content_beta=100.0,
            temporal_beta=100.0,
            feedback=0.75,
            max_iter=8,
            top_k=4,
        )

        self.assertEqual(result.top_label, "stair")
        self.assertGreaterEqual(len(result.trace), 2)
        self.assertTrue(all(step.top_label == "stair" for step in result.trace))
        self.assertIn("slip", [label for label, _ in result.temporal_items])


if __name__ == "__main__":
    unittest.main()
