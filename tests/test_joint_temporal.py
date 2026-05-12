import unittest

from energy_memory.experiments.synthetic_worlds import DISTRACTOR_STREAM, build_memory, distractor_vectors
from energy_memory.substrate import FHRR


class JointTemporalRecallTests(unittest.TestCase):
    def test_temporal_cue_disambiguates_tight_content_family(self):
        substrate = FHRR(dim=512, seed=23)
        vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=0.10)
        memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=2)
        temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])

        result = memory.joint_recall(
            vectors["stair"],
            temporal_query,
            content_beta=100.0,
            temporal_beta=100.0,
            top_k=4,
        )

        retrieved = [label for label, _ in result.temporal_items]
        self.assertEqual(result.top_label, "stair")
        self.assertIn("slip", retrieved)
        self.assertIn("doctor", retrieved)


if __name__ == "__main__":
    unittest.main()
