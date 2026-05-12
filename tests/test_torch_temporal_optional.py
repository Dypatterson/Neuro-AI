import unittest


class TorchTemporalOptionalTests(unittest.TestCase):
    def test_torch_temporal_if_available(self):
        try:
            from energy_memory.experiments.synthetic_worlds import DISTRACTOR_STREAM
            from energy_memory.memory.torch_temporal import TorchTemporalAssociationMemory
            from energy_memory.substrate.torch_fhrr import TorchFHRR

            substrate = TorchFHRR(dim=128, seed=23, device="cpu")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        labels = list(DISTRACTOR_STREAM)
        vectors = {label: substrate.random_vector() for label in labels}
        family_base = substrate.random_vector()
        for label in ["stair", "ladder", "ramp", "step", "escalator"]:
            vectors[label] = substrate.perturb(family_base, noise=0.10)
        memory = TorchTemporalAssociationMemory(substrate, window=2)
        memory.store_sequence(labels, [vectors[label] for label in labels])

        temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])
        result = memory.coupled_recall(vectors["stair"], temporal_query, content_beta=100.0, temporal_beta=100.0)

        self.assertEqual(result.top_label, "stair")
        self.assertIn("slip", [label for label, _ in result.temporal_items])


if __name__ == "__main__":
    unittest.main()

