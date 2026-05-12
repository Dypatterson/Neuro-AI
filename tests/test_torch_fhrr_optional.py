import unittest


class TorchFHRROptionalTests(unittest.TestCase):
    def test_torch_backend_round_trip_if_available(self):
        try:
            from energy_memory.substrate.torch_fhrr import TorchFHRR
            substrate = TorchFHRR(dim=256, seed=1)
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        role = substrate.random_vector()
        filler = substrate.random_vector()
        recovered = substrate.unbind(substrate.bind(role, filler), role)

        self.assertGreater(substrate.similarity(filler, recovered), 0.999)


if __name__ == "__main__":
    unittest.main()
