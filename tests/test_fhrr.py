import unittest

from energy_memory.substrate import FHRR


class FHRRTests(unittest.TestCase):
    def test_bind_unbind_round_trip(self):
        substrate = FHRR(dim=256, seed=1)
        role = substrate.random_vector()
        filler = substrate.random_vector()

        recovered = substrate.unbind(substrate.bind(role, filler), role)

        self.assertGreater(substrate.similarity(filler, recovered), 0.999)

    def test_bundle_keeps_members_recoverable(self):
        substrate = FHRR(dim=512, seed=2)
        vectors = substrate.random_vectors(5)
        bundled = substrate.bundle(vectors)

        member_scores = [substrate.similarity(bundled, vector) for vector in vectors]
        random_score = substrate.similarity(bundled, substrate.random_vector())

        self.assertGreater(min(member_scores), random_score + 0.1)


if __name__ == "__main__":
    unittest.main()

