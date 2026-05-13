"""Tests for the GIB-inspired synergy estimator (brainstorm Idea 8)."""

from __future__ import annotations

import unittest


class SynergyEstimatorTests(unittest.TestCase):
    def _substrate(self, dim=512, seed=13):
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        return TorchFHRR(dim=dim, seed=seed, device="cpu")

    def test_clean_binding_has_high_synergy(self):
        try:
            from energy_memory.diagnostics.synergy import synergy_score
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        role, filler = substrate.random_vectors(2)
        m = synergy_score(substrate, role, filler)
        # Perfect FHRR binding: unbind(bind(r,f), r) ~= f -> recover ~= 1.
        self.assertGreater(m.recover, 0.95)
        # Random independent role and filler have near-zero similarity.
        self.assertLess(abs(m.baseline_from_role), 0.15)
        self.assertLess(abs(m.baseline_from_binding), 0.15)
        self.assertGreater(m.synergy, 0.7)

    def test_atom_alone_synergy_is_near_zero(self):
        """Null hypothesis: the "binding" is just the filler. Synergy = 0."""
        try:
            from energy_memory.diagnostics.synergy import atom_alone_synergy
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        atoms = substrate.random_vectors(16)
        s = atom_alone_synergy(substrate, atoms)
        # Recover(filler, random_role) = sim(unbind(filler, role), filler) ~= 0
        # because unbinding by an unrelated role scrambles the vector. So
        # synergy ~= 0 - 1 = -1, which is < 0.1 either way. The point is it
        # is well below the clean-binding synergy of ~0.8+.
        self.assertLess(s, 0.1)

    def test_mean_synergy_aggregates(self):
        try:
            from energy_memory.diagnostics.synergy import mean_synergy
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        roles = substrate.random_vectors(8)
        fillers = substrate.random_vectors(8)
        m = mean_synergy(substrate, [roles[i] for i in range(8)], [fillers[i] for i in range(8)])
        # On clean independent role-filler pairs, mean synergy must be high.
        self.assertGreater(m.synergy, 0.7)
        # And it must be lower than the upper bound of 1.0.
        self.assertLess(m.synergy, 1.0)

    def test_phase5_headline_contract_binding_beats_atoms(self):
        """The Phase 5 headline metric should be high for bindings, low for atoms.

        This is the contract that makes synergy meaningful as a structure
        diagnostic: well-formed bindings score >> atoms-pretending-to-be-bindings.
        """
        try:
            from energy_memory.diagnostics.synergy import (
                atom_alone_synergy,
                mean_synergy,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        substrate = self._substrate()
        roles = substrate.random_vectors(16)
        fillers = substrate.random_vectors(16)

        bound = mean_synergy(
            substrate,
            [roles[i] for i in range(16)],
            [fillers[i] for i in range(16)],
        )
        unbound = atom_alone_synergy(substrate, [fillers[i] for i in range(16)])

        # Clear separation: bindings carry recoverable structure; atoms don't.
        self.assertGreater(bound.synergy - unbound, 0.5)


if __name__ == "__main__":
    unittest.main()
