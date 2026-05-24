"""Tests for the Phase 5 M1 role-energy stack."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestRoleBindingStats(unittest.TestCase):

    def test_weights_are_soft_frequency_normalized(self):
        from energy_memory.phase5.m1_role_energy import RoleBindingStats

        stats = RoleBindingStats.from_encoder_terms(
            [
                [(0, 1), (1, 1), (1, 2)],
                [(1, 1), (2, 2)],
            ],
            n_atoms=4,
            n_roles=3,
            device="cpu",
        )
        weights = stats.atom_role_weights(laplace=0.0)
        self.assertTrue(torch.allclose(weights[1], torch.tensor([1 / 3, 2 / 3, 0.0])))
        self.assertTrue(torch.allclose(weights[2], torch.tensor([0.0, 0.5, 0.5])))

    def test_unobserved_atoms_get_uniform_laplace_weights(self):
        from energy_memory.phase5.m1_role_energy import RoleBindingStats

        stats = RoleBindingStats.empty(n_atoms=2, n_roles=4, device="cpu")
        weights = stats.atom_role_weights(laplace=1.0)
        self.assertTrue(torch.allclose(weights[0], torch.full((4,), 0.25)))

    def test_from_traces_requires_provenance_by_default(self):
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.phase5.m1_role_energy import RoleBindingStats

        traces = [TrajectoryTrace(query=torch.zeros(8, dtype=torch.complex64))]
        with self.assertRaises(ValueError):
            RoleBindingStats.from_traces(
                traces, n_atoms=4, n_roles=2, device="cpu",
            )

    def test_from_pattern_encoder_terms_counts_roles_on_pattern_rows(self):
        from energy_memory.phase5.m1_role_energy import RoleBindingStats

        stats = RoleBindingStats.from_pattern_encoder_terms(
            [
                [(0, 10), (0, 11), (1, 12)],
                [(1, 10), (2, 12)],
                [(2, 99), (2, 100)],
            ],
            n_roles=3,
            device="cpu",
        )
        self.assertTrue(torch.equal(
            stats.counts.cpu(),
            torch.tensor([
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
            ]),
        ))

    def test_from_pattern_encoder_terms_requires_complete_by_default(self):
        from energy_memory.phase5.m1_role_energy import RoleBindingStats

        with self.assertRaises(ValueError):
            RoleBindingStats.from_pattern_encoder_terms(
                [[(0, 1)], None], n_roles=2, device="cpu",
            )


@unittest.skipIf(torch is None, "torch required")
class TestM1RoleEnergy(unittest.TestCase):

    def setUp(self):
        from energy_memory.phase2.encoding import build_position_vectors
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        self.substrate = TorchFHRR(dim=256, seed=23, device="cpu")
        self.patterns = self.substrate.random_vectors(12)
        self.roles = build_position_vectors(self.substrate, 3)
        counts = torch.ones((12, 3), dtype=torch.float32)
        for atom in range(12):
            counts[atom, atom % 3] += 5.0
        self.atom_role_weights = counts / counts.sum(dim=1, keepdim=True)

    def test_s2_weighted_mhn_check_passes_on_synthetic_substrate(self):
        from energy_memory.phase5.m1_role_energy import run_s2_weighted_mhn_check

        result = run_s2_weighted_mhn_check(
            seed=3, dim=256, n_atoms=24, n_roles=3, max_iter=8,
        )
        self.assertTrue(result.energy_trace)
        self.assertTrue(result.passed)
        self.assertLessEqual(result.max_energy_increase, 1e-5)
        self.assertGreaterEqual(result.final_magnitude, 0.7)
        self.assertLessEqual(result.final_magnitude, 1.3)

    def test_d3_cross_k_returns_one_state_per_branch(self):
        from energy_memory.phase5.m1_role_energy import d3_additive_cross_k_settle

        init = [self.patterns[0], self.patterns[1]]
        branch_weights = torch.stack([
            self.atom_role_weights[:, 0],
            self.atom_role_weights[:, 1],
        ])
        states, joint_trace = d3_additive_cross_k_settle(
            self.substrate,
            init,
            self.patterns,
            branch_weights,
            beta=8.0,
            max_iter=5,
            mix=0.5,
        )
        self.assertEqual(len(states), 2)
        self.assertEqual(len(joint_trace), 5)
        for state in states:
            self.assertEqual(state.shape, self.patterns[0].shape)
            self.assertAlmostEqual(float(state.abs().mean()), 1.0, places=5)

    def test_m1_stack_composes_role_energy_branches(self):
        from energy_memory.phase5.m1_role_energy import M1Config, run_m1_stack

        cue = self.substrate.bind(self.roles[0], self.patterns[0])
        result = run_m1_stack(
            self.substrate,
            cue,
            self.patterns,
            self.roles,
            self.atom_role_weights,
            branch_roles=[0, 1],
            config=M1Config(beta=8.0, max_iter=5, d3_mix=0.5, p3_saliency_gain=0.1),
        )
        self.assertEqual(len(result.branches), 2)
        self.assertEqual(len(result.joint_energy_trace), 5)
        self.assertEqual(result.branches[0].role_index, 0)
        for branch in result.branches:
            self.assertGreaterEqual(branch.top_index, 0)
            self.assertLess(branch.top_index, self.patterns.shape[0])
            self.assertEqual(len(branch.energy_trace), 5)


if __name__ == "__main__":
    unittest.main()
