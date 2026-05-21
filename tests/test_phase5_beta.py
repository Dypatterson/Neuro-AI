"""Tests for the β (path-3) role-fidelity-weighted prior.

See notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md and
src/energy_memory/phase5/role_fidelity.py.
"""
from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch required")
class TestRoleFidelity(unittest.TestCase):

    def _random_unit_phasor(self, d, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        phase = torch.rand((d,), generator=gen) * (2.0 * math.pi)
        return torch.polar(torch.ones((d,)), phase)

    def test_clean_role_binding_gives_high_fidelity(self):
        """W distinct random fillers → mean pairwise distance ≈ 1."""
        from energy_memory.phase5.role_fidelity import compute_role_fidelity
        D, W = 4096, 4
        fillers = torch.stack(
            [self._random_unit_phasor(D, seed=s) for s in range(W)], dim=0
        )
        # schema_bindings shape [N=1, W, D]
        bindings = fillers.unsqueeze(0)
        f = compute_role_fidelity(bindings)
        self.assertEqual(f.shape, (1,))
        # Random orthogonal-ish FHRR vectors at D=4096 have |G_jk| ≈ 1/sqrt(D)
        # ≈ 0.016, so 1 - |G_jk| ≈ 0.984. f should be near 1.
        self.assertGreater(float(f[0]), 0.95)

    def test_collapsed_decomposition_gives_low_fidelity(self):
        """All fillers identical → mean pairwise distance = 0."""
        from energy_memory.phase5.role_fidelity import compute_role_fidelity
        D, W = 4096, 4
        single = self._random_unit_phasor(D, seed=11)
        bindings = single.unsqueeze(0).repeat(1, W, 1)  # [1, W, D]
        # Actually need [N=1, W, D]; single is [D]
        bindings = single.unsqueeze(0).unsqueeze(0).repeat(1, W, 1)
        f = compute_role_fidelity(bindings)
        self.assertEqual(f.shape, (1,))
        self.assertLess(float(f[0]), 0.01)

    def test_single_position_returns_zero(self):
        """W=1 → no pairs → fidelity undefined; return 0."""
        from energy_memory.phase5.role_fidelity import compute_role_fidelity
        single = self._random_unit_phasor(4096, seed=3).unsqueeze(0).unsqueeze(0)
        # shape [N=1, W=1, D]
        f = compute_role_fidelity(single)
        self.assertEqual(f.shape, (1,))
        self.assertEqual(float(f[0]), 0.0)

    def test_empty_returns_empty(self):
        from energy_memory.phase5.role_fidelity import compute_role_fidelity
        empty = torch.zeros(0, 4, 4096, dtype=torch.complex64)
        f = compute_role_fidelity(empty)
        self.assertEqual(f.shape, (0,))

    def test_mixed_schemas_separate_correctly(self):
        """One clean + one collapsed schema → high/low fidelity respectively."""
        from energy_memory.phase5.role_fidelity import compute_role_fidelity
        D, W = 4096, 4
        clean = torch.stack(
            [self._random_unit_phasor(D, seed=s) for s in range(W)], dim=0
        )
        single = self._random_unit_phasor(D, seed=42)
        collapsed = single.unsqueeze(0).repeat(W, 1)
        bindings = torch.stack([clean, collapsed], dim=0)  # [2, W, D]
        f = compute_role_fidelity(bindings)
        self.assertEqual(f.shape, (2,))
        self.assertGreater(float(f[0]), 0.95)
        self.assertLess(float(f[1]), 0.01)


@unittest.skipIf(torch is None, "torch required")
class TestFidelityWeightedPrior(unittest.TestCase):

    def _random_unit_phasor(self, d, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        phase = torch.rand((d,), generator=gen) * (2.0 * math.pi)
        return torch.polar(torch.ones((d,)), phase)

    def test_output_is_unit_magnitude(self):
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        D, N = 1024, 8
        cue = self._random_unit_phasor(D, seed=17)
        schemas = torch.stack(
            [self._random_unit_phasor(D, seed=s) for s in range(N)], dim=0
        )
        fidelities = torch.rand(N)
        prior = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=fidelities, p=1.0, q=1.0,
        )
        self.assertEqual(prior.shape, (D,))
        # Each element should be near unit magnitude (FHRR convention)
        mag = prior.abs()
        self.assertTrue(torch.allclose(mag, torch.ones_like(mag), atol=1e-4))

    def test_q_zero_ignores_fidelity(self):
        """At q=0, the prior is content-only (same regardless of f_i values)."""
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        D, N = 1024, 8
        cue = self._random_unit_phasor(D, seed=23)
        schemas = torch.stack(
            [self._random_unit_phasor(D, seed=s) for s in range(N)], dim=0
        )
        f_a = torch.rand(N)
        f_b = torch.zeros(N)  # all zeros
        prior_a = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=f_a, p=1.0, q=0.0,
        )
        prior_b = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=f_b, p=1.0, q=0.0,
        )
        # At q=0, f^0 = 1 regardless of f → priors should match.
        self.assertTrue(torch.allclose(prior_a, prior_b, atol=1e-5))

    def test_q_one_upweights_high_fidelity_schema(self):
        """Two equally cue-matched schemas with different f: q=1 pulls prior toward high-f."""
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        # Use a fresh small dim for control; manually-aligned vectors
        D = 512
        gen = torch.Generator(device="cpu").manual_seed(101)
        phases_a = torch.rand((D,), generator=gen) * (2.0 * math.pi)
        phases_b = torch.rand((D,), generator=gen) * (2.0 * math.pi)
        s_a = torch.polar(torch.ones((D,)), phases_a)
        s_b = torch.polar(torch.ones((D,)), phases_b)
        # Cue: equally aligned to both via the average direction
        cue = torch.polar(torch.ones((D,)), (phases_a + phases_b) / 2.0)
        schemas = torch.stack([s_a, s_b], dim=0)
        # High-fidelity = schema a; low-fidelity = schema b
        fidelities = torch.tensor([0.9, 0.1], dtype=torch.float32)
        prior_q1 = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=fidelities, p=1.0, q=1.0,
        )
        prior_q0 = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=fidelities, p=1.0, q=0.0,
        )
        # At q=1, prior should align MORE with s_a than at q=0.
        align_q1_a = float((prior_q1.conj() * s_a).sum().real)
        align_q1_b = float((prior_q1.conj() * s_b).sum().real)
        align_q0_a = float((prior_q0.conj() * s_a).sum().real)
        align_q0_b = float((prior_q0.conj() * s_b).sum().real)
        # The ratio of s_a alignment to s_b alignment should be higher at q=1.
        self.assertGreater(align_q1_a / max(align_q1_b, 1e-6),
                           align_q0_a / max(align_q0_b, 1e-6))

    def test_anti_aligned_schemas_get_zero_weight(self):
        """A schema with negative cue cosine is clamped out (weight = 0)."""
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        D = 512
        cue = self._random_unit_phasor(D, seed=5)
        # Build: one aligned schema, one anti-aligned schema
        anti_cue = cue.conj()  # anti-aligned in FHRR (negative Re(<cue, anti>))
        # Actually, conj might not produce anti-aligned. Use -cue to flip phase.
        anti = torch.polar(torch.ones((D,)), torch.angle(cue) + math.pi)
        schemas = torch.stack([cue, anti], dim=0)
        fidelities = torch.tensor([1.0, 1.0])
        prior = fidelity_weighted_prior(
            cue=cue, schemas=schemas, fidelities=fidelities, p=1.0, q=1.0,
        )
        # Prior should align strongly with cue (the aligned schema), not anti.
        align_cue = float((prior.conj() * cue).sum().real)
        align_anti = float((prior.conj() * anti).sum().real)
        self.assertGreater(align_cue, 0.0)
        self.assertLess(align_anti, 0.0)
        # In fact prior should be ≈ cue itself (only weighted schema)
        self.assertGreater(align_cue / D, 0.99)

    def test_negative_p_or_q_raises(self):
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        D, N = 64, 2
        cue = self._random_unit_phasor(D, seed=1)
        schemas = torch.stack(
            [self._random_unit_phasor(D, seed=s) for s in range(N)], dim=0
        )
        fidelities = torch.tensor([0.5, 0.5])
        with self.assertRaises(ValueError):
            fidelity_weighted_prior(
                cue=cue, schemas=schemas, fidelities=fidelities,
                p=-1.0, q=1.0,
            )
        with self.assertRaises(ValueError):
            fidelity_weighted_prior(
                cue=cue, schemas=schemas, fidelities=fidelities,
                p=1.0, q=-0.5,
            )


if __name__ == "__main__":
    unittest.main()
