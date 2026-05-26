"""Tests for Phase 5' bridge readouts."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestBridgeReadouts(unittest.TestCase):

    def test_cue_conditioned_scene_energy_is_not_scene_self_energy(self):
        from energy_memory.phase5.bridge_readouts import (
            cue_conditioned_scene_energy,
            delta_e_content_minus_role,
        )

        cue = torch.tensor([1, 1, 1, 1], dtype=torch.complex64)
        same = torch.tensor([1, 1, 1, 1], dtype=torch.complex64)
        different = torch.tensor([1, -1, 1, -1], dtype=torch.complex64)

        role_energy = cue_conditioned_scene_energy(cue, same)
        content_energy = cue_conditioned_scene_energy(cue, different)
        delta_e = delta_e_content_minus_role(content_energy, role_energy)

        self.assertAlmostEqual(float(role_energy), -1.0, places=7)
        self.assertGreater(float(content_energy), -1.0)
        self.assertGreater(float(delta_e), 0.0)

    def test_cue_conditioned_scene_energy_rejects_bad_shape(self):
        from energy_memory.phase5.bridge_readouts import cue_conditioned_scene_energy

        cue = torch.ones(4, dtype=torch.complex64)
        scene = torch.ones(5, dtype=torch.complex64)

        with self.assertRaises(ValueError):
            cue_conditioned_scene_energy(cue, scene)


if __name__ == "__main__":
    unittest.main()
