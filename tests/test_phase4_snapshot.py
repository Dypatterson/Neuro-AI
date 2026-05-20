"""Tests for Phase 4 substrate snapshot save/load.

Round-trip preserves:
- Pattern matrix (bit-equal complex tensors after one save/load cycle).
- Consolidation u-chain, A, retrieval_count, below_threshold_steps,
  step_count.
- Indices in 1:1 alignment between memory._patterns[i] and
  consolidation.u[i].
- effective_strength() ordering.

This is the regression guard that Phase 5's get_schema_store reads the
exact same substrate post-load that the original Phase 4 run wrote.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _build_substrate_with_state(dim=32, n_atoms=5, seed=0):
    """Build a (memory, consolidation) pair with non-trivial state."""
    from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    from energy_memory.substrate.torch_fhrr import TorchFHRR
    torch.manual_seed(seed)
    substrate = TorchFHRR(dim=dim, device="cpu")
    mem = TorchHopfieldMemory(substrate)
    cfg = ConsolidationConfig(m=4, alpha=0.25, death_threshold=0.01,
                              death_window=50, inhibition_gain=0.05,
                              inhibition_decay=0.0, alpha_freq_lambda=0.5)
    cons = ConsolidationState(cfg, device="cpu")
    for i in range(n_atoms):
        p = substrate.normalize(torch.randn(dim, dtype=torch.complex64))
        mem.store(p, label=i)
        cons.add_pattern(novelty_strength=1.0 + 0.3 * i)
    # Run some dynamics so u-chain has spread, A has growth, retrieval_count
    # is non-zero per pattern.
    for step in range(10):
        cons.step_dynamics()
    for i in range(n_atoms):
        cons.reinforce(i, magnitude=0.05 * (i + 1))
        cons.accumulate_inhibition(i, magnitude=0.02)
    for step in range(5):
        cons.step_dynamics()
    return substrate, mem, cons


@unittest.skipIf(torch is None, "torch required")
class TestSnapshotRoundTrip(unittest.TestCase):

    def test_save_load_preserves_pattern_matrix(self):
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=0)
        orig_patterns = torch.stack(mem._patterns, dim=0).clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(
                memory=mem, consolidation=cons, path=path, label="test",
            )
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            new_mem, _, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertEqual(new_mem.stored_count, mem.stored_count)
            for i in range(new_mem.stored_count):
                self.assertTrue(torch.equal(new_mem._patterns[i], orig_patterns[i]))

    def test_save_load_preserves_consolidation_state(self):
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=1)
        orig_u = cons.u.clone()
        orig_a = cons.A.clone()
        orig_rc = cons.retrieval_count.clone()
        orig_bts = cons.below_threshold_steps.clone()
        orig_step = cons._step_count
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(
                memory=mem, consolidation=cons, path=path,
            )
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, new_cons, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertTrue(torch.equal(new_cons.u, orig_u))
            self.assertTrue(torch.equal(new_cons.A, orig_a))
            self.assertTrue(torch.equal(new_cons.retrieval_count, orig_rc))
            self.assertTrue(torch.equal(new_cons.below_threshold_steps, orig_bts))
            self.assertEqual(new_cons._step_count, orig_step)

    def test_save_load_preserves_consolidation_config(self):
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, new_cons, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertEqual(new_cons.config.m, cons.config.m)
            self.assertEqual(new_cons.config.alpha, cons.config.alpha)
            self.assertEqual(new_cons.config.inhibition_gain, cons.config.inhibition_gain)
            self.assertEqual(new_cons.config.alpha_freq_lambda, cons.config.alpha_freq_lambda)

    def test_effective_strength_ordering_preserved(self):
        """The schema-source rule top_k_by_effective_strength must select
        the same atoms after a save/load cycle."""
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=3, n_atoms=8)
        orig_strength = cons.effective_strength().abs().tolist()
        orig_ranking = sorted(range(len(orig_strength)),
                              key=lambda i: -orig_strength[i])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, new_cons, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            new_strength = new_cons.effective_strength().abs().tolist()
            new_ranking = sorted(range(len(new_strength)),
                                 key=lambda i: -new_strength[i])
            self.assertEqual(orig_ranking, new_ranking)
            for o, n in zip(orig_strength, new_strength):
                self.assertAlmostEqual(o, n, places=6)

    def test_label_and_metadata_round_trip(self):
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=4)
        meta = {"seed": 17, "step": 1500, "source": "exp_19"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(
                memory=mem, consolidation=cons, path=path,
                label="post_death", metadata=meta,
            )
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, _, info = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertEqual(info["label"], "post_death")
            self.assertEqual(info["metadata"], meta)
            self.assertEqual(info["version"], 1)

    def test_save_refuses_misaligned_state(self):
        """If memory and consolidation have different row counts, save
        raises rather than producing a corrupted snapshot."""
        from energy_memory.phase4.snapshot import save_substrate_snapshot
        substrate, mem, cons = _build_substrate_with_state(seed=5, n_atoms=4)
        # Corrupt the alignment by adding a pattern to memory only.
        mem.store(substrate.normalize(torch.randn(32, dtype=torch.complex64)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            with self.assertRaises(ValueError):
                save_substrate_snapshot(
                    memory=mem, consolidation=cons, path=path,
                )

    def test_load_rejects_dim_mismatch(self):
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=6, dim=32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            wrong_substrate = TorchFHRR(dim=64, device="cpu")
            with self.assertRaises(ValueError):
                load_substrate_snapshot(path=path, substrate=wrong_substrate)

    def test_positions_round_trip(self):
        """When positions are passed to save, load returns them in info."""
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.phase2.encoding import build_position_vectors
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=8)
        positions = build_position_vectors(substrate, count=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(
                memory=mem, consolidation=cons, path=path,
                positions=positions,
            )
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, _, info = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertIsNotNone(info["positions"])
            self.assertEqual(info["positions"].shape, (3, substrate.dim))
            for r in range(3):
                self.assertTrue(torch.equal(info["positions"][r], positions[r]))

    def test_positions_optional_old_snapshot_loads_none(self):
        """Snapshots saved without positions return None in info."""
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            _, _, info = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertIsNone(info["positions"])

    def test_empty_substrate_round_trips(self):
        """Edge case: a substrate with zero atoms saves and loads cleanly."""
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate = TorchFHRR(dim=16, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        cons = ConsolidationState(ConsolidationConfig(m=4), device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            new_substrate = TorchFHRR(dim=16, device="cpu")
            new_mem, new_cons, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            self.assertEqual(new_mem.stored_count, 0)
            self.assertEqual(new_cons.n_patterns, 0)


@unittest.skipIf(torch is None, "torch required")
class TestSnapshotIntegrationWithPhase5(unittest.TestCase):
    """Round-trip snapshot + use it as a Phase 5 schema source."""

    def test_get_schema_store_works_on_loaded_snapshot(self):
        import importlib.util, sys
        if "experiments_40" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "experiments_40", "experiments/40_phase5_branching.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["experiments_40"] = mod
            spec.loader.exec_module(mod)
        exp40 = sys.modules["experiments_40"]
        from energy_memory.phase4.snapshot import (
            save_substrate_snapshot, load_substrate_snapshot,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate, mem, cons = _build_substrate_with_state(seed=7, n_atoms=10)
        orig_schemas, orig_idx = exp40.get_schema_store(
            consolidation=cons, patterns=mem._pattern_matrix(),
            selection_rule="top_k_by_effective_strength", k=5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            save_substrate_snapshot(memory=mem, consolidation=cons, path=path)
            new_substrate = TorchFHRR(dim=substrate.dim, device="cpu")
            new_mem, new_cons, _ = load_substrate_snapshot(
                path=path, substrate=new_substrate,
            )
            new_schemas, new_idx = exp40.get_schema_store(
                consolidation=new_cons, patterns=new_mem._pattern_matrix(),
                selection_rule="top_k_by_effective_strength", k=5,
            )
            # Same indices selected, same schema vectors.
            self.assertTrue(torch.equal(orig_idx, new_idx))
            self.assertTrue(torch.equal(orig_schemas, new_schemas))


if __name__ == "__main__":
    unittest.main()
