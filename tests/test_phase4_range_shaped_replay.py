"""Tests for Phase 4 range-shaped replay."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestRangeShapedReplaySampler(unittest.TestCase):

    def setUp(self):
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace

        self.ReplayStore = ReplayStore
        self.TrajectoryTrace = TrajectoryTrace

    def _trace(self, terms):
        return self.TrajectoryTrace(
            query=torch.zeros(8, dtype=torch.complex64),
            encoder_terms=None if terms is None else list(terms),
        )

    def _store(self, rows):
        store = self.ReplayStore(capacity=100)
        for terms, gate in rows:
            store.add(self._trace(terms), gate_signal=gate)
        return store

    def test_seeded_sampling_is_deterministic(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            ([(0, 0)], 0.4),
            ([(0, 1)], 0.6),
            ([(1, 2)], 0.5),
            ([(1, 3)], 0.5),
        ])
        sampler = RangeShapedReplaySampler(store)
        g1 = torch.Generator(device="cpu").manual_seed(123)
        g2 = torch.Generator(device="cpu").manual_seed(123)

        self.assertEqual(
            sampler.sample_pairs(40, generator=g1),
            sampler.sample_pairs(40, generator=g2),
        )

    def test_factorized_sampling_expands_cross_role_atom_support(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            ([(0, 0)], 1.0),
            ([(0, 1)], 1.0),
            ([(1, 2)], 1.0),
            ([(1, 3)], 1.0),
        ])
        sampler = RangeShapedReplaySampler(store)
        g = torch.Generator(device="cpu").manual_seed(17)
        pairs = [(r, a) for r, a, _ in sampler.sample_pairs(1000, generator=g)]

        natural_support = {(0, 0), (0, 1), (1, 2), (1, 3)}
        reached = set(pairs)
        self.assertGreater(len(reached), len(natural_support))
        self.assertTrue(any(pair not in natural_support for pair in reached))

    def test_priority_weighted_backing_trace_selection(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            ([(0, 0)], 0.01),
            ([(0, 0)], 0.99),
        ])
        sampler = RangeShapedReplaySampler(store)
        g = torch.Generator(device="cpu").manual_seed(5)
        picks = [tix for _, _, tix in sampler.sample_pairs(500, generator=g)]

        self.assertGreater(picks.count(1), 450)
        self.assertLess(picks.count(0), 50)

    def test_missing_encoder_terms_are_ignored_safely(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            (None, 99.0),
            ([(0, 3)], 1.0),
        ])
        sampler = RangeShapedReplaySampler(store)
        diag = sampler.marginal_diagnostics()
        g = torch.Generator(device="cpu").manual_seed(2)

        self.assertEqual(diag["n_missing_encoder_terms"], 1)
        self.assertEqual(sampler.sample_pairs(10, generator=g), [(0, 3, 1)] * 10)

    def test_missing_pair_behavior_and_closest_fallback(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            ([(0, 0)], 1.0),
            ([(1, 1)], 1.0),
        ])
        sampler = RangeShapedReplaySampler(store)
        g = torch.Generator(device="cpu").manual_seed(19)
        triples = sampler.sample_pairs(200, generator=g)

        self.assertTrue(any(tix is None for _, _, tix in triples))
        g = torch.Generator(device="cpu").manual_seed(19)
        closest = sampler.sample_indices(200, generator=g, fallback="closest")
        self.assertTrue(closest)
        self.assertTrue(all(idx in {0, 1} for idx in closest))

    def test_atom_support_smoothing_expands_atom_support(self):
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler

        store = self._store([
            ([(0, 0)], 1.0),
            ([(1, 1)], 1.0),
        ])
        sampler = RangeShapedReplaySampler(
            store,
            atom_count=4,
            atom_smoothing_alpha=1.0,
        )
        g = torch.Generator(device="cpu").manual_seed(23)
        atoms = {a for _, a, _ in sampler.sample_pairs(1000, generator=g)}

        self.assertTrue({2, 3}.issubset(atoms))

    def test_rebind_synthesis_variants_preserve_encoder_terms(self):
        from energy_memory.phase2.encoding import build_position_vectors
        from energy_memory.phase4.range_shaped_replay import (
            synthesize_single_binding_trace,
            synthesize_window_preserving_trace,
        )
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=64, seed=7, device="cpu")
        positions = build_position_vectors(substrate, count=4)
        codebook = substrate.random_vectors(10)

        single = synthesize_single_binding_trace(
            substrate, positions, codebook, role=2, atom=5,
        )
        self.assertEqual(single.encoder_terms, [(2, 5)])
        self.assertEqual(tuple(single.query.shape), (64,))

        window = synthesize_window_preserving_trace(
            substrate, positions, codebook, [(0, 3), (2, 4), (1, 8)],
        )
        self.assertEqual(window.encoder_terms, [(0, 3), (2, 4), (1, 8)])
        self.assertEqual(tuple(window.query.shape), (64,))


@unittest.skipIf(torch is None, "torch required")
class TestRangeShapedReplayRegression(unittest.TestCase):

    def _rectangularity(self, pairs, n_roles, n_atoms):
        H = torch.zeros(n_roles, n_atoms, dtype=torch.float64)
        for role, atom in pairs:
            H[role, atom] += 1.0
        H = H / H.sum()
        factored = H.sum(dim=1, keepdim=True) @ H.sum(dim=0, keepdim=True)
        eps = 1e-9
        p = (H.flatten() + eps)
        q = (factored.flatten() + eps)
        p = p / p.sum()
        q = q / q.sum()
        return float((p * (p.log() - q.log())).sum())

    def test_small_report_068_regression_kl_drops_and_coverage_expands(self):
        from energy_memory.phase2.encoding import (
            build_position_vectors,
            encode_window_with_provenance,
        )
        from energy_memory.phase4.range_shaped_replay import RangeShapedReplaySampler
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        n_roles = 4
        n_atoms = 16
        skew = 2
        substrate = TorchFHRR(dim=128, seed=11, device="cpu")
        positions = build_position_vectors(substrate, count=n_roles)
        codebook = substrate.random_vectors(n_atoms)
        store = ReplayStore(capacity=100)

        for i in range(80):
            token_ids = [
                (role * skew + ((i + role) % skew)) % n_atoms
                for role in range(n_roles)
            ]
            query, terms = encode_window_with_provenance(
                substrate, positions, codebook, token_ids,
            )
            store.add(
                TrajectoryTrace(query=query, encoder_terms=terms),
                gate_signal=0.5 + (i % 5) * 0.1,
            )

        baseline_pairs = []
        for trace in store.traces:
            baseline_pairs.extend(trace.encoder_terms)
        baseline_rect = self._rectangularity(baseline_pairs, n_roles, n_atoms)
        baseline_cells = len(set(baseline_pairs))

        sampler = RangeShapedReplaySampler(store)
        g = torch.Generator(device="cpu").manual_seed(31)
        sampled_pairs = [
            (role, atom)
            for role, atom, _ in sampler.sample_pairs(2000, generator=g)
        ]
        sampled_rect = self._rectangularity(sampled_pairs, n_roles, n_atoms)
        sampled_cells = len(set(sampled_pairs))

        self.assertLess(sampled_rect, baseline_rect * 0.25)
        self.assertGreater(sampled_cells, baseline_cells)


@unittest.skipIf(torch is None, "torch required")
class TestRangeShapedReplayPhase4Wiring(unittest.TestCase):

    def test_replay_config_validates_static_range_shaped_knobs(self):
        from energy_memory.phase4.replay_loop import ReplayConfig

        ReplayConfig(
            replay_sampler="range_shaped",
            range_shaped_fallback="rebind",
            range_shaped_rebind_mode="window_preserving",
            range_shaped_smoothing_alpha=0.1,
        )
        with self.assertRaises(ValueError):
            ReplayConfig(replay_sampler="adaptive")
        with self.assertRaises(ValueError):
            ReplayConfig(range_shaped_fallback="best_metric")

    def test_run_replay_cycle_accepts_range_shaped_rebind_static_config(self):
        from energy_memory.phase2.encoding import (
            build_position_vectors,
            encode_window,
            encode_window_with_provenance,
        )
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig,
            ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import ReplayConfig, UnifiedReplayMemory
        from energy_memory.phase4.trajectory import TracedHopfieldMemory, TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=128, seed=13, device="cpu")
        positions = build_position_vectors(substrate, count=2)
        codebook = substrate.random_vectors(6)
        memory = TracedHopfieldMemory(substrate)
        for window in [(0, 1), (2, 3), (4, 5)]:
            memory.store(encode_window(substrate, positions, codebook, window))
        consolidation = ConsolidationState(
            ConsolidationConfig(m=3, alpha=0.25, death_window=100),
            device="cpu",
        )
        replay = UnifiedReplayMemory(
            substrate,
            memory,
            consolidation,
            config=ReplayConfig(
                replay_sampler="range_shaped",
                range_shaped_fallback="rebind",
                range_shaped_rebind_mode="single_binding",
                replay_batch_size=6,
                resolve_threshold=2.0,
            ),
            replay_position_vectors=positions,
            replay_codebook=codebook,
        )
        replay.attach_initial_patterns()
        for window, gate in [((0, 1), 0.9), ((2, 3), 0.8)]:
            query, terms = encode_window_with_provenance(
                substrate, positions, codebook, window,
            )
            replay.store.add(
                TrajectoryTrace(query=query, encoder_terms=terms),
                gate_signal=gate,
            )

        stats = replay.run_replay_cycle(beta=5.0, max_iter=3)

        self.assertIn("sampled", stats)
        self.assertGreaterEqual(stats["sampled"], 1)


if __name__ == "__main__":
    unittest.main()
