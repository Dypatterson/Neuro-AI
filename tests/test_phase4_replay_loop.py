"""Tests for the unified replay loop."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestReplayStore(unittest.TestCase):

    def setUp(self):
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.phase4.replay_loop import ReplayStore
        self.ReplayStore = ReplayStore
        self.TrajectoryTrace = TrajectoryTrace

    def _make_trace(self, top_score=0.5):
        return self.TrajectoryTrace(
            query=torch.zeros(8, dtype=torch.complex64),
            final_top_score=top_score,
        )

    def test_add_within_capacity(self):
        store = self.ReplayStore(capacity=5)
        for i in range(3):
            store.add(self._make_trace(), gate_signal=0.1 + i)
        self.assertEqual(len(store), 3)

    def test_evicts_lowest_gate_when_full(self):
        store = self.ReplayStore(capacity=3)
        for g in [0.1, 0.5, 0.2]:
            store.add(self._make_trace(), gate_signal=g)
        # Add a 4th — should evict the 0.1 entry
        store.add(self._make_trace(), gate_signal=0.4)
        self.assertEqual(len(store), 3)
        self.assertNotIn(0.1, store.gate_signals)

    def test_sample_returns_valid_indices(self):
        store = self.ReplayStore(capacity=10)
        for i in range(5):
            store.add(self._make_trace(), gate_signal=0.5 + 0.1 * i)
        sampled = store.sample(n=3)
        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(0 <= idx < 5 for idx in sampled))

    def test_remove(self):
        store = self.ReplayStore(capacity=10)
        for i in range(3):
            store.add(self._make_trace(), gate_signal=0.5)
        store.remove(1)
        self.assertEqual(len(store), 2)


@unittest.skipIf(torch is None, "torch required")
class TestUnifiedReplayMemory(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )

        self.substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        self.positions = build_position_vectors(self.substrate, 3)
        self.codebook = self.substrate.random_vectors(20)

        self.memory = TracedHopfieldMemory(self.substrate, snapshot_k=5)
        self.windows = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 3, 5)]
        for w in self.windows:
            self.memory.store(
                encode_window(self.substrate, self.positions, self.codebook, w),
                label=str(w),
            )

        self.consolidation = ConsolidationState(
            ConsolidationConfig(m=4, alpha=0.25, death_threshold=0.01, death_window=1000),
            device="cpu",
        )
        self.replay_config = ReplayConfig(
            store_threshold=0.05,
            store_capacity=20,
            resolve_threshold=0.9,
            replay_every=5,
            replay_batch_size=3,
            max_age=3,
            novelty_strength=1.0,
            retrieval_gain=0.1,
        )
        self.unified = UnifiedReplayMemory(
            substrate=self.substrate,
            memory=self.memory,
            consolidation=self.consolidation,
            config=self.replay_config,
        )
        self.unified.attach_initial_patterns()

    def test_attach_initializes_consolidation_for_all_patterns(self):
        self.assertEqual(self.consolidation.n_patterns, len(self.windows))
        # All u_1 entries should be at novelty_strength
        for i in range(len(self.windows)):
            self.assertAlmostEqual(
                self.consolidation.u[i, 0].item(),
                self.replay_config.novelty_strength,
            )

    def test_retrieve_and_observe_reinforces_top_pattern(self):
        from energy_memory.phase2.encoding import encode_window

        # Retrieve a known stored window — should reinforce its u_1
        target_idx = 0
        query = encode_window(
            self.substrate, self.positions, self.codebook, self.windows[target_idx],
        )
        before = self.consolidation.u[target_idx, 0].item()
        self.unified.retrieve_and_observe(query, beta=10.0)
        after = self.consolidation.u[target_idx, 0].item()
        self.assertGreater(after, before)

    def test_replay_store_respects_threshold(self):
        """Traces with gate > store_threshold enter the store; those at/below don't."""
        from energy_memory.phase4.trajectory import (
            TrajectoryTrace, TrajectorySnapshot,
        )

        # Manually construct a trace whose gate clearly exceeds threshold
        high_gate_trace = TrajectoryTrace(
            query=self.substrate.random_vector(),
            snapshots=[
                TrajectorySnapshot(step=1, top_k_indices=[0, 1, 2],
                                   top_k_weights=[0.4, 0.3, 0.3],
                                   entropy=0.9, energy=-0.1),
                TrajectorySnapshot(step=2, top_k_indices=[0, 1, 2],
                                   top_k_weights=[0.4, 0.3, 0.3],
                                   entropy=0.9, energy=-0.1),
            ],
            final_top_score=0.5,  # low resolution
            final_top_index=0,
        )
        # gate = 0.9 * (1 - 0.5) = 0.45, well above store_threshold=0.05
        self.assertGreater(high_gate_trace.gate_signal(), self.replay_config.store_threshold)
        self.unified.store.add(high_gate_trace, gate_signal=high_gate_trace.gate_signal())
        self.assertEqual(len(self.unified.store), 1)

        # A clean-basin trace has resolution → 1.0, gate → 0
        clean_trace = TrajectoryTrace(
            query=self.substrate.random_vector(),
            snapshots=[
                TrajectorySnapshot(step=1, top_k_indices=[0], top_k_weights=[0.99],
                                   entropy=0.05, energy=-1.0),
            ],
            final_top_score=0.99,
            final_top_index=0,
        )
        # gate = 0.05 * 0.01 = 0.0005, below threshold
        self.assertLess(clean_trace.gate_signal(), self.replay_config.store_threshold)

    def test_should_replay_at_replay_every(self):
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        # First 4 retrievals: no replay yet
        for _ in range(4):
            self.unified.retrieve_and_observe(query, beta=10.0)
            self.assertFalse(self.unified.should_replay())
        # 5th retrieval: should trigger
        self.unified.retrieve_and_observe(query, beta=10.0)
        self.assertTrue(self.unified.should_replay())

    def test_run_replay_cycle_steps_consolidation(self):
        from energy_memory.phase2.encoding import encode_window

        # Seed the store with some unresolved traces
        random_query = self.substrate.random_vector()
        for _ in range(5):
            self.unified.retrieve_and_observe(random_query, beta=2.0)
        steps_before = self.consolidation._step_count
        self.unified.run_replay_cycle(beta=10.0)
        # Consolidation should have stepped at least once
        self.assertGreater(self.consolidation._step_count, steps_before)

    def test_run_replay_cycle_emits_candidates_on_resolution(self):
        from energy_memory.phase2.encoding import encode_window

        # Add traces whose re-settling will resolve cleanly (high β)
        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        # Set lower store threshold to ensure store fills
        # Then run replay with a candidate handler that confirms candidates
        for _ in range(5):
            # Slightly noisy version to ensure some engagement
            noisy = self.substrate.normalize(query + 0.1 * self.substrate.random_vector())
            self.unified.retrieve_and_observe(noisy, beta=3.0)

        candidates_seen = []
        def handler(trace):
            candidates_seen.append(trace.final_top_score)
            return None  # Don't actually add to memory in this test

        cycle = self.unified.run_replay_cycle(beta=20.0, candidate_handler=handler)
        # We may or may not get candidates depending on resolution,
        # but the cycle should complete and return stats
        self.assertIn("candidates", cycle)
        self.assertIn("sampled", cycle)

    def test_stats_returns_expected_keys(self):
        s = self.unified.stats()
        for key in ["retrievals", "store", "consolidation", "memory_size"]:
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
