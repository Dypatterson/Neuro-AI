"""Tests for the trajectory trace mechanism."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestTrajectoryTrace(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory

        self.substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        self.positions = build_position_vectors(self.substrate, 3)
        self.codebook = self.substrate.random_vectors(20)
        self.memory = TracedHopfieldMemory(self.substrate, snapshot_k=5)

        windows = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 3, 5), (2, 4, 6)]
        for w in windows:
            self.memory.store(
                encode_window(self.substrate, self.positions, self.codebook, w),
                label=str(w),
            )
        self.windows = windows

    def test_trace_captures_per_step_snapshots(self):
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        result, trace = self.memory.retrieve_with_trace(
            query, beta=10.0, max_iter=8,
        )
        self.assertGreater(len(trace.snapshots), 0)
        self.assertEqual(len(trace.snapshots), result.iterations)
        for snap in trace.snapshots:
            self.assertGreaterEqual(len(snap.top_k_indices), 1)
            self.assertEqual(len(snap.top_k_indices), len(snap.top_k_weights))
            self.assertGreaterEqual(snap.entropy, 0.0)

    def test_trace_engagement_and_resolution(self):
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        _, trace = self.memory.retrieve_with_trace(query, beta=10.0, max_iter=8)

        # Engagement should be a real number ≥ 0
        e = trace.engagement()
        self.assertGreaterEqual(e, 0.0)
        # Resolution should be the final top score
        self.assertEqual(trace.resolution(), trace.final_top_score)
        # Gate signal = engagement * (1 - resolution)
        self.assertAlmostEqual(
            trace.gate_signal(),
            e * (1.0 - trace.final_top_score),
            places=5,
        )

    def test_high_beta_low_engagement(self):
        """At very high β, retrieval is sharp → low entropy → low engagement."""
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        _, trace_high = self.memory.retrieve_with_trace(query, beta=50.0, max_iter=8)
        _, trace_low = self.memory.retrieve_with_trace(query, beta=1.0, max_iter=8)
        # Higher β → lower engagement
        self.assertLess(trace_high.engagement(), trace_low.engagement())

    def test_trace_query_preserved(self):
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (4, 5, 6))
        _, trace = self.memory.retrieve_with_trace(query, beta=10.0)
        self.assertTrue(torch.allclose(trace.query, query, atol=1e-6))

    def test_retrieve_method_still_works(self):
        """The parent class retrieve() shouldn't be broken by the subclass."""
        from energy_memory.phase2.encoding import encode_window

        query = encode_window(self.substrate, self.positions, self.codebook, (1, 2, 3))
        result = self.memory.retrieve(query, beta=10.0, max_iter=8)
        self.assertIsNotNone(result.top_index)


if __name__ == "__main__":
    unittest.main()
