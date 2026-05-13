"""Tests for the tag-count + inhibition-of-return ReplayStore upgrades.

Brainstorm Idea 4a (Joo & Frank 2023): overlapping queries should
collapse onto a single trace with an incremented tag_count, not stored
as duplicates. Sampling weight then scales with re-tag frequency.

Brainstorm Idea 4c (Biderman SFMA 2023): inhibition of return — a
trace that is sampled for replay has its suppression multiplier
decayed; non-sampled traces recover. Prevents monopolization.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TagCountTests(unittest.TestCase):
    def setUp(self):
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        self.substrate = TorchFHRR(dim=128, seed=7, device="cpu")
        self.ReplayStore = ReplayStore
        self.TrajectoryTrace = TrajectoryTrace

    def _trace(self, query):
        return self.TrajectoryTrace(query=query, final_top_score=0.5)

    def test_overlap_increments_tag_count_instead_of_appending(self):
        base = self.substrate.random_vectors(1)[0]
        store = self.ReplayStore(
            capacity=10,
            substrate=self.substrate,
            tag_overlap_threshold=0.7,
        )
        store.add(self._trace(base), gate_signal=0.4)
        # Near-duplicate (very small perturbation): should collapse.
        nearby = self.substrate.perturb(base, noise=0.01)
        store.add(self._trace(nearby), gate_signal=0.6)

        self.assertEqual(len(store), 1)
        self.assertEqual(store.tag_counts[0], 2)
        # max gate is retained
        self.assertAlmostEqual(store.gate_signals[0], 0.6, places=6)

    def test_distinct_queries_do_not_collapse(self):
        a, b = self.substrate.random_vectors(2)
        store = self.ReplayStore(
            capacity=10,
            substrate=self.substrate,
            tag_overlap_threshold=0.7,
        )
        store.add(self._trace(a), gate_signal=0.4)
        store.add(self._trace(b), gate_signal=0.5)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.tag_counts, [1, 1])

    def test_disabled_when_threshold_is_none(self):
        # Backwards compatibility: substrate without threshold disables collapse.
        base = self.substrate.random_vectors(1)[0]
        store = self.ReplayStore(capacity=10)  # legacy ctor signature
        store.add(self._trace(base), gate_signal=0.4)
        store.add(self._trace(base), gate_signal=0.4)
        self.assertEqual(len(store), 2)

    def test_tag_count_boosts_sampling_weight(self):
        a, b = self.substrate.random_vectors(2)
        store = self.ReplayStore(
            capacity=10,
            substrate=self.substrate,
            tag_overlap_threshold=0.7,
        )
        # Trace b is observed many times (high tag_count) but with low gate;
        # trace a is observed once with mid gate. With multiplicative weight,
        # the heavily-tagged trace should be sampled the majority of the time.
        store.add(self._trace(a), gate_signal=0.4)
        store.add(self._trace(b), gate_signal=0.1)
        for _ in range(40):
            store.add(self._trace(self.substrate.perturb(b, noise=0.01)), gate_signal=0.1)

        self.assertEqual(len(store), 2)
        # Idx for b is 1 (added second)
        self.assertGreaterEqual(store.tag_counts[1], 40)
        gen = torch.Generator(device="cpu").manual_seed(13)
        b_hits = 0
        n_trials = 200
        for _ in range(n_trials):
            sampled = store.sample(n=1, generator=gen)
            if sampled and sampled[0] == 1:
                b_hits += 1
        self.assertGreater(b_hits / n_trials, 0.8)


@unittest.skipIf(torch is None, "torch required")
class InhibitionOfReturnTests(unittest.TestCase):
    def setUp(self):
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        self.substrate = TorchFHRR(dim=128, seed=23, device="cpu")
        self.ReplayStore = ReplayStore
        self.TrajectoryTrace = TrajectoryTrace

    def _trace(self):
        q = self.substrate.random_vectors(1)[0]
        return self.TrajectoryTrace(query=q, final_top_score=0.5)

    def test_sampling_decays_suppression_for_sampled_trace(self):
        store = self.ReplayStore(
            capacity=10,
            suppression_decay=0.5,
            suppression_recovery=0.1,
        )
        for _ in range(3):
            store.add(self._trace(), gate_signal=0.5)

        # Force index 0 to be picked by setting its priority very high.
        store.gate_signals[0] = 10.0
        gen = torch.Generator(device="cpu").manual_seed(1)
        store.sample(n=1, generator=gen)
        self.assertAlmostEqual(store.suppression[0], 0.5, places=6)
        # Non-sampled entries recover toward 1.0 (already at 1.0, capped)
        self.assertAlmostEqual(store.suppression[1], 1.0, places=6)

    def test_repeated_sampling_drives_suppression_toward_zero(self):
        store = self.ReplayStore(
            capacity=10,
            suppression_decay=0.5,
            suppression_recovery=0.0,
        )
        for _ in range(2):
            store.add(self._trace(), gate_signal=0.5)
        store.gate_signals[0] = 100.0
        gen = torch.Generator(device="cpu").manual_seed(2)
        # Sample five times; each picks index 0 (much higher gate), each halves suppression.
        for _ in range(5):
            store.sample(n=1, generator=gen)
        self.assertLess(store.suppression[0], 0.1)

    def test_default_config_is_no_op_when_disabled(self):
        store = self.ReplayStore(
            capacity=10,
            suppression_decay=1.0,
            suppression_recovery=0.0,
        )
        for _ in range(3):
            store.add(self._trace(), gate_signal=0.5)
        gen = torch.Generator(device="cpu").manual_seed(3)
        store.sample(n=2, generator=gen)
        self.assertEqual(store.suppression, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
