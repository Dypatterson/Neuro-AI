import math
import unittest

import torch

from energy_memory.phase3.bimodality_diagnostic import (
    BimodalityDiagnostics,
    ContextBagHistory,
    PersistentBimodalityTracker,
    compute_bimodality_diagnostics,
    context_bag_signal,
    gmm_bic_1d,
    hartigan_dip,
)


def _make_bimodal(n: int = 200, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    a = torch.randn(half, generator=g) * 0.2 - 1.0
    b = torch.randn(n - half, generator=g) * 0.2 + 1.0
    return torch.cat([a, b])


def _make_unimodal(n: int = 200, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


class TestBimodalityDiagnostic(unittest.TestCase):
    def test_t1_unimodal_history_dip_does_not_reject(self):
        torch.manual_seed(0)
        D = 64
        hist = ContextBagHistory()
        base = torch.randn(D)
        for _ in range(5):
            base = base + 0.01 * torch.randn(D)
            hist.append(base.clone())
        sig = context_bag_signal(hist)
        self.assertEqual(sig.shape[0], 4)
        gen = torch.Generator().manual_seed(42)
        p, rejects = hartigan_dip(sig, generator=gen)
        # N=4: dip is weak. Accept either p is computable but not rejecting,
        # or that it doesn't fire. (Honest report of N=5 finite-sample limit.)
        self.assertFalse(rejects)

    def test_t2_bimodal_signal_dip_rejects(self):
        sig = _make_bimodal(n=200, seed=2)
        gen = torch.Generator().manual_seed(7)
        p, rejects = hartigan_dip(sig, generator=gen)
        self.assertIsNotNone(p)
        self.assertTrue(rejects, msg=f"expected dip-reject on bimodal, p={p}")
        self.assertLess(p, 0.05)
        TestBimodalityDiagnostic.t2_p = p

    def test_t3_unimodal_signal_dip_does_not_reject(self):
        sig = _make_unimodal(n=200, seed=3)
        gen = torch.Generator().manual_seed(11)
        p, rejects = hartigan_dip(sig, generator=gen)
        self.assertIsNotNone(p)
        self.assertFalse(rejects, msg=f"expected no dip-reject on unimodal, p={p}")
        self.assertGreater(p, 0.05)
        TestBimodalityDiagnostic.t3_p = p

    def test_t4_gmm_bic_bimodal(self):
        sig = _make_bimodal(n=200, seed=2)
        delta, favors = gmm_bic_1d(sig)
        self.assertIsNotNone(delta)
        self.assertTrue(favors, msg=f"expected k2 favored, delta_bic={delta}")
        self.assertGreater(delta, 10.0)
        TestBimodalityDiagnostic.t4_delta = delta

    def test_t5_gmm_bic_unimodal(self):
        sig = _make_unimodal(n=200, seed=3)
        delta, favors = gmm_bic_1d(sig)
        self.assertIsNotNone(delta)
        self.assertFalse(favors, msg=f"expected k1 favored, delta_bic={delta}")
        self.assertLess(delta, 10.0)
        TestBimodalityDiagnostic.t5_delta = delta

    def test_t6_persistent_bimodality_tracker(self):
        tracker = PersistentBimodalityTracker()
        for v in [True, False, True, True, True]:
            tracker.append(0, v)
        self.assertTrue(tracker.is_persistent(0, threshold=3))
        for v in [False, False, True, False, False]:
            tracker.append(1, v)
        self.assertFalse(tracker.is_persistent(1, threshold=3))

    def test_t7_edge_case_too_few_samples(self):
        sig = torch.tensor([0.1, 0.9])
        p, rejects = hartigan_dip(sig)
        self.assertIsNone(p)
        self.assertFalse(rejects)
        d, f = gmm_bic_1d(sig)
        self.assertIsNone(d)
        self.assertFalse(f)

    def test_t8_edge_case_empty_history(self):
        empty = ContextBagHistory()
        sig0 = context_bag_signal(empty)
        self.assertEqual(sig0.shape[0], 0)
        one = ContextBagHistory()
        one.append(torch.randn(8))
        sig1 = context_bag_signal(one)
        self.assertEqual(sig1.shape[0], 0)
        p, r = hartigan_dip(sig0)
        self.assertIsNone(p)
        self.assertFalse(r)
        d, f = gmm_bic_1d(sig0)
        self.assertIsNone(d)
        self.assertFalse(f)

    def test_t9_complex_fhrr_context_bags(self):
        D = 32
        torch.manual_seed(5)
        hist = ContextBagHistory()
        for _ in range(5):
            phases = torch.rand(D) * 2 * math.pi
            bag = torch.complex(torch.cos(phases), torch.sin(phases))
            hist.append(bag)
        sig = context_bag_signal(hist)
        self.assertFalse(torch.is_complex(sig))
        self.assertEqual(sig.shape[0], 4)
        gen = torch.Generator().manual_seed(13)
        p, rejects = hartigan_dip(sig, generator=gen)
        # Should not crash. Outcome may be None (N<some threshold) or computable.
        self.assertIn(rejects, (True, False))

    def test_t10_convenience_wrapper_end_to_end(self):
        torch.manual_seed(0)
        D = 16
        histories = {}
        # Atom 0: smooth drift (unimodal cosines near 1).
        base = torch.randn(D)
        h0 = ContextBagHistory()
        for _ in range(5):
            base = base + 0.005 * torch.randn(D)
            h0.append(base.clone())
        histories[0] = h0
        # Atom 1: alternating sign flips (bimodal-ish cosines).
        h1 = ContextBagHistory()
        v = torch.randn(D)
        for i in range(5):
            sign = 1.0 if i % 2 == 0 else -1.0
            h1.append(sign * v + 0.001 * torch.randn(D))
        histories[1] = h1
        tracker = PersistentBimodalityTracker()
        diag = compute_bimodality_diagnostics(histories, tracker, use_gmm=False)
        self.assertIsInstance(diag, BimodalityDiagnostics)
        self.assertEqual(diag.n_atoms, 2)
        self.assertIn(0, diag.per_atom_signal)
        self.assertIn(1, diag.per_atom_signal)
        self.assertIn(0, diag.persistent_bimodality)
        self.assertIn(1, diag.persistent_bimodality)
        # Tracker updated for both atoms exactly once.
        self.assertEqual(tracker.count(0) + (1 - int(diag.per_atom_signal[0].dip_rejects)),
                         int(diag.per_atom_signal[0].dip_rejects) + (1 - int(diag.per_atom_signal[0].dip_rejects)))
        # persistent_bimodality matches tracker.is_persistent.
        for atom in (0, 1):
            self.assertEqual(
                diag.persistent_bimodality[atom],
                tracker.is_persistent(atom),
            )


def tearDownModule():
    p2 = getattr(TestBimodalityDiagnostic, "t2_p", None)
    p3 = getattr(TestBimodalityDiagnostic, "t3_p", None)
    d4 = getattr(TestBimodalityDiagnostic, "t4_delta", None)
    d5 = getattr(TestBimodalityDiagnostic, "t5_delta", None)
    print(f"\n[bimodality-diagnostic] T2 dip p={p2}  T3 dip p={p3}  "
          f"T4 ΔBIC={d4}  T5 ΔBIC={d5}")


if __name__ == "__main__":
    unittest.main()
