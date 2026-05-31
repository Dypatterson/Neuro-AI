"""Tests for the cue-space decorrelator (batch ZCA whitening)."""

import unittest

import torch

from energy_memory.phase3.basin_readout import top_index_hits
from energy_memory.phase4.decorrelator import CueDecorrelator
from energy_memory.phase4.hetero_write import (
    HeteroConsolidationBuffer, heteroassociative_write, recall_top_index,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _correlated_toy(D=256, N=128, C=16, rho=0.6, seed=0):
    fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
    keys = fhrr.random_vectors(N)
    shared = fhrr.random_vectors(1)
    keys = fhrr.normalize((1.0 - rho) * keys + rho * shared)  # correlated keys
    values = fhrr.random_vectors(C)
    vidx = torch.randint(0, C, (N,), generator=fhrr.generator)
    return fhrr, keys, values, vidx


def _write_recall_rate(fhrr, keys, values, vidx):
    buf = HeteroConsolidationBuffer(dim=keys.shape[1], device="cpu")
    for i in range(len(vidx)):
        buf.add(keys[i], int(vidx[i]))
    buf.freeze()
    H = heteroassociative_write(buf, values, lr=0.5, epochs=20)
    ti, _, _ = recall_top_index(fhrr, H, keys, values, beta=10.0, max_iter=12)
    return top_index_hits(ti, vidx) / len(vidx)


class DecorrelatorTest(unittest.TestCase):
    def test_offdiag_decreases(self):
        fhrr, keys, values, vidx = _correlated_toy()
        dec = CueDecorrelator(dim=256).fit(keys)
        self.assertLess(dec.offdiag_after, dec.offdiag_before * 0.5)

    def test_apply_unit_magnitude(self):
        fhrr, keys, values, vidx = _correlated_toy()
        dec = CueDecorrelator(dim=256).fit(keys)
        out = dec.apply(keys)
        self.assertTrue(torch.allclose(out.abs(), torch.ones_like(out.abs()), atol=1e-5))

    def test_apply_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            CueDecorrelator(dim=256).apply(torch.zeros(4, 256, dtype=torch.complex64))

    def test_decorrelation_rescues_correlated_collapse(self):
        """Load-bearing: the decorrelator rescues the correlated-key collapse the
        way closed-form whitening does (Report 049 §4)."""
        fhrr, keys, values, vidx = _correlated_toy(D=256, N=128, rho=0.6, seed=1)
        raw_rate = _write_recall_rate(fhrr, keys, values, vidx)
        dec = CueDecorrelator(dim=256).fit(keys)
        dec_rate = _write_recall_rate(fhrr, dec.apply(keys), values, vidx)
        self.assertGreater(dec_rate, 0.8)
        self.assertGreater(dec_rate, raw_rate + 0.3)


if __name__ == "__main__":
    unittest.main()
