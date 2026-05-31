"""Tests for the Phase-4 heteroassociative consolidation write."""

import unittest

import torch

from energy_memory.phase3.basin_readout import top_index_hits
from energy_memory.phase4.hetero_write import (
    HeteroConsolidationBuffer,
    heteroassociative_write,
    precommit_swap_negatives,
    recall_top_index,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _toy(D=256, N=512, C=16, seed=0):
    fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(C)
    vidx = torch.randint(0, C, (N,), generator=fhrr.generator)
    return fhrr, keys, values, vidx


class BufferTest(unittest.TestCase):
    def test_add_after_freeze_raises(self):
        fhrr, keys, values, vidx = _toy(N=4)
        buf = HeteroConsolidationBuffer(dim=256, device="cpu")
        buf.add(keys[0], int(vidx[0]))
        buf.freeze()
        with self.assertRaises(RuntimeError):
            buf.add(keys[1], int(vidx[1]))

    def test_write_before_freeze_raises(self):
        # AH condition 3: the write must see a frozen (batch-offline) buffer.
        fhrr, keys, values, vidx = _toy(N=4)
        buf = HeteroConsolidationBuffer(dim=256, device="cpu")
        buf.add(keys[0], int(vidx[0]))
        with self.assertRaises(RuntimeError):
            heteroassociative_write(buf, values)

    def test_freeze_stacks_tensors(self):
        fhrr, keys, values, vidx = _toy(N=8)
        buf = HeteroConsolidationBuffer(dim=256, device="cpu")
        for i in range(8):
            buf.add(keys[i], int(vidx[i]))
        buf.freeze()
        self.assertTrue(buf.frozen)
        self.assertEqual(tuple(buf.keys.shape), (8, 256))
        self.assertEqual(tuple(buf.vidx.shape), (8,))
        self.assertEqual(len(buf), 8)


class SwapNegativeTest(unittest.TestCase):
    def test_negatives_distinct_and_seed_fixed(self):
        vidx = torch.tensor([0, 1, 2, 3, 0, 1])
        n1 = precommit_swap_negatives(vidx, 8, seed=3)
        n2 = precommit_swap_negatives(vidx, 8, seed=3)
        self.assertTrue(torch.equal(n1, n2))  # seed-fixed (precommitted)
        self.assertTrue(bool((n1 != vidx).all()))  # always distinct from true

    def test_needs_two_values(self):
        with self.assertRaises(ValueError):
            precommit_swap_negatives(torch.tensor([0, 0]), 1, seed=0)


class RescueTest(unittest.TestCase):
    """The write must rescue key-only recovery where store-as-is (bundle) fails
    in the hard (N/D>1) regime — the Report 049 direction."""

    def _store_as_is_rate(self, fhrr, keys, values, vidx):
        bundle = fhrr.normalize(
            torch.stack([keys[i] * values[int(vidx[i])] for i in range(len(vidx))]).sum(0))
        u = torch.stack([fhrr.unbind(bundle, keys[i]) for i in range(len(vidx))])
        from energy_memory.phase4.hetero_write import batched_hopfield_topindex
        ti, _, _ = batched_hopfield_topindex(fhrr, values, u, beta=10.0, max_iter=12)
        return top_index_hits(ti, vidx) / len(vidx)

    def test_write_rescues_keyonly(self):
        fhrr, keys, values, vidx = _toy(D=256, N=512, C=16, seed=1)  # N/D=2
        chance = 1.0 / 16
        sa = self._store_as_is_rate(fhrr, keys, values, vidx)

        buf = HeteroConsolidationBuffer(dim=256, device="cpu")
        for i in range(len(vidx)):
            buf.add(keys[i], int(vidx[i]))
        buf.freeze()
        H = heteroassociative_write(buf, values, lr=0.5, epochs=20)
        ti, _, _ = recall_top_index(fhrr, H, keys, values, beta=10.0, max_iter=12)
        write_rate = top_index_hits(ti, vidx) / len(vidx)

        # store-as-is is well below the write; the write is near-perfect.
        self.assertGreater(write_rate, 0.8)
        self.assertGreater(write_rate, sa + 0.3)
        self.assertGreater(sa, 0.0)  # not a degenerate zero comparison

    def test_contrastive_also_rescues(self):
        fhrr, keys, values, vidx = _toy(D=256, N=512, C=16, seed=2)
        buf = HeteroConsolidationBuffer(dim=256, device="cpu")
        for i in range(len(vidx)):
            buf.add(keys[i], int(vidx[i]))
        buf.freeze()
        H = heteroassociative_write(
            buf, values, lr=0.5, epochs=20, contrastive=True, lr_push=0.1, neg_seed=2)
        ti, _, _ = recall_top_index(fhrr, H, keys, values, beta=10.0, max_iter=12)
        self.assertGreater(top_index_hits(ti, vidx) / len(vidx), 0.8)


if __name__ == "__main__":
    unittest.main()
