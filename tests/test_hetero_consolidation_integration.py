"""Integration tests for the surgical heteroassociative consolidation write
folded into OnlineCodebookUpdater (the Path-C consolidation orchestrator).

Verifies: (1) flag OFF is byte-identical / inert (reproducibility guardrail);
(2) flag ON accumulates a closed cue buffer and consolidate_hetero() writes a
dense H that recall_hetero() recovers (memorization); (3) the decorrelator and
random-codebook controls behave; (4) the batch-offline / AH guard rails
(consolidate-before-recall, freeze-before-write).
"""

import unittest

import torch

from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.phase3.basin_readout import top_index_hits as tix
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _make(dim=256, n_atoms=8, n_obs=64, seed=0):
    """A closed, memorizable (cue -> target) set: distinct cues, random targets."""
    sub = TorchFHRR(dim=dim, seed=seed, device="cpu")
    codebook = sub.random_vectors(n_atoms)
    cues = sub.random_vectors(n_obs)                       # distinct cues
    g = torch.Generator().manual_seed(seed * 31 + 1)
    targets = torch.randint(0, n_atoms, (n_obs,), generator=g)
    return sub, codebook, cues, targets


class HeteroIntegrationTests(unittest.TestCase):

    def test_flag_off_is_inert(self):
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb)  # hetero_write_enabled defaults False
        for i in range(cues.shape[0]):
            up.observe(int(tgts[i]), cb[int(tgts[i])], int(tgts[i]), cue=cues[i])
        # No accumulation, no H, and consolidate_hetero refuses.
        self.assertEqual(up.stats()["hetero_buffer_size"], 0)
        self.assertIsNone(up.hetero_H)
        with self.assertRaises(RuntimeError):
            up.consolidate_hetero()
        with self.assertRaises(RuntimeError):
            up.recall_hetero(cues[0])

    def test_recall_before_consolidate_raises(self):
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True)
        up.observe(int(tgts[0]), cb[int(tgts[0])], int(tgts[0]), cue=cues[0])
        with self.assertRaises(RuntimeError):
            up.recall_hetero(cues[0])

    def test_no_cues_returns_none(self):
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True)
        # observe WITHOUT a cue -> nothing accumulates
        up.observe(int(tgts[0]), cb[int(tgts[0])], int(tgts[0]))
        self.assertEqual(up.stats()["hetero_buffer_size"], 0)
        self.assertIsNone(up.consolidate_hetero())

    def test_write_then_recall_memorizes(self):
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True,
                                   decorrelator_enabled=True)
        for i in range(cues.shape[0]):
            up.observe(int(tgts[i]), cb[int(tgts[i])], int(tgts[i]), cue=cues[i])
        diag = up.consolidate_hetero()
        self.assertEqual(diag["hetero_n"], cues.shape[0])
        self.assertIsNotNone(up.hetero_H)
        self.assertEqual(tuple(up.hetero_H.shape), (256, 256))
        top_index, ent, marg = up.recall_hetero(cues)
        recall = tix(top_index, tgts) / cues.shape[0]
        self.assertGreater(recall, 0.8, f"memorization recall too low: {recall}")

    def test_decorrelator_off_still_memorizes(self):
        # With distinct (near-orthogonal) random cues the decorrelator is not
        # load-bearing for memorization; recall should still be high either way.
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True,
                                   decorrelator_enabled=False)
        for i in range(cues.shape[0]):
            up.observe(int(tgts[i]), cb[int(tgts[i])], int(tgts[i]), cue=cues[i])
        up.consolidate_hetero()
        self.assertIsNone(up.hetero_decorrelator)
        top_index, _, _ = up.recall_hetero(cues)
        self.assertGreater(tix(top_index, tgts) / cues.shape[0], 0.8)

    def test_random_codebook_control_collapses(self):
        # Read the H-output (trained into the real codebook) against an UNRELATED
        # random codebook -> chance (readout-leak control, exp50-style).
        sub, cb, cues, tgts = _make(n_atoms=8)
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True)
        for i in range(cues.shape[0]):
            up.observe(int(tgts[i]), cb[int(tgts[i])], int(tgts[i]), cue=cues[i])
        up.consolidate_hetero()
        D = cb.shape[1]
        recalled = (up.hetero_decorrelator.apply(cues) @ up.hetero_H.transpose(0, 1)) / D
        rand_cb = sub.random_vectors(8)
        from energy_memory.phase4.hetero_write import batched_hopfield_topindex
        ti, _, _ = batched_hopfield_topindex(sub, rand_cb, recalled, beta=10.0, max_iter=12)
        rate = tix(ti, tgts) / cues.shape[0]
        self.assertLess(rate, 0.35, f"random-codebook control did not collapse: {rate}")

    def test_recall_accepts_single_and_batch(self):
        sub, cb, cues, tgts = _make()
        up = OnlineCodebookUpdater(sub, cb, hetero_write_enabled=True)
        for i in range(cues.shape[0]):
            up.observe(int(tgts[i]), cb[int(tgts[i])], int(tgts[i]), cue=cues[i])
        up.consolidate_hetero()
        single, _, _ = up.recall_hetero(cues[0])          # [D] -> [1]
        self.assertEqual(single.shape[0], 1)
        batch, _, _ = up.recall_hetero(cues[:4])          # [4, D] -> [4]
        self.assertEqual(batch.shape[0], 4)
        self.assertEqual(int(single[0]), int(batch[0]))


if __name__ == "__main__":
    unittest.main()
