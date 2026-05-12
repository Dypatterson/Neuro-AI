import unittest

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch required")
class CodebookLearnerTests(unittest.TestCase):
    def _make_vocab_and_substrate(self, vocab_size=20):
        from energy_memory.phase2.corpus import Vocabulary
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        tokens = ["<UNK>", "<MASK>"] + [f"t{i}" for i in range(vocab_size - 2)]
        counts = {t: max(1, vocab_size - i) for i, t in enumerate(tokens)}
        vocab = Vocabulary(
            id_to_token=tokens,
            token_to_id={t: i for i, t in enumerate(tokens)},
            counts=counts,
        )
        substrate = TorchFHRR(dim=128, seed=42, device="cpu")
        codebook = substrate.random_vectors(vocab_size)
        return vocab, substrate, codebook

    def test_cooccurrence_matrix_shape_and_normalization(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5), (3, 4, 5, 6), (2, 6, 7, 8)]
        learner = CodebookLearner(substrate, codebook, vocab)
        C = learner.build_cooccurrence(windows)

        self.assertEqual(C.shape, (len(vocab.id_to_token), len(vocab.id_to_token)))
        row_sums = C.sum(dim=1)
        for tid in range(len(vocab.id_to_token)):
            if row_sums[tid] > 0:
                self.assertAlmostEqual(float(row_sums[tid]), 1.0, places=5)

    def test_special_tokens_zeroed_in_cooccurrence(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(0, 1, 2, 3), (1, 2, 3, 4)]
        learner = CodebookLearner(substrate, codebook, vocab)
        C = learner.build_cooccurrence(windows)

        self.assertEqual(float(C[vocab.unk_id].sum()), 0.0)
        self.assertEqual(float(C[vocab.mask_id].sum()), 0.0)

    def test_training_produces_different_codebook(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5)] * 50 + [(6, 7, 8, 9)] * 50
        learner = CodebookLearner(substrate, codebook, vocab, lr=0.05)

        for _ in learner.train(windows, epochs=3):
            pass

        diff = (learner.codebook - codebook).abs().mean().item()
        self.assertGreater(diff, 0.0)

    def test_cooccurring_tokens_become_more_similar(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5)] * 200

        sim_before = float(substrate.similarity(codebook[2], codebook[3]))

        learner = CodebookLearner(substrate, codebook, vocab, lr=0.05)
        for _ in learner.train(windows, epochs=5):
            pass

        sim_after = float(substrate.similarity(learner.codebook[2], learner.codebook[3]))
        self.assertGreater(sim_after, sim_before)

    def test_non_cooccurring_tokens_stay_dissimilar(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5)] * 200

        sim_before = float(substrate.similarity(codebook[2], codebook[10]))

        learner = CodebookLearner(substrate, codebook, vocab, lr=0.05)
        for _ in learner.train(windows, epochs=5):
            pass

        sim_after = float(substrate.similarity(learner.codebook[2], learner.codebook[10]))
        self.assertAlmostEqual(sim_before, sim_after, delta=0.15)

    def test_repulsion_prevents_collapse(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        # Force two tokens to be identical
        codebook[3] = codebook[2].clone()
        learner = CodebookLearner(
            substrate, codebook, vocab,
            repulsion_threshold=0.5, repulsion_strength=0.1,
        )

        count = learner._apply_repulsion()
        self.assertGreater(count, 0)
        sim = float(substrate.similarity(learner.codebook[2], learner.codebook[3]))
        self.assertLess(sim, 1.0)

    def test_diagnostics_yielded_per_epoch(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5)] * 20
        learner = CodebookLearner(substrate, codebook, vocab)
        log = list(learner.train(windows, epochs=3))

        self.assertEqual(len(log), 3)
        for entry in log:
            self.assertIn("epoch", entry)
            self.assertIn("lr", entry)
            self.assertIn("mean_drift", entry)
            self.assertIn("max_sim", entry)
            self.assertIn("repulsion_count", entry)

    def test_lr_decays_across_epochs(self):
        from energy_memory.phase2.codebook_learner import CodebookLearner

        vocab, substrate, codebook = self._make_vocab_and_substrate()
        windows = [(2, 3, 4, 5)] * 20
        learner = CodebookLearner(substrate, codebook, vocab, lr=0.1, lr_decay=0.5)
        log = list(learner.train(windows, epochs=3))

        self.assertAlmostEqual(log[0]["lr"], 0.1, places=5)
        self.assertAlmostEqual(log[1]["lr"], 0.05, places=5)
        self.assertAlmostEqual(log[2]["lr"], 0.025, places=5)


if __name__ == "__main__":
    unittest.main()
