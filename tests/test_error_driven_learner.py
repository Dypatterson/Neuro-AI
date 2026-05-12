import unittest

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch required")
class ErrorDrivenLearnerTests(unittest.TestCase):
    def _make_fixtures(self, vocab_size=20, dim=128):
        from energy_memory.phase2.corpus import Vocabulary
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        tokens = ["<UNK>", "<MASK>"] + [f"t{i}" for i in range(vocab_size - 2)]
        counts = {t: max(1, vocab_size - i) for i, t in enumerate(tokens)}
        vocab = Vocabulary(
            id_to_token=tokens,
            token_to_id={t: i for i, t in enumerate(tokens)},
            counts=counts,
        )
        substrate = TorchFHRR(dim=dim, seed=42, device="cpu")
        codebook = substrate.random_vectors(vocab_size)
        return vocab, substrate, codebook

    def _make_windows(self, n=50, window_size=4, vocab_range=(2, 15)):
        import random
        rng = random.Random(99)
        return [
            tuple(rng.randint(vocab_range[0], vocab_range[1]) for _ in range(window_size))
            for _ in range(n)
        ]

    def test_train_yields_consolidation_diagnostics(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=200, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=10,
            quality_threshold=0.99,
        )
        diagnostics = list(learner.train(
            landscape_windows=windows[:50],
            probe_windows=windows[50:],
            window_size=4,
        ))

        self.assertGreater(len(diagnostics), 0)
        for d in diagnostics:
            self.assertIn("consolidation", d)
            self.assertIn("pulled", d)
            self.assertIn("pushed", d)
            self.assertIn("mean_quality", d)
            self.assertIn("total_retrievals", d)
            self.assertIn("total_failures", d)
            self.assertIn("failure_rate", d)

    def test_codebook_changes_after_training(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        original = codebook.clone()
        windows = self._make_windows(n=200, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=10,
            quality_threshold=0.99,
        )
        for _ in learner.train(
            landscape_windows=windows[:50],
            probe_windows=windows[50:],
            window_size=4,
        ):
            pass

        diff = (learner.codebook - original).abs().mean().item()
        self.assertGreater(diff, 0.0)

    def test_consolidation_counter_increments(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=200, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=10,
            quality_threshold=0.99,
        )
        diagnostics = list(learner.train(
            landscape_windows=windows[:50],
            probe_windows=windows[50:],
            window_size=4,
        ))

        if len(diagnostics) >= 2:
            self.assertEqual(diagnostics[0]["consolidation"], 1)
            self.assertEqual(diagnostics[1]["consolidation"], 2)

    def test_buffer_clears_after_consolidation(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=200, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=10,
            quality_threshold=0.99,
        )
        for _ in learner.train(
            landscape_windows=windows[:50],
            probe_windows=windows[50:],
            window_size=4,
        ):
            self.assertEqual(len(learner._buffer), 0)

    def test_unk_targets_are_skipped(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        unk = vocab.unk_id
        windows_landscape = self._make_windows(n=20, window_size=4)
        windows_probe = [(unk, unk, unk, unk)] * 20

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=5,
            quality_threshold=0.99,
        )
        diagnostics = list(learner.train(
            landscape_windows=windows_landscape,
            probe_windows=windows_probe,
            window_size=4,
        ))

        self.assertEqual(len(diagnostics), 0)
        self.assertEqual(learner._total_retrievals, 0)

    def test_high_threshold_produces_failures(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=100, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=5,
            quality_threshold=0.99,
        )
        for _ in learner.train(
            landscape_windows=windows[:30],
            probe_windows=windows[30:],
            window_size=4,
        ):
            pass

        self.assertGreater(learner._total_failures, 0)

    def test_low_threshold_produces_few_failures(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=100, window_size=4)

        learner_high = ErrorDrivenLearner(
            substrate, codebook.clone(), vocab,
            consolidation_k=5,
            quality_threshold=0.99,
        )
        for _ in learner_high.train(
            landscape_windows=windows[:30],
            probe_windows=windows[30:],
            window_size=4,
        ):
            pass

        learner_low = ErrorDrivenLearner(
            substrate, codebook.clone(), vocab,
            consolidation_k=5,
            quality_threshold=-1.0,
        )
        for _ in learner_low.train(
            landscape_windows=windows[:30],
            probe_windows=windows[30:],
            window_size=4,
        ):
            pass

        self.assertLessEqual(
            learner_low._total_failures,
            learner_high._total_failures,
        )

    def test_codebook_vectors_remain_unit_modulus(self):
        from energy_memory.phase2.error_driven_learner import ErrorDrivenLearner

        vocab, substrate, codebook = self._make_fixtures()
        windows = self._make_windows(n=200, window_size=4)

        learner = ErrorDrivenLearner(
            substrate, codebook, vocab,
            consolidation_k=10,
            quality_threshold=0.99,
        )
        for _ in learner.train(
            landscape_windows=windows[:50],
            probe_windows=windows[50:],
            window_size=4,
        ):
            pass

        moduli = learner.codebook.abs()
        self.assertTrue(
            torch.allclose(moduli, torch.ones_like(moduli), atol=1e-4),
            f"Codebook vectors not unit modulus: max deviation {(moduli - 1.0).abs().max().item():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
