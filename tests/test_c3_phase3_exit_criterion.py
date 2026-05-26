"""Minimal unit tests for the C.3 Phase 3 exit-criterion driver.

These tests exercise:
  T1 — the driver module imports without error.
  T2 — ``compute_codebook_regime_diagnostics`` stratifies a tiny
       synthetic codebook the way the C.3 driver consumes it.
  T3 — the calibrated ``theta_prime_fn`` loads correctly from the JSON
       artifact at ``notes/emergent-codebook/theta_prime_calibration.json``.
  T4 — Recall@K computation on a tiny memorized landscape recovers the
       masked token from the standard codebook.
"""

from __future__ import annotations

import importlib
import unittest

import torch

from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase3.regime_diagnostic import (
    compute_codebook_regime_diagnostics,
)
from energy_memory.phase3.theta_prime_calibration import (
    load_theta_prime_calibration,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

# Import the driver via importlib so the test passes even when invoked
# from outside the repo root (tests live in tests/, driver lives in
# experiments/, which is not on sys.path by default; we add it inside
# the test).
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


class TestC3DriverImport(unittest.TestCase):
    """T1 — the driver imports cleanly."""

    def test_import_driver(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        self.assertTrue(hasattr(mod, "run"))
        self.assertTrue(hasattr(mod, "main"))
        # Public-shape helpers used inside the run loop.
        self.assertTrue(hasattr(mod, "_evaluate_recall_at_k"))
        self.assertTrue(hasattr(mod, "_aggregate_by_stratum"))


class TestRegimeDiagnosticIntegration(unittest.TestCase):
    """T2 — regime classification on a tiny synthetic codebook."""

    def test_tight_codebook_classified_tight(self):
        # Two near-identical atoms repeated several times → tight.
        torch.manual_seed(0)
        D = 64
        base = torch.complex(torch.cos(torch.zeros(D)), torch.sin(torch.zeros(D)))
        # Six near-duplicate copies plus tiny phase noise.
        codebook = torch.stack(
            [base * torch.exp(1j * 1e-4 * torch.randn(D)) for _ in range(6)],
            dim=0,
        ).to(torch.complex64)
        # Use beta=1.0 so default theta_prime = 1/beta = 1.0; anything
        # below d_bar=1.0 will be classified 'tight'.
        diag = compute_codebook_regime_diagnostics(codebook, k_nn=5, beta=1.0)
        # At least one atom should be classified — and most should be
        # 'tight' since the cluster is collapsed.
        self.assertEqual(len(diag.per_atom), 6)
        regimes = [a.regime for a in diag.per_atom.values()]
        self.assertGreater(regimes.count("tight"), 0)

    def test_spread_codebook_classified_spread(self):
        # I.i.d. random complex unit vectors → d_bar ≈ 1.0 → spread for
        # any theta_prime < 1.0.
        torch.manual_seed(1)
        D = 64
        N = 8
        phase = torch.rand(N, D) * 2.0 * 3.141592653589793
        codebook = torch.polar(torch.ones(N, D), phase).to(torch.complex64)
        # beta=10 → default theta_prime = 0.1, much smaller than the
        # spread cluster's d_bar ≈ 1.0.
        diag = compute_codebook_regime_diagnostics(codebook, k_nn=4, beta=10.0)
        regimes = [a.regime for a in diag.per_atom.values()]
        # Random codebook in low D should overwhelmingly classify as
        # 'spread'.
        self.assertGreater(regimes.count("spread"), len(regimes) // 2)


class TestThetaPrimeCalibrationLoad(unittest.TestCase):
    """T3 — calibrated theta_prime_fn loads from the JSON artifact."""

    def test_calibration_loads_and_returns_finite_values(self):
        fn = load_theta_prime_calibration()
        # The repo ships a calibration JSON at
        # notes/emergent-codebook/theta_prime_calibration.json, so this
        # MUST load on a fresh checkout.
        self.assertIsNotNone(fn, "calibration JSON not found on disk")
        # Exact calibrated betas. The JSON has been extended (2026-05-26
        # parallel task) to cover β ∈ {0.01, 0.1, 1.0, 3.0, 10.0, 30.0,
        # 100.0}. We assert finiteness + monotone-ish behavior rather than
        # nailing exact values — the underlying calibration sweep can be
        # rerun without breaking the test.
        for b in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
            v = fn(b)
            self.assertIsNotNone(v)
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # Low-β calibration finding (C.1.4): empirical θ′ is small at
        # β ≤ 0.1 (well below the 1/β prediction).
        self.assertLess(fn(0.01), 0.5)
        self.assertLess(fn(0.1), 0.5)
        # High-β calibration: empirical θ′ approaches the upper plateau.
        self.assertGreater(fn(1.0), 0.5)
        # In-range log-beta interpolation between 0.1 and 1.0 must lie
        # between the two endpoints.
        mid = fn(0.5)
        self.assertGreaterEqual(mid, fn(0.1))
        self.assertLessEqual(mid, fn(1.0))


class TestRecallAtKOnTinyExample(unittest.TestCase):
    """T4 — Recall@K on memorized window recovers the masked token."""

    def test_recall_at_k_recovers_memorized_token(self):
        # Memorize exactly one window. With K large enough, the masked
        # token MUST be in the top-K when the same window is queried.
        torch.manual_seed(7)
        substrate = TorchFHRR(dim=512, seed=7, device="cpu")
        vocab_size = 16
        codebook = substrate.random_vectors(vocab_size)
        window_size = 4
        positions = build_position_vectors(substrate, window_size)
        # Window: [3, 7, 2, 11], mask the last position.
        window = [3, 7, 2, 11]
        masked_idx = window_size - 1

        # Memorize via the same path the driver uses.
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory

        memory = TorchHopfieldMemory[str](substrate)
        memory.store(encode_window(substrate, positions, codebook, window), label="w0")

        # Build the masked cue using the driver's logic — a mask
        # placeholder vector for the masked slot.
        mask_vector = substrate.random_vector()
        cue_atoms = [codebook[t] for t in window]
        cue_atoms[masked_idx] = mask_vector
        terms = [substrate.bind(positions[i], cue_atoms[i]) for i in range(window_size)]
        cue = substrate.bundle(terms)

        # Retrieve, unbind the masked position, rank codebook atoms.
        result = memory.retrieve(cue, beta=10.0, max_iter=12)
        slot_query = substrate.unbind(result.state, positions[masked_idx])
        scores = substrate.similarity_matrix(slot_query, codebook)
        # K=5 → 16/5 baseline if random; the true masked token (11)
        # should rank within the top-5 with one stored pattern and
        # only one binding to disambiguate.
        topk = torch.topk(scores, k=5).indices.detach().cpu().tolist()
        self.assertIn(11, topk)


class TestPathAlphaCLIPropagation(unittest.TestCase):
    """T5 — ``--alpha-anti`` and ``--repulsion-step-size`` propagate.

    Path α (2026-05-26): both knobs must reach the substrate and the
    consolidation step so the inter-atom-separability force fires. This
    test runs a tiny n=1-seed sub-config of the driver and asserts the
    summary header records the knobs at their CLI values and that the
    per-cell ``consolidation_stats`` reflects the substrate state.
    """

    def test_alpha_anti_and_repulsion_step_size_propagate(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            summary = mod.run(
                seeds=[0],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=8,
                n_train_windows=32,
                vocab_size=16,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                standard_mode="consolidated",
                control_mode="shuffled-token",
                n_consolidation_events=20,
                alpha_anti=0.5,
                repulsion_step_size=0.1,
                device="cpu",
                output_dir=_Path(tmp),
                repo_root=REPO_ROOT,
            )
        header = summary["header"]
        self.assertEqual(header["alpha_anti"], 0.5)
        self.assertEqual(header["repulsion_step_size"], 0.1)
        self.assertTrue(header["substrate_repulsion_active"])
        self.assertEqual(header["control_mode"], "shuffled-token")

        # Both the standard row and the shuffled-token control row ran
        # consolidation, and at least one repulsion application fired.
        std_rows = [r for r in summary["per_cell_rows"] if not r["is_control"]]
        ctrl_rows = [r for r in summary["per_cell_rows"] if r["is_control"]]
        self.assertEqual(len(std_rows), 1)
        self.assertEqual(len(ctrl_rows), 1)
        for r in std_rows + ctrl_rows:
            cs = r["consolidation_stats"]
            self.assertIsNotNone(cs)
            self.assertEqual(cs["alpha_anti"], 0.5)
            self.assertEqual(cs["repulsion_step_size"], 0.1)
            self.assertGreater(
                cs["repulsion_applications"],
                0,
                "repulsion should fire at least once when both knobs > 0",
            )

    def test_alpha_anti_zero_disables_repulsion(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            summary = mod.run(
                seeds=[0],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=8,
                n_train_windows=32,
                vocab_size=16,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                standard_mode="consolidated",
                control_mode="shuffled-token",
                n_consolidation_events=20,
                alpha_anti=0.0,
                repulsion_step_size=0.05,
                device="cpu",
                output_dir=_Path(tmp),
                repo_root=REPO_ROOT,
            )
        self.assertFalse(summary["header"]["substrate_repulsion_active"])
        for r in summary["per_cell_rows"]:
            cs = r["consolidation_stats"]
            if cs is not None:
                self.assertEqual(cs["repulsion_applications"], 0)


class TestShuffledTokenControl(unittest.TestCase):
    """T6 — shuffled-token control is a permutation, not a fresh codebook.

    Path α (2026-05-26): per the task spec, the proper Phase 3 shuffled
    control re-uses the standard condition's atom set (same substrate
    seed) but applies a random row permutation to the codebook tensor
    before any training. Then the SAME consolidation pipeline is run.
    This test asserts the two key invariants:
      - the standard and control codebooks share the same atom set
        (each control row equals some standard row); and
      - the row order is not identity (it's an actual permutation).
    """

    def test_shuffled_codebook_is_permutation_of_standard(self):
        # Reproduce the driver's pre-consolidation codebook construction
        # for both standard and shuffled-token control at seed 0, and
        # check that one is a row-permutation of the other.
        D = 64
        vocab_size = 32
        seed = 0

        standard_substrate = TorchFHRR(dim=D, seed=seed, device="cpu", alpha_anti=0.0)
        standard_codebook = standard_substrate.random_vectors(vocab_size)

        control_substrate = TorchFHRR(dim=D, seed=seed, device="cpu", alpha_anti=0.0)
        control_codebook = control_substrate.random_vectors(vocab_size)
        import random as _random
        perm_rng = _random.Random(seed + 70000)
        perm_indices = list(range(vocab_size))
        perm_rng.shuffle(perm_indices)
        idx_tensor = torch.tensor(perm_indices, dtype=torch.long)
        control_codebook = control_codebook.index_select(0, idx_tensor).contiguous()

        # Sanity: pre-shuffle, the two substrates produce identical
        # atom sets (same seed). Post-shuffle, the control's row i
        # equals the standard's row perm_indices[i].
        for i in range(vocab_size):
            self.assertTrue(
                torch.allclose(control_codebook[i], standard_codebook[perm_indices[i]])
            )

        # Permutation must not be identity (probability under uniform
        # shuffle is ~ 1/vocab_size! — vanishing for vocab_size=32).
        self.assertNotEqual(perm_indices, list(range(vocab_size)))

        # As a set, the rows match.
        standard_rows = {tuple(standard_codebook[i].tolist()) for i in range(vocab_size)}
        control_rows = {tuple(control_codebook[i].tolist()) for i in range(vocab_size)}
        self.assertEqual(standard_rows, control_rows)

    def test_random_control_mode_preserves_legacy_no_consolidation(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            summary = mod.run(
                seeds=[0],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=8,
                n_train_windows=32,
                vocab_size=16,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                standard_mode="consolidated",
                control_mode="random",
                n_consolidation_events=20,
                alpha_anti=0.01,
                repulsion_step_size=0.05,
                device="cpu",
                output_dir=_Path(tmp),
                repo_root=REPO_ROOT,
            )
        # Standard ran consolidation; legacy random control did NOT.
        ctrl_rows = [r for r in summary["per_cell_rows"] if r["is_control"]]
        std_rows = [r for r in summary["per_cell_rows"] if not r["is_control"]]
        self.assertEqual(len(ctrl_rows), 1)
        self.assertEqual(len(std_rows), 1)
        self.assertIsNone(ctrl_rows[0]["consolidation_stats"])
        self.assertIsNotNone(std_rows[0]["consolidation_stats"])
        # Random-mode control gets the disjoint substrate seed.
        self.assertEqual(ctrl_rows[0]["substrate_seed"], 0 + 10000)


class TestWikiTextCorpusPathArgs(unittest.TestCase):
    """T7 — ``--corpus-source wikitext`` CLI flags wire through to ``run()``.

    Path β (2026-05-26): the driver gains a WikiText-2 corpus mode. This
    test only exercises argument parsing + corpus-loader invocation; the
    loader itself is replaced by a monkeypatched fake so the test does
    not actually fetch ~3 MB of corpus data. It verifies:

      - the CLI accepts ``--corpus-source wikitext``, ``--wikitext-name``,
        and ``--vocab-cap``;
      - the driver's ``run()`` calls ``load_corpus_splits`` with the
        right arguments when ``corpus_source='wikitext'``;
      - the header records the corpus block (source, name, vocab cap,
        token counts);
      - the effective vocab size is ``vocab_cap + 2`` (special tokens).
    """

    def _make_fake_corpus(self):
        """Build a tiny synthetic WikiText-shaped corpus for the test.

        The Phase 2 vocabulary builder takes a list of text strings, so
        the fake loader returns one long string per split with enough
        token diversity to populate the capped vocabulary.
        """
        # ~80 unique tokens, repeated; enough to satisfy vocab_cap=16.
        train_tokens = " ".join(
            f"tok{i % 80}" for i in range(2000)
        )
        val_tokens = " ".join(f"tok{i % 80}" for i in range(400))
        test_tokens = " ".join(f"tok{i % 80}" for i in range(400))
        return {
            "train": [train_tokens],
            "validation": [val_tokens],
            "test": [test_tokens],
        }

    def test_run_invokes_loader_with_wikitext_args(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path
        from unittest import mock

        fake_corpus = self._make_fake_corpus()
        loader_calls = []

        def fake_loader(source, repo_root, wikitext_name="wikitext-2-raw-v1"):
            loader_calls.append((source, str(repo_root), wikitext_name))
            return fake_corpus

        with mock.patch.object(mod, "load_corpus_splits", new=fake_loader):
            with tempfile.TemporaryDirectory() as tmp:
                summary = mod.run(
                    seeds=[0],
                    D=128,
                    landscape_size=4,
                    window_size=4,
                    n_test_windows=8,
                    n_train_windows=32,
                    vocab_size=200,  # ignored under wikitext mode
                    k=3,
                    beta=10.0,
                    theta_prime_mode="default",
                    standard_mode="consolidated",
                    control_mode="shuffled-token",
                    n_consolidation_events=10,
                    alpha_anti=0.0,
                    repulsion_step_size=0.0,
                    device="cpu",
                    output_dir=_Path(tmp),
                    repo_root=REPO_ROOT,
                    corpus_source="wikitext",
                    wikitext_name="wikitext-2-raw-v1",
                    vocab_cap=16,
                )
        # Loader called exactly once for the whole run.
        self.assertEqual(len(loader_calls), 1)
        src, _root, name = loader_calls[0]
        self.assertEqual(src, "wikitext")
        self.assertEqual(name, "wikitext-2-raw-v1")

        # Header records the corpus block.
        corpus = summary["header"]["corpus"]
        self.assertEqual(corpus["corpus_source"], "wikitext")
        self.assertEqual(corpus["wikitext_name"], "wikitext-2-raw-v1")
        self.assertEqual(corpus["vocab_cap"], 16)
        # vocab_cap=16 + <UNK>/<MASK> specials = 18 effective.
        self.assertEqual(corpus["effective_vocab_size"], 18)
        self.assertGreater(corpus["n_train_tokens"], 0)
        self.assertGreater(corpus["n_val_tokens"], 0)
        self.assertGreater(corpus["n_test_tokens"], 0)
        # Operating point reflects the effective vocab.
        self.assertEqual(summary["header"]["operating_point"]["vocab_size"], 18)

    def test_cli_parses_wikitext_flags(self):
        """``main`` accepts the three new CLI flags without explosion."""
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path
        from unittest import mock

        fake_corpus = self._make_fake_corpus()

        def fake_loader(source, repo_root, wikitext_name="wikitext-2-raw-v1"):
            return fake_corpus

        with mock.patch.object(mod, "load_corpus_splits", new=fake_loader):
            with tempfile.TemporaryDirectory() as tmp:
                rc = mod.main(
                    [
                        "--seeds", "0",
                        "--D", "128",
                        "--landscape-size", "4",
                        "--window", "4",
                        "--n-test-windows", "8",
                        "--n-train-windows", "32",
                        "--K", "3",
                        "--beta", "10.0",
                        "--theta-prime-mode", "default",
                        "--standard-mode", "consolidated",
                        "--control-mode", "shuffled-token",
                        "--n-consolidation-events", "10",
                        "--alpha-anti", "0.0",
                        "--repulsion-step-size", "0.0",
                        "--corpus-source", "wikitext",
                        "--wikitext-name", "wikitext-2-raw-v1",
                        "--vocab-cap", "16",
                        "--output-dir", tmp,
                    ]
                )
        self.assertEqual(rc, 0)

    def test_synthetic_mode_unchanged_backward_compat(self):
        """Default synthetic mode keeps working without WikiText loader."""
        mod = importlib.import_module("c3_phase3_exit_criterion")
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            summary = mod.run(
                seeds=[0],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=8,
                n_train_windows=32,
                vocab_size=16,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                standard_mode="consolidated",
                control_mode="shuffled-token",
                n_consolidation_events=10,
                alpha_anti=0.0,
                repulsion_step_size=0.0,
                device="cpu",
                output_dir=_Path(tmp),
                repo_root=REPO_ROOT,
                # corpus_source omitted -> defaults to "synthetic"
            )
        corpus = summary["header"]["corpus"]
        self.assertEqual(corpus["corpus_source"], "synthetic")
        self.assertIsNone(corpus["wikitext_name"])
        self.assertIsNone(corpus["vocab_cap"])
        # Synthetic mode keeps the requested vocab_size.
        self.assertEqual(summary["header"]["operating_point"]["vocab_size"], 16)


if __name__ == "__main__":
    unittest.main()
