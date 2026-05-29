"""Tests for Gate 0 (Frame A) — matched-world difference-in-differences.

  G1 (4a, MANDATORY) — the identity-permutation gauge control reproduces
       condition A byte-for-byte. This is the precommit's local gate; if it
       ever fails the gauge finding is wrong (G0->confound, STOP).
  G2 — the "shuffled" world applies a seed-deterministic global token-stream
       shuffle (a permutation of the real stream) BEFORE windowing.
  G3 — run_gate0 assembles all five arms, the per-seed DiD, secondaries, and
       the gauge confirmation, and never claims graduation.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _tiny_cell_kwargs(**overrides):
    """Shared tiny synthetic-corpus config for a single c3 condition cell."""
    base = dict(
        seed=0,
        is_control=False,
        theta_prime_mode="default",
        standard_mode="consolidated",
        control_mode="shuffled-token",
        n_consolidation_events=10,
        D=128,
        landscape_size=4,
        window_size=4,
        n_test_windows=12,
        vocab_size=16,
        n_train_windows=40,
        beta=10.0,
        k=3,
        alpha_anti=0.01,
        repulsion_step_size=0.05,
        lr_pull=0.1,
        lr_push=0.05,
        device="cpu",
        repo_root=REPO_ROOT,
        wikitext_corpus=None,
    )
    base.update(overrides)
    return base


def _make_fake_corpus(mod, vocab_size=20, n_train=400, n_held=160):
    """A fake _WikiTextCorpus with structured (non-i.i.d.) token streams."""
    vocab = types.SimpleNamespace(id_to_token=list(range(vocab_size)))
    # A repeating pattern → real co-occurrence structure the shuffle destroys.
    train_ids = [i % vocab_size for i in range(n_train)]
    held_ids = [i % vocab_size for i in range(n_held)]
    return mod._WikiTextCorpus(
        vocab=vocab,
        train_ids=train_ids,
        val_ids=held_ids[: n_held // 2],
        test_ids=held_ids[n_held // 2:],
    )


class TestGate0Identity4a(unittest.TestCase):
    """G1 — identity-permutation gauge control == condition A byte-for-byte."""

    def setUp(self):
        self.c3 = importlib.import_module("c3_phase3_exit_criterion")

    def test_identity_gauge_is_byte_identical_to_A(self):
        # Condition A: real world, standard consolidated, NOT control.
        a = self.c3._run_single_seed_condition(
            **_tiny_cell_kwargs(is_control=False, world="real")
        )
        # Gauge with the IDENTITY permutation: control arm, identity perm.
        e_id = self.c3._run_single_seed_condition(
            **_tiny_cell_kwargs(
                is_control=True, world="real", identity_permutation=True
            )
        )
        # Byte-identical per-stratum (succ, trials, recall) and regime counts.
        self.assertEqual(a["per_stratum"], e_id["per_stratum"])
        self.assertEqual(a["regime_counts"], e_id["regime_counts"])
        self.assertEqual(
            a["regime_counts_before_consolidation"],
            e_id["regime_counts_before_consolidation"],
        )
        # The row records the identity perm marker.
        self.assertEqual(e_id["shuffled_token_permutation_seed"], "identity")

    def test_nonidentity_gauge_takes_the_permutation_path(self):
        # A non-identity permutation runs the shuffled-token-with-
        # consolidation control under the overridden perm seed. We do NOT
        # assert the realized recall *differs* from A: on a structure-free
        # synthetic corpus the realized Δ can legitimately be 0 — that IS
        # the gauge-vacuity property (the permutation is a symmetry of the
        # distribution, so E[Δ]=0; any single realization may tie). The
        # mean-zero behaviour over many perm seeds is the 4b check.
        e_perm = self.c3._run_single_seed_condition(
            **_tiny_cell_kwargs(
                is_control=True, world="real", perm_seed_override=12345
            )
        )
        self.assertEqual(e_perm["shuffled_token_permutation_seed"], 12345)
        self.assertEqual(e_perm["control_mode"], "shuffled-token")
        # The gauge arm runs the SAME consolidation pipeline as A.
        self.assertIsNotNone(e_perm["consolidation_stats"])


class TestGate0WorldShuffle(unittest.TestCase):
    """G2 — the shuffled world is a seed-deterministic stream permutation."""

    def setUp(self):
        self.c3 = importlib.import_module("c3_phase3_exit_criterion")

    def _capture_streams(self, world, seed, corpus):
        """Run a cell, capturing the token streams handed to make_windows."""
        captured = []
        real_make_windows = self.c3.make_windows

        def _spy(stream, window_size):
            captured.append(list(stream))
            return real_make_windows(stream, window_size)

        from unittest import mock
        with mock.patch.object(self.c3, "make_windows", new=_spy):
            self.c3._run_single_seed_condition(
                **_tiny_cell_kwargs(
                    seed=seed, world=world, wikitext_corpus=corpus,
                    vocab_size=corpus.vocab_size,
                )
            )
        # First call = train stream, second = held-out stream.
        return captured[0], captured[1]

    def test_shuffled_world_is_permutation_and_deterministic(self):
        corpus = _make_fake_corpus(self.c3)
        real_train, real_held = self._capture_streams("real", 0, corpus)
        shuf_train, shuf_held = self._capture_streams("shuffled", 0, corpus)

        # Real world = the corpus streams verbatim.
        self.assertEqual(real_train, list(corpus.train_ids))

        # Shuffled = same multiset (unigram marginals preserved), different
        # order (co-occurrence destroyed).
        self.assertEqual(sorted(shuf_train), sorted(real_train))
        self.assertNotEqual(shuf_train, real_train)
        self.assertEqual(
            sorted(shuf_held),
            sorted(list(corpus.val_ids) + list(corpus.test_ids)),
        )

        # Seed-deterministic: arms B and D at the same seed see the SAME
        # shuffled world.
        shuf_train_2, shuf_held_2 = self._capture_streams("shuffled", 0, corpus)
        self.assertEqual(shuf_train, shuf_train_2)
        self.assertEqual(shuf_held, shuf_held_2)

        # Different atom seed → different shuffle.
        shuf_train_s1, _ = self._capture_streams("shuffled", 1, corpus)
        self.assertNotEqual(shuf_train, shuf_train_s1)


class TestGate0EndToEnd(unittest.TestCase):
    """G3 — run_gate0 assembles arms/DiD/gauge and never graduates."""

    def setUp(self):
        self.gate0 = importlib.import_module("gate0_frame_a")
        self.c3 = importlib.import_module("c3_phase3_exit_criterion")

    def test_run_gate0_structure_and_gates(self):
        corpus = _make_fake_corpus(self.c3, vocab_size=20)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.gate0.run_gate0(
                seeds=[0, 1, 2],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=16,
                n_train_windows=40,
                vocab_size=20,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                n_consolidation_events=10,
                device="cpu",
                output_dir=Path(tmp),
                repo_root=REPO_ROOT,
                corpus_source="wikitext",
                wikitext_corpus=corpus,
            )
            json_path, md_path = self.gate0.write_gate0_outputs(
                summary, Path(tmp)
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

        # Gate 0 NEVER graduates.
        self.assertTrue(summary["header"]["diagnostic_not_graduation"])

        # All five arms produced per-seed recall.
        for arm in ("A", "B", "C", "D"):
            self.assertEqual(len(summary["per_seed_recall"][arm]), 3)
        self.assertEqual(len(summary["per_seed_recall"]["E"]), 3)  # perm seeds

        # Primary DiD has both clauses + a bool pass.
        did = summary["primary_did"]
        self.assertIn("stats", did)
        self.assertIn("ci95_above_zero", did["stats"])
        self.assertIsInstance(did["passes"], bool)

        # Gauge confirmation: 4a byte-identity holds (sanity), and the
        # verdict is one of the pre-committed branches.
        self.assertTrue(summary["gauge_confirmation_E"]["byte_identical_4a"])
        self.assertIn(
            summary["verdict"],
            {
                "G0->pass", "G0->weak", "G0->null-cons",
                "G0->dead", "G0->confound",
            },
        )

    def test_markdown_renders(self):
        corpus = _make_fake_corpus(self.c3, vocab_size=20)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.gate0.run_gate0(
                seeds=[0, 1],
                D=128,
                landscape_size=4,
                window_size=4,
                n_test_windows=12,
                n_train_windows=40,
                vocab_size=20,
                k=3,
                beta=10.0,
                theta_prime_mode="default",
                n_consolidation_events=8,
                device="cpu",
                output_dir=Path(tmp),
                repo_root=REPO_ROOT,
                corpus_source="wikitext",
                wikitext_corpus=corpus,
            )
        md = self.gate0.format_gate0_markdown(summary)
        self.assertIn("Gate 0 (Frame A)", md)
        self.assertIn("VERDICT", md)
        self.assertIn("DiD", md)


if __name__ == "__main__":
    unittest.main()
