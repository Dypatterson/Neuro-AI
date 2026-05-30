"""Test 4a — codebook-only eval-isolation guard (Frame B prerequisite).

Frame B's exposure-slope design interposes READ-ONLY Recall@K evaluations
at intermediate checkpoints along a single consolidation trajectory. For
that to be sound, an interposed evaluation must be *non-perturbing*: it
must not mutate the codebook / buffer / C.2 consolidation state, and it
must not advance the RNG stream(s) that the consolidation ``observe()``
path also draws from. If it did, every checkpoint read would silently
change the trajectory it is trying to measure.

This module pins that isolation property on the REAL code path:

  * ``c3._evaluate_recall_at_k`` (the production evaluator) is invoked on
    a ``codebook.clone()`` so it cannot write through to the live codebook
    tensor; and it is given a SEPARATE eval substrate (its own
    ``torch.Generator``) so that the mask vector it draws via
    ``substrate.random_vector()`` (c3 ~line 282) cannot advance the
    consolidation substrate's generator.

    Substrate separation is used rather than a generator
    get_state/set_state snapshot because in this environment the CPU
    ``torch.Generator`` offset counter does not reliably survive a
    snapshot round-trip once MPS has been initialised — separation is the
    order-independent guarantee.

  * ``c3._consolidate_codebook`` (the production consolidation driver) is
    then run, and its final codebook is asserted bit-identical
    (``torch.equal``) with vs without the interposed eval.

We deliberately do NOT assert final-*recall* identity: Frame B's later
window-averaging step changes recall on purpose. The invariant under test
is strictly that a read-only eval leaves the consolidation trajectory
(codebook + RNG state) unperturbed.

The synthetic-corpus inputs are built with c3's OWN helpers
(``_generate_codebook``, ``build_position_vectors``, ``_make_synthetic_
windows``, ``encode_window``, ``TorchHopfieldMemory``) so the test
exercises the production construction path, not a re-implementation.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import c3_phase3_exit_criterion as c3  # noqa: E402

from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory  # noqa: E402
from energy_memory.phase2.encoding import (  # noqa: E402
    build_position_vectors,
    encode_window,
)


# Minimal synthetic-corpus operating point — small enough to run fast on
# CPU but exercising the real consolidation + eval primitives.
_D = 256
_VOCAB = 12
_WINDOW = 4
_N_LANDSCAPE = 6
_N_TRAIN = 20
_N_TEST = 8
_N_EVENTS = 15  # observe()/exposure count (NOT consolidation events)
_BETA = 10.0
_K = 3

# Fixed seed for the *global* torch RNG stream. ``c3._consolidate_codebook``
# is not a pure function of its arguments — the Hopfield retrieve / C.2
# dynamics it drives draw from the default torch generator as well as the
# substrate's per-instance generator. Pinning the global seed before each
# consolidation makes the eval-isolation comparison control that confound,
# so any final-codebook difference is attributable to the interposed eval.
_GLOBAL_RNG_PIN = 20260529

# Dedicated seed for the eval substrate (kept disjoint from the
# consolidation substrate's seed=0 stream).
_EVAL_SUBSTRATE_SEED = 12345


def _build_inputs(seed: int = 0) -> dict:
    """Construct a minimal but real consolidation + eval input set.

    Uses c3's own construction helpers. Two calls with the same ``seed``
    produce identical inputs because every random draw goes through the
    substrate's per-instance generator (seeded via ``TorchFHRR(seed=...)``)
    or a seeded ``random.Random`` for the window corpus.
    """
    substrate = TorchFHRR(dim=_D, seed=seed, device="cpu")

    codebook = c3._generate_codebook(substrate=substrate, vocab_size=_VOCAB)
    positions = build_position_vectors(substrate, _WINDOW)

    corpus_rng = random.Random(seed + 11111)
    train_windows = c3._make_synthetic_windows(
        n_windows=_N_TRAIN, window_size=_WINDOW, vocab_size=_VOCAB, rng=corpus_rng,
    )
    test_windows = c3._make_synthetic_windows(
        n_windows=_N_TEST, window_size=_WINDOW, vocab_size=_VOCAB, rng=corpus_rng,
    )
    landscape_windows = train_windows[:_N_LANDSCAPE]

    memory: TorchHopfieldMemory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(landscape_windows):
        memory.store(
            encode_window(substrate, positions, codebook, list(window)),
            label=f"window_{index}",
        )

    return {
        "substrate": substrate,
        "codebook": codebook,
        "positions": positions,
        "memory": memory,
        "landscape_windows": landscape_windows,
        "train_windows": train_windows,
        "test_windows": test_windows,
    }


def _run_consolidation(inp: dict) -> torch.Tensor:
    """Run the production consolidation driver; return the final codebook.

    The global torch RNG is pinned first (see ``_GLOBAL_RNG_PIN``) so the
    baseline run and the eval-interposed run start from an identical global
    stream.
    """
    torch.manual_seed(_GLOBAL_RNG_PIN)
    cb, _state, _upd, _stats = c3._consolidate_codebook(
        substrate=inp["substrate"],
        codebook=inp["codebook"],
        positions=inp["positions"],
        landscape_windows=inp["landscape_windows"],
        memory=inp["memory"],
        train_windows=inp["train_windows"],
        window_size=_WINDOW,
        beta=_BETA,
        vocab_size=_VOCAB,
        n_events=_N_EVENTS,
        device="cpu",
    )
    return cb


def _interpose_readonly_eval(inp: dict) -> list:
    """Run the REAL evaluator read-only, isolated from the live substrate.

    Two isolation guarantees, both exercised against the production
    evaluator ``c3._evaluate_recall_at_k``:

      1. The codebook passed in is a ``.clone()`` of the live tensor, so
         the eval cannot write through to the consolidation codebook.

      2. The eval runs on a SEPARATE ``TorchFHRR`` substrate (its own
         ``torch.Generator``). The evaluator draws one mask vector via
         ``substrate.random_vector()`` (c3 ~line 282); routing that draw
         through a dedicated eval-substrate means it cannot advance the
         generator that the consolidation ``observe()`` path draws from.

    The eval re-uses the live ``memory`` (read-only — Hopfield
    ``retrieve`` does not mutate stored patterns) and the live
    ``positions`` (read-only inputs to ``bind``).

    Returns the per-window outcomes so the test can confirm the eval
    actually ran (read-only) rather than being optimised away.
    """
    eval_substrate = TorchFHRR(dim=_D, seed=_EVAL_SUBSTRATE_SEED, device="cpu")

    outcomes = c3._evaluate_recall_at_k(
        substrate=eval_substrate,
        memory=inp["memory"],
        codebook=inp["codebook"].clone(),  # read-only: never write live tensor
        positions=inp["positions"],
        test_windows=inp["test_windows"],
        masked_idx=_WINDOW - 1,
        beta=_BETA,
        k=_K,
        regime_labels={tok: "borderline" for tok in range(_VOCAB)},
    )
    return outcomes


class TestFrameBEvalIsolation(unittest.TestCase):
    def test_interposed_readonly_eval_leaves_codebook_bit_identical(self):
        """Final codebook is byte-identical with vs without interposed eval."""
        baseline_inp = _build_inputs(seed=0)
        baseline_cb = _run_consolidation(baseline_inp)

        eval_inp = _build_inputs(seed=0)
        outcomes = _interpose_readonly_eval(eval_inp)
        # The eval must actually have run over every test window (read-only).
        self.assertEqual(len(outcomes), _N_TEST)
        eval_cb = _run_consolidation(eval_inp)

        self.assertTrue(
            torch.equal(baseline_cb, eval_cb),
            "interposing a read-only Recall@K eval perturbed the final "
            "consolidated codebook — eval isolation is broken.",
        )

    def test_interposed_eval_does_not_perturb_substrate_generator(self):
        """The consolidation substrate's generator survives the eval draw.

        Probed by the next-drawn-vector, NOT by ``generator.get_state()``:
        in this environment the CPU-generator offset counter does not
        reliably show up in a get_state() snapshot once MPS has been
        initialised, so a state-diff would be a silently-vacuous check. The
        next vector a substrate would draw is the ground-truth observable of
        its RNG position. If the eval is isolated, the consolidation
        substrate's next draw must equal the baseline's next draw.
        """
        baseline = _build_inputs(seed=0)
        v_baseline = baseline["substrate"].random_vector()

        inp = _build_inputs(seed=0)
        _interpose_readonly_eval(inp)
        v_after_eval = inp["substrate"].random_vector()

        self.assertTrue(
            torch.equal(v_baseline, v_after_eval),
            "the interposed eval advanced the live substrate generator "
            "(its next draw diverged from baseline) — the consolidation RNG "
            "stream is not isolated.",
        )

    def test_interposed_eval_does_not_mutate_live_codebook_or_positions(self):
        """The eval writes through neither the live codebook nor positions."""
        inp = _build_inputs(seed=0)
        cb_before = inp["codebook"].clone()
        pos_before = [p.clone() for p in inp["positions"]]
        _interpose_readonly_eval(inp)
        self.assertTrue(
            torch.equal(cb_before, inp["codebook"]),
            "eval mutated the live codebook in place (clone guard failed).",
        )
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(pos_before, inp["positions"])),
            "eval mutated the position vectors in place.",
        )

    def test_unisolated_eval_would_perturb_generator_control(self):
        """Control: run on the LIVE substrate and the eval DOES advance it.

        Negative control proving the isolation above is load-bearing: when
        the eval is pointed at the consolidation substrate directly (no
        separate eval-substrate), its mask draw advances that generator, so
        the substrate's next draw diverges from baseline. This shows the
        identity preserved in the isolated path is a real property of the
        isolation, not an accident of the eval never touching any generator.

        Probed by next-drawn-vector for the same reason as the positive
        test above (get_state() is unreliable here once MPS is warm).
        """
        baseline = _build_inputs(seed=0)
        v_baseline = baseline["substrate"].random_vector()

        inp = _build_inputs(seed=0)
        # Same eval, but pointed at the LIVE substrate (no isolation).
        c3._evaluate_recall_at_k(
            substrate=inp["substrate"],
            memory=inp["memory"],
            codebook=inp["codebook"].clone(),
            positions=inp["positions"],
            test_windows=inp["test_windows"],
            masked_idx=_WINDOW - 1,
            beta=_BETA,
            k=_K,
            regime_labels={tok: "borderline" for tok in range(_VOCAB)},
        )
        v_after_eval = inp["substrate"].random_vector()
        self.assertFalse(
            torch.equal(v_baseline, v_after_eval),
            "expected the un-isolated eval to advance the substrate "
            "generator (it draws a mask via substrate.random_vector()); if "
            "the next draw still matched baseline, the isolation test above "
            "is vacuous.",
        )


if __name__ == "__main__":
    unittest.main()
