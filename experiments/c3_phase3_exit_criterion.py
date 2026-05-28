"""C.3 — Phase 3 exit-criterion re-run on the instrumented codebook.

This is the **final Path C deliverable** per the precommit at
``notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md``.

Headline metric (per ``notes/emergent-codebook/phase-3-deep-dive.md:180-189``):
**Regime-stratified Recall@K on masked-token contextual completion, vs.
genuine shuffled-token control, on n ≥ 10 seeds.**

This driver is the C.3 graduation gate when run with ``--seeds``
containing 10+ entries. With fewer seeds the run is *diagnostic-only* and
the report header records that explicitly.

The driver:
  1. For each seed, generates a fresh **random codebook** as the main
     condition's token-to-hypervector assignment.
  2. Memorizes a small Hopfield landscape of masked-token windows over
     a synthetic source corpus (Phase 2 operating envelope).
  3. Computes regime diagnostics via
     ``compute_codebook_regime_diagnostics`` from C.1.3 to label each
     atom 'tight' / 'spread' / 'borderline'.
  4. Evaluates Recall@K on held-out windows, **stratified by the masked
     token's atom regime**.
  5. Repeats steps 1-4 with a **fresh shuffled-token control codebook**
     drawn from a disjoint seed prefix (``seed + 10000``). This is the
     genuine shuffled-token control the Path C precommit demands —
     **NOT** artifact reuse from Report 017.
  6. Runs both ``theta_prime_fn`` variants (default ``1/β`` and the
     calibrated lookup) per the Path C precommit binding.
  7. Aggregates per (theta_prime_mode, stratum) across seeds with
     Wilson CIs.

This driver does **not** modify any existing ``src/`` file. It only
calls existing primitives.

Note: at the Phase 2 operating point β=10, the calibration table at
``notes/emergent-codebook/theta_prime_calibration.json`` only covers
β ∈ {0.01, 0.1, 1.0}. The calibrated loader will warn and fall back to
``1/β`` outside the calibrated range. The two ``theta_prime_mode``
runs therefore agree numerically at β=10 unless the user supplies a
β inside the calibrated range. This is itself a binding finding to
report — see the smoke output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    Vocabulary,
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window,
    masked_window,
)
from energy_memory.phase2.metrics import wilson_interval
from energy_memory.phase3.regime_diagnostic import (
    CodebookRegimeDiagnostics,
    compute_codebook_regime_diagnostics,
)
from energy_memory.phase3.theta_prime_calibration import load_theta_prime_calibration
from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.phase4.consolidation import (
    ConsolidationConfig,
    ConsolidationState,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


# Sentinel constants -----------------------------------------------------------

STRATA = ("tight", "spread", "borderline")
THETA_PRIME_MODES = ("default", "calibrated")
STANDARD_MODES = ("random", "consolidated")
CORPUS_SOURCES = ("synthetic", "wikitext")
# Path α (2026-05-26): the proper Phase 3 shuffled-token control runs
# the SAME consolidation pipeline as the standard condition over the
# SAME training corpus, but with a random permutation of the
# token-to-hypervector assignment. ``random`` (the previous default)
# was actually a no-consolidation control — it built a fresh random
# codebook with no training and is preserved for backward compat only.
CONTROL_MODES = ("random", "shuffled-token")

# C.2 consolidation defaults for `--standard-mode consolidated` (per task spec):
#   C.2.1  anti-collapse force:              lambda_ac      = 0.5
#   C.2.2  splitting-tension modulation:     mu_T, tau_T    = 0.1, 0.5
#   C.2.3  cap-coverage gradient:            lambda_cc      = 0.5
#                                            theta_cc       = 0.5
#                                            tau_cc         = 0.1
#   C.2.4  metastability replay-priority:    metastability_obs_rate = 0.1
#                                            (metastability_gain / replay_decay
#                                             only affect REPLAY priority — they
#                                             have no effect here because this
#                                             driver does not run replay; the
#                                             EMA is still updated so the
#                                             dynamic is observable.)
#   C.2.5  drift replay-tension:             drift_ema_rate = 0.1
#                                            drift_replay_gain = 1.0
#                                            (gain affects replay priority only.)
C2_DEFAULTS = {
    "lambda_ac": 0.5,
    "mu_T": 0.1,
    "tau_T": 0.5,
    "lambda_cc": 0.5,
    "theta_cc": 0.5,
    "tau_cc": 0.1,
    "metastability_obs_rate": 0.1,
    "metastability_gain": 2.0,             # observed-only here (replay-store knob)
    "metastability_replay_decay": 0.5,     # observed-only here (replay-store knob)
    "drift_ema_rate": 0.1,
    "drift_replay_gain": 1.0,              # observed-only here (replay-store knob)
}


# Synthetic corpus -------------------------------------------------------------

@dataclass
class _WikiTextCorpus:
    """Pre-loaded WikiText-2 corpus for sharing across seeds in one run.

    Holds the vocabulary (top-``vocab_cap`` tokens + ``<UNK>`` + ``<MASK>``
    from the train split) and the encoded train / val / test token id
    streams. Each seed in the run draws its own train / test window
    samples from these streams (so different seeds see different sub-
    samples but the underlying corpus is fixed).

    The "effective" vocab_size used by the rest of the driver is
    ``len(vocab.id_to_token)`` (= vocab_cap + 2 special tokens). ``<MASK>``
    is the in-vocab token id used by the encoding/decoding utilities;
    the driver still uses ``vocab_size`` as the out-of-vocab "mask"
    sentinel for the masked-cue construction so it does not collide
    with the special tokens. The masked-token *answer* (the regime
    label and the topk membership) ranges over [0, vocab_size).
    """

    vocab: Vocabulary
    train_ids: List[int]
    val_ids: List[int]
    test_ids: List[int]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab.id_to_token)


def _load_wikitext_corpus(
    *,
    repo_root: Path,
    wikitext_name: str,
    vocab_cap: int,
) -> _WikiTextCorpus:
    """Load WikiText-2, build a capped vocabulary, encode all splits.

    Uses the same loader / builder / encoder Phase 2 uses
    (``experiments/02_phase2_retrieval_baseline.py:55-58``). Looks up
    the loader / builder / encoder names on the module at call time so
    tests can monkeypatch them (``mock.patch.object(mod,
    'load_corpus_splits', ...)``) without re-binding the default.
    """
    # Module-level lookups so ``mock.patch.object`` works at call time.
    module = sys.modules[__name__]
    corpus_loader = getattr(module, "load_corpus_splits")
    vocab_builder = getattr(module, "build_vocabulary")
    text_encoder = getattr(module, "encode_texts")
    splits = corpus_loader(
        "wikitext", repo_root, wikitext_name=wikitext_name,
    )
    vocab = vocab_builder(splits["train"], max_vocab=vocab_cap)
    train_ids = text_encoder(splits["train"], vocab)
    val_ids = text_encoder(splits["validation"], vocab)
    test_ids = text_encoder(splits["test"], vocab)
    return _WikiTextCorpus(
        vocab=vocab, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
    )


def _make_synthetic_windows(
    *,
    n_windows: int,
    window_size: int,
    vocab_size: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """Generate ``n_windows`` random windows of length ``window_size``.

    The "synthetic source distribution" here is uniform-random token ids
    in [0, vocab_size). This intentionally creates a corpus with no
    learned structure — the test of whether the random codebook supports
    contextual completion is whether *positional* binding alone can
    drive masked-token retrieval on memorized windows.

    For Phase 3's exit criterion, the *shuffled-token control* is the
    fresh-codebook permutation; the corpus itself does not need to carry
    structure for the headline to be interpretable, because what is
    being measured is whether regime-stratified Recall@K of the
    "standard" codebook beats the shuffled-codebook control.
    """
    windows: List[Tuple[int, ...]] = []
    for _ in range(n_windows):
        windows.append(tuple(rng.randrange(0, vocab_size) for _ in range(window_size)))
    return windows


# Codebook generation ----------------------------------------------------------

def _generate_codebook(*, substrate: TorchFHRR, vocab_size: int) -> torch.Tensor:
    """Generate a fresh random FHRR codebook of ``vocab_size`` atoms.

    Uses the substrate's own RNG (set via the constructor seed). The
    `random_vectors` call draws from `substrate.generator`, so two
    substrates constructed with different seeds produce disjoint
    codebooks. This is the operationalization of the "shuffled-token
    control" — the control gets a fresh substrate seed and therefore a
    fresh assignment of token ids to hypervectors.
    """
    return substrate.random_vectors(vocab_size)


# Recall@K evaluation ----------------------------------------------------------

@dataclass
class _CellOutcome:
    """Per-test-window evaluation outcome for one (mode, seed) cell."""

    masked_token: int
    masked_regime: str
    correct_at_k: bool


def _evaluate_recall_at_k(
    *,
    substrate: TorchFHRR,
    memory: TorchHopfieldMemory[str],
    codebook: torch.Tensor,
    positions: Sequence,
    test_windows: Sequence[Tuple[int, ...]],
    masked_idx: int,
    beta: float,
    k: int,
    regime_labels: Dict[int, str],
) -> List[_CellOutcome]:
    """Compute Recall@K for each held-out test window.

    For each test window:
      1. Mask the token at ``masked_idx`` (the last position by default).
      2. Encode the cue (window with mask placeholder).
      3. Hopfield-retrieve the settled state at temperature ``beta``.
      4. Unbind the masked position to get the cleaned-up query.
      5. Rank all codebook atoms by cosine similarity with the query.
      6. Mark "correct" if the true masked token is in the top-K.
      7. Tag with the regime label of the masked token's atom.

    No `if regime then route` logic anywhere — the regime is only used
    as a passive stratification label at aggregation time.
    """
    outcomes: List[_CellOutcome] = []
    vocab_size = codebook.shape[0]
    mask_id = vocab_size  # sentinel — out of vocab; encoded as zero-vector

    # Build a mask-id codebook entry by extending with a single neutral
    # vector. We use a random vector for the mask to keep the encoding
    # well-formed; the cue's masked position contributes a structural
    # binding but no information about the true token.
    mask_vector = substrate.random_vector()

    extended_codebook_list = [codebook[i] for i in range(vocab_size)] + [mask_vector]

    for window in test_windows:
        true_token = window[masked_idx]
        regime = regime_labels.get(true_token, "borderline")

        # Build the masked cue.
        cue_window = list(window)
        cue_window[masked_idx] = mask_id

        # Encode using the extended codebook (mask at end).
        terms = []
        for pos_idx, tok_id in enumerate(cue_window):
            terms.append(substrate.bind(positions[pos_idx], extended_codebook_list[tok_id]))
        cue = substrate.bundle(terms)

        # Hopfield-retrieve the settled state.
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        settled_state = result.state

        # Unbind the masked position to get the cleaned query for that slot.
        slot_query = substrate.unbind(settled_state, positions[masked_idx])

        # Rank codebook atoms by similarity to the slot query.
        scores = substrate.similarity_matrix(slot_query, codebook)
        topk = torch.topk(scores, k=min(k, vocab_size)).indices.detach().cpu().tolist()

        outcomes.append(
            _CellOutcome(
                masked_token=true_token,
                masked_regime=regime,
                correct_at_k=(true_token in topk),
            )
        )
    return outcomes


def _build_theta_prime_fn(
    mode: str,
    repo_root: Path,
) -> Tuple[Optional[Callable[[float], float]], Dict[str, object]]:
    """Return the ``theta_prime_fn`` for the given mode plus metadata.

    Returns ``(fn, info_dict)``. When ``mode == 'default'`` returns
    ``(None, {...})`` — the regime-diagnostic API treats ``None`` as
    the spec's ``1/β`` default. When ``mode == 'calibrated'`` returns
    the loaded callable from ``load_theta_prime_calibration()``.
    """
    if mode == "default":
        return None, {"mode": "default", "fn": "1/beta", "source": "spec_default"}
    if mode == "calibrated":
        # Calibration JSON lives at notes/emergent-codebook/theta_prime_calibration.json
        # at the repo root. The loader resolves the default path itself
        # when called with no argument.
        fn = load_theta_prime_calibration()
        if fn is None:
            return None, {
                "mode": "calibrated",
                "fn": "1/beta (fallback — calibration file missing)",
                "source": "calibration_missing",
            }
        return fn, {
            "mode": "calibrated",
            "fn": "JSON lookup with log-beta interpolation; out-of-range falls back to 1/beta",
            "source": "calibration_loaded",
        }
    raise ValueError(f"unknown theta_prime_mode: {mode!r}")


def _aggregate_by_stratum(
    outcomes: Sequence[_CellOutcome],
) -> Dict[str, Tuple[int, int]]:
    """Return per-stratum (successes, trials) counts."""
    succ: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for o in outcomes:
        total[o.masked_regime] += 1
        if o.correct_at_k:
            succ[o.masked_regime] += 1
    out: Dict[str, Tuple[int, int]] = {}
    for stratum in STRATA:
        out[stratum] = (succ.get(stratum, 0), total.get(stratum, 0))
    return out


def _wilson(successes: int, trials: int) -> Tuple[float, float, float]:
    """Return (mean, lower, upper). Mean is 0.0 if trials==0."""
    if trials == 0:
        return (0.0, 0.0, 0.0)
    lo, hi = wilson_interval(successes, trials)
    return (successes / trials, lo, hi)


# Consolidation orchestrator wiring (C.2 dynamics) ----------------------------

def _build_consolidation_state(
    *,
    vocab_size: int,
    device: str,
) -> ConsolidationState:
    """Build a ConsolidationState with all 5 C.2 dynamics turned on.

    Constants come from ``C2_DEFAULTS`` (the task spec's modest values).
    ``add_pattern()`` is called ``vocab_size`` times so per-atom slots
    (splitting_tension, drift_tension, metastability_ema) are sized to
    the codebook. The actuator-side semantics here treat each codebook
    atom as a "pattern" — the same conflation used in
    ``tests/test_drift_replay_tension.py`` where the C.2 actuators are
    exercised at the per-atom slot level.
    """
    cfg = ConsolidationConfig(
        # C.2.1
        lambda_ac=C2_DEFAULTS["lambda_ac"],
        # C.2.2
        mu_T=C2_DEFAULTS["mu_T"],
        tau_T=C2_DEFAULTS["tau_T"],
        # C.2.3
        lambda_cc=C2_DEFAULTS["lambda_cc"],
        theta_cc=C2_DEFAULTS["theta_cc"],
        tau_cc=C2_DEFAULTS["tau_cc"],
        # C.2.4 — observed only (replay-store knobs not applicable here).
        metastability_obs_rate=C2_DEFAULTS["metastability_obs_rate"],
        # C.2.5
        drift_ema_rate=C2_DEFAULTS["drift_ema_rate"],
        drift_replay_gain=C2_DEFAULTS["drift_replay_gain"],
    )
    state = ConsolidationState(cfg, device=device)
    for _ in range(vocab_size):
        state.add_pattern(novelty_strength=1.0)
    return state


def _consolidate_codebook(
    *,
    substrate: TorchFHRR,
    codebook: torch.Tensor,
    positions: Sequence,
    landscape_windows: Sequence[Tuple[int, ...]],
    memory: TorchHopfieldMemory[str],
    train_windows: Sequence[Tuple[int, ...]],
    window_size: int,
    beta: float,
    vocab_size: int,
    n_events: int,
    device: str,
    consolidation_k: int = 100,
    quality_threshold: float = 0.15,
    lr_pull: float = 0.1,
    lr_push: float = 0.05,
    repulsion_step_size: float = 0.0,
    use_context_residual: bool = False,
    lr_cr: float = 0.1,
    use_pull_push: bool = True,
) -> Tuple[torch.Tensor, ConsolidationState, OnlineCodebookUpdater, dict]:
    """Run ``n_events`` consolidation observations over training windows.

    For each event:
      1. Sample a training window and mask its last position (matches
         the test-time evaluation protocol).
      2. Encode the masked cue with a mask placeholder vector.
      3. Hopfield-retrieve the settled state at the given β.
      4. Unbind the masked position → slot_query.
      5. Compute predicted_id = argmax(sim(slot_query, codebook)).
      6. Append a basin-trace tuple ``(settled_state, predicted_id)`` to
         the C.2.1 substrate buffer so the next consolidation can see
         basin-geometry signals.
      7. Call ``state.update_metastability(metastability_contribution)``
         (C.2.4 EMA — observed only here since no replay is running).
      8. Call ``updater.observe(target_id, slot_query, predicted_id)``;
         when the buffer fills, ``consolidate_if_ready()`` fires all
         five C.2 dynamics (pull/push + anti-collapse + cap-coverage +
         splitting-tension + drift-EMA).

    Returns ``(consolidated_codebook, state, updater, stats)``.
    """
    state = _build_consolidation_state(vocab_size=vocab_size, device=device)
    updater = OnlineCodebookUpdater(
        substrate=substrate,
        codebook=codebook,
        lr_pull=lr_pull,
        lr_push=lr_push,
        consolidation_k=consolidation_k,
        quality_threshold=quality_threshold,
        consolidation_state=state,
        use_pull_push=use_pull_push,
        use_context_residual=use_context_residual,
        lr_cr=lr_cr,
    )

    masked_idx = window_size - 1
    mask_id = vocab_size  # out-of-vocab sentinel — same convention as eval
    mask_vector = substrate.random_vector()
    extended_codebook = [codebook[i] for i in range(vocab_size)] + [mask_vector]

    n_train = len(train_windows)
    consolidation_events = 0

    # Path α: track the substrate-side repulsion contribution so we can
    # report it in the consolidation_stats. Mirrors the replay-loop
    # snippet at src/energy_memory/phase4/replay_loop.py:786-798 — the
    # repulsion fires whenever both ``substrate.alpha_anti > 0.0`` AND
    # ``repulsion_step_size > 0.0``, applied to the full codebook
    # matrix after each consolidation event so the C.2 within-basin
    # tightening is balanced by the substrate-level inter-basin
    # separation force from the 2026-05-09 reformulation.
    repulsion_applications = 0

    def _apply_repulsion_to_codebook() -> int:
        if substrate.alpha_anti <= 0.0 or repulsion_step_size <= 0.0:
            return 0
        if codebook.shape[0] < 2:
            return 0
        force = substrate.repulsion_force(codebook)
        new_patterns = substrate.normalize(codebook + repulsion_step_size * force)
        codebook[:] = new_patterns
        # OnlineCodebookUpdater holds the SAME tensor reference (we
        # passed `codebook=` into its constructor and only ever
        # mutate it in place), so no rebinding is needed.
        return 1

    observations_made = 0
    for event_idx in range(n_events):
        window = train_windows[event_idx % n_train]
        true_token = window[masked_idx]

        cue_window = list(window)
        cue_window[masked_idx] = mask_id

        terms = []
        for pos_idx, tok_id in enumerate(cue_window):
            terms.append(
                substrate.bind(positions[pos_idx], extended_codebook[tok_id])
            )
        cue = substrate.bundle(terms)

        result = memory.retrieve(cue, beta=beta, max_iter=12)
        slot_query = substrate.unbind(result.state, positions[masked_idx])

        scores = substrate.similarity_matrix(slot_query, codebook)
        predicted_id = int(scores.argmax().detach().cpu())

        state.record_retrieval(result.state, predicted_id)

        if result.metastability_contribution is not None:
            try:
                state.update_metastability(result.metastability_contribution)
            except Exception:
                pass

        observations_made += 1
        if updater.observe(
            target_id=int(true_token),
            slot_query=slot_query,
            predicted_id=predicted_id,
        ):
            diag = updater.consolidate_if_ready()
            if diag is not None:
                consolidation_events += 1
                repulsion_applications += _apply_repulsion_to_codebook()

    # Force one final consolidation if there's anything buffered.
    final_diag = updater.force_consolidate()
    if final_diag is not None:
        consolidation_events += 1
        repulsion_applications += _apply_repulsion_to_codebook()

    stats = {
        "n_events_requested": int(n_events),
        "observations_made": int(observations_made),
        "consolidations_fired": int(consolidation_events),
        "buffered_at_end": int(updater.stats()["buffer_size"]),
        "total_failures": int(updater.stats()["total_failures"]),
        "failure_rate": float(updater.stats()["failure_rate"]),
        "c2_config": dict(C2_DEFAULTS),
        "alpha_anti": float(substrate.alpha_anti),
        "repulsion_step_size": float(repulsion_step_size),
        "repulsion_applications": int(repulsion_applications),
    }
    return codebook, state, updater, stats


# Run loop ---------------------------------------------------------------------

def _run_single_seed_condition(
    *,
    seed: int,
    is_control: bool,
    theta_prime_mode: str,
    standard_mode: str,
    control_mode: str,
    n_consolidation_events: int,
    D: int,
    landscape_size: int,
    window_size: int,
    n_test_windows: int,
    vocab_size: int,
    n_train_windows: int,
    beta: float,
    k: int,
    alpha_anti: float,
    repulsion_step_size: float,
    lr_pull: float,
    lr_push: float,
    device: str,
    repo_root: Path,
    wikitext_corpus: Optional[_WikiTextCorpus] = None,
    use_context_residual: bool = False,
    lr_cr: float = 0.1,
    use_pull_push: bool = True,
) -> Dict[str, object]:
    """Run one (seed, mode, condition) cell.

    Two control-mode operationalizations are supported:

      ``random`` (legacy, backward-compat): the control gets a
      **disjoint substrate seed** (``seed + 10000``) and skips
      consolidation. This was the pre-Path-α default — see the
      2026-05-26 STATUS.md walk-back which flagged it as actually a
      no-consolidation control rather than a shuffled-token control.

      ``shuffled-token`` (Path α default per the precommit at
      ``notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md``):
      the control shares the SAME substrate seed (so identical atoms)
      and the SAME training/test windows, but its codebook is a random
      permutation of the standard codebook's rows (token-id π(i) gets
      atom i). The control then runs the SAME consolidation pipeline
      (all 5 C.2 dynamics + alpha_anti + repulsion_step_size) over the
      SAME training corpus. The eval ranks slot_query against the
      control's own (shuffled-then-trained) codebook with the standard
      token-id ground truth — so any structure reflected in Recall@K
      under the control is corpus-statistical artefact rather than
      learned token-meaning, per
      ``notes/emergent-codebook/phase-3-deep-dive.md:217-218``.
    """
    # Substrate / corpus seeds. Main and control share the corpus draws
    # (so identical windows are evaluated).
    corpus_rng = random.Random(seed)  # shared by main + control

    if is_control and control_mode == "random":
        substrate_seed = seed + 10000
    else:
        # Both ``standard`` and ``shuffled-token`` control share the
        # substrate seed so the atom set is identical; the control's
        # permutation reshuffles row-id → atom assignment on top of
        # that shared atom set.
        substrate_seed = seed

    substrate = TorchFHRR(
        dim=D, seed=substrate_seed, device=device, alpha_anti=alpha_anti,
    )
    codebook = _generate_codebook(substrate=substrate, vocab_size=vocab_size)
    # Path α shuffled-token control: permute the codebook row order so
    # token-id i is assigned atom π(i). Permutation seed is deterministic
    # in the substrate seed (``seed + 70000`` keeps it disjoint from any
    # other seed-derived RNG in this driver). The permutation is fixed
    # for the lifetime of this cell, so memorize / consolidate / eval
    # all see the same shuffled assignment.
    if is_control and control_mode == "shuffled-token":
        perm_rng = random.Random(seed + 70000)
        perm_indices = list(range(vocab_size))
        perm_rng.shuffle(perm_indices)
        idx_tensor = torch.tensor(perm_indices, dtype=torch.long, device=codebook.device)
        codebook = codebook.index_select(0, idx_tensor).contiguous()
    positions = build_position_vectors(substrate, window_size)

    # Generate corpus (training + test windows). Both main and control
    # see the same training-window indices and the same test windows.
    # ``train_windows`` is sampled down to ``landscape_size`` for the
    # Hopfield landscape.
    #
    # Two corpus modes:
    #   synthetic (``wikitext_corpus is None``): uniform-random token
    #     id windows from [0, vocab_size).
    #   wikitext  (``wikitext_corpus`` provided): sliding-window
    #     extraction from the WikiText-2 train split (for the
    #     training pool, which the landscape and consolidation share)
    #     and from the val+test split concatenation (for the
    #     held-out test pool). Both main and control share the same
    #     seeded window subsamples so the comparison is matched.
    if wikitext_corpus is None:
        train_windows = _make_synthetic_windows(
            n_windows=n_train_windows,
            window_size=window_size,
            vocab_size=vocab_size,
            rng=corpus_rng,
        )
        test_windows = _make_synthetic_windows(
            n_windows=n_test_windows,
            window_size=window_size,
            vocab_size=vocab_size,
            rng=corpus_rng,
        )
    else:
        # WikiText: build all non-overlapping windows once over the
        # encoded streams, then per-seed subsample. Use
        # ``sample_windows`` from the Phase 2 corpus module — same
        # primitive Phase 2 uses (deterministic given a seed).
        all_train_windows = make_windows(
            wikitext_corpus.train_ids, window_size,
        )
        held_out_ids = list(wikitext_corpus.val_ids) + list(
            wikitext_corpus.test_ids
        )
        all_test_windows = make_windows(held_out_ids, window_size)
        if not all_train_windows or not all_test_windows:
            raise RuntimeError(
                "WikiText corpus produced no windows at the requested "
                f"window_size={window_size}."
            )
        # Per-seed window sampling. Subsample seeds are disjoint from
        # the substrate / shuffle / control RNG streams used above.
        train_windows = sample_windows(
            all_train_windows,
            min(n_train_windows, len(all_train_windows)),
            seed=seed + 50000,
        )
        test_windows = sample_windows(
            all_test_windows,
            min(n_test_windows, len(all_test_windows)),
            seed=seed + 60000,
        )
    if landscape_size > len(train_windows):
        landscape_size = len(train_windows)
    landscape_windows = train_windows[:landscape_size]

    # Memorize the landscape into Hopfield memory.
    memory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(landscape_windows):
        memory.store(
            encode_window(substrate, positions, codebook, list(window)),
            label=f"window_{index}",
        )

    # Compute regime diagnostics on the *initial* (pre-consolidation)
    # codebook. For ``random`` standard-mode this is the only diagnostic.
    # For ``consolidated`` standard-mode this is the BEFORE snapshot.
    theta_fn, theta_info = _build_theta_prime_fn(theta_prime_mode, repo_root)
    regime_diag_before: CodebookRegimeDiagnostics = (
        compute_codebook_regime_diagnostics(
            codebook,
            k_nn=min(8, vocab_size - 1),
            beta=beta,
            theta_prime_fn=theta_fn,
        )
    )

    # Phase 3 consolidation pass.
    #
    # Path α policy: the standard condition runs consolidation when
    # standard_mode == 'consolidated'. The control runs consolidation
    # when control_mode == 'shuffled-token' (so the comparison is
    # consolidated-vs-consolidated with only the codebook permutation
    # differing) and skips it when control_mode == 'random' (legacy
    # no-consolidation control kept for backward compat). The two
    # condition-vs-control matchups are:
    #   standard='consolidated' × control='shuffled-token'
    #     → real Phase 3 graduation gate
    #   standard='consolidated' × control='random'
    #     → legacy consolidated-vs-fresh-random (the 2026-05-26 smoke's
    #       methodology gap)
    #   standard='random'       × *                       → pre-Path-C
    consolidation_stats: Optional[Dict[str, object]] = None
    run_consolidation = False
    if (not is_control) and standard_mode == "consolidated":
        run_consolidation = True
    if is_control and control_mode == "shuffled-token":
        run_consolidation = True
    if run_consolidation:
        # Consolidation training corpus: drawn from the training pool but
        # disjoint from the landscape (already memorized) and from the
        # test windows (drawn from the corpus RNG after train). Slice
        # train_windows[landscape_size:] to guarantee train != landscape;
        # corpus_rng has already advanced past these for test_windows.
        cons_train = list(train_windows[landscape_size:]) or list(train_windows)
        codebook, _cstate, _cupd, consolidation_stats = _consolidate_codebook(
            substrate=substrate,
            codebook=codebook,
            positions=positions,
            landscape_windows=landscape_windows,
            memory=memory,
            train_windows=cons_train,
            window_size=window_size,
            beta=beta,
            vocab_size=vocab_size,
            n_events=n_consolidation_events,
            device=device,
            lr_pull=lr_pull,
            lr_push=lr_push,
            repulsion_step_size=repulsion_step_size,
            use_context_residual=use_context_residual,
            lr_cr=lr_cr,
            use_pull_push=use_pull_push,
        )

    # Recompute regime diagnostics on the (possibly consolidated) codebook
    # — this is the *evaluation-time* codebook regime.
    regime_diag: CodebookRegimeDiagnostics = compute_codebook_regime_diagnostics(
        codebook, k_nn=min(8, vocab_size - 1), beta=beta, theta_prime_fn=theta_fn,
    )
    regime_labels: Dict[int, str] = {
        atom_id: ar.regime for atom_id, ar in regime_diag.per_atom.items()
    }

    # Recall@K with the masked position at the last index of each window.
    masked_idx = window_size - 1
    outcomes = _evaluate_recall_at_k(
        substrate=substrate,
        memory=memory,
        codebook=codebook,
        positions=positions,
        test_windows=test_windows,
        masked_idx=masked_idx,
        beta=beta,
        k=k,
        regime_labels=regime_labels,
    )

    per_stratum = _aggregate_by_stratum(outcomes)

    return {
        "seed": seed,
        "substrate_seed": substrate_seed,
        "is_control": is_control,
        "control_mode": control_mode if is_control else None,
        "shuffled_token_permutation_seed": (
            (seed + 70000) if (is_control and control_mode == "shuffled-token") else None
        ),
        "alpha_anti": float(alpha_anti),
        "repulsion_step_size": float(repulsion_step_size),
        "standard_mode": standard_mode,
        "theta_prime_mode": theta_prime_mode,
        "theta_prime_info": theta_info,
        # Evaluation-time regime (after consolidation if applied; else
        # identical to "before").
        "regime_counts": dict(regime_diag.regime_counts),
        "regime_summary": regime_diag.summary,
        # Pre-consolidation regime — for the consolidated standard
        # condition, lets us see whether the C.2 dynamics moved any
        # atoms out of 'spread' into 'tight' / 'borderline'.
        "regime_counts_before_consolidation": dict(
            regime_diag_before.regime_counts
        ),
        "regime_summary_before_consolidation": regime_diag_before.summary,
        "consolidation_stats": consolidation_stats,
        "per_stratum": {
            stratum: {
                "successes": int(s),
                "trials": int(t),
                "recall_at_k": (s / t) if t > 0 else 0.0,
            }
            for stratum, (s, t) in per_stratum.items()
        },
        "n_outcomes": len(outcomes),
    }


def run(
    *,
    seeds: Sequence[int],
    D: int,
    landscape_size: int,
    window_size: int,
    n_test_windows: int,
    n_train_windows: int,
    vocab_size: int,
    k: int,
    beta: float,
    theta_prime_mode: str,
    standard_mode: str = "consolidated",
    control_mode: str = "shuffled-token",
    n_consolidation_events: int = 1000,
    alpha_anti: float = 0.0,
    repulsion_step_size: float = 0.0,
    lr_pull: float = 0.1,
    lr_push: float = 0.05,
    use_context_residual: bool = False,
    lr_cr: float = 0.1,
    use_pull_push: bool = True,
    device: str,
    output_dir: Path,
    repo_root: Path,
    corpus_source: str = "synthetic",
    wikitext_name: str = "wikitext-2-raw-v1",
    vocab_cap: int = 1000,
    wikitext_corpus: Optional[_WikiTextCorpus] = None,
) -> dict:
    """Run the full C.3 experiment.

    Iterates over ``theta_prime_mode`` ∈ {default, calibrated} when
    ``theta_prime_mode == 'both'``. For each (mode, seed) pair runs the
    main and shuffled-token-control condition. Returns the assembled
    summary dict written to ``c3_summary.json``.
    """
    if theta_prime_mode == "both":
        modes = list(THETA_PRIME_MODES)
    else:
        modes = [theta_prime_mode]

    if corpus_source not in CORPUS_SOURCES:
        raise ValueError(
            f"unknown corpus_source: {corpus_source!r}; expected one of "
            f"{CORPUS_SOURCES}."
        )

    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load WikiText-2 once for the whole run if requested. The vocabulary
    # is built from the train split; vocab_size used by everything
    # downstream becomes ``len(vocab.id_to_token)`` (= vocab_cap + 2
    # special tokens — <UNK>, <MASK>). The caller can also inject a
    # pre-built ``wikitext_corpus`` (used by tests to mock the loader).
    effective_vocab_size = vocab_size
    corpus_info: Dict[str, object] = {
        "corpus_source": corpus_source,
        "wikitext_name": wikitext_name if corpus_source == "wikitext" else None,
        "vocab_cap": vocab_cap if corpus_source == "wikitext" else None,
    }
    if corpus_source == "wikitext" and wikitext_corpus is None:
        wikitext_corpus = _load_wikitext_corpus(
            repo_root=repo_root,
            wikitext_name=wikitext_name,
            vocab_cap=vocab_cap,
        )
    if wikitext_corpus is not None:
        effective_vocab_size = wikitext_corpus.vocab_size
        corpus_info.update(
            {
                "effective_vocab_size": effective_vocab_size,
                "n_train_tokens": len(wikitext_corpus.train_ids),
                "n_val_tokens": len(wikitext_corpus.val_ids),
                "n_test_tokens": len(wikitext_corpus.test_ids),
                "unk_token_id": int(wikitext_corpus.vocab.unk_id),
                "mask_token_id": int(wikitext_corpus.vocab.mask_id),
            }
        )

    per_cell_rows: List[Dict[str, object]] = []

    for mode in modes:
        for seed in seeds:
            for is_control in (False, True):
                row = _run_single_seed_condition(
                    seed=seed,
                    is_control=is_control,
                    theta_prime_mode=mode,
                    standard_mode=standard_mode,
                    control_mode=control_mode,
                    n_consolidation_events=n_consolidation_events,
                    D=D,
                    landscape_size=landscape_size,
                    window_size=window_size,
                    n_test_windows=n_test_windows,
                    vocab_size=effective_vocab_size,
                    n_train_windows=n_train_windows,
                    beta=beta,
                    k=k,
                    alpha_anti=alpha_anti,
                    repulsion_step_size=repulsion_step_size,
                    lr_pull=lr_pull,
                    lr_push=lr_push,
                    use_context_residual=use_context_residual,
                    lr_cr=lr_cr,
                    use_pull_push=use_pull_push,
                    device=device,
                    repo_root=repo_root,
                    wikitext_corpus=wikitext_corpus,
                )
                per_cell_rows.append(row)

    # Aggregate per (mode, condition, stratum) across seeds. Pool all
    # trial counts across seeds before computing the Wilson interval;
    # this is the per-stratum n.
    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in modes:
        aggregated[mode] = {}
        for is_control in (False, True):
            condition_key = "shuffled_control" if is_control else "standard"
            aggregated[mode][condition_key] = {}
            for stratum in STRATA:
                tot_s = 0
                tot_t = 0
                for row in per_cell_rows:
                    if row["theta_prime_mode"] != mode:
                        continue
                    if row["is_control"] != is_control:
                        continue
                    cell = row["per_stratum"][stratum]
                    tot_s += int(cell["successes"])
                    tot_t += int(cell["trials"])
                mean_v, lo, hi = _wilson(tot_s, tot_t)
                aggregated[mode][condition_key][stratum] = {
                    "successes": tot_s,
                    "trials": tot_t,
                    "recall_at_k": mean_v,
                    "wilson_lower": lo,
                    "wilson_upper": hi,
                }

        # Per-stratum delta and CI: delta is standard - shuffled_control.
        # CI for the delta is the standard's CI shifted by the control's
        # point estimate — *not* a paired test (different test windows
        # under each codebook draw share the same indices but evaluate
        # under different codebooks, which makes a clean paired analysis
        # subtle). For the smoke we report both CIs honestly and the
        # point-estimate delta.
        aggregated[mode]["delta_standard_minus_control"] = {}
        for stratum in STRATA:
            std_cell = aggregated[mode]["standard"][stratum]
            ctrl_cell = aggregated[mode]["shuffled_control"][stratum]
            delta = std_cell["recall_at_k"] - ctrl_cell["recall_at_k"]
            aggregated[mode]["delta_standard_minus_control"][stratum] = {
                "delta_recall_at_k": delta,
                "standard_trials": std_cell["trials"],
                "control_trials": ctrl_cell["trials"],
                # Disjoint-CI gate: standard's lower > control's upper.
                "ci_disjoint_standard_beats_control": (
                    std_cell["wilson_lower"] > ctrl_cell["wilson_upper"]
                ),
            }

    summary = {
        "header": {
            "diagnostic_not_graduation": len(seeds) < 10,
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "phase_3_exit_criterion_source": (
                "notes/emergent-codebook/phase-3-deep-dive.md:180-189"
            ),
            "path_c_precommit_source": (
                "notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md"
            ),
            "headline_metric": (
                "Regime-stratified Recall@K on masked-token contextual "
                "completion vs. genuine shuffled-token control"
            ),
            "graduation_gate_n_seeds": 10,
            "standard_mode": standard_mode,
            "control_mode": control_mode,
            "n_consolidation_events": (
                n_consolidation_events if standard_mode == "consolidated" else 0
            ),
            "c2_config": (
                dict(C2_DEFAULTS) if standard_mode == "consolidated" else None
            ),
            # Path α (2026-05-26): the inter-atom separability half of
            # the 2026-05-09 NC1/NC2 reformulation. Both must be > 0 to
            # fire, mirroring the gate at replay_loop.py:780-781. The
            # consolidation pipeline applies repulsion to the full
            # codebook matrix after each consolidate_if_ready() event.
            "alpha_anti": float(alpha_anti),
            "repulsion_step_size": float(repulsion_step_size),
            "substrate_repulsion_active": bool(
                alpha_anti > 0.0 and repulsion_step_size > 0.0
            ),
            "lr_pull": float(lr_pull),
            "lr_push": float(lr_push),
            "use_context_residual": bool(use_context_residual),
            "lr_cr": float(lr_cr),
            "use_pull_push": bool(use_pull_push),
            "operating_point": {
                "D": D,
                "landscape_size": landscape_size,
                "window_size": window_size,
                "vocab_size": effective_vocab_size,
                "n_train_windows": n_train_windows,
                "n_test_windows": n_test_windows,
                "beta": beta,
                "K": k,
            },
            "corpus": corpus_info,
            "theta_prime_modes_run": modes,
            "device": device,
            "wall_clock_seconds": None,  # filled in below
        },
        "per_cell_rows": per_cell_rows,
        "aggregated": aggregated,
    }

    elapsed = time.time() - start
    summary["header"]["wall_clock_seconds"] = elapsed

    return summary


# Reporting --------------------------------------------------------------------

def _format_markdown(summary: dict) -> str:
    header = summary["header"]
    aggregated = summary["aggregated"]
    lines: List[str] = []
    lines.append("# C.3 — Phase 3 Exit Criterion Re-run")
    lines.append("")
    if header["diagnostic_not_graduation"]:
        lines.append(
            "**DIAGNOSTIC, NOT GRADUATION EVIDENCE.** "
            f"This run used n_seeds={header['n_seeds']} (< 10). The "
            "Phase 3 exit criterion requires n ≥ 10 seeds; the smoke "
            "output below is diagnostic-only and CIs are wide."
        )
    else:
        lines.append(
            f"Run with n_seeds={header['n_seeds']} (≥ 10). "
            "Graduation gate eligible if at least one stratum's standard "
            "Wilson lower > control Wilson upper."
        )
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    op = header["operating_point"]
    lines.append(f"- D = `{op['D']}` (Phase 2 baseline envelope)")
    lines.append(f"- landscape_size = `{op['landscape_size']}`")
    lines.append(f"- window_size = `{op['window_size']}`")
    lines.append(f"- vocab_size = `{op['vocab_size']}`")
    lines.append(f"- n_train_windows = `{op['n_train_windows']}`")
    lines.append(f"- n_test_windows = `{op['n_test_windows']}`")
    lines.append(f"- β = `{op['beta']}`")
    lines.append(f"- K = `{op['K']}`")
    lines.append(f"- seeds = `{header['seeds']}`")
    lines.append(f"- theta_prime_modes = `{header['theta_prime_modes_run']}`")
    lines.append(
        f"- standard_mode = `{header.get('standard_mode', 'random')}`"
    )
    lines.append(
        f"- control_mode = `{header.get('control_mode', 'random')}`"
    )
    if header.get("standard_mode") == "consolidated":
        lines.append(
            f"- n_consolidation_events = `{header['n_consolidation_events']}`"
        )
        lines.append(
            f"- C.2 dynamics config = `{header.get('c2_config')}`"
        )
    lines.append(
        f"- alpha_anti = `{header.get('alpha_anti', 0.0)}`"
        f"  repulsion_step_size = `{header.get('repulsion_step_size', 0.0)}`"
        f"  substrate_repulsion_active = `{header.get('substrate_repulsion_active', False)}`"
    )
    lines.append(f"- device = `{header['device']}`")
    lines.append(f"- wall_clock = `{header['wall_clock_seconds']:.1f}s`")
    corpus = header.get("corpus")
    if corpus is not None:
        lines.append(f"- corpus_source = `{corpus.get('corpus_source')}`")
        if corpus.get("corpus_source") == "wikitext":
            lines.append(f"- wikitext_name = `{corpus.get('wikitext_name')}`")
            lines.append(f"- vocab_cap = `{corpus.get('vocab_cap')}`")
            lines.append(
                f"- effective_vocab_size = `{corpus.get('effective_vocab_size')}`"
            )
            lines.append(f"- n_train_tokens = `{corpus.get('n_train_tokens')}`")
            lines.append(f"- n_val_tokens = `{corpus.get('n_val_tokens')}`")
            lines.append(f"- n_test_tokens = `{corpus.get('n_test_tokens')}`")
    lines.append("")
    lines.append("## Headline table — per-mode, per-stratum")
    lines.append("")
    lines.append(
        "| Mode | Stratum | Standard Recall@K [Wilson CI] | "
        "Shuffled-control Recall@K [Wilson CI] | Δ (std − ctrl) | "
        "CI-disjoint (std lower > ctrl upper)? | n_std | n_ctrl |"
    )
    lines.append(
        "|---|---|---|---|---:|:--:|---:|---:|"
    )
    for mode in header["theta_prime_modes_run"]:
        for stratum in STRATA:
            std = aggregated[mode]["standard"][stratum]
            ctrl = aggregated[mode]["shuffled_control"][stratum]
            dlt = aggregated[mode]["delta_standard_minus_control"][stratum]
            std_cell = (
                f"{std['recall_at_k']:.3f} "
                f"[{std['wilson_lower']:.3f}, {std['wilson_upper']:.3f}]"
            )
            ctrl_cell = (
                f"{ctrl['recall_at_k']:.3f} "
                f"[{ctrl['wilson_lower']:.3f}, {ctrl['wilson_upper']:.3f}]"
            )
            lines.append(
                f"| {mode} | {stratum} | {std_cell} | {ctrl_cell} | "
                f"{dlt['delta_recall_at_k']:+.3f} | "
                f"{'YES' if dlt['ci_disjoint_standard_beats_control'] else 'no'} | "
                f"{std['trials']} | {ctrl['trials']} |"
            )
    lines.append("")
    lines.append("## Regime classifier agreement diagnostic")
    lines.append("")
    lines.append(
        "Per the C.1.4 calibration finding (`1/β` off by 2-3 orders at "
        "low β), the regime classifier *can* disagree between the "
        "`default` (1/β) and `calibrated` modes. At β=10 the "
        "calibration JSON does not cover the operating point (its grid "
        "is β ∈ {0.01, 0.1, 1.0}), so the calibrated loader falls "
        "back to 1/β with a stderr warning; the two modes therefore "
        "agree numerically at this β."
    )
    lines.append("")
    # Regime BEFORE vs AFTER consolidation — only meaningful for the
    # consolidated standard condition.
    if header.get("standard_mode") == "consolidated":
        lines.append(
            "## Regime distribution BEFORE vs AFTER consolidation (STD + CTRL)"
        )
        lines.append("")
        lines.append(
            "| Seed | Cond | Mode | Before (t/s/b) | After (t/s/b) | "
            "Δtight | Δspread | Δborderline | cons fired | repulsion fires |"
        )
        lines.append("|---:|:--:|---|---|---|---:|---:|---:|---:|---:|")
        for row in summary["per_cell_rows"]:
            rb = row.get("regime_counts_before_consolidation", {})
            ra = row.get("regime_counts", {})
            cs = row.get("consolidation_stats") or {}
            d_tight = ra.get("tight", 0) - rb.get("tight", 0)
            d_spread = ra.get("spread", 0) - rb.get("spread", 0)
            d_border = ra.get("borderline", 0) - rb.get("borderline", 0)
            cond = "CTRL" if row["is_control"] else "STD"
            lines.append(
                f"| {row['seed']} | {cond} | {row['theta_prime_mode']} | "
                f"{rb.get('tight',0)}/{rb.get('spread',0)}/{rb.get('borderline',0)} | "
                f"{ra.get('tight',0)}/{ra.get('spread',0)}/{ra.get('borderline',0)} | "
                f"{d_tight:+d} | {d_spread:+d} | {d_border:+d} | "
                f"{cs.get('consolidations_fired', 0)} | "
                f"{cs.get('repulsion_applications', 0)} |"
            )
        lines.append("")

    lines.append("## Per-cell rows (per-seed)")
    lines.append("")
    lines.append(
        "| Seed | Substrate seed | Control? | Mode | tight/spread/borderline counts | tight Recall@K | spread Recall@K | borderline Recall@K |"
    )
    lines.append("|---:|---:|:--:|---|---|---:|---:|---:|")
    for row in summary["per_cell_rows"]:
        rc = row["regime_counts"]
        ps = row["per_stratum"]
        lines.append(
            f"| {row['seed']} | {row['substrate_seed']} | "
            f"{'CTRL' if row['is_control'] else 'STD'} | "
            f"{row['theta_prime_mode']} | "
            f"{rc.get('tight',0)}/{rc.get('spread',0)}/{rc.get('borderline',0)} | "
            f"{ps['tight']['recall_at_k']:.3f} ({ps['tight']['trials']}) | "
            f"{ps['spread']['recall_at_k']:.3f} ({ps['spread']['trials']}) | "
            f"{ps['borderline']['recall_at_k']:.3f} ({ps['borderline']['trials']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_stdout_table(summary: dict) -> str:
    """A short ASCII table of the headline for stdout."""
    header = summary["header"]
    aggregated = summary["aggregated"]
    lines: List[str] = []
    lines.append("=== C.3 HEADLINE ===")
    if header["diagnostic_not_graduation"]:
        lines.append(
            f"[diagnostic, n_seeds={header['n_seeds']} < 10 — not graduation]"
        )
    lines.append(
        f"D={header['operating_point']['D']} "
        f"L={header['operating_point']['landscape_size']} "
        f"W={header['operating_point']['window_size']} "
        f"vocab={header['operating_point']['vocab_size']} "
        f"test_windows={header['operating_point']['n_test_windows']} "
        f"K={header['operating_point']['K']} β={header['operating_point']['beta']}"
    )
    sm = header.get("standard_mode", "random")
    cm = header.get("control_mode", "random")
    lines.append(
        f"standard_mode={sm}  control_mode={cm}"
        + (
            f"  n_consolidation_events={header['n_consolidation_events']}"
            if sm == "consolidated"
            else ""
        )
    )
    lines.append(
        f"alpha_anti={header.get('alpha_anti', 0.0)}  "
        f"repulsion_step_size={header.get('repulsion_step_size', 0.0)}  "
        f"substrate_repulsion_active={header.get('substrate_repulsion_active', False)}"
    )
    corpus = header.get("corpus")
    if corpus is not None:
        src = corpus.get("corpus_source")
        if src == "wikitext":
            lines.append(
                f"corpus=wikitext({corpus.get('wikitext_name')}) "
                f"vocab_cap={corpus.get('vocab_cap')} "
                f"effective_vocab={corpus.get('effective_vocab_size')} "
                f"n_train_tok={corpus.get('n_train_tokens')} "
                f"n_val_tok={corpus.get('n_val_tokens')} "
                f"n_test_tok={corpus.get('n_test_tokens')}"
            )
        else:
            lines.append(f"corpus={src}")
    for mode in header["theta_prime_modes_run"]:
        lines.append(f"\n  theta_prime_mode = {mode}")
        lines.append(
            f"    {'stratum':<12} {'std R@K':>9}  {'std CI':>16}  "
            f"{'ctrl R@K':>9}  {'ctrl CI':>16}  {'Δ':>7}  disjoint?"
        )
        for stratum in STRATA:
            std = aggregated[mode]["standard"][stratum]
            ctrl = aggregated[mode]["shuffled_control"][stratum]
            dlt = aggregated[mode]["delta_standard_minus_control"][stratum]
            std_ci = f"[{std['wilson_lower']:.3f},{std['wilson_upper']:.3f}]"
            ctrl_ci = f"[{ctrl['wilson_lower']:.3f},{ctrl['wilson_upper']:.3f}]"
            lines.append(
                f"    {stratum:<12} {std['recall_at_k']:>9.3f}  {std_ci:>16}  "
                f"{ctrl['recall_at_k']:>9.3f}  {ctrl_ci:>16}  "
                f"{dlt['delta_recall_at_k']:>+7.3f}  "
                f"{'YES' if dlt['ci_disjoint_standard_beats_control'] else 'no'}"
            )
    return "\n".join(lines)


def write_outputs(summary: dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "c3_summary.json"
    md_path = output_dir / "c3_summary.md"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    md_path.write_text(_format_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# CLI --------------------------------------------------------------------------

def _parse_seeds(raw: str) -> List[int]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        raise ValueError("expected at least one seed")
    return [int(p) for p in parts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C.3 — Phase 3 exit-criterion re-run.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--n-test-windows", type=int, default=200)
    parser.add_argument("--n-train-windows", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=200)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument(
        "--theta-prime-mode",
        choices=["default", "calibrated", "both"],
        default="both",
    )
    parser.add_argument(
        "--standard-mode",
        choices=list(STANDARD_MODES),
        default="consolidated",
        help=(
            "Composition of the 'standard' (non-control) condition's "
            "codebook. 'random' = fresh random codebook (the existing "
            "smoke / backward-compat behavior — methodology gap noted in "
            "the 2026-05-26 C.3 partial smoke). 'consolidated' = run "
            "Phase 3 consolidation orchestrator (OnlineCodebookUpdater + "
            "ConsolidationState with all 5 C.2 dynamics on) over "
            "--n-consolidation-events retrievals, then evaluate."
        ),
    )
    parser.add_argument(
        "--n-consolidation-events",
        type=int,
        default=1000,
        help=(
            "Number of training-window retrievals to drive through the "
            "consolidation orchestrator before evaluating "
            "(--standard-mode consolidated only)."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=list(CONTROL_MODES),
        default="shuffled-token",
        help=(
            "Operationalization of the control condition. "
            "'shuffled-token' (Path α default) shares the standard "
            "condition's substrate seed and atom set but permutes the "
            "row order of the codebook (token-id π(i) gets atom i) and "
            "runs the SAME consolidation pipeline over the SAME training "
            "corpus. This is the proper Phase 3 shuffled-token control "
            "per notes/emergent-codebook/phase-3-deep-dive.md:217-218. "
            "'random' (legacy backward-compat) uses a disjoint substrate "
            "seed (seed + 10000) and skips consolidation — the 2026-05-26 "
            "smoke's methodology gap."
        ),
    )
    parser.add_argument(
        "--alpha-anti",
        type=float,
        default=0.01,
        help=(
            "Substrate-side anti-collapse strength α for the "
            "H_anti = -α·log(d_eff) repulsion energy. Both alpha_anti > 0 "
            "AND --repulsion-step-size > 0 must hold for the repulsion "
            "force to fire on the codebook after each consolidation event "
            "(mirrors src/energy_memory/phase4/replay_loop.py:780-798). "
            "Default 0.01 is the modest Path α value chosen for the first "
            "D=4096 exercise at the Phase 2 operating point; the existing "
            "test_phase5_ab_death_dynamic.py exercises 1.0 at D=512. The "
            "audit's §4.1 [F] flagged alpha_anti as 'wired but "
            "underspecified' before this driver — this is the first D=4096 "
            "exercise. Set to 0.0 to reproduce the 2026-05-26 partial "
            "smoke (no inter-atom-separability force)."
        ),
    )
    parser.add_argument(
        "--lr-pull",
        type=float,
        default=0.1,
        help=(
            "Per-event consolidation pull learning rate (OnlineCodebookUpdater "
            "lr_pull). Default 0.1 matches the existing Path α smoke. Sweep "
            "above this to test whether consolidation strength is too weak "
            "to express corpus-specific learning at the synthetic operating "
            "point."
        ),
    )
    parser.add_argument(
        "--lr-push",
        type=float,
        default=0.05,
        help=(
            "Per-event consolidation push learning rate (OnlineCodebookUpdater "
            "lr_push). Default 0.05 matches the existing Path α smoke."
        ),
    )
    parser.add_argument(
        "--use-context-residual",
        action="store_true",
        help=(
            "Activate Γ1.c context-residual consolidation as the base "
            "update (Path γ leader candidate). Per the precommit at "
            "notes/notes/2026-05-27-path-gamma-gamma1-context-residual-"
            "precommit.md: gradient descent on the per-event repulsion "
            "energy E_cr over confused atom pairs. Default off preserves "
            "Path C reproducibility byte-identically. The Γ1 headline "
            "condition sets this flag AND --no-pull-push."
        ),
    )
    parser.add_argument(
        "--lr-cr",
        type=float,
        default=0.1,
        help=(
            "Γ1.c context-residual learning rate. Default 0.1 matches "
            "Path C's lr_pull and is the single value pre-committed for "
            "the Γ1 headline gate (no sweep at headline scale per H3 in "
            "the precommit)."
        ),
    )
    parser.add_argument(
        "--no-pull-push",
        action="store_true",
        help=(
            "Disable the pull/push base update at the OnlineCodebookUpdater "
            "level. The Γ1 headline condition sets this flag together "
            "with --use-context-residual. Default off (pull/push active) "
            "preserves Path C reproducibility."
        ),
    )
    parser.add_argument(
        "--repulsion-step-size",
        type=float,
        default=0.05,
        help=(
            "Step size for the per-consolidation-event substrate update "
            "under the H_anti = -α·log(d_eff) gradient. Companion to "
            "--alpha-anti. Default 0.05 is modest; the existing "
            "test_phase5_ab_death_dynamic.py exercises 50.0 at D=256."
        ),
    )
    parser.add_argument(
        "--corpus-source",
        choices=list(CORPUS_SOURCES),
        default="synthetic",
        help=(
            "Source corpus for training and held-out windows. "
            "'synthetic' (default, Path α / pre-Path β) uses uniform-"
            "random token-id windows over [0, vocab_size). 'wikitext' "
            "(Path β) loads WikiText-2 via the same loader Phase 2 "
            "uses (experiments/02_phase2_retrieval_baseline.py:55-58), "
            "builds a top --vocab-cap vocabulary from the train split "
            "(with <UNK>+<MASK> special tokens), and uses sliding-"
            "window extraction from the train split for training/"
            "consolidation and from the val+test split for held-out "
            "test windows. The shuffled-token control still applies a "
            "random row permutation to the codebook — it tests whether "
            "the standard condition's edge depends on the corpus-"
            "specific token-to-hypervector assignment."
        ),
    )
    parser.add_argument(
        "--wikitext-name",
        default="wikitext-2-raw-v1",
        help="HuggingFace WikiText config name (default wikitext-2-raw-v1).",
    )
    parser.add_argument(
        "--vocab-cap",
        type=int,
        default=1000,
        help=(
            "For --corpus-source wikitext, cap vocabulary to the top-K "
            "most frequent train-split tokens (default 1000). The "
            "effective vocab_size (incl. <UNK>+<MASK>) becomes "
            "--vocab-cap + 2. Out-of-vocab tokens map to <UNK>. "
            "Ignored when --corpus-source synthetic."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Compute device. Default auto-detects MPS if available, "
            "else CPU. Pass 'cpu' explicitly to force CPU."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write c3_summary.{json,md} into. Default: "
            "reports/c3_smoke_<timestamp>/."
        ),
    )

    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    seeds = _parse_seeds(args.seeds)
    if args.device is None:
        args.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "reports" / f"c3_smoke_{ts}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir

    summary = run(
        seeds=seeds,
        D=args.D,
        landscape_size=args.landscape_size,
        window_size=args.window,
        n_test_windows=args.n_test_windows,
        n_train_windows=args.n_train_windows,
        vocab_size=args.vocab_size,
        k=args.K,
        beta=args.beta,
        theta_prime_mode=args.theta_prime_mode,
        standard_mode=args.standard_mode,
        control_mode=args.control_mode,
        n_consolidation_events=args.n_consolidation_events,
        alpha_anti=args.alpha_anti,
        repulsion_step_size=args.repulsion_step_size,
        lr_pull=args.lr_pull,
        lr_push=args.lr_push,
        use_context_residual=args.use_context_residual,
        lr_cr=args.lr_cr,
        use_pull_push=not args.no_pull_push,
        device=args.device,
        output_dir=output_dir,
        repo_root=repo_root,
        corpus_source=args.corpus_source,
        wikitext_name=args.wikitext_name,
        vocab_cap=args.vocab_cap,
    )
    json_path, md_path = write_outputs(summary, output_dir)
    print(_format_stdout_table(summary))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
