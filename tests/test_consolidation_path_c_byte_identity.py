"""Baseline parity test — Γ1.c refactor preserves Path C byte-identity.

Per the precommit at
notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md
§"Required pre-code gates" #3. The default config flags
(use_pull_push=True, use_context_residual=False) must produce
byte-identical results to the pre-Γ1.c pull/push implementation on a
representative buffer / seed set.

Strategy: compute the expected post-consolidation codebook by reproducing
the original pull/push math inline, then assert torch.allclose at
float32 numerical tolerance. This locks down behavior without needing
a snapshot fixture committed to the repo.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _expected_pull_push(substrate, codebook, buffer_entries, lr_pull, lr_push):
    """Reproduce the original pull/push math from online_codebook.py.

    Returns a new codebook tensor (copy; doesn't mutate the input).
    Mirrors the exact ordering and normalization of the pre-Γ1.c
    implementation at L144-164 of the original file.
    """
    from collections import defaultdict

    cb = codebook.detach().clone()
    pull_targets = defaultdict(list)
    push_targets = defaultdict(list)
    for entry in buffer_entries:
        pull_targets[entry["target_id"]].append(entry["slot_query"])
        if entry["predicted_id"] != entry["target_id"]:
            push_targets[entry["predicted_id"]].append(entry["slot_query"])

    for tid, queries in pull_targets.items():
        avg_dir = substrate.normalize(torch.stack(queries).sum(dim=0))
        cb[tid] = substrate.normalize(
            (1.0 - lr_pull) * cb[tid] + lr_pull * avg_dir
        )
    for wid, queries in push_targets.items():
        avg_dir = substrate.normalize(torch.stack(queries).sum(dim=0))
        cb[wid] = substrate.normalize(
            (1.0 + lr_push) * cb[wid] - lr_push * avg_dir
        )
    return cb


@unittest.skipIf(torch is None, "torch required")
class TestPathCByteIdentity(unittest.TestCase):

    def test_default_flags_match_inline_pull_push_math(self):
        """The cleanest possible baseline parity check: compute expected
        pull/push output inline, assert the refactored code matches.
        """
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase34.online_codebook import (
            OnlineCodebookUpdater, _BufferedFailure,
        )

        substrate = TorchFHRR(dim=128, seed=42, device="cpu")
        codebook = substrate.random_vectors(20)
        original_codebook = codebook.clone()

        # Fixed buffer: a mix of (target, predicted) pairs covering
        # correct retrievals (predicted == target), wrong retrievals,
        # and multiple events per target.
        buffer_spec = [
            {"target_id": 3, "predicted_id": 7, "slot_query": substrate.random_vector()},
            {"target_id": 3, "predicted_id": 7, "slot_query": substrate.random_vector()},
            {"target_id": 5, "predicted_id": 5, "slot_query": substrate.random_vector()},
            {"target_id": 7, "predicted_id": 3, "slot_query": substrate.random_vector()},
            {"target_id": 9, "predicted_id": 14, "slot_query": substrate.random_vector()},
        ]

        expected_codebook = _expected_pull_push(
            substrate, original_codebook, buffer_spec,
            lr_pull=0.1, lr_push=0.05,
        )

        # Run through the refactored updater with default flags.
        updater = OnlineCodebookUpdater(
            substrate=substrate,
            codebook=codebook,
            lr_pull=0.1,
            lr_push=0.05,
            consolidation_k=len(buffer_spec),
            quality_threshold=0.5,
        )
        for spec in buffer_spec:
            updater._buffer.append(_BufferedFailure(
                target_id=spec["target_id"],
                predicted_id=spec["predicted_id"],
                slot_query=spec["slot_query"],
                quality=0.0,
            ))
        updater.force_consolidate()

        self.assertTrue(
            torch.allclose(codebook, expected_codebook, atol=1e-7),
            f"max diff: {(codebook - expected_codebook).abs().max().item()}",
        )

    def test_default_flags_no_context_residual_movement(self):
        """When use_pull_push=True, use_context_residual=False (defaults),
        cr_updated == 0 and the diagnostics report no Γ1.c activity.
        """
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase34.online_codebook import (
            OnlineCodebookUpdater, _BufferedFailure,
        )

        substrate = TorchFHRR(dim=64, seed=17, device="cpu")
        codebook = substrate.random_vectors(10)
        updater = OnlineCodebookUpdater(
            substrate=substrate,
            codebook=codebook,
            consolidation_k=3,
            quality_threshold=0.5,
        )
        for _ in range(3):
            updater._buffer.append(_BufferedFailure(
                target_id=2,
                predicted_id=5,
                slot_query=substrate.random_vector(),
                quality=0.0,
            ))
        diag = updater.force_consolidate()
        self.assertEqual(diag["context_residual_updated"], 0)
        self.assertGreater(diag["pulled"], 0)
        self.assertGreater(diag["pushed"], 0)

    def test_both_flags_on_compose_additively(self):
        """When both use_pull_push=True AND use_context_residual=True,
        both forces fire; the result differs from either alone. This is
        the precommit's claim that composition is additive, not switched.
        """
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase34.online_codebook import (
            OnlineCodebookUpdater, _BufferedFailure,
        )

        substrate = TorchFHRR(dim=64, seed=23, device="cpu")
        base_codebook = substrate.random_vectors(10)

        def _run(use_pull_push, use_context_residual):
            cb = base_codebook.clone()
            upd = OnlineCodebookUpdater(
                substrate=substrate,
                codebook=cb,
                consolidation_k=3,
                quality_threshold=0.5,
                use_pull_push=use_pull_push,
                use_context_residual=use_context_residual,
                lr_cr=0.1,
            )
            torch.manual_seed(0)
            queries = [substrate.random_vector() for _ in range(3)]
            for q in queries:
                upd._buffer.append(_BufferedFailure(
                    target_id=2, predicted_id=5,
                    slot_query=q, quality=0.0,
                ))
            upd.force_consolidate()
            return cb

        pull_only = _run(use_pull_push=True, use_context_residual=False)
        cr_only = _run(use_pull_push=False, use_context_residual=True)
        both = _run(use_pull_push=True, use_context_residual=True)

        # All three should differ from each other on codebook[2]
        # (pull/push moves codebook[2] toward avg query;
        # Γ1.c moves it toward codebook[2] + lr_cr * (cb[2] − cb[5]);
        # both moves it under composed forces — additive vectors).
        self.assertFalse(
            torch.allclose(pull_only[2], cr_only[2], atol=1e-4),
            "pull-only and cr-only produced identical codebook[2]",
        )
        self.assertFalse(
            torch.allclose(pull_only[2], both[2], atol=1e-4),
            "pull-only and both produced identical codebook[2]",
        )
        self.assertFalse(
            torch.allclose(cr_only[2], both[2], atol=1e-4),
            "cr-only and both produced identical codebook[2]",
        )


if __name__ == "__main__":
    unittest.main()
